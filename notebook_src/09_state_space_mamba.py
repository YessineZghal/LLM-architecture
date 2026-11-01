# %% [md]
# # 09 · State-space models: Mamba
#
# An SSM keeps a hidden state like an RNN, but is defined by a continuous linear system discretised with a step size Δ:
#
# `h_t = Ā h_{t-1} + B̄ x_t`,  `y_t = C h_t`  with  `Ā = exp(ΔA)`.
#
# **Mamba's insight (selectivity):** make Δ, B and C *functions of the input*, so the model can decide per token what to
# remember or forget (large Δ → "read this", small Δ → "ignore it"). Linear-time, constant memory, no attention at all.
# %%
import time
import torch, torch.nn as nn, torch.nn.functional as F, matplotlib.pyplot as plt, numpy as np
from llmarch import *
setup_style()
from llmarch.blocks import *
set_seed(0); device = get_device(); STEPS = 500
data = load_text_data("char"); train, val, V = data["train"], data["val"], data["vocab_size"]
# %%
block_diagram([("output", "out"), ("× SiLU(z) gate  →  out_proj", "ssm"), ("selective SSM scan  (Δ,B,C from x)", "ssm"), ("SiLU", "ssm"), ("causal depthwise conv1d (k=4)", "ssm"),
               ("in_proj → (x, z)", "ssm"), ("RMSNorm", "norm"), ("input", "other")][::-1], title="Mamba block (replaces attention)")
# %% [md]
# ## The block (pure PyTorch, sequential scan)
# The python `for t in range(T)` loop is the *exact* recurrence. Real Mamba uses a fused parallel-scan CUDA kernel.
# %%
show_source(MambaBlock)
# %% [md]
# ## Model: Mamba mixer + SwiGLU MLP in the same residual shell
# %%
d = 96
model = DecoderLM(V, d, [Block(d, MambaBlock(d, d_state=16), SwiGLU(d)) for _ in range(3)])
print(f"{count_params(model):,} params"); _ = show_summary(model, input_data=torch.randint(0, V, (1, 64)), depth=2)
hist = train_lm(model, train, val, steps=STEPS, block_size=64, batch_size=32, lr=3e-3, device=device, eval_every=200)
plot_losses(hist, "Mamba LM"); save_result("09 Mamba (SSM)", model, hist)
start = torch.zeros((1, 1), dtype=torch.long, device=device)
print(data["decode"](generate(model, start, 300, temperature=0.8, top_k=10, block_size=128)[0].tolist()))
# %% [md]
# ## Selectivity: where does the step size Δ spike?
# Mean Δ over channels for each character of a sentence. The model changes how fast its state moves depending on the token.
# %%
seq = val[500:564]; model.eval(); model(seq[None].to(device)); lab = [("⏎" if c == "\n" else c) for c in data["decode"](seq.tolist())]
fig, axs = plt.subplots(3, 1, figsize=(15, 6.5))
for l, ax in enumerate(axs):
    dt = model.blocks[l].mixer.last_dt[0].cpu()                             # (T, d_inner)
    ax.imshow(dt.T[:64], aspect="auto", cmap="magma"); ax.set_xticks(range(64)); ax.set_xticklabels(lab, fontsize=8); ax.set_yticks([]); ax.set_ylabel(f"layer {l}\nΔ (64 channels)")
plt.suptitle("input-dependent step size Δ_t"); plt.tight_layout(); plt.show()
# %%
fig, ax = plt.subplots(1, 2, figsize=(12, 3.6))
for l in range(3):
    dtm = model.blocks[l].mixer.last_dt[0].mean(-1).cpu(); ax[0].plot(dtm, label=f"layer {l}")
ax[0].set_xticks(range(64)); ax[0].set_xticklabels(lab, fontsize=6); ax[0].set_title("mean Δ per character"); ax[0].legend()
A = -torch.exp(model.blocks[0].mixer.A_log.detach()).cpu(); ax[1].hist(A.flatten().numpy(), bins=40); ax[1].set_title("A (state decay rates), layer 0 — initialised as −1…−16"); plt.tight_layout(); plt.show()
# %% [md]
# ## Memory of the state: how much does the first token still matter?
# We perturb the first character and measure the change of the hidden output at later positions: the SSM state decays
# at input-dependent rates.
# %%
x = val[:64][None].clone().to(device); x2 = x.clone(); x2[0, 0] = (x2[0, 0] + 7) % V
with torch.no_grad():
    def feats(z):
        h = model.tok(z)
        for b in model.blocks: h, _ = b(h)
        return h
    diff = (feats(x) - feats(x2))[0].norm(dim=-1).cpu()
plt.figure(figsize=(6.5, 3.5)); plt.semilogy(diff[1:] + 1e-9); plt.xlabel("position"); plt.ylabel("‖Δ hidden‖"); plt.title("influence of the 1st token fades over time"); plt.grid(alpha=.3, which="both"); plt.show()
# %% [md]
# ## Scaling: attention vs SSM at inference
# Attention's per-step cost and memory grow with the context; the SSM's stay constant (state = d_inner × N).
# %%
Ts = np.array([1, 1e3, 1e4, 1e5, 1e6]); dm = 4096; N = 16
fig, ax = plt.subplots(1, 2, figsize=(12, 3.6))
ax[0].loglog(Ts, Ts * 2 * dm * 2 * 32 / 1e9, "o-", label="Attention KV cache (32 layers, d=4096, fp16)"); ax[0].loglog(Ts, np.full_like(Ts, 2 * dm * N * 2 * 32 / 1e9), "o-", label="Mamba state (expand 2, N=16)")
ax[0].set(xlabel="context tokens", ylabel="GB", title="inference memory"); ax[0].legend(fontsize=7); ax[0].grid(alpha=.3, which="both")
tt = {"attention": [], "mamba (python loop)": []}; Tl = [32, 64, 128, 256]
att = Block(96, Attention(96, 4, rope=True), SwiGLU(96)).eval(); mb = Block(96, MambaBlock(96), SwiGLU(96)).eval()
for T in Tl:
    z = torch.randn(4, T, 96)
    for k, blk in [("attention", att), ("mamba (python loop)", mb)]:
        t0 = time.perf_counter(); [blk(z) for _ in range(3)]; tt[k].append((time.perf_counter() - t0) / 3)
for k, v in tt.items(): ax[1].plot(Tl, v, "o-", label=k)
ax[1].set(xlabel="sequence length", ylabel="seconds / forward", title="measured on this CPU (loop impl. is slow, but linear)"); ax[1].legend(); ax[1].grid(alpha=.3)
plt.tight_layout(); plt.show()
