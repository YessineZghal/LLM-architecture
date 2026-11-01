# %% [md]
# # 04 · The modern decoder-only recipe (LLaMA / Mistral / Qwen style)
#
# Today's open LLMs are still GPT-shaped, but almost every component was swapped for a better one:
#
# | GPT-2 | → | LLaMA-style |
# |---|---|---|
# | LayerNorm (+bias) | → | **RMSNorm** (no mean, no bias) |
# | learned absolute positions | → | **RoPE** rotary positions inside attention |
# | GELU MLP (4d) | → | **SwiGLU** gated MLP (≈ 8/3 d) |
# | multi-head attention | → | **GQA** grouped-query attention (fewer KV heads) |
# | biases everywhere | → | no biases |
#
# Below we build both from the same `DecoderLM` shell and compare them at (nearly) equal parameter count.
# %%
import torch, torch.nn as nn, torch.nn.functional as F, matplotlib.pyplot as plt, numpy as np
from llmarch import *
setup_style()
from llmarch.blocks import *
set_seed(0); device = get_device(); STEPS = 1200
data = load_text_data("char"); train, val, V = data["train"], data["val"], data["vocab_size"]
# %%
block_diagram([("next-token logits", "out"), ("RMSNorm", "norm"), ("SwiGLU MLP", "ffn*"), ("RMSNorm", "norm*"), ("GQA attention + RoPE on Q,K", "attn*"),
               ("RMSNorm", "norm*"), ("token embedding (no pos emb!)", "embed"), ("token ids", "other")][::-1], title="LLaMA-style decoder", repeat="N ×")
# %% [md]
# ## The building blocks
# %%
show_source(RMSNorm, SwiGLU, rope_cos_sin, apply_rope)
# %% [md]
# ### RMSNorm vs LayerNorm
# %%
x = torch.randn(1000, 64) * 3 + 2
ln, rn = nn.LayerNorm(64)(x), RMSNorm(64)(x)
fig, ax = plt.subplots(1, 3, figsize=(13, 3.2))
ax[0].hist(x.flatten(), bins=60); ax[0].set_title("input (mean≈2, std≈3)")
ax[1].hist(ln.flatten().detach(), bins=60); ax[1].set_title("LayerNorm → mean 0, std 1")
ax[2].hist(rn.flatten().detach(), bins=60); ax[2].set_title("RMSNorm → keeps the mean, rms 1")
plt.tight_layout(); plt.show()
# %% [md]
# ### SwiGLU vs GELU
# %%
z = torch.linspace(-5, 5, 400)
plt.figure(figsize=(5.5, 3.5)); plt.plot(z, F.gelu(z), label="GELU"); plt.plot(z, F.silu(z), label="SiLU / swish (the gate in SwiGLU)"); plt.plot(z, F.relu(z), "--", label="ReLU", alpha=.5)
plt.legend(); plt.grid(alpha=.3); plt.title("activations"); plt.show()
# %% [md]
# ### RoPE: positions as rotations
# Each pair of channels is rotated by angle `pos · θ_i`, with different θ_i per pair (fast → slow). The dot-product
# `q_m · k_n` then depends only on the **relative distance m − n**.
# %%
hd, T = 32, 64; cos, sin = rope_cos_sin(T, hd)
ang = torch.atan2(sin, cos)
fig, ax = plt.subplots(1, 2, figsize=(12, 3.8))
plot_heatmap(sin.T, title="sin(pos·θ_i)  (rows: frequency pair, cols: position)", ax=ax[0], cmap="coolwarm")
q = torch.randn(1, 1, 1, hd).expand(1, 1, T, hd).contiguous(); k = q.clone()
qr, kr = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
score = (qr[0, 0] @ kr[0, 0].T)                       # same content everywhere -> only position matters
plot_heatmap(score, title="q_m·k_n with identical content: depends on (m−n) only", ax=ax[1])
plt.tight_layout(); plt.show()
print("Toeplitz check (diagonals constant):", torch.allclose(score[10, 5], score[30, 25], atol=1e-4))
# %% [md]
# ## Build both models
# %%
d, H = 96, 4
def gpt2_like():
    return DecoderLM(V, d, [Block(d, Attention(d, H, bias=True), GELUMLP(d), norm=nn.LayerNorm) for _ in range(3)], norm=nn.LayerNorm, learned_pos=128)
def llama_like(kv=2):
    return DecoderLM(V, d, [Block(d, Attention(d, H, n_kv_head=kv, rope=True, bias=False), SwiGLU(d), norm=RMSNorm) for _ in range(3)])
m_gpt, m_llama = gpt2_like(), llama_like()
print(f"GPT-2-like : {count_params(m_gpt):,} params\nLLaMA-like : {count_params(m_llama):,} params")
_ = show_summary(m_llama, input_data=torch.randint(0, V, (1, 64)), depth=2)
# %% [md]
# ## Train both for the same number of steps
# %%
res = {}
for name, m in [("GPT-2-like", m_gpt), ("LLaMA-like", m_llama)]:
    print("==", name); set_seed(0); res[name] = train_lm(m, train, val, steps=STEPS, block_size=64, batch_size=32, lr=3e-3, device=device, eval_every=200)
save_result("04 LLaMA-style decoder (RMSNorm+RoPE+SwiGLU+GQA)", m_llama, res["LLaMA-like"])
fig, ax = plt.subplots(figsize=(6.5, 4))
for n, h in res.items(): ax.plot(h["val_step"], h["val"], "o-", label=f"{n} ({count_params(m_gpt if n.startswith('GPT') else m_llama)/1e3:.0f}k)")
ax.set(xlabel="step", ylabel="val loss", title="Same budget, different recipe"); ax.legend(); ax.grid(alpha=.3); plt.show()
# %% [md]
# ## Sample from the LLaMA-style model
# %%
start = torch.zeros((1, 1), dtype=torch.long, device=device)
print(data["decode"](generate(m_llama, start, 400, temperature=0.8, top_k=10, block_size=128)[0].tolist()))
# %% [md]
# ## GQA attention maps: 4 query heads share 2 KV heads
# Heads (0,1) read the same keys/values, heads (2,3) read the others — yet they still attend differently.
# %%
ids = val[:48][None].to(device); txt = [("⏎" if c == "\n" else c) for c in data["decode"](val[:48].tolist())]
for b in m_llama.blocks: b.mixer.keep_attn = True
m_llama.eval(); m_llama(ids)
fig, axs = plt.subplots(1, 4, figsize=(20, 5))
for h in range(4): plot_heatmap(m_llama.blocks[2].mixer.last_attn[0, h].cpu(), txt, txt, f"last layer · q-head {h} → kv-head {h // 2}", ax=axs[h])
plt.tight_layout(); plt.show()
