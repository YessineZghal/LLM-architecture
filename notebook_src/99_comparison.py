# %% [md]
# # 99 · All architectures side by side
# Collects the numbers every notebook saved to `data/results/` (run notebooks 01–13 first).
# All next-token models used the same data (char-level tiny-Shakespeare), context 64 and AdamW + one-cycle LR.
#
# **Read this table carefully:**
# * most models train for **1 200 steps**; the two slowest ones (Mamba: 500 steps, hybrid: 400 steps — their selective scan is a plain Python loop) got a **smaller budget**, see the `steps` column;
# * model sizes differ (0.2 M – 1.0 M parameters; the MoE and hybrid have many more *total* parameters than *active* ones);
# * one seed, tiny scale.
#
# So treat the numbers as intuition about each family, **not** as a leaderboard.
# %%
import json
import pandas as pd, numpy as np, matplotlib.pyplot as plt, seaborn as sns
from llmarch import DATA_DIR, setup_style
setup_style()
res = pd.DataFrame([json.loads(f.read_text()) for f in sorted((DATA_DIR / "results").glob("*.json"))]).set_index("name")
res.index.name = "notebook / model"
comparable = res[res.note.isna()] if "note" in res else res
# %% [md]
# ## Next-token models (directly comparable)
# %%
c = comparable.sort_values("val_loss"); c = c.assign(val_ppl=np.exp(c.val_loss.astype(float)), params_k=c.params / 1e3)
display(c[["params_k", "steps", "val_loss", "val_ppl"]].round(3))
fig, ax = plt.subplots(1, 2, figsize=(14, 4.5))
sns.barplot(y=c.index, x=c.val_loss.astype(float), ax=ax[0], palette="viridis"); ax[0].set(xlabel="validation cross-entropy (nats/char)", title="lower is better"); ax[0].set_xlim(c.val_loss.astype(float).min() - .2)
ax[1].scatter(c.params_k, c.val_loss.astype(float), s=90)
for n, r in c.iterrows(): ax[1].annotate(n.split(" ")[0] + " " + n.split(" ")[1], (r.params_k, float(r.val_loss)), fontsize=7, xytext=(3, 3), textcoords="offset points")
ax[1].set(xlabel="parameters (k)", ylabel="val loss", title="quality vs size"); ax[1].grid(alpha=.3); plt.tight_layout(); plt.show()
# %% [md]
# ## Everything else (different objectives — not comparable with the above)
# %%
res[res.note.notna()][["params", "val_loss", "note"]] if "note" in res else None
# %% [md]
# ## Cheat-sheet: when does each family win?
# %%
pd.DataFrame([
 ["Decoder-only Transformer", "general LLMs, chat, code", "strongest quality/scaling, parallel training", "KV cache & O(T²) attention"],
 ["Encoder-only", "embeddings, classification, retrieval", "bidirectional context", "not generative"],
 ["Encoder-decoder", "translation, summarisation, speech", "clean input/output split; cross-attention", "more complex, less used for chat"],
 ["MoE", "scaling capacity cheaply", "more params at same FLOPs", "memory footprint, routing instability"],
 ["RNN / LSTM / GRU", "tiny streaming models", "constant memory", "sequential training, forgetting"],
 ["Mamba / SSM", "very long sequences, streaming", "linear time, constant state", "weaker exact recall"],
 ["Linear attn / RetNet / RWKV", "efficient inference with parallel training", "RNN inference, Transformer-like training", "recall limited by fixed state"],
 ["Hybrid (Jamba)", "long-context production LLMs", "mixes recall + efficiency", "engineering complexity"],
 ["ViT / VLM", "images, documents, multimodal", "reuses the Transformer for pixels", "needs lots of data"],
 ["Diffusion LM", "parallel decoding, infilling", "bidirectional, any-order generation", "multiple denoising steps, young ecosystem"],
], columns=["family", "typical use", "strength", "weakness"])
