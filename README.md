<div align="center">

# LLM Architecture — A Hands-On Tour in PyTorch

**Small, readable, runnable implementations of every major family of language-model architecture —
each with diagrams, parameter breakdowns, training curves and "look inside" visualisations.**

![Python](https://img.shields.io/badge/python-3.12-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c)
![uv](https://img.shields.io/badge/env-uv-6e56cf)
![Notebooks](https://img.shields.io/badge/notebooks-15-orange)
![Runs on](https://img.shields.io/badge/runs%20on-laptop%20CPU-green)

</div>

---

## Table of contents

1. [Overview](#1-overview)
2. [What is covered](#2-what-is-covered)
3. [Quick start](#3-quick-start)
4. [Repository layout](#4-repository-layout)
5. [Datasets](#5-datasets)
6. [Notebook guide](#6-notebook-guide)
7. [The `llmarch` helper library](#7-the-llmarch-helper-library)
8. [Experimental protocol](#8-experimental-protocol)
9. [Results](#9-results)
10. [Visualisation toolbox](#10-visualisation-toolbox)
11. [Extending the project](#11-extending-the-project)
12. [Design decisions and limitations](#12-design-decisions-and-limitations)
13. [Troubleshooting](#13-troubleshooting)
14. [References](#14-references)

---

## 1. Overview

Modern LLMs look very different from the original Transformer: grouped-query attention, rotary positions,
mixture-of-experts, state-space layers, hybrids, even diffusion-style generation. Reading the papers gives you the
equations; this repository gives you **working code you can run, break and inspect** for each idea.

Design goals:

| Goal | How it is met |
|------|---------------|
| **Coverage** | One notebook per architecture family, from RNNs (1997) to masked-diffusion LMs (2025). |
| **Readability** | Pure PyTorch, no hidden framework magic; the key class of each notebook fits on one screen. |
| **Visual understanding** | Every notebook has architecture diagrams, parameter treemaps, attention / routing / gate / state visualisations. |
| **Comparability** | A shared dataset, trainer and (mostly) training budget, so notebooks can be compared in `99_comparison`. |
| **Reproducibility** | `uv`-locked environment, cached deterministic data, seeded runs, a one-command notebook builder. |
| **Low barrier** | Everything trains on a laptop CPU in seconds to minutes — no GPU required. |

> **Scope note.** The models have ~0.2–1 M parameters. They are *teaching models*: they exhibit the characteristic
> behaviour of each architecture (attention patterns, router specialisation, decay spectra …) but are not meant to
> reproduce production-scale quality.

## 2. What is covered

| Family | Representative models | Token mixer | Train-time cost | Inference memory | Notebook |
|---|---|---|---|---|---|
| Decoder-only Transformer | GPT-2/3, LLaMA, Qwen | causal self-attention | O(T²) | KV cache ∝ T | `01` |
| Encoder-only Transformer | BERT, RoBERTa, DeBERTa | bidirectional attention | O(T²) | – (no generation) | `02` |
| Encoder–decoder Transformer | T5, BART, Whisper | self- + cross-attention | O(T²) | KV cache + memory | `03` |
| Modern decoder recipe | LLaMA 2/3, Mistral, Gemma | attention + RoPE/GQA/SwiGLU/RMSNorm | O(T²) | smaller KV cache | `04` |
| Attention variants | MQA, GQA, sliding window, MLA | – | – | KV cache 1× → ~0.05× | `05` |
| Positional encodings | sinusoidal, learned, RoPE, ALiBi, NoPE | – | – | – | `06` |
| Mixture of Experts | Switch, Mixtral, DeepSeek-MoE | attention + sparse MLP | FLOPs ∝ active params | memory ∝ total params | `07` |
| Recurrent networks | RNN, LSTM, GRU | recurrence | O(T), sequential | O(1) state | `08` |
| State-space models | S4, Mamba | selective scan | O(T) | O(1) state | `09` |
| Linear attention / retention / RWKV | Linear Transformer, RetNet, RWKV | decayed recurrence | O(T) | O(1) state | `10` |
| Hybrids | Jamba, Zamba | Mamba + attention (+ MoE) | mostly O(T) | few KV layers | `11` |
| Vision & multimodal | ViT, LLaVA | patch tokens, projector | O(P²) | prefix tokens | `12` |
| Diffusion LMs | LLaDA, MDLM | bidirectional denoiser | O(T²) × steps | none | `13` |

## 3. Quick start

### Requirements

* [`uv`](https://docs.astral.sh/uv/) (tested with 0.8) — it installs Python 3.12 and every dependency for you
* ~1.5 GB of disk for the environment (PyTorch + Jupyter + visualisation stack), ~25 MB for data and notebooks
* Any recent laptop; no GPU needed (Apple-silicon MPS / CUDA are used automatically only if you ask for them)

### Install and run

```bash
git clone https://github.com/YessineZghal/LLM-architecture.git
cd LLM-architecture

uv sync                                   # create .venv with the locked dependencies
uv run python scripts/prepare_data.py     # download / generate and cache the datasets in data/
uv run python -m ipykernel install --user \
    --name llm-architecture --display-name "LLM Architecture (uv)"   # one-time kernel registration
uv run jupyter lab                        # open Notebook/ and choose the "LLM Architecture (uv)" kernel
```

### Execute every notebook headlessly

```bash
uv run python scripts/build_notebooks.py              # rebuild + execute all notebooks (4 parallel jobs)
uv run python scripts/build_notebooks.py 07 09 -j 2   # only notebooks whose name contains 07 or 09
uv run python scripts/build_notebooks.py --no-exec    # regenerate .ipynb without running them
```

The builder reads `notebook_src/*.py`, executes each notebook with the project kernel, and writes the results to
`Notebook/*.ipynb`. A full run takes on the order of 15–25 minutes on a laptop CPU (notebook 11 dominates).

### Inspect model graphs interactively

```bash
uv run netron artifacts/gpt.onnx          # produced by notebook 01
```

## 4. Repository layout

```
LLM-architecture/
├── Notebook/                 Executed notebooks (generated – open these)
├── notebook_src/             Notebook sources, percent format ("# %%" cells) – edit these
├── llmarch/                  Shared helper package (installed in the uv env)
│   ├── data.py               dataset loaders
│   ├── train.py              seeding, device choice, generic LM trainer, sampling, result logging
│   ├── blocks.py             RMSNorm, SwiGLU, RoPE, Attention (MHA/MQA/GQA/window), MoE, MambaBlock, DecoderLM
│   ├── viz.py                block diagrams, heatmaps, torchinfo, fx graph, ONNX export, source display
│   └── style.py              global plotting style and warning filter
├── scripts/
│   ├── prepare_data.py       builds data/ (idempotent)
│   ├── build_notebooks.py    sources → executed notebooks, adds banner/TOC/takeaways/references
│   └── notebook_meta.py      per-notebook metadata (family, models, takeaways, references)
├── data/                     cached tensors, BPE tokenizer, per-notebook results (large files git-ignored)
├── artifacts/                exported ONNX graphs
├── pyproject.toml / uv.lock  dependencies (single source of truth for the environment)
└── README.md
```

## 5. Datasets

All datasets are produced by `scripts/prepare_data.py` and described in `data/data_card.json`.

| Dataset | Used by | Description |
|---|---|---|
| **Tiny Shakespeare (char level)** | 01, 02, 04–11, 13 | 1.1 M characters, 65-symbol vocabulary, 90/10 train/val split. |
| **Tiny Shakespeare (byte-level BPE)** | 00 (demo) | Same text with a 512-token BPE tokenizer trained locally (`data/shakespeare_bpe_tokenizer.json`). |
| **Reverse-a-word seq2seq** | 03 | 6 000 synthetic pairs: reverse 3–8 random letters. Requires cross-attention to solve. |
| **sklearn digits** | 12 | 1 797 grayscale 8×8 handwritten digits, 10 classes (85/15 split). |

The text corpus is downloaded from the public
[`char-rnn`](https://github.com/karpathy/char-rnn/tree/master/data/tinyshakespeare) repository; all other data is generated
deterministically.

## 6. Notebook guide

Each executed notebook starts with an **At a glance** table (family, models, core idea, cost, runtime), a list of learning
objectives and a clickable table of contents; it ends with **Key takeaways**, **References** and previous/next navigation.

| # | Notebook | You will see |
|---|----------|--------------|
| 00 | `00_data_and_taxonomy` | dataset statistics, BPE vocabulary, architecture taxonomy table |
| 01 | `01_decoder_only_gpt` | causal attention from scratch · parameter treemap (plotly) · `torch.fx` graph · attention maps · position-embedding structure · ONNX export |
| 02 | `02_encoder_only_bert` | causal vs bidirectional masks · masked-LM training · fill-in-the-blank demo · bidirectional attention · PCA of contextual embeddings |
| 03 | `03_encoder_decoder_transformer` | encoder/decoder diagrams · teacher forcing · greedy decoding (100 % exact match) · **cross-attention anti-diagonal** |
| 04 | `04_llama_style_modern_decoder` | RMSNorm vs LayerNorm · SwiGLU vs GELU · RoPE rotations & Toeplitz proof · GPT-2 vs LLaMA recipe · GQA head sharing |
| 05 | `05_attention_variants` | mask shapes · head-layout diagrams · MLA · KV-cache size for a 70B model · receptive field of sliding windows |
| 06 | `06_positional_encodings` | five encodings visualised · **length extrapolation** 32→512 tokens · shuffle test |
| 07 | `07_mixture_of_experts` | router, load balancing · dense vs Switch vs Mixtral vs DeepSeek layers · expert load · P(expert \| character class) · routing per character |
| 08 | `08_recurrent_rnn_lstm_gru` | gradient-vs-distance (vanishing gradient) · hand-computed LSTM gates · constant-state vs KV-cache memory |
| 09 | `09_state_space_mamba` | selective scan · **input-dependent Δ** heat-maps · state decay · measured speed vs attention |
| 10 | `10_linear_attention_retnet_rwkv` | numerical proof *parallel ≡ recurrent* · decay matrices · RetNet / RWKV training · learned decay spectra |
| 11 | `11_hybrid_jamba` | layer-pattern diagrams · all-attention vs all-Mamba vs hybrid vs hybrid+MoE · attention, routing and Δ in one model |
| 12 | `12_vision_transformer_and_vlm` | patchification · ViT training · `[CLS]` attention overlays · LLaVA-style captioner |
| 13 | `13_diffusion_language_model` | forward noising · 1/t-weighted objective · parallel unmasking · step-count trade-off · infilling |
| 99 | `99_comparison` | all next-token models side by side, cheat-sheet of strengths and weaknesses |

Recommended reading order: **00 → 01 → 04 → 05 → 06** (Transformer core), then **02, 03** (other Transformer shapes),
then **07** (sparsity), **08 → 09 → 10 → 11** (the path to linear-time models), **12, 13** (modalities and new paradigms),
and finish with **99**.

## 7. The `llmarch` helper library

Importable in any notebook (`from llmarch import *`). The architecture-defining code is intentionally kept readable and is
displayed inside the notebooks via `show_source(...)`.

| Module | Main contents |
|---|---|
| `llmarch.data` | `load_text_data("char" \| "bpe")`, `load_seq2seq_data()`, `load_digits_data()` |
| `llmarch.train` | `set_seed`, `get_device`, `train_lm` (AdamW + one-cycle LR, grad clipping, optional auxiliary loss), `generate` (temperature, top-k), `save_result` |
| `llmarch.blocks` | `RMSNorm`, `GELUMLP`, `SwiGLU`, `rope_cos_sin` / `apply_rope`, `Attention` (MHA / MQA / GQA / sliding window / bidirectional, optional attention capture), `MoE` (top-k routing + balance loss + shared expert), `MambaBlock`, generic `Block` and `DecoderLM` |
| `llmarch.viz` | `block_diagram`, `plot_heatmap`, `plot_losses`, `show_summary` (torchinfo), `fx_graph_plot`, `export_onnx`, `show_source` |
| `llmarch.style` | `setup_style()` – consistent seaborn/matplotlib theme and quiet warnings |

Minimal example — build and train a LLaMA-style model in a few lines:

```python
from llmarch import *
from llmarch.blocks import *

data = load_text_data("char")
d = 96
model = DecoderLM(
    data["vocab_size"], d,
    [Block(d, Attention(d, n_head=4, n_kv_head=2, rope=True, bias=False), SwiGLU(d)) for _ in range(3)],
)
hist = train_lm(model, data["train"], data["val"], steps=1200)
plot_losses(hist)
```

## 8. Experimental protocol

* **Task**: next-character prediction on tiny-Shakespeare, context length 64, batch size 32.
* **Optimiser**: AdamW (weight decay 0.01), peak LR 3e-3, one-cycle schedule (10 % warm-up), gradient clipping at 1.0.
* **Budget**: 1 200 steps for most notebooks. The two with a pure-Python selective scan (09 Mamba, 11 hybrid) use fewer steps
  (500 and 400) to keep runtime reasonable — the `steps` column in the results table makes this explicit.
* **Seed**: `set_seed(0)` before every model construction.
* **Metric**: validation cross-entropy in nats per character (perplexity = `exp(loss)`).
* **Not comparable** (different objectives, reported separately): BERT (masked-LM loss), encoder–decoder (reversal task),
  diffusion LM (masked-token cross-entropy).

## 9. Results

<!-- RESULTS_TABLE -->

## 10. Visualisation toolbox

All of the following are installed in the uv environment.

| Tool | Used for |
|---|---|
| `matplotlib`, `seaborn` | training curves, heat-maps, custom block diagrams |
| `plotly` | interactive treemaps of parameter budgets |
| `torchinfo` | layer-by-layer summaries with shapes and parameter counts |
| `torch.fx` + `networkx` | computation graphs drawn without any system dependency |
| `onnx` + `onnxscript` + `netron` | ONNX export and interactive graph viewer (`uv run netron <file>.onnx`) |
| `bertviz`, `transformers` | attention visualisation for Hugging Face models (ready to use in your own experiments) |
| `tokenizers` | BPE training for the text corpus |
| `scikit-learn` | PCA of embeddings, digits dataset |

## 11. Extending the project

**Add a new architecture notebook**

1. Create `notebook_src/NN_my_architecture.py`. Use `# %% [md]` for markdown and `# %%` for code cells; start with a
   `# # NN · Title` cell and use `## Section` headings (the builder numbers them and builds the table of contents).
2. Re-use `llmarch.blocks` (e.g. `DecoderLM(vocab, d, [Block(d, my_mixer, my_ffn) ...])`) and `train_lm`, then call
   `save_result("NN my model", model, hist)` so it appears in `99_comparison`.
3. Add an entry (family, models, idea, cost, objectives, takeaways, references) to `scripts/notebook_meta.py`.
4. Run `uv run python scripts/build_notebooks.py NN` and then `99` to refresh the comparison.

**Add a dependency**: `uv add <package>` (updates `pyproject.toml` and `uv.lock`).

**Use a GPU**: pass `device=get_device(prefer_gpu=True)` to `train_lm` (CUDA, else Apple MPS). Scale the model with
the `d`, `n_layer` and `steps` variables at the top of a notebook.

## 12. Design decisions and limitations

* **Pure PyTorch, small models.** Optimised for reading, not speed; e.g. the Mamba scan is a Python loop, whereas real
  implementations use a fused parallel-scan kernel. The notebooks say so explicitly.
* **No Graphviz dependency.** `torchview` / `torchviz` need the system `dot` binary, so computation graphs are drawn with
  `torch.fx` + `networkx` and ONNX/Netron instead. Install Graphviz and add those packages if you prefer them.
* **Simplified variants.** MLA omits the decoupled-RoPE detail; the BERT masking recipe is simplified (100 % `[MASK]`);
  RetNet has no output gate or xPos; RWKV uses a naive (non-log-space) WKV recurrence. Each simplification is stated in the
  corresponding notebook.
* **Small-scale comparisons are noisy.** One seed, ~0.3 M parameters, character-level text. Rankings can change with
  scale, data, tokenisation and tuning — use them to build intuition, not to pick an architecture.
* **Pre-generated outputs.** Notebooks in `Notebook/` contain executed outputs (figures included); re-running them changes
  only timings and, marginally, floating-point details.

## 13. Troubleshooting

| Problem | Fix |
|---|---|
| Kernel "LLM Architecture (uv)" not listed | `uv run python -m ipykernel install --user --name llm-architecture --display-name "LLM Architecture (uv)"` |
| `FileNotFoundError: ... run scripts/prepare_data.py` | run `uv run python scripts/prepare_data.py` (needs internet once for the corpus) |
| `ModuleNotFoundError: llmarch` | start Jupyter through `uv run jupyter lab` so the project package is on the path |
| Plotly treemap does not render | use JupyterLab/VS Code with a recent version; the other figures are static images |
| Notebook 09/11 feel slow | lower `STEPS` at the top of the notebook – the scan is a Python loop by design |
| Out-of-memory | reduce `batch_size` in `train_lm(...)` or the model width `d` |

## 14. References

**Transformers and variants** — Vaswani et al., *Attention Is All You Need* ([1706.03762](https://arxiv.org/abs/1706.03762)) ·
Devlin et al., *BERT* ([1810.04805](https://arxiv.org/abs/1810.04805)) · Raffel et al., *T5* ([1910.10683](https://arxiv.org/abs/1910.10683)) ·
Touvron et al., *LLaMA* ([2302.13971](https://arxiv.org/abs/2302.13971)) · Zhang & Sennrich, *RMSNorm* ([1910.07467](https://arxiv.org/abs/1910.07467)) ·
Shazeer, *GLU Variants* ([2002.05202](https://arxiv.org/abs/2002.05202))

**Attention and positions** — Shazeer, *MQA* ([1911.02150](https://arxiv.org/abs/1911.02150)) · Ainslie et al., *GQA* ([2305.13245](https://arxiv.org/abs/2305.13245)) ·
DeepSeek-V2, *MLA* ([2405.04434](https://arxiv.org/abs/2405.04434)) · Beltagy et al., *Longformer* ([2004.05150](https://arxiv.org/abs/2004.05150)) ·
Su et al., *RoPE* ([2104.09864](https://arxiv.org/abs/2104.09864)) · Press et al., *ALiBi* ([2108.12409](https://arxiv.org/abs/2108.12409)) ·
Kazemnejad et al., *NoPE* ([2305.19466](https://arxiv.org/abs/2305.19466))

**Sparsity** — Fedus et al., *Switch Transformers* ([2101.03961](https://arxiv.org/abs/2101.03961)) · Jiang et al., *Mixtral* ([2401.04088](https://arxiv.org/abs/2401.04088)) ·
Dai et al., *DeepSeekMoE* ([2401.06066](https://arxiv.org/abs/2401.06066))

**Recurrent, SSM and linear-time models** — Hochreiter & Schmidhuber, *LSTM* (1997) · Cho et al., *GRU* ([1406.1078](https://arxiv.org/abs/1406.1078)) ·
Gu & Dao, *Mamba* ([2312.00752](https://arxiv.org/abs/2312.00752)) · Gu et al., *S4* ([2111.00396](https://arxiv.org/abs/2111.00396)) ·
Katharopoulos et al., *Transformers are RNNs* ([2006.16236](https://arxiv.org/abs/2006.16236)) · Sun et al., *RetNet* ([2307.08621](https://arxiv.org/abs/2307.08621)) ·
Peng et al., *RWKV* ([2305.13048](https://arxiv.org/abs/2305.13048)) · Lieber et al., *Jamba* ([2403.19887](https://arxiv.org/abs/2403.19887))

**Multimodal and diffusion** — Dosovitskiy et al., *ViT* ([2010.11929](https://arxiv.org/abs/2010.11929)) · Liu et al., *LLaVA* ([2304.08485](https://arxiv.org/abs/2304.08485)) ·
Nie et al., *LLaDA* ([2502.09992](https://arxiv.org/abs/2502.09992)) · Sahoo et al., *MDLM* ([2406.07524](https://arxiv.org/abs/2406.07524))

---

<div align="center">
Maintained by <b>Yessine Zghal</b> · Built with PyTorch and uv
</div>
