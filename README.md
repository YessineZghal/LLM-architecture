# LLM Architecture — a hands-on tour in PyTorch

Small, readable, **runnable** implementations of every major family of language-model architecture, each in its own
notebook with architecture diagrams, parameter breakdowns, training curves and "look inside" visualisations
(attention maps, router load, SSM step sizes, LSTM gates, cross-attention alignments …).
Everything trains on a laptop CPU in seconds-to-minutes.

## Notebooks

| # | Notebook | Family | Highlights |
|---|----------|--------|------------|
| 00 | `00_data_and_taxonomy` | overview | datasets, BPE tokenizer, architecture taxonomy |
| 01 | `01_decoder_only_gpt` | decoder-only Transformer | attention from scratch, fx graph, ONNX export |
| 02 | `02_encoder_only_bert` | encoder-only | masked-LM, bidirectional attention, embeddings PCA |
| 03 | `03_encoder_decoder_transformer` | encoder–decoder | cross-attention alignment on a reversal task |
| 04 | `04_llama_style_modern_decoder` | modern decoder | RMSNorm · RoPE · SwiGLU · GQA vs GPT-2 recipe |
| 05 | `05_attention_variants` | attention | MHA / MQA / GQA / sliding window / MLA, KV-cache maths |
| 06 | `06_positional_encodings` | positions | sinusoidal, learned, RoPE, ALiBi, NoPE + length extrapolation |
| 07 | `07_mixture_of_experts` | MoE | Switch / Mixtral / DeepSeek layers, router analysis |
| 08 | `08_recurrent_rnn_lstm_gru` | recurrent | vanishing gradients, LSTM gates |
| 09 | `09_state_space_mamba` | SSM | selective scan, input-dependent Δ |
| 10 | `10_linear_attention_retnet_rwkv` | linear-time | parallel ≡ recurrent proofs, RetNet, RWKV |
| 11 | `11_hybrid_jamba` | hybrid | Mamba + attention + MoE layer patterns |
| 12 | `12_vision_transformer_and_vlm` | multimodal | ViT, LLaVA-style VLM |
| 13 | `13_diffusion_language_model` | diffusion LM | masked diffusion, parallel decoding, infilling |
| 99 | `99_comparison` | summary | all models under one training budget |

## Quick start

```bash
uv sync                                        # create .venv with torch + visualisation stack
uv run python scripts/prepare_data.py          # download / generate the datasets (cached in data/)
uv run jupyter lab                             # pick the kernel "LLM Architecture (uv)"
```

Rebuild and re-execute every notebook headlessly (4 parallel jobs):

```bash
uv run python scripts/build_notebooks.py            # all
uv run python scripts/build_notebooks.py 07 09      # selected
```

If the kernel is missing: `uv run python -m ipykernel install --user --name llm-architecture --display-name "LLM Architecture (uv)"`.

## Layout

```
Notebook/        executed notebooks (generated)
notebook_src/    notebook sources in percent format (edit these)
llmarch/         shared helpers: data loading, trainer, building blocks, visualisation, plot style
scripts/         prepare_data.py, build_notebooks.py, notebook_meta.py (banners / takeaways / references)
data/            cached datasets (git-ignored where large) and per-notebook results
artifacts/       exported ONNX graphs — open with `uv run netron artifacts/gpt.onnx`
```

## Visualisation toolbox (all installed in the uv env)

`matplotlib`, `seaborn`, `plotly` (interactive treemaps), `torchinfo` (layer tables), `torch.fx` + `networkx`
(computation graphs, no Graphviz needed), `onnx` + `netron` (interactive graph viewer), `bertviz` (attention
viewer for Hugging Face models), `transformers`, `tokenizers`.

## Conventions

* Same data (char-level tiny-Shakespeare), context 64, AdamW + one-cycle LR and ~0.3 M parameters for every
  next-token model, so notebook 99 is a fair — but small-scale — comparison.
* Models are written for clarity, not speed (e.g. the Mamba scan is a Python loop).
