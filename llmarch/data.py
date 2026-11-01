"""Data loading. Run `uv run python scripts/prepare_data.py` once to build the cache."""
from pathlib import Path
import json
import torch

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"


def _need(path):
    if not path.exists():
        raise FileNotFoundError(f"{path} missing - run `uv run python scripts/prepare_data.py` first")
    return path


def load_text_data(kind="char"):
    """Tiny-Shakespeare language-modelling data.

    kind: 'char' (vocab ~65) or 'bpe' (vocab 512, trained BPE).
    Returns dict(train=LongTensor, val=LongTensor, vocab_size=int, itos=list[str], decode=callable).
    """
    d = torch.load(_need(DATA_DIR / f"shakespeare_{kind}.pt"), weights_only=False)
    itos = d["itos"]
    d["decode"] = lambda ids: "".join(itos[int(i)] for i in ids)
    d["encode_char"] = {c: i for i, c in enumerate(itos)} if kind == "char" else None
    return d


def load_seq2seq_data():
    """Synthetic seq2seq task (for encoder-decoder models): reverse a short word sequence.
    Returns dict(train=(src,tgt), val=(src,tgt), vocab_size, itos, pad, bos, eos)."""
    return torch.load(_need(DATA_DIR / "seq2seq.pt"), weights_only=False)


def load_digits_data():
    """8x8 handwritten digits (sklearn) as tensors for vision/multimodal notebooks."""
    return torch.load(_need(DATA_DIR / "digits.pt"), weights_only=False)


def data_card():
    return json.loads(_need(DATA_DIR / "data_card.json").read_text())
