"""Download / generate and cache all datasets used by the notebooks. Idempotent.

    uv run python scripts/prepare_data.py
"""
import json, random, urllib.request
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parent.parent
RAW, OUT = ROOT / "data" / "raw", ROOT / "data"
RAW.mkdir(parents=True, exist_ok=True)
URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


def shakespeare():
    p = RAW / "shakespeare.txt"
    if not p.exists():
        print("downloading tiny-shakespeare ...")
        urllib.request.urlretrieve(URL, p)
    text = p.read_text()
    # char-level
    chars = sorted(set(text)); stoi = {c: i for i, c in enumerate(chars)}
    ids = torch.tensor([stoi[c] for c in text], dtype=torch.long); n = int(.9 * len(ids))
    torch.save(dict(train=ids[:n], val=ids[n:], vocab_size=len(chars), itos=chars), OUT / "shakespeare_char.pt")
    # BPE (vocab 512)
    from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
    tok = Tokenizer(models.BPE(unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False); tok.decoder = decoders.ByteLevel()
    tok.train_from_iterator([text], trainers.BpeTrainer(vocab_size=512, special_tokens=["<unk>"],
                            initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
    tok.save(str(OUT / "shakespeare_bpe_tokenizer.json"))
    enc = torch.tensor(tok.encode(text).ids, dtype=torch.long); n2 = int(.9 * len(enc))
    itos = [tok.id_to_token(i) for i in range(tok.get_vocab_size())]
    torch.save(dict(train=enc[:n2], val=enc[n2:], vocab_size=tok.get_vocab_size(), itos=itos), OUT / "shakespeare_bpe.pt")
    return dict(chars=len(text), char_vocab=len(chars), bpe_tokens=len(enc), bpe_vocab=tok.get_vocab_size())


def seq2seq(n=6000, seed=0):
    """Reverse a sequence of 3-8 symbols (a-z). Needs an encoder-decoder with cross-attention to solve."""
    rnd = random.Random(seed)
    letters = list("abcdefghijklmnopqrstuvwxyz")
    itos = ["<pad>", "<bos>", "<eos>"] + letters; stoi = {c: i for i, c in enumerate(itos)}
    L = 8 + 2
    def mk():
        k = rnd.randint(3, 8); s = [rnd.choice(letters) for _ in range(k)]
        src = [stoi[c] for c in s] + [2]
        tgt = [1] + [stoi[c] for c in reversed(s)] + [2]
        return src + [0] * (L - len(src)), tgt + [0] * (L - len(tgt))
    pairs = [mk() for _ in range(n)]
    src = torch.tensor([p[0] for p in pairs]); tgt = torch.tensor([p[1] for p in pairs]); m = int(.9 * n)
    torch.save(dict(train=(src[:m], tgt[:m]), val=(src[m:], tgt[m:]), vocab_size=len(itos), itos=itos,
                    pad=0, bos=1, eos=2), OUT / "seq2seq.pt")
    return dict(pairs=n, vocab=len(itos), max_len=L)


def digits():
    from sklearn.datasets import load_digits
    d = load_digits(); X = torch.tensor(d.images, dtype=torch.float32) / 16.0; y = torch.tensor(d.target)
    g = torch.Generator().manual_seed(0); perm = torch.randperm(len(X), generator=g); m = int(.85 * len(X))
    torch.save(dict(x_train=X[perm[:m]], y_train=y[perm[:m]], x_val=X[perm[m:]], y_val=y[perm[m:]],
                    image_size=8, classes=list(range(10))), OUT / "digits.pt")
    return dict(images=len(X), size="8x8", classes=10)


if __name__ == "__main__":
    card = dict(shakespeare=shakespeare(), seq2seq=seq2seq(), digits=digits())
    (OUT / "data_card.json").write_text(json.dumps(card, indent=2)); print(json.dumps(card, indent=2))
