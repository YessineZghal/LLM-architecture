import random, time
import numpy as np
import torch
import torch.nn.functional as F


def set_seed(s=0):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)


def get_device(prefer_gpu=False):
    """CPU by default (small models, deterministic). Set prefer_gpu=True for Apple MPS / CUDA."""
    if prefer_gpu:
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
    return torch.device("cpu")


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def get_batch(data, block_size, batch_size, device):
    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)


@torch.no_grad()
def eval_loss(model, data, block_size, batch_size, device, iters=10):
    model.eval()
    ls = []
    for _ in range(iters):
        x, y = get_batch(data, block_size, batch_size, device)
        out = model(x)
        logits = out[0] if isinstance(out, tuple) else out
        ls.append(F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1)).item())
    model.train()
    return float(np.mean(ls))


def train_lm(model, train, val, steps=300, block_size=64, batch_size=32, lr=3e-3,
             device=None, eval_every=50, aux_loss_fn=None, verbose=True):
    """Generic next-token trainer. Model: forward(x)->logits (B,T,V) or (logits, aux).
    If the model returns (logits, aux) and aux is a tensor it is added to the loss (e.g. MoE balance loss)."""
    device = device or get_device()
    model.to(device).train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.1)
    hist = {"step": [], "train": [], "val_step": [], "val": []}
    t0 = time.time()
    for step in range(1, steps + 1):
        x, y = get_batch(train, block_size, batch_size, device)
        out = model(x)
        aux = None
        if isinstance(out, tuple):
            logits, aux = out[0], out[1]
        else:
            logits = out
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        total = loss + (aux if torch.is_tensor(aux) and aux.ndim == 0 else 0)
        opt.zero_grad(set_to_none=True)
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        hist["step"].append(step); hist["train"].append(loss.item())
        if step % eval_every == 0 or step == steps:
            v = eval_loss(model, val, block_size, batch_size, device)
            hist["val_step"].append(step); hist["val"].append(v)
            if verbose:
                print(f"step {step:4d} | train {loss.item():.3f} | val {v:.3f} | {time.time()-t0:.0f}s")
    return hist


@torch.no_grad()
def generate(model, idx, max_new_tokens=200, temperature=1.0, top_k=None, block_size=None):
    model.eval()
    for _ in range(max_new_tokens):
        ctx = idx if block_size is None else idx[:, -block_size:]
        out = model(ctx)
        logits = (out[0] if isinstance(out, tuple) else out)[:, -1, :] / temperature
        if top_k:
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[:, [-1]]] = -float("inf")
        idx = torch.cat([idx, torch.multinomial(F.softmax(logits, -1), 1)], dim=1)
    model.train()
    return idx


def save_result(name, model, hist, extra=None):
    """Record final val loss / params (one json per notebook entry) so notebook 99 can compare every architecture."""
    import json, re
    from .data import DATA_DIR
    d = DATA_DIR / "results"; d.mkdir(exist_ok=True)
    rec = dict(name=name, params=count_params(model), steps=len(hist["train"]), val_loss=float(hist["val"][-1]),
               train_loss=float(np.mean(hist["train"][-20:])), **(extra or {}))
    (d / (re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_") + ".json")).write_text(json.dumps(rec, indent=2))
    return rec
