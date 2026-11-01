"""Reusable architecture building blocks (pure PyTorch, small & readable).

Notebooks import these and show their source with `llmarch.show_source`, so you always see the real code.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------- normalisation & MLPs
class RMSNorm(nn.Module):
    """LLaMA-style norm: x / rms(x) * g  (no mean subtraction, no bias)."""
    def __init__(self, d, eps=1e-6):
        super().__init__(); self.eps = eps; self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


class GELUMLP(nn.Module):
    def __init__(self, d, mult=4):
        super().__init__(); self.up = nn.Linear(d, mult * d); self.down = nn.Linear(mult * d, d)

    def forward(self, x):
        return self.down(F.gelu(self.up(x)))


class SwiGLU(nn.Module):
    """Gated MLP: down( silu(gate(x)) * up(x) ). hidden ~ 8/3 d keeps the parameter count of a 4d GELU MLP."""
    def __init__(self, d, hidden=None):
        super().__init__(); hidden = hidden or int(8 * d / 3)
        self.gate = nn.Linear(d, hidden, bias=False); self.up = nn.Linear(d, hidden, bias=False)
        self.down = nn.Linear(hidden, d, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


# ---------------------------------------------------------------- rotary embeddings
def rope_cos_sin(T, hd, base=10000.0, device=None):
    inv = 1.0 / (base ** (torch.arange(0, hd, 2, device=device).float() / hd))
    ang = torch.outer(torch.arange(T, device=device).float(), inv)          # (T, hd/2)
    return ang.cos(), ang.sin()


def apply_rope(x, cos, sin):
    """x: (B, H, T, hd). Rotate each (even, odd) pair by a position-dependent angle."""
    x1, x2 = x[..., 0::2], x[..., 1::2]
    out = torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
    return out.flatten(-2)


# ---------------------------------------------------------------- attention (MHA / MQA / GQA / sliding window)
class Attention(nn.Module):
    """n_kv_head == n_head -> MHA, == 1 -> MQA, in between -> GQA. window -> sliding-window causal attention.
    Set `keep_attn=True` to store the last attention probabilities in `self.last_attn` (B, H, T, T)."""
    def __init__(self, d, n_head, n_kv_head=None, rope=False, window=None, causal=True, bias=True):
        super().__init__()
        self.h, self.hkv = n_head, n_kv_head or n_head
        assert d % n_head == 0 and n_head % self.hkv == 0
        self.hd = d // n_head; self.rope, self.window, self.causal = rope, window, causal
        self.q = nn.Linear(d, n_head * self.hd, bias=bias)
        self.k = nn.Linear(d, self.hkv * self.hd, bias=bias)
        self.v = nn.Linear(d, self.hkv * self.hd, bias=bias)
        self.o = nn.Linear(d, d, bias=bias)
        self.keep_attn = False; self.last_attn = None

    def mask(self, T, device):
        i = torch.arange(T, device=device)[:, None]; j = torch.arange(T, device=device)[None, :]
        m = (j <= i) if self.causal else torch.ones(T, T, dtype=torch.bool, device=device)
        if self.window is not None:
            m = m & (i - j < self.window) if self.causal else m & ((i - j).abs() < self.window)
        return m

    def forward(self, x, pad_mask=None):
        B, T, _ = x.shape
        q = self.q(x).view(B, T, self.h, self.hd).transpose(1, 2)
        k = self.k(x).view(B, T, self.hkv, self.hd).transpose(1, 2)
        v = self.v(x).view(B, T, self.hkv, self.hd).transpose(1, 2)
        if self.rope:
            cos, sin = rope_cos_sin(T, self.hd, device=x.device)
            q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        if self.hkv != self.h:                                            # share KV heads across query groups
            k = k.repeat_interleave(self.h // self.hkv, 1); v = v.repeat_interleave(self.h // self.hkv, 1)
        m = self.mask(T, x.device)[None, None]
        if pad_mask is not None:                                          # (B, T) True = real token
            m = m & pad_mask[:, None, None, :]
        if self.keep_attn:
            s = (q @ k.transpose(-1, -2)) / math.sqrt(self.hd)
            p = s.masked_fill(~m, float("-inf")).softmax(-1)
            self.last_attn = p.detach(); y = p @ v
        else:
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=m)
        return self.o(y.transpose(1, 2).reshape(B, T, -1))


# ---------------------------------------------------------------- mixture of experts
class MoE(nn.Module):
    """Token-choice top-k routing over `n_exp` SwiGLU experts (+ optional always-on shared expert).
    forward returns (y, aux) where aux is the Switch-style load-balancing loss."""
    def __init__(self, d, n_exp=4, top_k=1, hidden=None, shared=False, aux_coef=0.01):
        super().__init__()
        self.n_exp, self.top_k, self.aux_coef = n_exp, top_k, aux_coef
        self.router = nn.Linear(d, n_exp, bias=False)
        self.experts = nn.ModuleList([SwiGLU(d, hidden) for _ in range(n_exp)])
        self.shared = SwiGLU(d, hidden) if shared else None
        self.last_probs = None; self.last_idx = None

    def forward(self, x):
        B, T, D = x.shape; xf = x.reshape(-1, D)
        probs = self.router(xf).softmax(-1)                               # (N, E)
        w, idx = probs.topk(self.top_k, dim=-1); w = w / w.sum(-1, keepdim=True)
        out = torch.zeros_like(xf)
        for e, expert in enumerate(self.experts):
            tok, slot = (idx == e).nonzero(as_tuple=True)
            if tok.numel():
                out[tok] += w[tok, slot, None] * expert(xf[tok])
        if self.shared is not None:
            out = out + self.shared(xf)
        frac = F.one_hot(idx[:, 0], self.n_exp).float().mean(0)           # fraction routed (top-1)
        aux = self.aux_coef * self.n_exp * (frac * probs.mean(0)).sum()
        self.last_probs, self.last_idx = probs.detach().view(B, T, -1), idx.detach().view(B, T, -1)
        return out.view(B, T, D), aux


# ---------------------------------------------------------------- state-space model (Mamba-style selective SSM)
class MambaBlock(nn.Module):
    """Simplified Mamba: in_proj -> causal depthwise conv -> selective SSM (input-dependent dt, B, C) -> gate -> out_proj.
    The scan is a plain python loop over time: slow but crystal clear."""
    def __init__(self, d, d_state=16, expand=2, d_conv=4):
        super().__init__()
        self.di, self.N = expand * d, d_state
        self.in_proj = nn.Linear(d, 2 * self.di, bias=False)
        self.conv = nn.Conv1d(self.di, self.di, d_conv, groups=self.di, padding=d_conv - 1)
        self.x_proj = nn.Linear(self.di, 1 + 2 * d_state, bias=False)     # -> dt (rank 1 broadcast), B, C
        self.dt_proj = nn.Linear(1, self.di)
        self.A_log = nn.Parameter(torch.log(torch.arange(1, d_state + 1).float()).repeat(self.di, 1))
        self.D = nn.Parameter(torch.ones(self.di))
        self.out_proj = nn.Linear(self.di, d, bias=False)
        self.last_dt = None

    def forward(self, x):
        B, T, _ = x.shape
        xz = self.in_proj(x); u, z = xz.chunk(2, -1)
        u = F.silu(self.conv(u.transpose(1, 2))[..., :T].transpose(1, 2))
        dt, Bm, Cm = self.x_proj(u).split([1, self.N, self.N], -1)
        dt = F.softplus(self.dt_proj(dt))                                 # (B,T,di)  input-dependent step size
        self.last_dt = dt.detach()
        A = -torch.exp(self.A_log)                                        # (di, N)  stable (negative) dynamics
        h = x.new_zeros(B, self.di, self.N); ys = []
        for t in range(T):
            dA = torch.exp(dt[:, t, :, None] * A)                         # discretised A
            dBu = dt[:, t, :, None] * Bm[:, t, None, :] * u[:, t, :, None]
            h = dA * h + dBu                                              # h_t = A h_{t-1} + B u_t
            ys.append((h * Cm[:, t, None, :]).sum(-1))                    # y_t = C h_t
        y = torch.stack(ys, 1) + u * self.D
        return self.out_proj(y * F.silu(z))


# ---------------------------------------------------------------- generic pre-norm block + decoder LM wrapper
class Block(nn.Module):
    """x = x + mixer(norm1(x));  x = x + ffn(norm2(x)).  ffn may return (y, aux) (MoE)."""
    def __init__(self, d, mixer, ffn, norm=RMSNorm):
        super().__init__(); self.n1, self.n2 = norm(d), norm(d); self.mixer, self.ffn = mixer, ffn

    def forward(self, x):
        x = x + self.mixer(self.n1(x))
        y = self.ffn(self.n2(x)); aux = x.new_zeros(())
        if isinstance(y, tuple):
            y, aux = y
        return x + y, aux


class DecoderLM(nn.Module):
    """token emb (+ optional learned positions) -> blocks -> norm -> tied head.
    Returns logits, or (logits, aux_loss) when `aux` blocks (MoE) are present."""
    def __init__(self, vocab, d, blocks, norm=RMSNorm, learned_pos=0, has_aux=False):
        super().__init__()
        self.tok = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(learned_pos, d) if learned_pos else None
        self.blocks = nn.ModuleList(blocks); self.norm = norm(d)
        self.head = nn.Linear(d, vocab, bias=False); self.head.weight = self.tok.weight
        self.has_aux = has_aux
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)
            if getattr(m, "bias", None) is not None:
                nn.init.zeros_(m.bias)

    def forward(self, idx):
        x = self.tok(idx)
        if self.pos is not None:
            x = x + self.pos(torch.arange(idx.size(1), device=idx.device))
        aux = x.new_zeros(())
        for b in self.blocks:
            x, a = b(x); aux = aux + a
        logits = self.head(self.norm(x))
        return (logits, aux) if self.has_aux else logits
