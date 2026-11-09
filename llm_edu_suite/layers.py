"""
Building blocks of a modern decoder-only transformer.

Contents
--------
RMSNorm                      the normalization used by Llama, Mistral, DeepSeek and Gemma
precompute_freqs_cis         RoPE frequency table (complex), with optional Llama-3.1 style scaling
apply_rope                   rotates q/k by absolute position, which gives relative-position attention for free
Attention variants           all share forward(x, freqs_cis, start_pos=None) -> (B, T, C)
    GroupedQueryAttention    MHA / GQA / MQA with KV-cache (+ optional QK-norm)
    MultiHeadLatentAttention DeepSeek MLA with compressed KV-cache
    SlidingWindowAttention   local attention (Mistral, Gemma local layers)
    LongformerAttention      sliding window + global tokens
    BlockSparseAttention     content-dependent block selection (NSA/DSA-style, simplified)
    LinearAttention          O(T) attention with a recurrent state (parallel + step forms)
SwiGLU                       gated FFN
MoE                          mixture of experts with aux loss and optional shared expert
AuxFreeRouter                DeepSeek-V3 style bias-balanced router

Shape conventions used throughout: B is the batch, T the sequence length, C is d_model, H the number
of heads and Dh the head dimension (C // H).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# Normalization
class RMSNorm(nn.Module):
    """y = x / sqrt(mean(x^2) + eps) * gamma.

    Unlike LayerNorm there is no mean-centering and no bias; rescaling alone turns out to be enough
    (Zhang & Sennrich 2019) and it is a little cheaper.
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))  # gamma; starts at 1 so the layer begins as a pure rescale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute in fp32 even when x is bf16, because the sum of squares needs the extra precision.
        xf = x.float()
        rms = torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (xf * rms).type_as(x) * self.weight


# Rotary position embeddings
def precompute_freqs_cis(
    head_dim: int,
    max_seq_len: int,
    base: float = 10000.0,
    scaling: dict | None = None,
) -> torch.Tensor:
    """
    Return e^{i * m * theta_i} for every position m and pair i as a complex tensor
    of shape (max_seq_len, head_dim // 2).

    With theta_i = base^(-2i / head_dim), the first pairs rotate quickly and give fine local
    resolution, while the last pairs rotate slowly and carry long-range signal. Think of a clock
    with many hands turning at different speeds.

    `scaling` (optional) applies Llama-3.1 style frequency scaling to extend the context window:
        {"factor": 8.0, "orig_len": 8192, "low": 1.0, "high": 4.0}
    High-frequency pairs are left alone, low-frequency pairs are divided by `factor`, and the ones
    in between are blended smoothly (a form of NTK-aware / YaRN scaling).
    """
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))  # (Dh/2,)
    if scaling is not None:
        factor, orig_len = scaling["factor"], scaling["orig_len"]
        low, high = scaling.get("low", 1.0), scaling.get("high", 4.0)
        wavelen = 2 * math.pi / inv_freq                       # how many tokens one full rotation takes
        low_wl, high_wl = orig_len / low, orig_len / high
        new = torch.empty_like(inv_freq)
        for i, (f, w) in enumerate(zip(inv_freq, wavelen)):
            if w < high_wl:                                    # high frequency: keep as is
                new[i] = f
            elif w > low_wl:                                   # low frequency: stretch it
                new[i] = f / factor
            else:                                              # in between: interpolate smoothly
                s = (orig_len / w - low) / (high - low)
                new[i] = (1 - s) * f / factor + s * f
        inv_freq = new
    positions = torch.arange(max_seq_len).float()
    angles = torch.outer(positions, inv_freq)                  # (T, Dh/2): m * theta_i
    return torch.polar(torch.ones_like(angles), angles)        # cos + i sin


def apply_rope(x: torch.Tensor, freqs_cis: torch.Tensor, start_pos: int = 0) -> torch.Tensor:
    """
    Rotate pairs of dimensions of x according to their absolute position.
    x: (B, T, H, Dh), real. The result has the same shape.

    Each pair (x_{2i}, x_{2i+1}) is treated as a complex number and multiplied by e^{i m theta_i}.
    Since rotations compose by adding angles, the dot product q_m · k_n ends up depending only on
    the offset (n - m), which is exactly the relative-position behaviour we want.
    """
    B, T, H, Dh = x.shape
    assert Dh % 2 == 0, "RoPE needs an even head dimension"
    # x may be a non-contiguous split or view (MLA's rotary slice, for example). view_as_complex needs
    # an even storage offset and even strides, and a fresh contiguous copy guarantees both.
    xc = torch.view_as_complex(x.float().contiguous().reshape(B, T, H, Dh // 2, 2))
    f = freqs_cis[start_pos : start_pos + T].reshape(1, T, 1, Dh // 2)
    return torch.view_as_real(xc * f).flatten(-2).type_as(x)


# Masks
def causal_mask(T: int, device=None) -> torch.Tensor:
    """(T, T) bool mask that is True where query i may attend to key j, i.e. j <= i."""
    return torch.tril(torch.ones(T, T, dtype=torch.bool, device=device))


def sliding_window_mask(T: int, window: int, device=None) -> torch.Tensor:
    """Causal mask that also requires |i - j| < window."""
    i = torch.arange(T, device=device)[:, None]
    j = torch.arange(T, device=device)[None, :]
    return (j <= i) & (i - j < window)


def _causal_args(Tq: int, Tk: int, device=None):
    """
    Return (attn_mask, is_causal) for SDPA, given query length Tq and key length Tk (Tk >= Tq).

    There are three cases:
      Tq == Tk     training or prefill. SDPA's built-in is_causal=True is enough, no explicit mask.
      Tq == 1      decoding a single new token. No mask at all, since that token may see the whole cache.
      1 < Tq < Tk  several new tokens appended to an existing cache. We need an explicit mask: query i
                   sits at absolute position Tk - Tq + i, so it may see keys j <= Tk - Tq + i.
    """
    if Tq == Tk:
        return None, Tq > 1
    if Tq == 1:
        return None, False
    mask = torch.ones(Tq, Tk, dtype=torch.bool, device=device).tril(Tk - Tq)
    return mask, False


def longformer_mask(T: int, window: int, global_idx, device=None) -> torch.Tensor:
    """Sliding window plus a few global positions that every token can see and that can see every
    (earlier) token themselves."""
    m = sliding_window_mask(T, window, device)
    c = causal_mask(T, device)
    g = torch.zeros(T, dtype=torch.bool, device=device)
    idx = [i for i in global_idx if i < T]
    if idx:
        g[idx] = True
    m = m | (c & g[None, :])  # global columns: everyone can look at the global tokens
    m = m | (c & g[:, None])  # global rows: the global tokens can look at everyone before them
    return m


# Attention variants
class GroupedQueryAttention(nn.Module):
    """
    Causal attention where groups of query heads share n_kv_heads key/value heads.
    With n_kv_heads == n_heads this is plain MHA, with n_kv_heads == 1 it is MQA, and anything
    in between is GQA (Llama 3, Mistral). Fewer KV heads means a proportionally smaller KV-cache.

    It supports a KV-cache for inference: call it with start_pos=0 on the prompt (prefill), then
    with start_pos=t and one new token per step (decode).
    The optional QK-norm (an RMSNorm on q and k per head) keeps attention logits from growing
    without bound, which helps training stability.
    """

    def __init__(self, d_model: int, n_heads: int, n_kv_heads: int, qk_norm: bool = False, dropout: float = 0.0,
                 causal: bool = True):
        super().__init__()
        assert d_model % n_heads == 0 and n_heads % n_kv_heads == 0
        self.causal = causal                                       # False: every token sees every other (an encoder)
        self.H, self.Hkv, self.Dh = n_heads, n_kv_heads, d_model // n_heads
        self.rep = n_heads // n_kv_heads
        self.wq = nn.Linear(d_model, n_heads * self.Dh, bias=False)
        self.wk = nn.Linear(d_model, n_kv_heads * self.Dh, bias=False)
        self.wv = nn.Linear(d_model, n_kv_heads * self.Dh, bias=False)
        self.wo = nn.Linear(n_heads * self.Dh, d_model, bias=False)
        self.q_norm = RMSNorm(self.Dh) if qk_norm else None
        self.k_norm = RMSNorm(self.Dh) if qk_norm else None
        self.dropout = dropout
        self.cache_k: torch.Tensor | None = None
        self.cache_v: torch.Tensor | None = None
        self.supports_cache = causal                               # an encoder reads the whole input at once

    def clear_cache(self) -> None:
        self.cache_k = self.cache_v = None

    def kv_cache_bytes_per_token(self, dtype_bytes: int = 2) -> int:
        return 2 * self.Hkv * self.Dh * dtype_bytes

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor, start_pos: int | None = None) -> torch.Tensor:
        B, T, _ = x.shape
        q = self.wq(x).view(B, T, self.H, self.Dh)
        k = self.wk(x).view(B, T, self.Hkv, self.Dh)
        v = self.wv(x).view(B, T, self.Hkv, self.Dh)
        if self.q_norm is not None:
            q, k = self.q_norm(q), self.k_norm(k)

        pos = start_pos or 0
        q, k = apply_rope(q, freqs_cis, pos), apply_rope(k, freqs_cis, pos)

        if start_pos is not None:                                  # inference with a KV-cache
            if start_pos == 0 or self.cache_k is None:
                self.cache_k, self.cache_v = k, v
            else:
                self.cache_k = torch.cat([self.cache_k, k], dim=1)
                self.cache_v = torch.cat([self.cache_v, v], dim=1)
            k, v = self.cache_k, self.cache_v

        # Repeat the KV heads so they line up with the query heads. Optimized kernels index into the
        # shared heads instead of copying them; the copy here just keeps the code easy to read.
        k = k.repeat_interleave(self.rep, dim=2)
        v = v.repeat_interleave(self.rep, dim=2)

        q, k, v = (t.transpose(1, 2) for t in (q, k, v))           # (B, H, T, Dh)
        mask, is_causal = _causal_args(q.shape[2], k.shape[2], q.device) if self.causal else (None, False)
        y = F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, is_causal=is_causal, dropout_p=self.dropout if self.training else 0.0
        )
        return self.wo(y.transpose(1, 2).reshape(B, T, -1))


class MultiHeadLatentAttention(nn.Module):
    """
    Multi-head Latent Attention (DeepSeek-V2/V3).

    The keys and values of all heads are derived from a single low-rank latent c_kv
    (kv_lora_rank dims), and that latent is the only thing we cache. RoPE cannot be pushed through
    the compression, because the rotation depends on position while the up-projection does not.
    So each head is split into a position-free part (qk_nope_dim, reconstructed from the latent)
    and a small rotary part (qk_rope_dim) that is computed directly and shared across heads, MQA-style.

    The cache costs kv_lora_rank + qk_rope_dim values per token per layer. For DeepSeek-V3 that is
    512 + 64 = 576, compared with 2 * 128 * 128 = 32768 for a 128-head MHA.

    This is the "naive" form, which rebuilds k and v from the cached latent. Production code uses
    the "absorb" trick instead: it folds the up-projection into the query so attention runs against
    the latent directly.
    """

    def __init__(self, d_model: int, n_heads: int, kv_lora_rank: int, qk_nope_dim: int, qk_rope_dim: int, v_dim: int,
                 q_lora_rank: int | None = None):
        super().__init__()
        self.H, self.Dn, self.Dr, self.Dv, self.r = n_heads, qk_nope_dim, qk_rope_dim, v_dim, kv_lora_rank
        qdim = n_heads * (qk_nope_dim + qk_rope_dim)
        if q_lora_rank:                                            # DeepSeek compresses q as well
            self.wq = nn.Sequential(nn.Linear(d_model, q_lora_rank, bias=False), RMSNorm(q_lora_rank),
                                    nn.Linear(q_lora_rank, qdim, bias=False))
        else:
            self.wq = nn.Linear(d_model, qdim, bias=False)
        self.wkv_a = nn.Linear(d_model, kv_lora_rank + qk_rope_dim, bias=False)   # x -> [c_kv | k_pe]
        self.kv_norm = RMSNorm(kv_lora_rank)
        self.wkv_b = nn.Linear(kv_lora_rank, n_heads * (qk_nope_dim + v_dim), bias=False)  # c_kv -> k_nope, v
        self.wo = nn.Linear(n_heads * v_dim, d_model, bias=False)
        self.scale = 1.0 / math.sqrt(qk_nope_dim + qk_rope_dim)
        self.cache_ckv: torch.Tensor | None = None
        self.cache_kpe: torch.Tensor | None = None
        self.supports_cache = True

    def clear_cache(self) -> None:
        self.cache_ckv = self.cache_kpe = None

    def kv_cache_bytes_per_token(self, dtype_bytes: int = 2) -> int:
        return (self.r + self.Dr) * dtype_bytes

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor, start_pos: int | None = None) -> torch.Tensor:
        B, T, _ = x.shape
        pos = start_pos or 0
        # The rotary sub-dimension only needs part of the RoPE table, namely the first Dr/2 frequency pairs.
        f = freqs_cis[:, : self.Dr // 2]

        q = self.wq(x).view(B, T, self.H, self.Dn + self.Dr)
        q_nope, q_pe = q.split([self.Dn, self.Dr], dim=-1)
        q_pe = apply_rope(q_pe, f, pos)

        kv = self.wkv_a(x)
        c_kv, k_pe = kv.split([self.r, self.Dr], dim=-1)
        c_kv = self.kv_norm(c_kv)                                              # (B, T, r)      <- cached
        k_pe = apply_rope(k_pe.view(B, T, 1, self.Dr), f, pos)                 # (B, T, 1, Dr)  <- cached

        if start_pos is not None:
            if start_pos == 0 or self.cache_ckv is None:
                self.cache_ckv, self.cache_kpe = c_kv, k_pe
            else:
                self.cache_ckv = torch.cat([self.cache_ckv, c_kv], dim=1)
                self.cache_kpe = torch.cat([self.cache_kpe, k_pe], dim=1)
            c_kv, k_pe = self.cache_ckv, self.cache_kpe
        Tk = c_kv.shape[1]

        kv_up = self.wkv_b(c_kv).view(B, Tk, self.H, self.Dn + self.Dv)       # decompress the latent (naive form)
        k_nope, v = kv_up.split([self.Dn, self.Dv], dim=-1)
        k_pe = k_pe.expand(B, Tk, self.H, self.Dr)

        q_full = torch.cat([q_nope, q_pe], dim=-1).transpose(1, 2)            # (B, H, T, Dn+Dr)
        k_full = torch.cat([k_nope, k_pe], dim=-1).transpose(1, 2)            # (B, H, Tk, Dn+Dr)
        v = v.transpose(1, 2)
        mask, is_causal = _causal_args(T, Tk, x.device)
        y = F.scaled_dot_product_attention(q_full, k_full, v, attn_mask=mask, is_causal=is_causal, scale=self.scale)
        return self.wo(y.transpose(1, 2).reshape(B, T, self.H * self.Dv))


class _ProjMHA(nn.Module):
    """The q/k/v/o projections (plain MHA) shared by the mask-based attention variants below."""

    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.H, self.Dh = n_heads, d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)

    def split(self, x: torch.Tensor, freqs_cis: torch.Tensor):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=-1)
        q = apply_rope(q.view(B, T, self.H, self.Dh), freqs_cis).transpose(1, 2)
        k = apply_rope(k.view(B, T, self.H, self.Dh), freqs_cis).transpose(1, 2)
        v = v.view(B, T, self.H, self.Dh).transpose(1, 2)
        return q, k, v                                                           # (B, H, T, Dh)

    def merge(self, y: torch.Tensor) -> torch.Tensor:
        B, H, T, Dh = y.shape
        return self.out(y.transpose(1, 2).reshape(B, T, H * Dh))


class SlidingWindowAttention(nn.Module):
    """
    Each token attends only to the previous `window` tokens, so the cost is O(T * window).

    Be aware that with a boolean mask SDPA still computes the full T x T score matrix and then throws
    most of it away. The real savings come from kernels that skip whole blocks (FlexAttention) or
    from truncating the KV-cache to `window` entries. This version has no KV-cache and simply
    recomputes over the full input.
    """

    def __init__(self, d_model: int, n_heads: int, window: int):
        super().__init__()
        self.p, self.window = _ProjMHA(d_model, n_heads), window
        self.supports_cache = False

    def forward(self, x, freqs_cis, start_pos=None):
        q, k, v = self.p.split(x, freqs_cis)
        mask = sliding_window_mask(x.shape[1], self.window, x.device)
        return self.p.merge(F.scaled_dot_product_attention(q, k, v, attn_mask=mask))


class LongformerAttention(nn.Module):
    """Sliding window plus fixed global positions (Longformer, 2020).

    We only reproduce the attention pattern; the paper also gives the global tokens their own
    q/k/v projections.
    """

    def __init__(self, d_model: int, n_heads: int, window: int, global_idx=(0,)):
        super().__init__()
        self.p, self.window, self.global_idx = _ProjMHA(d_model, n_heads), window, tuple(global_idx)
        self.supports_cache = False

    def forward(self, x, freqs_cis, start_pos=None):
        q, k, v = self.p.split(x, freqs_cis)
        mask = longformer_mask(x.shape[1], self.window, self.global_idx, x.device)
        return self.p.merge(F.scaled_dot_product_attention(q, k, v, attn_mask=mask))


class BlockSparseAttention(nn.Module):
    """
    Content-dependent block-sparse attention, a simplified take on NSA / DSA:
      1. group the keys into blocks of `block` tokens,
      2. let a cheap indexer score each block for each query (here, q · mean of the keys in the block),
      3. have each query attend to its top-`top_blocks` blocks plus a local window, causally.
    The pattern is dynamic, because the model decides it, and block-structured, which suits GPUs.
    We build it as a mask to keep things readable; real kernels gather only the selected blocks.
    """

    def __init__(self, d_model: int, n_heads: int, block: int = 16, top_blocks: int = 2, local_window: int = 32):
        super().__init__()
        self.p = _ProjMHA(d_model, n_heads)
        self.block, self.top_blocks, self.local = block, top_blocks, local_window
        self.supports_cache = False
        self.last_mask: torch.Tensor | None = None  # kept around so notebooks can inspect or plot it

    def forward(self, x, freqs_cis, start_pos=None):
        q, k, v = self.p.split(x, freqs_cis)
        B, H, T, Dh = q.shape
        nb = math.ceil(T / self.block)
        pad = nb * self.block - T
        k_blocks = F.pad(k, (0, 0, 0, pad)).view(B, H, nb, self.block, Dh).mean(dim=3)        # (B, H, nb, Dh)
        scores = torch.einsum("bhtd,bhnd->bhtn", q, k_blocks) / math.sqrt(Dh)                 # (B, H, T, nb)
        q_pos = torch.arange(T, device=x.device)[:, None]
        block_start = (torch.arange(nb, device=x.device) * self.block)[None, :]
        scores = scores.masked_fill(block_start > q_pos, float("-inf"))                         # never pick a block from the future
        top = scores.topk(min(self.top_blocks, nb), dim=-1).indices
        sel = torch.zeros(B, H, T, nb, dtype=torch.bool, device=x.device).scatter_(-1, top, True)
        sel_tokens = sel.repeat_interleave(self.block, dim=-1)[..., :T]
        mask = (sel_tokens | sliding_window_mask(T, self.local, x.device)) & causal_mask(T, x.device)
        self.last_mask = mask.detach()
        return self.p.merge(F.scaled_dot_product_attention(q, k, v, attn_mask=mask))


class LinearAttention(nn.Module):
    """
    Causal linear attention (Katharopoulos et al. 2020). Replacing softmax(qk^T) with phi(q) phi(k)^T
    lets us rewrite the sum over the past as a running state S_t = sum_{s<=t} phi(k_s) v_s^T of size
    Dh x Dh.

    During training we use the parallel form with cumsum; at inference step() updates the state with
    constant memory and no KV-cache. The price is that it loses softmax's sharp "exact lookup" ability,
    which is why it shows up mostly in hybrids (Qwen3-Next, Kimi Linear). RoPE is skipped here, since
    linear-attention models tend to use other position schemes or none at all.
    """

    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.H, self.Dh = n_heads, d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)
        self.supports_cache = False

    @staticmethod
    def phi(x: torch.Tensor) -> torch.Tensor:
        return F.elu(x) + 1.0                       # feature map that is always positive

    def _split(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=-1)
        f = lambda t: t.view(B, T, self.H, self.Dh).transpose(1, 2)
        return self.phi(f(q)), self.phi(f(k)), f(v)

    def forward(self, x, freqs_cis=None, start_pos=None):
        q, k, v = self._split(x)                                        # (B, H, T, Dh)
        kv = torch.einsum("bhtd,bhte->bhtde", k, v)                     # outer product k_t v_t^T at every step
        S = kv.cumsum(dim=2)                                            # running state at each t
        z = k.cumsum(dim=2)                                             # running normalizer
        num = torch.einsum("bhtd,bhtde->bhte", q, S)
        den = torch.einsum("bhtd,bhtd->bht", q, z).unsqueeze(-1) + 1e-6
        y = num / den
        B, H, T, Dh = y.shape
        return self.out(y.transpose(1, 2).reshape(B, T, H * Dh))

    def init_state(self, batch_size: int, device=None):
        S = torch.zeros(batch_size, self.H, self.Dh, self.Dh, device=device)
        z = torch.zeros(batch_size, self.H, self.Dh, device=device)
        return S, z

    @torch.no_grad()
    def step(self, x_t: torch.Tensor, state):
        """Process one token at a time, carrying a fixed-size state (S, z) instead of a growing cache."""
        q, k, v = self._split(x_t)                                      # (B, H, 1, Dh)
        S, z = state
        S = S + torch.einsum("bhtd,bhte->bhde", k, v)
        z = z + k.squeeze(2)
        num = torch.einsum("bhtd,bhde->bhte", q, S)
        den = torch.einsum("bhtd,bhd->bht", q, z).unsqueeze(-1) + 1e-6
        y = num / den
        B, H, T, Dh = y.shape
        return self.out(y.transpose(1, 2).reshape(B, T, H * Dh)), (S, z)


# Feed-forward
class SwiGLU(nn.Module):
    """FFN(x) = W_down( SiLU(W_gate x) * W_up x ).

    This uses three matrices instead of two, so d_ff is set to about 8/3 * d_model to keep the
    parameter count the same as a classic 4 * d_model FFN.
    """

    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.w_gate = nn.Linear(d_model, d_ff, bias=False)
        self.w_up = nn.Linear(d_model, d_ff, bias=False)
        self.w_down = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


def swiglu_hidden_dim(d_model: int, multiple_of: int = 64, ffn_dim_multiplier: float | None = None) -> int:
    """Llama's rule: take 2/3 of 4 * d_model, optionally scale it, then round up to a multiple of `multiple_of`."""
    hidden = int(2 * (4 * d_model) / 3)
    if ffn_dim_multiplier is not None:
        hidden = int(ffn_dim_multiplier * hidden)
    return multiple_of * ((hidden + multiple_of - 1) // multiple_of)


# Mixture of Experts
class AuxFreeRouter(nn.Module):
    """
    DeepSeek-V3 style router: sigmoid scores plus a per-expert bias that only affects which experts
    get *selected*. The bias is nudged online (not by gradient descent) to even out the load, so no
    auxiliary loss is needed.
    """

    def __init__(self, d_model: int, n_experts: int, top_k: int, bias_lr: float = 0.01):
        super().__init__()
        self.E, self.k, self.bias_lr = n_experts, top_k, bias_lr
        self.weight = nn.Linear(d_model, n_experts, bias=False)
        self.register_buffer("bias", torch.zeros(n_experts))

    def forward(self, xf: torch.Tensor):
        scores = torch.sigmoid(self.weight(xf))                              # (N, E)
        _, topk_i = (scores + self.bias).topk(self.k, dim=-1)                # the bias decides which experts run...
        topk_s = scores.gather(-1, topk_i)                                   # ...the raw score decides how much they count
        topk_w = topk_s / (topk_s.sum(-1, keepdim=True) + 1e-9)
        if self.training:
            with torch.no_grad():
                load = F.one_hot(topk_i, self.E).sum((0, 1)).float()
                self.bias -= self.bias_lr * torch.sign(load - load.mean())
        return topk_i, topk_w, scores


class MoE(nn.Module):
    """
    Mixture of Experts FFN with top-k routing.

    routing="softmax" uses a softmax router with the load-balancing auxiliary loss
                      E * sum_e f_e * P_e (Switch Transformer, Mixtral).
    routing="auxfree" uses the AuxFreeRouter (DeepSeek-V3), and the returned aux_loss is 0.
    n_shared > 0 adds shared experts that see every token (DeepSeekMoE).

    Dispatch is a plain Python loop over experts, which is easy to follow but slow. Production
    code uses grouped GEMMs and expert parallelism.
    """

    def __init__(self, d_model: int, d_ff: int, n_experts: int, top_k: int, n_shared: int = 0, routing: str = "softmax"):
        super().__init__()
        self.E, self.k, self.routing = n_experts, top_k, routing
        if routing == "softmax":
            self.router = nn.Linear(d_model, n_experts, bias=False)
        elif routing == "auxfree":
            self.router = AuxFreeRouter(d_model, n_experts, top_k)
        else:
            raise ValueError(routing)
        self.experts = nn.ModuleList(SwiGLU(d_model, d_ff) for _ in range(n_experts))
        self.shared = SwiGLU(d_model, d_ff * n_shared) if n_shared > 0 else None
        self.last_load: torch.Tensor | None = None   # routed slots per expert divided by N tokens; sums to top_k, and k/E each when perfectly balanced

    def forward(self, x: torch.Tensor):
        B, T, C = x.shape
        xf = x.reshape(-1, C)                                                   # (N, C)

        if self.routing == "softmax":
            probs = self.router(xf).softmax(dim=-1)                              # (N, E)
            topk_p, topk_i = probs.topk(self.k, dim=-1)
            topk_p = topk_p / topk_p.sum(dim=-1, keepdim=True)
        else:
            topk_i, topk_p, probs = self.router(xf)

        out = torch.zeros_like(xf)
        for e in range(self.E):
            tok_idx, slot = (topk_i == e).nonzero(as_tuple=True)
            if tok_idx.numel() == 0:
                continue
            y_e = self.experts[e](xf[tok_idx])
            out.index_add_(0, tok_idx, y_e * topk_p[tok_idx, slot, None])

        if self.shared is not None:
            out = out + self.shared(xf)

        with torch.no_grad():
            f_load = F.one_hot(topk_i, self.E).sum(1).float().mean(0)          # (E,), sums to k because each token fills k slots
            # Like the HF Mixtral implementation, we don't divide f by k, so at perfect balance the aux loss is
            # about k rather than 1. That only rescales the effective aux-loss coefficient; the gradient still
            # pushes toward balance.
            self.last_load = f_load
        if self.routing == "softmax":
            P = probs.mean(0)
            aux_loss = self.E * (f_load * P).sum()
        else:
            aux_loss = torch.zeros((), device=x.device)
        return out.view(B, T, C), aux_loss
