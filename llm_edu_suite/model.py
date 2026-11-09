"""
A configurable decoder-only transformer language model.

    tokens -> embedding -> N x Block -> RMSNorm -> lm_head -> logits

Block (pre-norm):
    x = x + Attn(RMSNorm(x))              # mixes information across tokens
    x = x + FFN(RMSNorm(x))               # transforms each token (dense SwiGLU or MoE)

Every design choice lives in `ModelConfig`: the attention variant (gqa / mla / swa / longformer /
blocksparse / linear), hybrid layer patterns, a dense or MoE FFN, QK-norm and sandwich norm.
`generate()` samples with temperature, top-k and top-p. It uses the KV-cache when every layer
supports one, and otherwise falls back to recomputing over the full sequence.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import (
    RMSNorm, precompute_freqs_cis,
    GroupedQueryAttention, MultiHeadLatentAttention, SlidingWindowAttention,
    LongformerAttention, BlockSparseAttention, LinearAttention,
    SwiGLU, MoE, swiglu_hidden_dim,
)


@dataclass
class ModelConfig:
    vocab_size: int = 512
    max_seq_len: int = 256
    d_model: int = 128
    n_layers: int = 4
    n_heads: int = 8
    n_kv_heads: int = 2                 # GQA groups; set it to n_heads for MHA or to 1 for MQA
    d_ff: int | None = None             # None means "use Llama's rule" (about 8/3 * d_model, rounded)
    dropout: float = 0.0
    # attention
    attn: str = "gqa"                   # gqa | mla | swa | longformer | blocksparse | linear
    hybrid_pattern: str | None = None   # e.g. "LLLG": L = `attn`, G = dense GQA (global)
    window: int = 64                    # local window for swa, longformer and blocksparse
    global_idx: tuple = (0,)            # longformer global positions
    block: int = 16                     # blocksparse block size
    top_blocks: int = 2                 # blocksparse selected blocks per query
    qk_norm: bool = False
    sandwich_norm: bool = False         # Gemma-2 style: also normalize sub-block outputs
    # MLA
    kv_lora_rank: int = 32
    qk_nope_dim: int = 16
    qk_rope_dim: int = 8                # must fit in the head dim (<= d_model // n_heads)
    v_dim: int = 16
    # MoE
    moe: bool = False
    n_experts: int = 4
    top_k: int = 2
    n_shared_experts: int = 0
    moe_routing: str = "softmax"        # softmax | auxfree
    aux_loss_coef: float = 0.01
    moe_every: int = 1                  # put MoE on every k-th layer (1 means all of them); the rest stay dense
    # positions
    rope_base: float = 10000.0
    rope_scaling: dict | None = None    # {"factor": 8.0, "orig_len": 8192}
    tie_embeddings: bool = True
    norm_eps: float = 1e-6              # RMSNorm epsilon (Llama and SmolLM2 use 1e-5)

    def __post_init__(self):
        if self.d_ff is None:
            self.d_ff = swiglu_hidden_dim(self.d_model, multiple_of=8)

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    def to_dict(self) -> dict:
        return asdict(self)


def build_attention(cfg: ModelConfig, layer_idx: int) -> nn.Module:
    kind = cfg.attn
    if cfg.hybrid_pattern:
        kind = "gqa" if cfg.hybrid_pattern[layer_idx % len(cfg.hybrid_pattern)] == "G" else cfg.attn
    if kind == "gqa":
        return GroupedQueryAttention(cfg.d_model, cfg.n_heads, cfg.n_kv_heads, qk_norm=cfg.qk_norm, dropout=cfg.dropout)
    if kind == "mla":
        assert cfg.qk_rope_dim <= cfg.head_dim, "qk_rope_dim must fit in head_dim (shares the RoPE table)"
        return MultiHeadLatentAttention(cfg.d_model, cfg.n_heads, cfg.kv_lora_rank, cfg.qk_nope_dim, cfg.qk_rope_dim, cfg.v_dim)
    if kind == "swa":
        return SlidingWindowAttention(cfg.d_model, cfg.n_heads, cfg.window)
    if kind == "longformer":
        return LongformerAttention(cfg.d_model, cfg.n_heads, cfg.window, cfg.global_idx)
    if kind == "blocksparse":
        return BlockSparseAttention(cfg.d_model, cfg.n_heads, cfg.block, cfg.top_blocks, cfg.window)
    if kind == "linear":
        return LinearAttention(cfg.d_model, cfg.n_heads)
    raise ValueError(f"unknown attention kind: {kind}")


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig, layer_idx: int):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.attn = build_attention(cfg, layer_idx)
        self.ffn_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        use_moe = cfg.moe and (layer_idx % cfg.moe_every == cfg.moe_every - 1)
        self.ffn = (MoE(cfg.d_model, cfg.d_ff, cfg.n_experts, cfg.top_k, cfg.n_shared_experts, cfg.moe_routing)
                    if use_moe else SwiGLU(cfg.d_model, cfg.d_ff))
        self.post_attn_norm = RMSNorm(cfg.d_model, cfg.norm_eps) if cfg.sandwich_norm else nn.Identity()
        self.post_ffn_norm = RMSNorm(cfg.d_model, cfg.norm_eps) if cfg.sandwich_norm else nn.Identity()
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor, start_pos: int | None = None):
        x = x + self.drop(self.post_attn_norm(self.attn(self.attn_norm(x), freqs_cis, start_pos)))
        h = self.ffn_norm(x)
        if isinstance(self.ffn, MoE):
            y, aux = self.ffn(h)
        else:
            y, aux = self.ffn(h), torch.zeros((), device=x.device)
        return x + self.drop(self.post_ffn_norm(y)), aux


class TransformerLM(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList(Block(cfg, i) for i in range(cfg.n_layers))
        self.norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.tok_emb.weight
        self.register_buffer(
            "freqs_cis",
            precompute_freqs_cis(cfg.head_dim, cfg.max_seq_len, cfg.rope_base, cfg.rope_scaling),
            persistent=False,
        )
        self.apply(self._init_weights)
        # The GPT-2 trick: shrink the projections that write into the residual stream by
        # 1/sqrt(2 * n_layers), so that adding up N blocks does not blow up its variance.
        for name, p in self.named_parameters():
            if name.endswith(("wo.weight", "w_down.weight", "out.weight")):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layers))

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    # Forward pass
    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None, start_pos: int | None = None):
        """
        idx: (B, T) token ids. targets: (B, T) next tokens (-100 means ignore), or None.
        start_pos is None during training; for cached inference it is the position of idx[:, 0].
        Returns (logits of shape (B, T, V), loss or None).
        """
        B, T = idx.shape
        assert (start_pos or 0) + T <= self.cfg.max_seq_len, "sequence longer than max_seq_len"
        x = self.tok_emb(idx)
        aux_total = torch.zeros((), device=idx.device)
        for blk in self.blocks:
            x, aux = blk(x, self.freqs_cis, start_pos)
            aux_total = aux_total + aux
        logits = self.lm_head(self.norm(x))
        loss = None
        if targets is not None:
            ce = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-100)
            loss = ce + self.cfg.aux_loss_coef * aux_total / len(self.blocks)
        return logits, loss

    # Helpers
    @property
    def supports_cache(self) -> bool:
        return all(getattr(b.attn, "supports_cache", False) for b in self.blocks)

    def clear_cache(self) -> None:
        for b in self.blocks:
            if hasattr(b.attn, "clear_cache"):
                b.attn.clear_cache()

    def num_params(self, trainable_only: bool = False) -> int:
        return sum(p.numel() for p in self.parameters() if (p.requires_grad or not trainable_only))

    def num_active_params(self) -> int:
        """Parameters actually used per token. With MoE only the top_k chosen experts count."""
        total = self.num_params()
        inactive = 0
        for b in self.blocks:
            if isinstance(b.ffn, MoE):
                per_expert = sum(p.numel() for p in b.ffn.experts[0].parameters())
                inactive += (b.ffn.E - b.ffn.k) * per_expert
        return total - inactive

    def kv_cache_bytes_per_token(self, dtype_bytes: int = 2) -> int:
        return sum(getattr(b.attn, "kv_cache_bytes_per_token", lambda d: 0)(dtype_bytes) for b in self.blocks)

    # Generation
    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0,
                 top_k: int | None = None, top_p: float | None = None, eos_id: int | None = None,
                 repetition_penalty: float = 1.0) -> torch.Tensor:
        """
        Autoregressive sampling. When every layer supports a KV-cache we do one prefill pass and then
        decode one token at a time. Otherwise we recompute the whole sequence at each step, which is
        still correct but costs O(T^2) per token.
        """
        self.eval()
        use_cache = self.supports_cache
        if use_cache:
            self.clear_cache()
            logits, _ = self(idx, start_pos=0)                  # prefill the cache with the prompt
        else:
            logits, _ = self(idx)
        pos = idx.shape[1]
        for _ in range(max_new_tokens):
            next_logits = logits[:, -1, :]
            if repetition_penalty != 1.0:
                next_logits = apply_repetition_penalty(next_logits, idx, repetition_penalty)
            nxt = sample_next_token(next_logits, temperature, top_k, top_p)
            idx = torch.cat([idx, nxt], dim=1)
            if eos_id is not None and (nxt == eos_id).all():
                break
            if pos >= self.cfg.max_seq_len:
                break
            if use_cache:
                logits, _ = self(nxt, start_pos=pos)            # decode: feed only the new token
            else:
                logits, _ = self(idx[:, -self.cfg.max_seq_len:])
            pos += 1
        return idx

    # Saving and loading
    def save(self, path: str) -> None:
        torch.save({"config": self.cfg.to_dict(), "state_dict": self.state_dict()}, path)

    @classmethod
    def load(cls, path: str, map_location="cpu") -> "TransformerLM":
        ckpt = torch.load(path, map_location=map_location)
        cfg = ModelConfig(**ckpt["config"])
        model = cls(cfg)
        model.load_state_dict(ckpt["state_dict"])
        return model


# Sampling utilities
def next_token_distribution(logits: torch.Tensor, temperature: float = 1.0, top_k: int | None = None,
                            top_p: float | None = None) -> torch.Tensor:
    """
    The distribution a sampler actually draws from once temperature, top-k and top-p are applied.
    logits: (B, V). Returns probs (B, V) that sum to 1, with filtered tokens at exactly 0.
      temperature  divides the logits; below 1 it sharpens (more conservative), above 1 it flattens (more diverse)
      top_k        keeps only the k most likely tokens
      top_p        keeps the smallest set of tokens whose cumulative probability reaches p (the "nucleus")
    """
    logits = logits / max(temperature, 1e-6)
    if top_k is not None and top_k < logits.size(-1):
        kth = logits.topk(top_k, dim=-1).values[:, -1:]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    if top_p is not None and top_p < 1.0:                                       # nucleus sampling
        sorted_logits, sorted_idx = logits.sort(dim=-1, descending=True)
        probs = sorted_logits.softmax(dim=-1)
        cum = probs.cumsum(dim=-1)
        remove = (cum - probs) > top_p                                          # keep the token that crosses p, drop the rest
        sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
        logits = torch.full_like(logits, float("-inf")).scatter(-1, sorted_idx, sorted_logits)
    return logits.softmax(dim=-1)


def sample_next_token(logits: torch.Tensor, temperature: float = 1.0, top_k: int | None = None,
                      top_p: float | None = None) -> torch.Tensor:
    """Sample one id per row from logits (B, V) at the last position, returning (B, 1).
    A temperature <= 0 means greedy decoding."""
    if temperature <= 0:
        return logits.argmax(dim=-1, keepdim=True)                              # greedy
    return torch.multinomial(next_token_distribution(logits, temperature, top_k, top_p), num_samples=1)


def apply_repetition_penalty(logits: torch.Tensor, generated: torch.Tensor, penalty: float) -> torch.Tensor:
    """Discourage tokens we have already generated (CTRL paper): positive logits are divided by `penalty`
    and negative ones multiplied by it, so both move toward "less likely"."""
    logits = logits.clone()
    for b in range(logits.size(0)):
        ids = generated[b].unique()
        vals = logits[b, ids]
        logits[b, ids] = torch.where(vals > 0, vals / penalty, vals * penalty)
    return logits


# From scratch to real weights: Hugging Face Llama-family checkpoints in TransformerLM
#
# Llama, Mistral, Qwen2 and SmolLM2 share the block this file builds (pre-norm RMSNorm, RoPE, grouped-query
# attention, SwiGLU, no biases), so their weights can be copied into TransformerLM tensor by tensor. Two details
# need care: the names of the tensors, and the order of the q/k rows, because two RoPE conventions exist.

def config_from_hf_llama(hf_config, max_seq_len: int | None = None) -> ModelConfig:
    """ModelConfig equivalent to a Hugging Face LlamaConfig (or any config with the same fields)."""
    rope = getattr(hf_config, "rope_parameters", None) or getattr(hf_config, "rope_scaling", None) or {}
    rope_theta = getattr(hf_config, "rope_theta", None) or rope.get("rope_theta", 10000.0)
    if rope.get("rope_type", "default") != "default":
        raise NotImplementedError(f"rope_type {rope.get('rope_type')!r} is not supported by this converter")
    if getattr(hf_config, "attention_bias", False) or getattr(hf_config, "mlp_bias", False):
        raise NotImplementedError("biases in attention or MLP are not supported (Llama-family models have none)")
    return ModelConfig(
        vocab_size=hf_config.vocab_size,
        max_seq_len=max_seq_len or hf_config.max_position_embeddings,
        d_model=hf_config.hidden_size,
        n_layers=hf_config.num_hidden_layers,
        n_heads=hf_config.num_attention_heads,
        n_kv_heads=getattr(hf_config, "num_key_value_heads", None) or hf_config.num_attention_heads,
        d_ff=hf_config.intermediate_size,
        dropout=0.0,
        attn="gqa",
        rope_base=float(rope_theta),
        tie_embeddings=bool(getattr(hf_config, "tie_word_embeddings", False)),
        norm_eps=float(hf_config.rms_norm_eps),
    )


def rope_rotate_half_to_interleaved(w: torch.Tensor, n_heads: int, head_dim: int) -> torch.Tensor:
    """
    Reorder the rows of a q or k projection from Hugging Face's RoPE layout to ours.

    Hugging Face rotates dimension i together with i + head_dim/2 ("rotate_half", the GPT-NeoX convention);
    TransformerLM, like Meta's original Llama code, rotates adjacent pairs (2i, 2i+1). Both compute the same
    attention scores if the rows of W_q and W_k are permuted consistently: within each head, our row 2i is
    their row i and our row 2i+1 is their row i + head_dim/2. (Hugging Face's own Llama conversion script
    applies the inverse permutation.) V and the output projection are not rotated, so they need no change.
    """
    d_in = w.shape[1]
    return w.view(n_heads, 2, head_dim // 2, d_in).transpose(1, 2).reshape(n_heads * head_dim, d_in)


def load_hf_llama(model: "TransformerLM", hf_state_dict: dict, permute_rope: bool = True) -> list[str]:
    """
    Copy a Hugging Face Llama-family state dict into `model` (built with config_from_hf_llama). Returns the
    names of our parameters that were filled. Raises if any tensor is missing or has the wrong shape.
    `permute_rope=False` skips the q/k reordering, which is useful only to see what goes wrong without it.
    """
    cfg = model.cfg
    sd = {k[len("model."):] if k.startswith("model.") else k: v for k, v in hf_state_dict.items()}
    fix = (lambda w, h: rope_rotate_half_to_interleaved(w, h, cfg.head_dim)) if permute_rope else (lambda w, h: w)
    mine = {"tok_emb.weight": sd["embed_tokens.weight"], "norm.weight": sd["norm.weight"]}
    if not cfg.tie_embeddings:
        mine["lm_head.weight"] = sd["lm_head.weight"]
    for i in range(cfg.n_layers):
        p, q = f"layers.{i}.", f"blocks.{i}."
        mine.update({
            q + "attn_norm.weight": sd[p + "input_layernorm.weight"],
            q + "ffn_norm.weight": sd[p + "post_attention_layernorm.weight"],
            q + "attn.wq.weight": fix(sd[p + "self_attn.q_proj.weight"], cfg.n_heads),
            q + "attn.wk.weight": fix(sd[p + "self_attn.k_proj.weight"], cfg.n_kv_heads),
            q + "attn.wv.weight": sd[p + "self_attn.v_proj.weight"],
            q + "attn.wo.weight": sd[p + "self_attn.o_proj.weight"],
            q + "ffn.w_gate.weight": sd[p + "mlp.gate_proj.weight"],
            q + "ffn.w_up.weight": sd[p + "mlp.up_proj.weight"],
            q + "ffn.w_down.weight": sd[p + "mlp.down_proj.weight"],
        })
    own = model.state_dict()
    for k, v in mine.items():
        if k not in own or own[k].shape != v.shape:
            raise ValueError(f"cannot load {k}: ours {tuple(own[k].shape) if k in own else None}, theirs {tuple(v.shape)}")
    missing, unexpected = model.load_state_dict({k: v.to(own[k].dtype) for k, v in mine.items()}, strict=False)
    allowed_missing = {"lm_head.weight"} if cfg.tie_embeddings else set()   # tied: same tensor as tok_emb
    if set(missing) - allowed_missing or unexpected:
        raise ValueError(f"missing {sorted(set(missing) - allowed_missing)}, unexpected {sorted(unexpected)}")
    return sorted(mine)
