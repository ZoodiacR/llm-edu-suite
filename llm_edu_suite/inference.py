"""
Inference techniques beyond plain sampling.

speculative_decode     a small draft model proposes k tokens and the target checks them all in a single
                       forward pass. Rejection sampling decides which ones to keep, in a way that
                       preserves the target's distribution exactly (Leviathan et al. 2023, Chen et al. 2023).
kv_cache_memory        a back-of-the-envelope KV-cache size for a given model, context and batch
batched_generate       left-padded batched generation. It recomputes everything and uses no padding
                       mask, so pad tokens get attended to and RoPE positions shift per row. That is
                       fine for a demo; production engines use proper masks.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _probs(model, ids: torch.Tensor, temperature: float) -> torch.Tensor:
    """Run a full-sequence forward pass (no cache) and return softmax probs for every position: (B, T, V)."""
    logits, _ = model(ids)
    return F.softmax(logits / max(temperature, 1e-6), dim=-1)


@torch.no_grad()
def speculative_decode(target, draft, idx: torch.Tensor, max_new_tokens: int, k: int = 4, temperature: float = 1.0,
                       eos_id: int | None = None) -> tuple[torch.Tensor, dict]:
    """
    Speculative decoding for batch size 1. We favour clarity over throughput here, so both models
    recompute the whole sequence instead of using KV caches.

    Each round:
      1. The draft samples k tokens one at a time: x_1..x_k with probabilities q_i(x_i).
      2. The target computes p_i for all k+1 positions in a single forward pass.
      3. For i = 1..k, accept x_i with probability min(1, p_i(x_i) / q_i(x_i)). At the first
         rejection, sample a replacement from norm(max(0, p_i - q_i)) and end the round.
      4. If all k were accepted, sample one bonus token from p_{k+1}.
    The output follows exactly the same distribution as sampling from the target alone. The speed-up
    comes from the target checking k tokens for roughly the cost of one decode step, because decoding
    is memory-bound rather than compute-bound.
    """
    assert idx.shape[0] == 1, "batch size 1 for clarity"
    target.eval(); draft.eval()
    stats = {"proposed": 0, "accepted": 0, "rounds": 0}
    max_len = min(getattr(getattr(target, "cfg", None), "max_seq_len", 10**9), getattr(getattr(draft, "cfg", None), "max_seq_len", 10**9))
    start_len = idx.shape[1]
    while True:
        n0 = idx.shape[1]
        generated = n0 - start_len
        remaining = min(max_new_tokens - generated, max_len - n0)
        if remaining <= 0:
            break
        if remaining == 1:                                   # with one token left, a plain target step is enough
            idx = torch.cat([idx, torch.multinomial(_probs(target, idx, temperature)[:, -1], 1)], dim=1)
            stats["rounds"] += 1
            break
        # The k proposals plus the bonus token must fit in the remaining budget.
        k_eff = min(k, remaining - 1)
        # Step 1: the draft proposes k tokens.
        draft_ids = idx.clone()
        q_list = []
        for _ in range(k_eff):
            q = _probs(draft, draft_ids, temperature)[:, -1]              # (1, V)
            x = torch.multinomial(q, 1)
            q_list.append(q)
            draft_ids = torch.cat([draft_ids, x], dim=1)
        proposed = draft_ids[:, n0:]                                          # (1, k)
        # Step 2: the target scores every proposal in one pass.
        p_all = _probs(target, draft_ids, temperature)[:, n0 - 1 :]           # the positions that predict tokens n0..n0+k
        # Step 3: accept or reject each proposal in order.
        out_tokens, n_accepted = [], 0
        for i in range(k_eff):
            x = proposed[0, i].item()
            p_i, q_i = p_all[0, i], q_list[i][0]
            ratio = (p_i[x] / q_i[x].clamp(min=1e-12)).item()
            if torch.rand(()).item() < min(1.0, ratio):
                out_tokens.append(x)
                n_accepted += 1
            else:
                resid = (p_i - q_i).clamp(min=0)                               # the residual distribution keeps sampling unbiased
                resid = resid / resid.sum() if resid.sum() > 0 else p_i
                out_tokens.append(torch.multinomial(resid, 1).item())
                break
        else:
            # Step 4: everything was accepted, so take a free bonus token from the target's (k+1)-th distribution.
            out_tokens.append(torch.multinomial(p_all[0, k_eff], 1).item())
        stats["proposed"] += k_eff
        stats["accepted"] += n_accepted
        stats["rounds"] += 1
        idx = torch.cat([idx, torch.tensor([out_tokens], device=idx.device)], dim=1)
        if eos_id is not None and eos_id in out_tokens:
            break
    stats["acceptance_rate"] = stats["accepted"] / max(1, stats["proposed"])
    stats["tokens_per_round"] = (idx.shape[1] - start_len) / max(1, stats["rounds"])
    return idx, stats


def kv_cache_memory(n_layers: int, n_kv_heads: int, head_dim: int, context: int, batch: int = 1, dtype_bytes: int = 2) -> dict:
    """KV-cache size in bytes: 2 (one K and one V) * layers * kv_heads * head_dim * context * batch * dtype_bytes."""
    per_token = 2 * n_layers * n_kv_heads * head_dim * dtype_bytes
    total = per_token * context * batch
    return {"bytes_per_token": per_token, "total_bytes": total, "total_gb": total / 2**30}


@torch.no_grad()
def batched_generate(model, prompts: list[list[int]], max_new_tokens: int, pad_id: int, temperature: float = 0.0,
                     eos_id: int | None = None) -> list[list[int]]:
    """
    Simple batched decoding with left padding, so every prompt's last token sits at the final position.
    The catch is that RoPE positions are shifted differently in each row. That is acceptable for a demo;
    production engines use per-sequence position ids and PagedAttention. We recompute the whole sequence
    each step (no cache) to keep things obviously correct.
    """
    from .model import sample_next_token

    model.eval()
    device = next(model.parameters()).device
    T = max(len(p) for p in prompts)
    ids = torch.full((len(prompts), T), pad_id, dtype=torch.long, device=device)
    for i, p in enumerate(prompts):
        ids[i, T - len(p):] = torch.tensor(p, device=device)
    done = torch.zeros(len(prompts), dtype=torch.bool, device=device)
    for _ in range(max_new_tokens):
        logits, _ = model(ids[:, -model.cfg.max_seq_len:])
        nxt = sample_next_token(logits[:, -1], temperature)
        nxt[done] = pad_id
        ids = torch.cat([ids, nxt], dim=1)
        if eos_id is not None:
            done |= nxt.squeeze(1) == eos_id
            if done.all():
                break
    out = []
    for i, p in enumerate(prompts):
        gen = ids[i, T:].tolist()
        if eos_id is not None and eos_id in gen:
            gen = gen[: gen.index(eos_id)]
        out.append([t for t in gen if t != pad_id])
    return out
