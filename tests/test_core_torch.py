"""Tests for the from-scratch PyTorch core (layers, model, training, inference). Skipped if torch is missing."""
import copy
import math
import random

import pytest

torch = pytest.importorskip("torch")

from llm_edu_suite.layers import (RMSNorm, precompute_freqs_cis, apply_rope, GroupedQueryAttention, MultiHeadLatentAttention,
                             SlidingWindowAttention, LongformerAttention, BlockSparseAttention, LinearAttention, MoE, AuxFreeRouter,
                             sliding_window_mask, longformer_mask)
from llm_edu_suite.model import ModelConfig, TransformerLM, sample_next_token
from llm_edu_suite.data import synthetic_structured_corpus, train_val_split, get_batch
from llm_edu_suite.training import TrainConfig, train_lm, lr_schedule, make_optimizer
from llm_edu_suite.inference import speculative_decode, kv_cache_memory, batched_generate
from llm_edu_suite.evaluation import perplexity, exact_match_rate

torch.manual_seed(0)
C, H = 64, 8


# Layers

def test_rmsnorm_matches_torch_and_is_scale_invariant():
    x = torch.randn(2, 5, 16) * 7 + 3
    rms = RMSNorm(16)
    assert torch.allclose(rms(x), torch.nn.RMSNorm(16)(x), atol=1e-5)
    assert torch.allclose(rms(3 * x), rms(x), atol=1e-5)


def test_rope_relative_position_only():
    Dh = 32
    f = precompute_freqs_cis(Dh, 512)
    q, k = torch.randn(Dh), torch.randn(Dh)
    def score(m, n):
        qm = apply_rope(q.view(1, 1, 1, Dh), f, m); kn = apply_rope(k.view(1, 1, 1, Dh), f, n)
        return (qm * kn).sum().item()
    assert abs(score(5, 0) - score(205, 200)) < 1e-4
    assert torch.allclose(q.norm(), apply_rope(q.view(1, 1, 1, Dh), f, 9).norm(), atol=1e-5)
    assert torch.allclose(apply_rope(q.view(1, 1, 1, Dh), f, 0).flatten(), q, atol=1e-6)


def test_rope_scaling_leaves_high_freq_intact():
    a = precompute_freqs_cis(64, 4)[1].angle()
    b = precompute_freqs_cis(64, 4, scaling={"factor": 8.0, "orig_len": 8192})[1].angle()
    assert torch.allclose(a[:4], b[:4])
    assert torch.allclose(a[-4:] / 8, b[-4:], atol=1e-6)


@pytest.mark.parametrize("n_kv", [8, 2, 1])
def test_gqa_kv_cache_matches_full_forward(n_kv):
    attn = GroupedQueryAttention(C, H, n_kv)
    f = precompute_freqs_cis(C // H, 64)
    x = torch.randn(1, 12, C)
    full = attn(x, f)
    attn.clear_cache()
    parts = [attn(x[:, :8], f, start_pos=0)] + [attn(x[:, t:t + 1], f, start_pos=t) for t in range(8, 12)]
    assert torch.allclose(full, torch.cat(parts, 1), atol=1e-4)
    # several new tokens appended at once (needs the explicit causal mask inside the cache path)
    attn.clear_cache()
    parts = [attn(x[:, :6], f, start_pos=0), attn(x[:, 6:9], f, start_pos=6), attn(x[:, 9:12], f, start_pos=9)]
    assert torch.allclose(full, torch.cat(parts, 1), atol=1e-4)


def test_mla_kv_cache_matches_full_forward():
    attn = MultiHeadLatentAttention(C, H, kv_lora_rank=16, qk_nope_dim=6, qk_rope_dim=4, v_dim=8)
    f = precompute_freqs_cis(C // H, 64)
    x = torch.randn(2, 10, C)
    full = attn(x, f)
    attn.clear_cache()
    parts = [attn(x[:, :6], f, start_pos=0)] + [attn(x[:, t:t + 1], f, start_pos=t) for t in range(6, 10)]
    assert torch.allclose(full, torch.cat(parts, 1), atol=1e-4)
    assert attn.kv_cache_bytes_per_token() < GroupedQueryAttention(C, H, 2).kv_cache_bytes_per_token()


def test_mask_variants_shapes_and_causality():
    f = precompute_freqs_cis(C // H, 64)
    x = torch.randn(2, 24, C)
    for m in (SlidingWindowAttention(C, H, 5), LongformerAttention(C, H, 5, (0, 10)), BlockSparseAttention(C, H, 4, 2, 4)):
        y = m(x, f)
        assert y.shape == x.shape
    # masks are causal
    assert not sliding_window_mask(10, 3).triu(1).any()
    assert not longformer_mask(10, 3, [0, 5]).triu(1).any()
    bs = BlockSparseAttention(C, H, 4, 2, 4); bs(x, f)
    assert not bs.last_mask.triu(1).any()


def test_linear_attention_parallel_equals_recurrent():
    lin = LinearAttention(C, H)
    x = torch.randn(2, 9, C)
    y_par = lin(x)
    state = lin.init_state(2)
    outs = []
    for t in range(9):
        o, state = lin.step(x[:, t:t + 1], state)
        outs.append(o)
    assert torch.allclose(y_par, torch.cat(outs, 1), atol=1e-4)


def test_moe_routes_every_token_and_aux_loss_is_finite():
    moe = MoE(C, 32, n_experts=4, top_k=2, n_shared=1)
    y, aux = moe(torch.randn(3, 7, C))
    assert y.shape == (3, 7, C) and torch.isfinite(aux)
    assert abs(moe.last_load.sum().item() - 2.0) < 1e-5          # each token goes to exactly top_k experts
    moe2 = MoE(C, 32, n_experts=4, top_k=2, routing="auxfree")
    y2, aux2 = moe2(torch.randn(3, 7, C))
    assert y2.shape == (3, 7, C) and aux2.item() == 0.0


def test_auxfree_router_balances_load():
    r = AuxFreeRouter(C, 8, 2, bias_lr=0.1).train()
    with torch.no_grad():
        r.weight.weight[0] += 3.0
    xf = torch.randn(1024, C)
    loads = []
    for _ in range(10):
        idx, _, _ = r(xf)
        loads.append(torch.nn.functional.one_hot(idx, 8).sum((0, 1)).float())
    assert loads[-1].max() / loads[-1].mean() < loads[0].max() / loads[0].mean()


# Model

@pytest.mark.parametrize("kw", [dict(attn="gqa"), dict(attn="mla", qk_rope_dim=4, qk_nope_dim=4, kv_lora_rank=8, v_dim=8), dict(attn="swa", window=8),
                                dict(attn="longformer", window=8), dict(attn="blocksparse", block=4, top_blocks=2, window=4),
                                dict(attn="linear"), dict(attn="swa", window=8, hybrid_pattern="LLLG"), dict(moe=True, n_experts=4, top_k=2),
                                dict(qk_norm=True, sandwich_norm=True), dict(tie_embeddings=False)])
def test_model_forward_loss_and_generate(kw):
    cfg = ModelConfig(vocab_size=128, max_seq_len=64, d_model=32, n_layers=2, n_heads=4, n_kv_heads=2, **kw)
    m = TransformerLM(cfg)
    idx = torch.randint(0, 128, (2, 17))
    logits, loss = m(idx[:, :-1], idx[:, 1:])
    assert logits.shape == (2, 16, 128)
    assert abs(loss.item() - math.log(128)) < 0.5                 # ~ uniform at init
    out = m.generate(idx[:, :5], 6, temperature=0.8, top_k=10)
    assert out.shape == (2, 11)
    out = m.generate(idx[:, :5], 6, temperature=0.0)
    assert out.shape == (2, 11)


def test_no_future_leak():
    m = TransformerLM(ModelConfig(vocab_size=128, max_seq_len=64, d_model=32, n_layers=2, n_heads=4, n_kv_heads=2))
    x = torch.randint(0, 128, (2, 12))
    x2 = x.clone(); x2[:, -1] = (x2[:, -1] + 1) % 128
    with torch.no_grad():
        a, _ = m(x); b, _ = m(x2)
    assert torch.allclose(a[:, :-1], b[:, :-1], atol=1e-5)
    assert not torch.allclose(a[:, -1], b[:, -1])


def test_cached_generate_equals_recompute_greedy():
    m = TransformerLM(ModelConfig(vocab_size=128, max_seq_len=64, d_model=32, n_layers=2, n_heads=4, n_kv_heads=2))
    idx = torch.randint(0, 128, (1, 6))
    cached = m.generate(idx, 10, temperature=0)
    cur = idx.clone()
    for _ in range(10):
        logits, _ = m(cur)
        cur = torch.cat([cur, logits[:, -1].argmax(-1, keepdim=True)], 1)
    assert torch.equal(cached, cur)


def test_sampling_helpers():
    logits = torch.tensor([[0.0, 10.0, -5.0, 1.0]])
    assert sample_next_token(logits, temperature=0).item() == 1
    for _ in range(20):
        assert sample_next_token(logits, 1.0, top_k=1).item() == 1
        assert sample_next_token(logits, 1.0, top_p=0.01).item() == 1


def test_save_load_roundtrip(tmp_path):
    m = TransformerLM(ModelConfig(vocab_size=64, max_seq_len=32, d_model=16, n_layers=1, n_heads=2, n_kv_heads=1))
    p = tmp_path / "m.pt"
    m.save(str(p))
    m2 = TransformerLM.load(str(p))
    x = torch.randint(0, 64, (1, 8))
    with torch.no_grad():
        assert torch.allclose(m(x)[0], m2(x)[0])


# Training

def test_lr_schedule_shapes():
    assert lr_schedule(0, 1.0, 10, 100) == pytest.approx(0.1)
    assert lr_schedule(10, 1.0, 10, 100) == pytest.approx(1.0)
    assert lr_schedule(99, 1.0, 10, 100) < 0.2
    assert lr_schedule(50, 1.0, 10, 100, kind="wsd") == 1.0
    assert lr_schedule(99, 1.0, 10, 100, kind="wsd") < 0.2


def test_optimizer_groups_exclude_norms_from_decay():
    m = TransformerLM(ModelConfig(vocab_size=64, max_seq_len=32, d_model=16, n_layers=1, n_heads=2, n_kv_heads=1))
    opt = make_optimizer(m, 1e-3, fused=False)
    assert opt.param_groups[0]["weight_decay"] > 0 and opt.param_groups[1]["weight_decay"] == 0
    assert all(p.dim() < 2 for p in opt.param_groups[1]["params"])


def test_training_reduces_loss():
    data = synthetic_structured_corpus(n_tokens=20_000, vocab=128)
    tr, va = train_val_split(data)
    m = TransformerLM(ModelConfig(vocab_size=128, max_seq_len=64, d_model=32, n_layers=2, n_heads=4, n_kv_heads=2))
    res = train_lm(m, tr, va, TrainConfig(steps=60, batch_size=16, block_size=32, max_lr=3e-3, warmup=5, eval_every=30, log=False, amp=False), "cpu")
    assert res.val_losses[-1][1] < res.val_losses[0][1]
    assert perplexity(m, va, block_size=32, batch_size=16, n_batches=3) < 128


# Fine-tuning

def test_speculative_decode_runs_and_returns_requested_length():
    cfg = dict(vocab_size=64, max_seq_len=64, n_heads=2, n_kv_heads=1)
    target = TransformerLM(ModelConfig(d_model=32, n_layers=2, **cfg))
    draft = TransformerLM(ModelConfig(d_model=16, n_layers=1, **cfg))
    idx = torch.randint(0, 64, (1, 5))
    out, stats = speculative_decode(target, draft, idx, max_new_tokens=12, k=3)
    assert out.shape[1] == 5 + 12                                   # always generates exactly max_new_tokens (no eos)
    out1, _ = speculative_decode(target, draft, idx, max_new_tokens=1, k=3)
    assert out1.shape[1] == 6
    assert 0.0 <= stats["acceptance_rate"] <= 1.0 and stats["rounds"] > 0
    # with draft == target, every proposal should be accepted
    out2, stats2 = speculative_decode(target, target, idx, max_new_tokens=12, k=3)
    assert stats2["acceptance_rate"] > 0.9


def test_kv_cache_memory_and_batched_generate():
    m = kv_cache_memory(n_layers=32, n_kv_heads=8, head_dim=128, context=8192)
    assert m["bytes_per_token"] == 2 * 32 * 8 * 128 * 2
    model = TransformerLM(ModelConfig(vocab_size=64, max_seq_len=64, d_model=32, n_layers=1, n_heads=2, n_kv_heads=1))
    outs = batched_generate(model, [[1, 2, 3], [4, 5]], 6, pad_id=0, temperature=0.0)
    assert len(outs) == 2 and all(len(o) <= 6 for o in outs)

