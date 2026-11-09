"""Notebook 06 - Inference and serving

Builds 06_inference_serving.ipynb. Run: python tools/build_notebooks.py 06
"""
from _common import *  # noqa: F401,F403

# Notebook 06 - Inference and serving
nb = NB()
nb.md(r'''
# 06 · Inference: sampling, KV-cache, speculative decoding, serving math

**What you will learn**
- Prefill vs decode and why decode is memory-bandwidth-bound
- Sampling strategies (greedy, temperature, top-k, top-p, repetition penalty) and their effect
- Measuring the KV-cache speed-up, and sizing cache memory for real models
- **Speculative decoding** implemented from scratch, with acceptance-rate measurement
- Batched generation; what vLLM/SGLang add (PagedAttention, continuous batching, prefix caching)
- Evaluation: TTFT, tokens/s, and checking that optimizations do not change outputs

**Source:** `llm_edu_suite/model.py` (`generate`, `sample_next_token`), `llm_edu_suite/inference.py`.
''')
nb.code(SETUP)
nb.code(r'''
# Train a small target model and an even smaller draft model on the same data. Each takes a few seconds.
from llm_edu_suite.model import ModelConfig, TransformerLM
from llm_edu_suite.data import synthetic_structured_corpus, train_val_split
from llm_edu_suite.training import TrainConfig, train_lm

data = synthetic_structured_corpus(n_tokens=150_000, vocab=512, n_words=80)
train_data, val_data = train_val_split(data)
tcfg = TrainConfig(steps=400, batch_size=32, block_size=64, max_lr=3e-3, warmup=40, eval_every=400, log=False)

seed_everything(0); target = TransformerLM(ModelConfig(vocab_size=512, d_model=192, n_layers=6, n_heads=6, n_kv_heads=2)); train_lm(target, train_data, val_data, tcfg, device)
seed_everything(0); draft  = TransformerLM(ModelConfig(vocab_size=512, d_model=64,  n_layers=2, n_heads=4, n_kv_heads=2)); train_lm(draft,  train_data, val_data, tcfg, device)
print(f"target {human(target.num_params())} params | draft {human(draft.num_params())} params")
''')
nb.md(r'''
## 1. Sampling
The model gives a distribution over `V`; the decoding strategy picks a token. Typical: chat `T=0.7, top_p=0.9`; code/math `T≈0`; RL rollouts `T=1`.

| strategy | what it does to the distribution | effect |
|---|---|---|
| greedy (`T=0`) | always the argmax | deterministic, prone to loops |
| temperature `T` | `softmax(logits / T)` | `T<1` sharper, `T>1` flatter |
| top-k | keep the k most likely tokens | cuts the long tail at a fixed size |
| top-p (nucleus) | keep the smallest set with cumulative prob ≥ p | adaptive: few tokens when confident, many when unsure |
| repetition penalty | down-weights tokens already generated | fewer loops, can hurt when repetition is correct |

In our synthetic language a "word" is a fixed sequence of 2–4 tokens and words are separated by `0`. **Inside** a word the next token is almost certain (all strategies agree); **right after a `0`** the model must choose among ~80 words — that is where the strategies differ. We look at that position.
''')
nb.code(r'''
from llm_edu_suite.model import next_token_distribution

# Find a context that ends with the separator 0, so the next token starts a new word and the model is genuinely uncertain.
pos = int((val_data[40:] == 0).nonzero()[0]) + 40          # start at 40 so the 32-token window below fits
ctx = val_data[pos - 31 : pos + 1].unsqueeze(0).to(device)
with torch.no_grad():
    logits = target(ctx)[0][:, -1, :]

strategies = {"T=1.0": dict(temperature=1.0), "T=0.7": dict(temperature=0.7), "T=0.7 top-k 20": dict(temperature=0.7, top_k=20),
              "T=0.7 top-p 0.9": dict(temperature=0.7, top_p=0.9), "T=1.5": dict(temperature=1.5)}
fig, ax = plt.subplots(figsize=(11, 3.5))
order = logits[0].argsort(descending=True)[:40]
for name, kw in strategies.items():
    p = next_token_distribution(logits, **kw)[0]
    ent = -(p[p > 0] * p[p > 0].log()).sum().item()
    print(f"{name:16s}: candidates with p>0 {int((p > 0).sum()):4d} | top-1 prob {p.max().item():.3f} | entropy {ent:.2f} nats")
    ax.plot(range(40), p[order].cpu(), "o-", ms=3, label=name)
ax.set_xlabel("candidate rank (by model probability)"); ax.set_ylabel("sampling probability")
ax.set_title("the same next-token distribution after each decoding strategy (start of a new word)"); ax.legend(); ax.grid(alpha=.3)
plt.show()
''')
nb.code(r'''
# Measure diversity across 16 continuations of the same prompt. distinct-2 = unique bigrams / all bigrams.
def distinct_n(seqs, n=2):
    grams = [tuple(s[i:i + n]) for s in seqs for i in range(len(s) - n + 1)]
    return len(set(grams)) / max(1, len(grams))

prompt = ctx[:, -8:]
rows = {"greedy": dict(temperature=0), **strategies, "T=0.7 rep. penalty 1.5": dict(temperature=0.7, repetition_penalty=1.5)}
for name, kw in rows.items():
    torch.manual_seed(0)
    outs = [target.generate(prompt, 24, **kw)[0, 8:].tolist() for _ in range(16)]
    print(f"{name:24s} distinct-2 {distinct_n(outs):.2f} | unique continuations {len(set(map(tuple, outs))):2d}/16 | e.g. {outs[0][:12]}")
''')
nb.md(r'''
**Reading the results.** Lower temperature and tighter top-k/top-p concentrate the probability mass (higher top-1 prob, lower entropy) and reduce diversity; `T=1.5` flattens the distribution and gives the most varied — and least faithful — continuations. Greedy produces one continuation 16 times. Pick the strategy per task and **measure** diversity and quality together: more diversity is only good while the outputs stay correct.
''')
nb.md(r'''
## 2. KV-cache: same output, far less work
Without the cache, generating N tokens recomputes k/v for every past token at every step (O(N²) projections). With it, O(N). We verify equality and time both.
''')
nb.code(r'''
import time
def generate_no_cache(model, idx, n):
    model.eval()
    for _ in range(n):
        logits, _ = model(idx)                                  # recomputes the whole prefix at every step
        idx = torch.cat([idx, logits[:, -1].argmax(-1, keepdim=True)], 1)
    return idx

prompt = val_data[:16].unsqueeze(0).to(device)
t0 = time.perf_counter(); a = generate_no_cache(target, prompt, 100); t_nc = time.perf_counter() - t0
t0 = time.perf_counter(); b = target.generate(prompt, 100, temperature=0); t_c = time.perf_counter() - t0
print("identical outputs:", torch.equal(a, b))
print(f"no cache {t_nc*1000:.0f} ms | with cache {t_c*1000:.0f} ms | speed-up {t_nc / t_c:.1f}x  (grows with sequence length)")
''')
nb.code(r'''
from llm_edu_suite.inference import kv_cache_memory
for name, args in {"Llama-2-70B (MHA)": dict(n_layers=80, n_kv_heads=64, head_dim=128), "Llama-3-70B (GQA 8)": dict(n_layers=80, n_kv_heads=8, head_dim=128),
                   "Llama-3-8B (GQA 8)": dict(n_layers=32, n_kv_heads=8, head_dim=128)}.items():
    m = kv_cache_memory(context=8192, batch=1, **args)
    print(f"{name:20s} {m['bytes_per_token']/1024:6.0f} KB/token -> {m['total_gb']:5.1f} GB for one 8k-token sequence")
print("\nDeepSeek-V3 MLA: 61 layers x 576 values x 2 bytes =", 61 * 576 * 2 / 1024, "KB/token")
''')
nb.md(r'''
## 3. Speculative decoding

Decode is memory-bound: verifying `k` tokens costs about the same as generating 1. So let a cheap **draft** propose `k` tokens, and have the **target** verify them in one forward pass, accepting each with probability `min(1, p/q)` and resampling from `norm(max(0, p−q))` at the first rejection. The output distribution is *exactly* the target's; speed comes from accepting several tokens per target call.
''')
nb.code(r'''
from llm_edu_suite.inference import speculative_decode

prompt = val_data[:8].unsqueeze(0).to(device)
torch.manual_seed(0)
out, stats = speculative_decode(target, draft, prompt, max_new_tokens=60, k=4, temperature=1.0)
print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in stats.items()})
print("generated:", out[0, 8:].tolist()[:30], "...")
print(f"\n-> {stats['tokens_per_round']:.2f} tokens per target forward pass (plain decoding = 1.0). "
      f"Acceptance rate {stats['acceptance_rate']:.0%}: the better the draft approximates the target, the bigger the win.")
''')
nb.md(r'''
### Does speculative decoding change the distribution?
It must not. We sample many next-token continuations with both methods and compare the empirical distributions over the first generated token (a statistical check, not a proof — the proof is in the paper).
''')
nb.code(r'''
from collections import Counter
torch.manual_seed(1)
N = 400
plain = Counter(target.generate(prompt, 1, temperature=1.0)[0, -1].item() for _ in range(N))
spec = Counter(speculative_decode(target, draft, prompt, max_new_tokens=2, k=1, temperature=1.0)[0][0, prompt.shape[1]].item() for _ in range(N))
top = [t for t, _ in (plain + spec).most_common(6)]
print("token  : plain  spec")
for t in top:
    print(f"{t:6d} : {plain[t]/N:5.2f}  {spec[t]/N:5.2f}")
''')
nb.md(r'''
## 4. Batched generation
Serving many users means batching: one weight read serves many sequences. The simple version left-pads prompts; real engines (vLLM) use **PagedAttention** (cache in non-contiguous pages, no fragmentation), **continuous batching** (new requests join mid-flight), **prefix caching** (shared system prompts computed once), and **chunked prefill** so long prompts don't stall others.
''')
nb.code(r'''
from llm_edu_suite.inference import batched_generate
prompts = [val_data[i:i+8].tolist() for i in range(0, 64, 8)]
t0 = time.perf_counter(); outs = batched_generate(target, prompts, 32, pad_id=0, temperature=0.0); dt = time.perf_counter() - t0
print(f"{len(prompts)} prompts x 32 tokens in {dt*1000:.0f} ms -> {len(prompts)*32/dt:.0f} tokens/s aggregate (no cache, recompute mode)")
print("first output:", outs[0][:16])
''')
nb.md(r'''
## 5. Evaluation: latency and throughput

The serving metrics: **TTFT** (time to first token; prefill-bound), **tokens/s per sequence** (decode-bound), **aggregate throughput** (grows with batch), and **cost per 1M tokens**. Always report them *together with* a quality check — an optimization that changes outputs is a bug, not a speed-up.
''')
nb.code(r'''
from llm_edu_suite.evaluation import benchmark_generation
for B in (1, 4, 16):
    p = val_data[:8].repeat(B, 1).to(device)
    r = benchmark_generation(target, p, max_new_tokens=48, n_runs=2, temperature=0)
    print(f"batch {B:2d}: TTFT {r['ttft_ms']:6.1f} ms | {r['tokens_per_s']:7.0f} tokens/s aggregate")
print("\nAggregate throughput grows almost linearly with batch size while per-sequence speed barely drops: decode is memory-bound.")
''')
nb.md(r'''
**A GPU caveat that is worth understanding.** With a model this small, each generated token launches dozens of tiny GPU kernels, and the fixed launch cost dominates the actual math. That is why, on a GPU, single-sequence decoding of a toy model can be no faster than on a CPU and the KV-cache speed-up looks modest: there is almost no compute to save. Batching amortizes the launch cost over many sequences, and that is exactly what serving engines do (vLLM's continuous batching, plus CUDA graphs to remove launch overhead). With real models (billions of parameters) the picture flips: the work per token is large, and the KV-cache and batching gains are dramatic.

**Takeaways**
- KV-cache is mandatory; GQA/MLA/quantized caches decide how many sequences fit.
- Speculative decoding gives 2–3× at low batch; its value shrinks as batch grows (the GPU is no longer idle).
- For production: `vllm serve <model> --enable-prefix-caching --kv-cache-dtype fp8` and measure TTFT/TPOT/throughput on *your* traffic.

**Next:** `07_finetuning_sft_lora.ipynb`.
''')
nb.save(notebook_path("06_inference_serving.ipynb"))

print("06_inference_serving.ipynb written")
