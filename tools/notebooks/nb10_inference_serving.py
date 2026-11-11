"""Notebook 10 - Inference and serving

Builds 10_inference_serving.ipynb. Run: python tools/build_notebooks.py 10
"""
from _common import *  # noqa: F401,F403

# Notebook 10 - Inference and serving
nb = NB()
nb.md(r'''
# 10 · Inference: sampling, KV-cache, speculative decoding, serving math

**What you will learn**
- Prefill vs decode and why decode is memory-bandwidth-bound
- Sampling strategies (greedy, temperature, top-k, top-p, repetition penalty) and their effect
- Measuring the KV-cache speed-up, and sizing cache memory for real models
- **Speculative decoding** implemented from scratch, with acceptance-rate measurement
- Batched generation; what vLLM/SGLang add (PagedAttention, continuous batching, prefix caching)
- Evaluation: TTFT, tokens/s, and checking that optimizations do not change outputs

**Source:** `llm_edu_suite/model.py` (`generate`, `sample_next_token`), `llm_edu_suite/inference.py`.
''')
nb.md(r'''
## Background: the two phases of generation

A causal LM produces one token per forward pass, feeds it back, and repeats. That loop has two very different phases:

```
prompt (P tokens)          generated tokens, one per step
[ t1 t2 ... tP ]  ---->  y1 ---> y2 ---> y3 ---> ... ---> yN
 \___ prefill ___/        \__________ decode _____________/
 one pass, all P tokens   N passes, one new token each
 matrix x matrix          matrix x vector
 compute-bound            memory-bandwidth-bound
```

- **Prefill** processes the whole prompt in one pass. Every weight matrix is multiplied by a $P \times C$ block of activations, so each weight read from memory is used $P$ times. The GPU's arithmetic units are busy: prefill is **compute-bound** and determines the **time to first token (TTFT)**.
- **Decode** produces one token per pass. Every weight is read from memory to be used for a single vector (or for $B$ vectors with batch size $B$). Reading the weights takes far longer than the arithmetic: decode is **memory-bandwidth-bound** and determines the **time per output token (TPOT)**.

### Arithmetic intensity, in one line

For a model with $N$ parameters stored in bf16 (2 bytes each), one decode step for a batch of $B$ sequences performs about $2NB$ FLOPs and reads at least $2N$ bytes of weights:

$$\text{arithmetic intensity} \approx \frac{2NB}{2N} = B \ \text{FLOPs per byte}$$

A modern datacenter GPU can do on the order of a hundred or more FLOPs in the time it reads one byte from memory (its "ridge point" in the roofline model of Williams et al., 2009). At $B = 1$ the GPU is therefore idle most of the time, waiting for memory. This single fact explains most of this notebook:
- the speed ceiling at batch 1 is roughly **memory bandwidth / model size in bytes** (a 7B model in bf16 is 14 GB, so a GPU with 1 TB/s cannot exceed about 70 tokens/s for one sequence, whatever its FLOPs);
- **batching** is almost free until $B$ approaches the ridge point, because the same weight read serves $B$ sequences;
- **quantization** (notebook 11) speeds up decode by reducing the bytes, not the FLOPs;
- **speculative decoding** exploits the idle compute by checking several tokens per weight read.

Pope et al. (2022), *Efficiently Scaling Transformer Inference*, is the standard reference for this analysis.

## Background: decoding strategies

The model outputs logits $z \in \mathbb{R}^V$ for the next token. A **decoding strategy** turns them into a choice. There are two families.

**Mode-seeking** methods look for the most probable text. **Greedy** decoding takes $\arg\max_i z_i$ at each step. **Beam search** keeps the $b$ best partial sequences and approximates the most probable *sequence*. They work well when there is one correct answer (translation, extraction, short factual answers), but for open-ended text they produce bland, repetitive output. Holtzman et al. (2020) showed that maximizing likelihood leads to text that is *more* predictable than anything a human writes, and that repetition is self-reinforcing: once a phrase repeats, the model assigns it even higher probability.

**Sampling** methods draw from the distribution, usually after reshaping it:

**Temperature** divides the logits before the softmax:

$$p_i(T) = \frac{e^{z_i / T}}{\sum_j e^{z_j / T}}$$

$T < 1$ sharpens the distribution (as $T \to 0$ it becomes greedy), $T > 1$ flattens it (as $T \to \infty$ it becomes uniform). Temperature never changes the *ranking* of tokens, only how much mass the top tokens get. The name and the formula come from the Boltzmann distribution in statistical physics.

**Truncation** removes the unreliable tail. The tail of an LLM's distribution contains thousands of tokens that are each very unlikely but together can hold a few percent of the mass; sample from it once and the text can derail, after which the model conditions on its own mistake.
- **Top-k** (Fan, Lewis, Dauphin, 2018) keeps the $k$ most likely tokens. Its weakness is that $k$ is fixed: too many candidates when the model is sure, too few when many continuations are reasonable.
- **Top-p / nucleus** (Holtzman et al., 2020) keeps the smallest set $V_p$ whose total probability reaches $p$, then renormalizes:
$$V_p = \text{smallest } S \text{ with } \sum_{i \in S} p_i \ge p, \qquad \tilde p_i = \frac{p_i \,\mathbb{1}[i \in V_p]}{\sum_{j \in V_p} p_j}$$
  The candidate set adapts to the model's confidence, which is why top-p replaced top-k as the default in most APIs.
- **Min-p** (Nguyen et al., 2024) keeps tokens with $p_i \ge p_{min} \cdot \max_j p_j$, a threshold relative to the top token. It is popular in open-source inference tools, particularly at higher temperatures; how much it improves over top-p is still discussed.

**Repetition control.** The repetition penalty from CTRL (Keskar et al., 2019) divides the positive logits of already-generated tokens by $\theta > 1$ (and multiplies negative ones), making them less likely. The OpenAI-style *presence* and *frequency* penalties instead subtract a constant, or a constant times the count, from those logits. Both are blunt: they also punish repetition that is correct, such as a variable name in code or a name in a story.

| strategy | parameter | adapts to confidence? | typical use |
|---|---|---|---|
| greedy / beam | beam width | no | extraction, translation, deterministic evals |
| temperature | $T$ | no (rescales everything) | always combined with one of the below |
| top-k | $k$ | no | older default (GPT-2 era) |
| top-p | $p$ | yes | default for chat (often $T \approx 0.7$, $p \approx 0.9$) |
| min-p | $p_{min}$ | yes | open-source runtimes, creative writing |

Two practical notes. The **order** in which libraries apply temperature, top-k and top-p differs, so the same parameter values can give slightly different distributions in different engines. And **$T = 0$ is not perfectly deterministic** in a production server: the floating-point result of a matmul can depend on the batch it was computed in, so near-ties can flip between runs. Structured output (notebook 16) is also a decoding strategy: it masks the logits of every token that would violate a grammar or JSON schema.

## Background: the KV-cache

In causal attention, the key and value vectors of a past position never change once computed, because they depend only on tokens up to that position. Recomputing them at every step is pure waste, so every inference engine stores them in a **KV-cache**. Without it, step $t$ recomputes projections for all $t$ previous tokens, $O(N^2)$ projection work for $N$ tokens; with it, each step computes projections for one new token and attends over the cached ones, $O(N)$ projection work in total (the attention itself still reads all $t$ cached entries per step).

The cache is not free. Per token it stores keys and values for every layer and KV head:

$$\text{bytes per token} = 2 \times n_{layers} \times n_{kv\_heads} \times d_{head} \times \text{bytes per value}$$

and the total grows with context length times batch size. In serving, the cache, not the weights, usually decides how many sequences fit on a GPU. That is why so many architecture changes target it: multi-query attention (Shazeer, 2019), grouped-query attention (Ainslie et al., 2023), multi-head latent attention (DeepSeek-V2, 2024), sliding-window attention, and fp8 KV-cache quantization.

## Background: serving many users

| technique | problem it solves | key reference |
|---|---|---|
| static batching | weight reads are amortized over the batch | — |
| continuous (iteration-level) batching | in static batching, finished sequences leave idle slots until the longest one ends; here the scheduler adds and removes sequences at every decode step | Yu et al. (2022), *Orca* |
| PagedAttention | contiguous per-sequence cache buffers waste memory to fragmentation and over-reservation; storing the cache in fixed-size blocks with a block table (like virtual memory pages) nearly eliminates the waste and allows sharing blocks | Kwon et al. (2023), *Efficient Memory Management for Large Language Model Serving with PagedAttention* (vLLM) |
| prefix caching | many requests share a system prompt or few-shot examples; reuse their KV blocks instead of re-running prefill | Zheng et al. (2024), *SGLang* (RadixAttention) |
| chunked prefill | a long prompt's prefill stalls all decoding sequences; split it into chunks mixed with decode steps | Agrawal et al. (2024), *Sarathi-Serve* |
| prefill/decode disaggregation | the two phases want different hardware trade-offs; run them on separate GPU pools | Zhong et al. (2024), *DistServe* |
| CUDA graphs | kernel launch overhead dominates small decode steps; record the step once and replay it | — |

vLLM, SGLang and TensorRT-LLM combine most of these. The metrics they are tuned for are TTFT, TPOT, throughput at a given latency target (often called "goodput"), and cost per million tokens.

## Background: speculative decoding and why it is exact

Decode leaves the GPU's compute idle, and the target model can score $k+1$ positions in one pass for nearly the cost of one. Speculative decoding (Leviathan, Kalman, Matias, 2023; and independently Chen et al., 2023, under the name speculative sampling; an earlier greedy-only form is Stern, Shazeer, Uszkoreit, 2018) turns that into speed:

```
draft  (small, fast):  proposes  d1 d2 d3 d4          (k = 4 cheap steps)
target (big, exact):   one pass scores positions 1..5 in parallel
                       accept d1  accept d2  reject d3 -> resample from residual
output this round:     d1 d2 r3                       (3 tokens for 1 target pass)
```

For each proposed token $x \sim q(\cdot)$ (draft distribution) with target distribution $p(\cdot)$ at the same position:
1. accept $x$ with probability $\min\big(1, \tfrac{p(x)}{q(x)}\big)$;
2. at the first rejection, sample a replacement from the **residual distribution** $p'(x) = \dfrac{\max(0,\, p(x) - q(x))}{\sum_y \max(0,\, p(y) - q(y))}$ and stop the round;
3. if all $k$ are accepted, sample one bonus token from the target's distribution at position $k+1$, which the same pass already computed.

**Why the output distribution is exactly $p$.** Let $\beta = \sum_y \min(p(y), q(y))$ be the total acceptance probability. The probability of emitting token $x$ is the probability of proposing and accepting it, plus the probability of rejecting and then resampling it:

$$P(x) = q(x)\min\Big(1, \frac{p(x)}{q(x)}\Big) + (1-\beta)\,p'(x) = \min(p(x), q(x)) + \max(0, p(x) - q(x)) = p(x)$$

using $1 - \beta = \sum_y \max(0, p(y) - q(y))$. No approximation is involved: a bad draft makes the method slow, never wrong. Note that $\beta = 1 - \text{TV}(p, q)$, so the per-token acceptance rate is one minus the total variation distance between draft and target.

**Expected speed-up.** If each token is accepted independently with probability $\alpha$, the expected number of tokens produced per target pass is

$$\mathbb{E}[\text{tokens per round}] = \frac{1 - \alpha^{k+1}}{1 - \alpha}$$

which tends to $k+1$ as $\alpha \to 1$. The wall-clock gain is smaller, because the draft's $k$ forward passes are not free; Leviathan et al. derive the improvement factor $\frac{1-\alpha^{k+1}}{(1-\alpha)(kc+1)}$ where $c$ is the cost of one draft step relative to one target step. So you want a draft that is both cheap (small $c$) and well aligned with the target (large $\alpha$), and $k$ is tuned between the two.

**When it does not help.** At large batch sizes decode is no longer memory-bound, the spare compute disappears, and the extra verification work competes with real requests. Acceptance also drops at high temperature (both distributions are flatter and disagree more) and on text the draft model knows poorly. The draft must use the same tokenizer as the target.

**Modern variants** remove the separate draft model: Medusa (Cai et al., 2024) adds extra decoding heads to the target, EAGLE (Li et al., 2024) drafts from the target's own hidden features, prompt-lookup / n-gram drafting copies spans from the context (very effective for editing and RAG), and DeepSeek-V3's multi-token-prediction module can be reused as a draft. All of them keep the same accept/reject rule, which is what guarantees correctness.

### Key papers

- Holtzman, Buys, Du, Forbes, Choi (2020), *The Curious Case of Neural Text Degeneration*.
- Fan, Lewis, Dauphin (2018), *Hierarchical Neural Story Generation*.
- Keskar et al. (2019), *CTRL: A Conditional Transformer Language Model for Controllable Generation*.
- Pope et al. (2022), *Efficiently Scaling Transformer Inference*.
- Shazeer (2019), *Fast Transformer Decoding: One Write-Head is All You Need*.
- Ainslie et al. (2023), *GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints*.
- Yu et al. (2022), *Orca: A Distributed Serving System for Transformer-Based Generative Models*.
- Kwon et al. (2023), *Efficient Memory Management for Large Language Model Serving with PagedAttention*.
- Leviathan, Kalman, Matias (2023), *Fast Inference from Transformers via Speculative Decoding*.
- Chen et al. (2023), *Accelerating Large Language Model Decoding with Speculative Sampling*.
- Cai et al. (2024), *Medusa*; Li et al. (2024), *EAGLE*.
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

**What the cell measures.** For each strategy it prints how many tokens keep non-zero probability (truncation), the probability of the top token (sharpness) and the **entropy** $H = -\sum_i p_i \log p_i$ in nats. Entropy is the cleanest single number for "how random will the next token be": $e^{H}$ is the effective number of equally likely choices, so a uniform choice among 80 words would give $H = \ln 80 \approx 4.4$ nats. Temperature changes entropy smoothly and keeps every candidate alive; top-k and top-p cut candidates to zero and renormalize the survivors, which also raises their individual probabilities.
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

The equality check is the important half. A KV-cache is an *exact* optimization: with greedy decoding, the cached and uncached paths must produce the same tokens, because they compute the same function. Small floating-point differences between the two code paths can in principle flip a near-tie, which is why serious engines test cache correctness on many prompts. The second cell then sizes the cache with the formula from the Background, $2 \times n_{layers} \times n_{kv\_heads} \times d_{head} \times 2$ bytes in bf16, to show how much KV-sharing (GQA) and compression (MLA) reduce it.
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
for name, args in {"70B with MHA (64 KV)": dict(n_layers=80, n_kv_heads=64, head_dim=128), "Llama-3-70B (GQA 8)": dict(n_layers=80, n_kv_heads=8, head_dim=128),
                   "Llama-3-8B (GQA 8)": dict(n_layers=32, n_kv_heads=8, head_dim=128)}.items():
    m = kv_cache_memory(context=8192, batch=1, **args)
    print(f"{name:20s} {m['bytes_per_token']/1024:6.0f} KB/token -> {m['total_gb']:5.1f} GB for one 8k-token sequence")
print("\nDeepSeek-V3 MLA: 61 layers x 576 values x 2 bytes =", 61 * 576 * 2 / 1024, "KB/token")
''')
nb.md(r'''
## 3. Speculative decoding

Decode is memory-bound: verifying `k` tokens costs about the same as generating 1. So let a cheap **draft** propose `k` tokens, and have the **target** verify them in one forward pass, accepting each with probability `min(1, p/q)` and resampling from `norm(max(0, p−q))` at the first rejection. The output distribution is *exactly* the target's; speed comes from accepting several tokens per target call.

**What to expect from the numbers.** With $k = 4$ the ceiling is $k + 1 = 5$ tokens per target pass, reached only when every draft token is accepted. Under the independence assumption from the Background, the expected value is $(1 - \alpha^{5}) / (1 - \alpha)$ for acceptance rate $\alpha$, so you can check the measured `tokens_per_round` against the measured `acceptance_rate`. Our synthetic language is highly predictable inside words, which favours a high acceptance rate; real chat text at temperature 1 with a much smaller draft typically gives lower values. Remember that tokens per round is not wall-clock speed-up: on real hardware you would also pay for the draft's $k$ forward passes.
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

**How close is "close enough"?** Each frequency is an estimate from $N = 400$ samples, so its standard error is $\sqrt{p(1-p)/N}$, at most $\sqrt{0.25/400} = 0.025$. Two correct samplers will therefore disagree by a couple of percentage points just by chance; a systematic gap several times larger than that would point to a bug, for example forgetting to renormalize the residual distribution. A more formal version of this test is a chi-squared test on the two count tables.
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

**Why left padding.** In a batch, prompts have different lengths, but a decoder generates at the *end* of each row. Padding on the left aligns every prompt's last token in the same column, so each step appends one new column for the whole batch. The padded positions must be masked out of attention (and position ids adjusted), otherwise the model would attend to pad tokens and its outputs would differ from unbatched generation. The demo `batched_generate` keeps things minimal and does neither (its docstring says so); in the cell below every prompt has the same 8 tokens, so no padding actually occurs and the issue does not arise. Padding also wastes compute on positions that carry no information, which is one of the inefficiencies that continuous batching and PagedAttention remove: they let each sequence occupy exactly the cache it needs.
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

**The latency-throughput trade-off.** Total generation time for one request is roughly $\text{TTFT} + N_{out} \times \text{TPOT}$. Batching raises aggregate throughput (tokens per second across all users) almost for free while decode is memory-bound, but once the batch is large enough to make each step compute-bound, every extra sequence makes the step slower and TPOT grows for everyone. A serving system therefore picks an operating point: the largest batch that still meets the latency target (for example, a TTFT and TPOT budget at the 99th percentile). Benchmarks that report only peak throughput, or only single-request latency, hide that choice.
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

''')
nb.md(r'''
## 6. From scratch to real weights: SmolLM2 inside our TransformerLM

Everything so far ran on models we trained ourselves on a toy corpus. The question a skeptic should ask is: *is this code really an LLM, or something that only looks like one?* The strongest answer is to take a real, published model, copy its weights into **our** `TransformerLM`, and check that it computes the same thing. That is what Raschka does with GPT-2 in *Build a Large Language Model (From Scratch)* (chapter 5), and what every framework that runs other people's models (llama.cpp, MLX, vLLM, TensorRT-LLM) does when it adds a new architecture.

### What a checkpoint is

A published model is three things:

| part | file | what it holds |
|---|---|---|
| architecture | code (`modeling_llama.py` in `transformers`, `model.py` here) | the computation: which layers, in which order |
| hyperparameters | `config.json` | sizes and constants: hidden size, layers, heads, KV heads, RoPE base, norm epsilon, tied embeddings |
| weights | `model.safetensors` | a flat dictionary from a tensor **name** to a tensor, nothing else |

`safetensors` is the current standard format: a header with names, shapes and dtypes followed by raw bytes, so it loads without running any code (unlike Python pickle, which could) and can be memory-mapped. Porting a model is therefore a mapping problem: build the same architecture from the config, then put each named tensor where your code expects it.

SmolLM2 (Hugging Face, 2024; Apache 2.0) is a Llama-family model: pre-norm RMSNorm, RoPE, grouped-query attention, SwiGLU, no biases, tied input and output embeddings. That is exactly the block built in notebooks 03 and 05, so the mapping is one to one:

| Hugging Face (`config.json`) | ours (`ModelConfig`) | SmolLM2-135M |
|---|---|---|
| `hidden_size` | `d_model` | 576 |
| `num_hidden_layers` | `n_layers` | 30 |
| `num_attention_heads` / `num_key_value_heads` | `n_heads` / `n_kv_heads` | 9 / 3 (GQA, 3 query heads per KV head) |
| `intermediate_size` | `d_ff` | 1536 |
| `rope_parameters.rope_theta` | `rope_base` | 100000 |
| `rms_norm_eps` | `norm_eps` | 1e-5 |
| `tie_word_embeddings` | `tie_embeddings` | true |

| Hugging Face tensor (`model.layers.{i}.`) | ours (`blocks.{i}.`) |
|---|---|
| `input_layernorm` | `attn_norm` |
| `self_attn.q_proj` / `k_proj` / `v_proj` / `o_proj` | `attn.wq` / `wk` / `wv` / `wo` |
| `post_attention_layernorm` | `ffn_norm` |
| `mlp.gate_proj` / `up_proj` / `down_proj` | `ffn.w_gate` / `w_up` / `w_down` |
| `model.embed_tokens`, `model.norm` | `tok_emb`, `norm` (and `lm_head`, which shares `tok_emb`'s tensor) |

### The one subtle part: two RoPE conventions

RoPE rotates pairs of dimensions of each query and key (notebook 03). There are two ways to choose the pairs:

- **interleaved** (Meta's original Llama code, and ours): dimensions $(2i, 2i+1)$ form plane $i$;
- **rotate_half** (GPT-NeoX, and Hugging Face's Llama): dimensions $(i, i + d_h/2)$ form plane $i$.

Each convention is self-consistent, and they are equivalent up to a reordering of the dimensions inside each head. Let $P$ be the permutation that sends position $2i \mapsto i$ and $2i+1 \mapsto i + d_h/2$. The attention score is a dot product, and a dot product does not change when you permute both vectors the same way:

$$q^\top k = (Pq)^\top (Pk)$$

So if we permute the **rows** of $W_q$ and $W_k$ inside every head, our interleaved rotation acts on exactly the planes Hugging Face's rotate_half acts on, with the same frequency on each plane, and every attention score comes out identical. $W_v$ and $W_o$ are never rotated and are copied as they are. (Hugging Face's script that converted Meta's checkpoints applied the inverse permutation, which is why this step exists at all.) Forget it and nothing crashes: the model loads and runs, and because the content of each token still flows through attention, much of what it says still looks plausible. Only the position information is scrambled, so the model gets quietly worse. Silent degradation is the dangerous kind of bug, and the reason porting code must always be checked numerically.

### How to check equivalence

Two implementations of the same model never agree bit for bit: the order of floating-point additions differs (another kernel, another fused operation), and rounding errors accumulate over 30 layers. So the test is statistical, done in fp32 where rounding is small:

- the **largest absolute difference** between the two sets of logits (expected around $10^{-4}$ for logits of size ~10);
- **top-1 agreement**: the share of positions where both models pick the same next token (expected 100 %);
- the **KL divergence** between the two next-token distributions (expected near 0);
- the decisive end-to-end test: **greedy generation** from the same prompt must produce the same text.

In bf16 the differences grow by orders of magnitude and greedy outputs can diverge after a few dozen tokens on near-ties, which is normal and is the reason equivalence is tested in fp32.
''')
nb.code(r'''
from transformers import AutoModelForCausalLM, AutoTokenizer
from llm_edu_suite.model import config_from_hf_llama, load_hf_llama, rope_rotate_half_to_interleaved
import inspect

HF_ID = "HuggingFaceTB/SmolLM2-135M-Instruct"
tok = AutoTokenizer.from_pretrained(HF_ID)
hf = AutoModelForCausalLM.from_pretrained(HF_ID, dtype=torch.float32).to(device).eval()

cfg_real = config_from_hf_llama(hf.config, max_seq_len=2048)   # 2048 positions is plenty here; SmolLM2 allows 8192
print({k: getattr(cfg_real, k) for k in ["vocab_size", "d_model", "n_layers", "n_heads", "n_kv_heads", "d_ff", "rope_base", "norm_eps", "tie_embeddings"]})
ours = TransformerLM(cfg_real).to(device).eval()
copied = load_hf_llama(ours, hf.state_dict())
n_hf = sum(p.numel() for p in hf.parameters())
print(f"{len(copied)} tensors copied | parameters: ours {ours.num_params():,} vs Hugging Face {n_hf:,}")
print("\nThe q/k row permutation, from llm_edu_suite/model.py:\n")
print(inspect.getsource(rope_rotate_half_to_interleaved))
''')
nb.code(r'''
def compare(model_a, model_b, prompts):
    # max |logit difference|, top-1 agreement and mean KL(a || b) over every position of every prompt
    diffs, agree, kls = [], [], []
    with torch.no_grad():
        for p in prompts:
            ids = tok(p, return_tensors="pt").input_ids.to(device)
            la = model_a(ids).logits if hasattr(model_a, "config") else model_a(ids)[0]
            lb = model_b(ids).logits if hasattr(model_b, "config") else model_b(ids)[0]
            diffs.append((la - lb).abs().max().item())
            agree.append((la.argmax(-1) == lb.argmax(-1)).float().mean().item())
            kls.append(F.kl_div(lb.log_softmax(-1), la.log_softmax(-1), log_target=True, reduction="none").sum(-1).mean().item())
    return max(diffs), float(np.mean(agree)), float(np.mean(kls))

PROMPTS = ["The capital of France is", "def fibonacci(n):\n    ", "Large language models are trained to predict",
           "In 1969, the first humans to walk on the Moon were", "Photosynthesis converts sunlight into"]
d, a, k = compare(hf, ours, PROMPTS)
print(f"ours vs Hugging Face, fp32:          max |Δlogit| {d:.2e} | top-1 agreement {a:.1%} | KL {k:.2e}")

wrong = TransformerLM(cfg_real).to(device).eval()
load_hf_llama(wrong, hf.state_dict(), permute_rope=False)        # the same weights, without the q/k reordering
d, a, k = compare(hf, wrong, PROMPTS)
print(f"without the RoPE permutation:        max |Δlogit| {d:.2e} | top-1 agreement {a:.1%} | KL {k:.2e}")
''')
nb.code(r'''
messages = [{"role": "user", "content": "Explain in three sentences what a transformer language model does."}]
ids = tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt", return_dict=True)["input_ids"].to(device)
N_NEW = 80
with torch.no_grad():
    out_hf = hf.generate(ids, max_new_tokens=N_NEW, do_sample=False, pad_token_id=tok.eos_token_id)[0, ids.shape[1]:]
out_ours = ours.generate(ids, max_new_tokens=N_NEW, temperature=0.0, eos_id=tok.eos_token_id)[0, ids.shape[1]:]
out_wrong = wrong.generate(ids, max_new_tokens=N_NEW, temperature=0.0, eos_id=tok.eos_token_id)[0, ids.shape[1]:]
n = min(len(out_hf), len(out_ours))
print("Hugging Face :", tok.decode(out_hf, skip_special_tokens=True))
print("ours         :", tok.decode(out_ours, skip_special_tokens=True))
print(f"identical greedy tokens: {torch.equal(out_hf[:n], out_ours[:n])} ({n} tokens compared)")
print("no permutation:", tok.decode(out_wrong, skip_special_tokens=True))
''')
nb.code(r'''
# Speed: the same weights, our generate() with its KV-cache vs Hugging Face's generate(), greedy, fp32, batch 1
def tokens_per_s(fn, reps=3):
    fn()                                                   # warm-up
    times = []
    for _ in range(reps):
        if device == "cuda": torch.cuda.synchronize()
        t0 = time.time(); fn()
        if device == "cuda": torch.cuda.synchronize()
        times.append(time.time() - t0)
    return 64 / min(times)
with torch.no_grad():
    t_hf = tokens_per_s(lambda: hf.generate(ids, max_new_tokens=64, min_new_tokens=64, do_sample=False, pad_token_id=tok.eos_token_id))
t_ours = tokens_per_s(lambda: ours.generate(ids, max_new_tokens=64, temperature=0.0))
print(f"Hugging Face generate: {t_hf:5.0f} tokens/s | our TransformerLM.generate: {t_ours:5.0f} tokens/s")
del hf, ours, wrong
if device == "cuda": torch.cuda.empty_cache()
''')
nb.md(r'''
**What the numbers say.** All 272 tensors map one to one, and the parameter count is identical (134,515,008). In fp32 the two implementations agree to within $3 \times 10^{-5}$ on logits of size ~10, pick the same next token at 100 % of positions, and their next-token distributions differ by a KL of $2.4 \times 10^{-8}$: what is left is floating-point rounding from different kernels and orders of addition. The end-to-end test passes too. From the same chat prompt, our `generate()` (with the KV-cache of section 2) and Hugging Face's produce the same 76 tokens, word for word. The code of notebooks 02 to 05 is a real LLM.

Without the q/k permutation the model still loads and runs, and that is the instructive part. It still picks the same top token at 68 % of positions, because a lot can be guessed from the content of the tokens alone, but the logits differ by up to 24 and the KL is 0.56. Asked the same question, it answers "You are an AI assistant.": fluent, plausible and unrelated to the question. Nothing raises an error, so only a numerical comparison against the reference catches the bug.

Speed is the same for both, about 17 to 18 tokens per second here. At batch 1, a 30-layer model in fp32 is limited by the cost of launching hundreds of small GPU kernels for every token (the caveat of section 5), not by the arithmetic, so a clean implementation and the reference library end up in the same place. Production engines are faster because they fuse kernels, capture the whole decode step in a CUDA graph, quantize and batch many requests (vLLM, llama.cpp).

## Results

| question | measured | lesson |
|---|---|---|
| what sampling does to the distribution (after a word boundary) | entropy T=1.0 4.25 nats, T=0.7 4.18, T=1.5 4.36; top-k 20 keeps 20 candidates (2.89 nats), top-p 0.9 keeps 61 (4.01) | temperature reshapes, truncation cuts |
| diversity over 16 continuations | greedy: 1 unique, distinct-2 0.02; sampling: 16 unique, distinct-2 0.42–0.64 | greedy collapses to one answer |
| KV-cache | identical outputs; 1193 → 1044 ms (1.1×) at this short length | exact, and the speed-up grows with length |
| KV memory for one 8k-token sequence | 70B with MHA 20.0 GB; Llama-3-70B (GQA 8) 2.5 GB; Llama-3-8B 1.0 GB; DeepSeek-V3 MLA 68.625 KB per token | cache size decides how many users fit |
| speculative decoding | acceptance 98 % (47 of 48), 4.62 tokens per target forward pass | exact speed-up when the draft is good |
| batching | batch 1 → 16: 88 → 1520 tokens/s aggregate, TTFT 20–26 ms | decode is memory-bound; batching is nearly free |
| porting SmolLM2-135M into our model | 134,515,008 params match; max \|Δlogit\| 3.05e-05, top-1 agreement 100 %, KL 2.4e-08; 76 identical greedy tokens | the code of notebooks 02–05 is a real LLM |
| without the q/k permutation | top-1 agreement 68.2 %, KL 0.557, max \|Δlogit\| 23.6; fluent but unrelated answer | silent bugs need a numerical check |
| speed at batch 1, fp32 | Hugging Face 18 tokens/s vs ours 17 | kernel launches, not arithmetic, set the pace |

## In production

**Serving engines.**
- vLLM, SGLang and TensorRT-LLM for GPU servers; llama.cpp and Ollama for local and CPU or Apple hardware; managed APIs when you do not run models yourself.
- They all use paged KV-caches, continuous batching, prefix caching, fused kernels and CUDA graphs. A typical start: `vllm serve <model> --enable-prefix-caching --kv-cache-dtype fp8`.
- KV-cache is mandatory; GQA, MLA and quantized caches decide how many sequences fit.

**Making decode faster.**
- Speculative decoding (draft models, EAGLE-style heads, n-gram lookup) usually gives 2–3× at low batch; its value shrinks as batch grows, because the GPU is no longer idle.
- Separating prefill and decode onto different GPUs (disaggregated serving) at large scale.
- Quantized weights and caches (notebook 11).

**Measure on your traffic.** TTFT, time per output token, throughput and cost per million tokens, at the batch sizes and prompt lengths you actually see (notebook 12 load-tests this).

**Sampling settings.** Chat usually runs around T=0.7 with top-p 0.9; code, extraction and evaluation runs near T=0; RL rollouts at T=1. Reasoning models often ship recommended settings: use them.

**Failure modes.**
- A ported or converted model that runs but is subtly wrong (RoPE layout, chat template, tokenizer). Always compare logits against the reference, as done here.
- Greedy decoding loops in open-ended generation.
- Out-of-memory under load because KV-cache size was not budgeted for long prompts.
- Benchmarks at batch 1 that say nothing about behaviour under concurrency.

**Next:** `11_quantization.ipynb`, which shrinks the bytes that decode has to read.
''')
nb.save(notebook_path("10_inference_serving.ipynb"))

print("10_inference_serving.ipynb written")
