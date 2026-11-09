"""Notebook 02 - Attention, RoPE and normalization

Builds 02_attention_rope_norm.ipynb. Run: python tools/build_notebooks.py 02
"""
from _common import *  # noqa: F401,F403

# Notebook 02 - Attention, RoPE and normalization
nb = NB()
nb.md(r'''
# 02 · Normalization, RoPE and attention (MHA → GQA → MLA)

**What you will learn**
- RMSNorm vs LayerNorm and why pre-norm won
- RoPE: how rotating q/k encodes *relative* position, verified numerically
- Scaled dot-product attention computed by hand and checked against PyTorch's fused kernel
- Multi-head → grouped-query → multi-head-latent attention and what each does to the KV-cache
- KV-cache correctness: incremental decoding must equal the full forward pass

**Source:** `llm_edu_suite/layers.py`.
''')
nb.md(r'''
## Background: the transformer block, piece by piece

A decoder-only language model is a stack of $L$ identical blocks sitting between an embedding table and an output head. Each block in a modern model (Llama, Mistral, Qwen, DeepSeek, Gemma) has the same pre-norm structure:

$$h = x + \text{Attn}\big(\text{Norm}(x)\big), \qquad y = h + \text{FFN}\big(\text{Norm}(h)\big)$$

```
x ──┬──► Norm ──► Attention (+RoPE on q, k) ──►(+)──┬──► Norm ──► FFN ──►(+)──► y
    └──────────────── residual ───────────────────┘└────── residual ────────┘
```

The two sub-layers do different jobs. **Attention** is the only place where positions exchange information: it moves information *between* tokens. The **FFN** (notebook 04) processes each position independently: it transforms information *within* a token. The **residual stream** $x$ is the running sum that every sub-layer reads from and writes into, and the **norm** keeps the scale of what each sub-layer reads under control.

This notebook dissects the parts that are about attention and its plumbing:

| component | problem it solves | what the industry uses today | what it replaced |
|---|---|---|---|
| normalization | activations drift in scale through deep stacks, making optimization unstable | pre-norm RMSNorm (sometimes plus QK-norm) | post-norm LayerNorm (original Transformer, BERT) |
| position encoding | attention by itself ignores order | RoPE, often with a scaling rule for long context | sinusoidal or learned absolute embeddings (Transformer, GPT-2, BERT) |
| attention heads | one big attention map can only express one kind of relationship at a time | multi-head attention with grouped K/V (GQA) or latent K/V (MLA) | full multi-head attention (MHA) |
| KV-cache | autoregressive decoding would recompute every past key and value at each step | cache K/V, and shrink it architecturally | no cache (quadratic recomputation) |

The key papers behind this notebook: Vaswani et al. (2017) "Attention Is All You Need"; Ba, Kiros & Hinton (2016) "Layer Normalization"; Zhang & Sennrich (2019) "Root Mean Square Layer Normalization"; Xiong et al. (2020) "On Layer Normalization in the Transformer Architecture"; Su et al. (2021) "RoFormer: Enhanced Transformer with Rotary Position Embedding"; Shazeer (2019) "Fast Transformer Decoding: One Write-Head is All You Need"; Ainslie et al. (2023) "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints"; DeepSeek-AI (2024) "DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model".

### Why the pieces fit the hardware

It helps to keep in mind where the time goes. During **training and prefill**, all $T$ positions are processed at once, so the work is large matrix multiplications and the GPU is limited by compute. During **decoding**, one token is processed per step, so each step does little arithmetic but must read every weight and the entire KV-cache from memory: it is limited by memory bandwidth. Much of the architectural evolution in this notebook (GQA, MLA) is about the second regime: it reduces the bytes read per generated token without hurting what the model can learn.
''')
nb.code(SETUP)
nb.md(r'''
### Normalization: from LayerNorm to pre-norm RMSNorm

**LayerNorm** (Ba et al., 2016) normalizes each token vector to zero mean and unit variance, then applies a learned gain $\gamma$ and bias $\beta$:

$$\text{LayerNorm}(x) = \gamma \odot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta, \qquad \mu = \tfrac{1}{C}\textstyle\sum_i x_i,\ \ \sigma^2 = \tfrac{1}{C}\textstyle\sum_i (x_i - \mu)^2$$

Without some normalization, the residual stream grows as layers add to it, and the inputs each sub-layer sees change scale during training, which forces small learning rates. **RMSNorm** (Zhang & Sennrich, 2019) drops the mean subtraction and the bias. Its authors hypothesised that re-scaling, not re-centering, is what matters, and showed comparable quality with less computation; later large models confirmed it works at scale. The saving is modest in FLOPs but real in practice because normalization is memory-bound (one fewer reduction over the vector).

**Where** the norm sits matters more than which norm it is. The original Transformer used **post-norm**: $y = \text{Norm}(x + \text{Sublayer}(x))$. Xiong et al. (2020) showed that with post-norm, gradients near the output are large at initialization, which is why post-norm models needed a careful learning-rate warmup and became hard to train when deep. **Pre-norm**, $y = x + \text{Sublayer}(\text{Norm}(x))$, leaves an un-normalized identity path from the input to the output, so gradients flow through the residual stream directly. It trains stably with less tuning, and almost every large LLM since GPT-2 uses it.

Pre-norm is not free of criticism. Because each block adds to an ever-growing residual, later layers contribute relatively less, and some argue this wastes depth. Variants exist: a final norm before the LM head (always used with pre-norm), norms both before and after each sub-layer (Gemma 2 does this), and **QK-norm**, which normalizes queries and keys inside attention to stop attention logits from growing during training (used in OLMo 2 and Gemma 3, among others). Which variant is best is still being studied.
''')
nb.md(r'''
## 1. RMSNorm

LayerNorm re-centers and re-scales each token vector; RMSNorm only re-scales:

$$\text{RMSNorm}(x) = \gamma \odot \frac{x}{\sqrt{\tfrac{1}{C}\sum_i x_i^2 + \epsilon}}$$

Zhang & Sennrich (2019) hypothesised that the re-scaling is what stabilizes training and that the re-centering is dispensable, and showed comparable quality at lower cost; large models have since confirmed it in practice. One reduction less, no bias, equally good → the standard in Llama / Mistral / DeepSeek / Gemma.
''')
nb.code(r'''
from llm_edu_suite.layers import RMSNorm

x = torch.randn(2, 5, 16) * 7 + 3                      # deliberately large scale and offset
ln, rms = nn.LayerNorm(16), RMSNorm(16)
for name, y in [("input", x), ("LayerNorm", ln(x)), ("RMSNorm", rms(x))]:
    print(f"{name:10s} mean {y.mean():6.2f}   rms {y.pow(2).mean().sqrt():5.2f}")
print("matches torch.nn.RMSNorm:", torch.allclose(rms(x), nn.RMSNorm(16)(x), atol=1e-5))
print("scale invariance: RMSNorm(3x) == RMSNorm(x):", torch.allclose(rms(3 * x), rms(x), atol=1e-5))
''')
nb.md(r'''
### Position information: the problem and the options

Self-attention computes a weighted sum over a set. If you permute the input tokens (without a mask), the outputs are permuted the same way and nothing else changes; the operation is permutation-equivariant. "Dog bites man" and "man bites dog" would look the same. The causal mask breaks the symmetry a little (a token can tell roughly how many tokens precede it), but models without explicit position information are weaker and less predictable, so position must be injected somewhere.

| approach | idea | relative? | extra parameters | main drawback |
|---|---|---|---|---|
| sinusoidal absolute (Vaswani et al., 2017) | add fixed sines and cosines of the position to the embedding | only indirectly | none | position mixed into content; weak extrapolation |
| learned absolute (GPT-2, BERT) | add a learned vector per position | no | $T_{\max} \times C$ | cannot go beyond $T_{\max}$ at all |
| relative bias (Shaw et al., 2018; T5, Raffel et al., 2020) | add a learned bias $b_{m-n}$ to each attention logit | yes | small | modifies the $T \times T$ score matrix, which complicates fused kernels |
| ALiBi (Press et al., 2022) | add a fixed linear penalty $-\lambda\,|m-n|$ to logits | yes | none | strong built-in recency bias |
| **RoPE** (Su et al., 2021) | rotate $q$ and $k$ by angles proportional to position | yes | none | needs a scaling rule to extend context |

RoPE became the default for three reasons. It makes the attention score depend on relative position (which is what language mostly needs), it adds no parameters, and it acts on $q$ and $k$ *before* attention, so the attention kernel itself stays a plain dot-product softmax that FlashAttention and other fused kernels can run unchanged. It also plays well with the KV-cache: a key is rotated once, by its own absolute position, and never needs updating.
''')
nb.md(r'''
## 2. RoPE — rotary position embeddings

Attention is permutation-invariant, so position must be injected. RoPE rotates the query at position *m* by angle *mθ* and the key at *n* by *nθ*. Because rotations compose by adding angles:

$$(R(m\theta)q)^\top (R(n\theta)k) = q^\top R((n-m)\theta)\,k$$

**The score depends only on the distance n − m.** Each head's `Dh` dims are split into `Dh/2` pairs with frequencies $\theta_i = \text{base}^{-2i/D_h}$: fast "hands" for local resolution, slow ones for long range.

### Rotary position embeddings in detail

Take one pair of dimensions $(q_{2i}, q_{2i+1})$ of a query at position $m$. RoPE applies a 2D rotation:

$$\begin{pmatrix} q'_{2i} \\ q'_{2i+1} \end{pmatrix} = \begin{pmatrix} \cos m\theta_i & -\sin m\theta_i \\ \sin m\theta_i & \cos m\theta_i \end{pmatrix} \begin{pmatrix} q_{2i} \\ q_{2i+1} \end{pmatrix}$$

and the same to the key at position $n$ with angle $n\theta_i$. The neat way to see it is with complex numbers: treat each pair as $z_i = q_{2i} + \mathrm{i}\,q_{2i+1}$; then rotation is multiplication by $e^{\mathrm{i} m\theta_i}$, and the real dot product of the two rotated vectors is

$$\langle q', k' \rangle = \mathrm{Re}\sum_i z^{(q)}_i\, \overline{z^{(k)}_i}\; e^{\mathrm{i}(m-n)\theta_i}$$

Only $m - n$ survives. This is exactly what `precompute_freqs_cis` and `apply_rope` do: the table holds $e^{\mathrm{i} t\theta_i}$ for every position $t$ and pair $i$, and applying RoPE is an element-wise complex multiplication.

A few properties worth knowing:

- **Norm preservation.** Rotations do not change vector length, so RoPE never changes the scale of $q$ or $k$; it only changes the angle between them.
- **A spectrum of frequencies.** With $\theta_i = \text{base}^{-2i/D_h}$, the first pairs rotate by about one radian per token and distinguish neighbouring positions; the last pairs rotate so slowly that they barely move over thousands of tokens and act almost like position-free channels. The base controls this spectrum: the original paper used 10,000; Llama 3 raised it to 500,000 to make the slow pairs slower for long context.
- **Long-range decay.** Averaged over random vectors, the score between a vector and its rotated copy falls and oscillates as distance grows (the plot below). Su et al. present this as a useful inductive bias towards locality. It is a statistical tendency, not a hard rule: a trained model can still attend sharply to distant tokens when $q$ and $k$ align strongly.
- **Partial RoPE.** Some models rotate only a fraction of each head's dimensions and leave the rest position-free. MLA (below) uses a version of this out of necessity.

Not everyone agrees position encodings are needed at all in causal decoders: Kazemnejad et al. (2023) found that models with no positional encoding ("NoPE") can generalise to longer lengths in some settings, and some recent models interleave layers without RoPE. RoPE remains the mainstream choice.
''')
nb.code(r'''
from llm_edu_suite.layers import precompute_freqs_cis, apply_rope

Dh = 64
freqs = precompute_freqs_cis(Dh, max_seq_len=1024)              # (T, Dh/2) complex
print("freqs_cis:", freqs.shape, freqs.dtype)

q, k = torch.randn(Dh), torch.randn(Dh)
def score(m, n):
    qm = apply_rope(q.view(1, 1, 1, Dh), freqs, start_pos=m)
    kn = apply_rope(k.view(1, 1, 1, Dh), freqs, start_pos=n)
    return (qm * kn).sum().item()

print("norm preserved:", torch.allclose(q.norm(), apply_rope(q.view(1,1,1,Dh), freqs, 7).norm(), atol=1e-5))
print(f"distance 5 at (5,0):     {score(5, 0):+.4f}")
print(f"distance 5 at (105,100): {score(105, 100):+.4f}")
print(f"distance 5 at (905,900): {score(905, 900):+.4f}   <- identical: only relative position matters")
''')
nb.code(r'''
import matplotlib.pyplot as plt
# Dot product of a vector with a rotated copy of itself, as a function of distance. The decay is RoPE's built-in locality bias.
ds = list(range(0, 400, 4)); curve = []
for d in ds:
    vals = []
    for _ in range(64):
        v = torch.randn(Dh)
        a = apply_rope(v.view(1,1,1,Dh), freqs, d); b = apply_rope(v.view(1,1,1,Dh), freqs, 0)
        vals.append((a * b).sum().item() / (v @ v).item())
    curve.append(sum(vals) / len(vals))
plt.plot(ds, curve); plt.axhline(0, color="k", lw=.5)
plt.xlabel("distance between query and key"); plt.ylabel("relative score q·q_rotated"); plt.title("RoPE: score decays and oscillates with distance"); plt.grid(alpha=.3); plt.show()
''')
nb.md(r'''
### Context extension
Frequencies are fixed for the training length. Beyond it, the slow pairs reach angles never seen. Llama-3.1 style scaling divides the *low* frequencies by a factor and leaves the high ones alone (`scaling={"factor": 8, "orig_len": 8192}`) — only `precompute_freqs_cis` changes, the model does not.

**Why naive extrapolation fails and how the fixes work.** At position $m$, pair $i$ has angle $m\theta_i$. During training with length $T_{\text{train}}$, the slow pairs only ever see a small arc of the circle. Running at $m > T_{\text{train}}$ produces angles in those pairs that no weight was trained on, and perplexity usually blows up rather than degrading gracefully. The fixes all change the angle table:

| method | change | effect |
|---|---|---|
| Position Interpolation (Chen et al., 2023) | use $m / s$ instead of $m$ for all pairs | every angle stays in the trained range, but neighbouring tokens become harder to tell apart; needs a short fine-tune |
| "NTK-aware" scaling (community proposal, 2023) | increase the base | stretches slow pairs more than fast ones |
| YaRN (Peng et al., 2023) | interpolate only low-frequency pairs, keep high ones, ramp in between, and slightly rescale attention temperature | keeps local resolution; reaches long context with little fine-tuning |
| Llama 3.1 scaling | divide low frequencies by a factor, keep high ones, smooth transition | same family of ideas as YaRN, the variant implemented here |

The common insight: high-frequency pairs already cycle many times within the training length, so they have seen every angle and need no change; only the slow pairs need stretching. Scaling rules make longer context *possible*; models are still usually fine-tuned on some long documents to use it well.
''')
nb.code(r'''
f_base = precompute_freqs_cis(64, 16)
f_scaled = precompute_freqs_cis(64, 16, scaling={"factor": 8.0, "orig_len": 8192})
a_base, a_scaled = f_base[1].angle(), f_scaled[1].angle()
print("high-frequency pairs unchanged:", torch.allclose(a_base[:8], a_scaled[:8]))
print("low-frequency pairs divided by 8:", torch.allclose(a_base[-7:] / 8, a_scaled[-7:], atol=1e-6))   # the 7 slowest pairs are fully stretched, while pairs ~21-24 sit in the smooth transition
''')
nb.md(r'''
## 3. Scaled dot-product attention, by hand

$$\text{Attn}(Q,K,V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{D_h}} + M\right)V$$

- `QKᵀ`: affinity of every query with every key, shape `(T, T)`
- `/√Dh`: keeps logits O(1) so the softmax does not saturate
- `M`: causal mask, `-inf` above the diagonal → exactly zero weight on future tokens
- the output of each token is a weighted average of the values it attends to

### Attention as a soft dictionary lookup

Each token produces three vectors from its residual-stream state $x$: a query $q = W_Q x$ ("what am I looking for"), a key $k = W_K x$ ("what do I contain, for matching purposes") and a value $v = W_V x$ ("what I hand over if selected"). A hard dictionary would return the value of the single key equal to the query. Attention returns a *blend* of all values, weighted by how well each key matches, and the softmax makes that blend differentiable so $W_Q, W_K, W_V$ can be learned by gradient descent. When one score dominates, the softmax approaches a hard lookup; when scores are similar, it averages.

**Why divide by $\sqrt{D_h}$.** If the components of $q$ and $k$ are independent with mean 0 and variance 1, then $q^\top k = \sum_{j=1}^{D_h} q_j k_j$ has variance $D_h$. With $D_h = 128$, raw logits would have a standard deviation around 11, the softmax would put almost all weight on one key, and its gradient with respect to the other logits would be nearly zero. Dividing by $\sqrt{D_h}$ restores unit variance at initialization. (Some models use a different scale, and QK-norm adds a learned one; the principle is the same.)

**Multiple heads.** The model dimension $C$ is split into $H$ heads of size $D_h = C/H$, each with its own projections and its own attention pattern; their outputs are concatenated and mixed by $W_O$. One softmax can only produce one weighting per query, so heads let a layer track several relationships at once (the previous token, the subject of the sentence, a matching bracket). Interpretability work has found specific, reusable head types, such as the induction heads in notebook 03.

**Cost.** Computing $QK^\top$ and multiplying by $V$ takes $O(T^2 D_h)$ per head, so $O(T^2 C)$ per layer, and a naive implementation stores the $T \times T$ weight matrix per head. **FlashAttention** (Dao et al., 2022; FlashAttention-2, 2023; FlashAttention-3, 2024) computes *exactly* the same result without ever materializing that matrix: it processes $K$ and $V$ in tiles that fit in fast on-chip memory and keeps a running ("online") softmax, rescaling partial sums as new maxima appear. Memory becomes $O(T)$ and the speed-up comes from avoiding slow GPU memory traffic. FLOPs stay quadratic; reducing those is the subject of notebook 03. `F.scaled_dot_product_attention` dispatches to such a kernel when one is available.

**The mask.** Setting future logits to $-\infty$ before the softmax gives them weight $e^{-\infty} = 0$. This is what lets one forward pass train on all $T$ next-token predictions at once without leaking the answers.
''')
nb.code(r'''
def attention_manual(q, k, v, verbose=False):
    B, H, T, Dh = q.shape
    scores = q @ k.transpose(-2, -1) / math.sqrt(Dh)
    allowed = torch.tril(torch.ones(T, T, dtype=torch.bool))
    scores = scores.masked_fill(~allowed, float("-inf"))
    w = scores.softmax(-1)
    if verbose:
        print("attention weights (batch 0, head 0) — rows sum to 1, upper triangle is 0:")
        print(w[0, 0].round(decimals=2))
    return w @ v

q, k, v = torch.randn(3, 1, 1, 5, 8)
out = attention_manual(q, k, v, verbose=True)
ref = F.scaled_dot_product_attention(q, k, v, is_causal=True)
print("\nmatches F.scaled_dot_product_attention:", torch.allclose(out, ref, atol=1e-5))
''')
nb.md(r'''
## 4. MHA → GQA → MLA: the KV-cache problem

During generation every past token's **k** and **v** are cached per layer. A model shaped like Llama-2-70B (80 layers, 64 heads × 128 dims) would need 2.6 MB per token in fp16 if every head kept its own K and V (plain multi-head attention): with 4k tokens of context, **10 GB per sequence**. The real Llama-2-70B shares K/V across groups of heads (GQA with 8 KV heads, section 4), which cuts that by 8x, and that saving is precisely why GQA was adopted. The cache, not compute, limits how many users a GPU serves.

| | k/v heads | cache per token per layer | used by |
|---|---|---|---|
| MHA | H | 2·H·Dh | GPT-2/3 |
| MQA | 1 | 2·Dh | PaLM, Falcon |
| **GQA** | Hkv (H/4–H/8) | 2·Hkv·Dh | Llama 2/3, Mistral, Qwen |
| **MLA** | latent | r + Dr (512 + 64) | DeepSeek V2/V3/R1 |

`GroupedQueryAttention` implements all of the first three via `n_kv_heads`; `MultiHeadLatentAttention` compresses k and v of *all* heads into one low-rank latent and caches that.

### KV-cache arithmetic

With causal attention, the key and value of a past token never change once computed, because they depend only on that token and the tokens before it. So during generation we store them and, at each step, compute $q, k, v$ only for the new token. Per sequence, the cache holds

$$\text{bytes} = 2 \times L \times H_{kv} \times D_h \times T \times b$$

where the 2 counts $K$ and $V$, $L$ is the number of layers, $H_{kv}$ the number of key/value heads, $T$ the context length and $b$ the bytes per element (2 for fp16/bf16). Each decode step must read all of it, plus all the weights, to produce a single token. That is why the cache size sets both the maximum batch size (memory) and the per-token latency at long context (bandwidth). Shrinking $H_{kv} \times D_h$ is the lever all the variants below pull.

### Multi-query and grouped-query attention

**MQA** (Shazeer, 2019) keeps $H$ query heads but shares a single key head and a single value head across all of them. The cache shrinks by a factor of $H$. The cost is quality: all heads must now look through the same keys and read the same values, which removes a lot of expressiveness, and the GQA paper reports both some quality loss and training instability for MQA.

**GQA** (Ainslie et al., 2023) interpolates: $H$ query heads are split into $H_{kv}$ groups, and each group shares one key/value head. $H_{kv} = H$ is MHA, $H_{kv} = 1$ is MQA. With 8 KV heads for 32 or 64 query heads, quality is close to MHA and the cache shrinks 4 to 8 times. The paper also showed an existing MHA checkpoint can be converted by mean-pooling the key and value heads of each group and then training briefly ("uptraining"). GQA became the standard for dense open models because it is a simple change, compatible with every attention kernel, with a good quality/memory balance.

### Multi-head latent attention

**MLA** (DeepSeek-V2, 2024) attacks the cache differently: instead of sharing heads, it stores a single compressed vector per token and reconstructs full per-head keys and values from it.

$$c_t = W^{DKV} x_t \in \mathbb{R}^{d_c}, \qquad k^{(h)}_t = W^{UK}_h c_t, \qquad v^{(h)}_t = W^{UV}_h c_t$$

Only $c_t$ is cached ($d_c = 512$ in DeepSeek-V2/V3, versus $2 \times 128 \times 128$ values per token per layer for their full MHA). The trick that makes this cheap at inference is **weight absorption**: since $q^{(h)\top} k^{(h)}_t = q^{(h)\top} W^{UK}_h c_t = (W^{UK\top}_h q^{(h)})^\top c_t$, the up-projection can be folded into the query side, and attention runs directly against the cached latents without ever materializing the full keys. The same works for values and $W_O$.

RoPE breaks absorption, because a position-dependent rotation would sit between $W^{UK}_h$ and $c_t$ and could no longer be folded into a fixed matrix. DeepSeek's solution is **decoupled RoPE**: each head's key gets a small extra part ($d^R = 64$ dims) that carries RoPE, computed directly from $x_t$ and shared across heads, so it is cached alongside $c_t$. That is the "512 + 64" in the table above.

| | cache per token per layer | quality (as reported) | complexity |
|---|---|---|---|
| MHA | $2 H D_h$ | baseline | simple |
| MQA | $2 D_h$ | noticeably lower, less stable | simple |
| GQA | $2 H_{kv} D_h$ | close to MHA | simple, kernel-friendly |
| MLA | $d_c + d^R$ | on par with or better than MHA in DeepSeek's ablations | more projections, absorption, decoupled RoPE, custom kernels |

MLA's trade-off is engineering complexity and somewhat more compute per decode step in exchange for a much smaller cache. Whether its quality advantage over GQA holds generally, or depends on DeepSeek's setup, is not yet settled by independent studies.
''')
nb.code(r'''
from llm_edu_suite.layers import GroupedQueryAttention, MultiHeadLatentAttention

C, H = 128, 8
freqs = precompute_freqs_cis(C // H, 256)
variants = {
    "MHA (8 kv heads)": GroupedQueryAttention(C, H, n_kv_heads=8),
    "GQA (2 kv heads)": GroupedQueryAttention(C, H, n_kv_heads=2),
    "MQA (1 kv head) ": GroupedQueryAttention(C, H, n_kv_heads=1),
    "MLA (rank 16)   ": MultiHeadLatentAttention(C, H, kv_lora_rank=16, qk_nope_dim=12, qk_rope_dim=4, v_dim=16),
}
x = torch.randn(2, 10, C)
for name, m in variants.items():
    y = m(x, freqs)
    print(f"{name}  params {count_params(m):>7,}  kv-cache/token/layer {m.kv_cache_bytes_per_token():>5} bytes  out {tuple(y.shape)}")

# For reference, these are DeepSeek-V3's actual sizes:
print(f"\nDeepSeek-V3 per token per layer: MHA(128 heads x 128) = {2*128*128*2:,} bytes | MLA = {(512+64)*2:,} bytes  (÷{2*128*128/(512+64):.0f})")
''')
nb.md(r'''
## 5. KV-cache correctness

The cache must not change the result. We compare the full forward pass with prefill (8 tokens) + 4 single-token decode steps.
Note the RoPE position bookkeeping: a new token at position `t` is rotated with `freqs_cis[t]`, cached keys keep their original rotation, and the angle *difference* comes out right automatically.

**Prefill and decode.** Inference has two phases. *Prefill* runs the whole prompt in one parallel pass (compute-bound, like training) and fills the cache. *Decode* then adds one token per step (memory-bound). The equivalence tested below is the contract between them: cached decoding is an optimization and must give the same numbers as recomputing everything, up to floating-point rounding.

**Common bugs this test catches:** an off-by-one `start_pos` (the new token gets the wrong rotation, so all relative distances shift by one and quality degrades without any crash); applying RoPE to the cached keys a second time; forgetting the causal mask inside a multi-token prefill; and, for MLA, caching the up-projected keys instead of the latent, which works but wastes all the memory MLA was supposed to save. Another inference-time effect worth knowing: trained models tend to put a large share of attention on the very first tokens ("attention sinks", Xiao et al., 2023), so cache-eviction schemes that drop the first tokens break models in surprising ways.
''')
nb.code(r'''
for name, attn in [("GQA", variants["GQA (2 kv heads)"]), ("MLA", variants["MLA (rank 16)   "])]:
    x = torch.randn(1, 12, C)
    y_full = attn(x, freqs)                                            # training-style forward pass without a cache
    attn.clear_cache()
    parts = [attn(x[:, :8], freqs, start_pos=0)]                       # prefill
    for t in range(8, 12):
        parts.append(attn(x[:, t:t+1], freqs, start_pos=t))            # then decode one token at a time
    y_inc = torch.cat(parts, dim=1)
    print(f"{name}: full == prefill+decode:", torch.allclose(y_full, y_inc, atol=1e-4), f"(max diff {(y_full - y_inc).abs().max():.2e})")
''')
nb.md(r'''
## 6. Evaluation: attention variants on a real objective

Metrics so far were structural (parameters, cache bytes, equivalence). The question that matters: **does GQA/MLA hurt modeling quality?** We train tiny LMs with each variant on the same synthetic corpus for the same number of steps and compare validation perplexity. (At this scale differences are noisy — the point is the methodology; run longer / larger to get reliable comparisons.)

**Comparing architectures fairly.** Perplexity, $\exp$ of the mean next-token cross-entropy on held-out data, is the standard intrinsic metric for language models, but architecture comparisons are easy to get wrong. Fewer KV heads also means fewer parameters in $W_K$ and $W_V$, so a GQA model is slightly smaller than its MHA counterpart; published comparisons often compensate by widening the FFN to match parameter counts. Differences between attention variants are also typically small next to run-to-run noise at small scale, so a serious comparison uses several seeds, a matched compute budget and downstream tasks (in particular long-context retrieval, where cache compression is most likely to hurt).
''')
nb.code(r'''
from llm_edu_suite.model import ModelConfig, TransformerLM
from llm_edu_suite.data import synthetic_structured_corpus, train_val_split
from llm_edu_suite.training import TrainConfig, train_lm
from llm_edu_suite.evaluation import perplexity

data = synthetic_structured_corpus(n_tokens=120_000, vocab=512)
train_data, val_data = train_val_split(data)
tcfg = TrainConfig(steps=250, batch_size=32, block_size=64, max_lr=3e-3, warmup=25, eval_every=250, log=False)

results = {}
for name, kw in {"MHA": dict(attn="gqa", n_kv_heads=8), "GQA-2": dict(attn="gqa", n_kv_heads=2),
                 "MQA": dict(attn="gqa", n_kv_heads=1), "MLA": dict(attn="mla", kv_lora_rank=16, qk_nope_dim=12, qk_rope_dim=4, v_dim=16)}.items():
    seed_everything(0)
    model = TransformerLM(ModelConfig(vocab_size=512, d_model=128, n_layers=4, n_heads=8, **kw))
    res = train_lm(model, train_data, val_data, tcfg, device)
    ppl = perplexity(model, val_data, block_size=64, n_batches=20, device=device)
    results[name] = (ppl, model.kv_cache_bytes_per_token(), res.wall_time)
    print(f"{name:6s} val ppl {ppl:7.2f} | kv-cache {model.kv_cache_bytes_per_token():5d} B/token | {res.wall_time:.1f}s")
''')
nb.md(r'''
**How to read this:** perplexity should be similar across variants (GQA ≈ MHA is the published finding; MLA ≥ MHA in DeepSeek's experiments) while the cache column shrinks 4–10×. If one variant is clearly worse here, increase `steps` before concluding — 250 steps on a toy corpus is a smoke test, not a benchmark.

**Next:** `03_efficient_attention.ipynb` — what to do when the sequence gets long.
''')
nb.save(notebook_path("02_attention_rope_norm.ipynb"))
print("02_attention_rope_norm.ipynb written")
