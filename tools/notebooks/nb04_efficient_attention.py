"""Notebook 04 - Efficient attention

Builds 04_efficient_attention.ipynb. Run: python tools/build_notebooks.py 04
"""
from _common import *  # noqa: F401,F403

# Notebook 04 - Efficient attention
nb = NB()
nb.md(r'''
# 04 · Efficient attention for long contexts

Dense attention costs O(T²). This notebook implements and compares the main alternatives, all with the
same interface `forward(x, freqs_cis) -> (B, T, C)` so they drop into the same model:

| variant | pattern | cost | used by |
|---|---|---|---|
| dense causal | everything in the past | O(T²) | default; FlashAttention makes it memory-efficient |
| sliding window (SWA) | last `w` tokens | O(T·w) | Mistral, Gemma local layers, gpt-oss |
| Longformer | window + global tokens | O(T·(w+g)) | 2020 encoders; ancestor of hybrids |
| content-based sparse (NSA, DSA) | top-k blocks (NSA) or top-k tokens (DSA) chosen *by content* + local | O(T·k·b) | DeepSeek-V3.2, Kimi |
| linear | recurrent state, no softmax | O(T) | hybrids: Qwen3-Next, Kimi Linear, MiniMax |
| hybrid | mostly cheap layers + few dense | ~O(T) | Gemma 3 (5:1), Qwen3-Next (3:1) |

**Source:** `llm_edu_suite/layers.py`.
''')
nb.md(r'''
## Background: the quadratic wall

### Where the cost comes from

For one layer with model width $C$ and sequence length $T$, the work splits into two parts:

- **Projections and FFN**, which act on each token independently. With the usual 4x FFN, that is roughly $24\,C^2$ FLOPs per token (Q, K, V, O projections plus the two FFN matrices, counting a multiply-add as 2 FLOPs). This grows linearly in $T$.
- **The attention mixing itself**, $QK^\top$ and the weighted sum with $V$, which is about $4\,T\,C$ FLOPs per token for full attention (half of that with a causal mask). Summed over the sequence this grows as $T^2$.

So attention overtakes the rest of the layer once $T$ is roughly an order of magnitude larger than $C$. For a model with $C = 4096$ that happens in the tens of thousands of tokens. Below that point, making attention cheaper barely changes total cost; above it, attention dominates and keeps growing. (These are back-of-the-envelope counts; exact constants depend on the architecture.)

Inference has a second wall. Each generated token must read the whole KV-cache, which grows linearly with $T$, so generating at long context is limited by memory capacity (how many sequences fit) and memory bandwidth (how fast each step is). Notebook 03 shrank the cache *per token* (GQA, MLA). This notebook shrinks *how many tokens* each query looks at.

### What FlashAttention does and does not solve

FlashAttention (Dao et al., 2022) is often mentioned in the same breath as efficient attention, but it is a different kind of fix. It computes **exact** dense attention while never storing the $T \times T$ matrix, by tiling and an online softmax. That removes the $O(T^2)$ *memory* problem and makes dense attention much faster in practice, which is a big reason dense attention is still the default up to fairly long contexts. It does not remove the $O(T^2)$ *compute*, and it does not shrink the KV-cache. Everything else in this notebook changes the computation itself, and therefore changes what the model can represent.

### The families of solutions

| family | key idea | representative papers | what you give up |
|---|---|---|---|
| fixed sparse patterns | each query attends to a predefined subset (local window, strided positions, a few global tokens) | Sparse Transformer (Child et al., 2019); Longformer (Beltagy, Peters & Cohan, 2020); BigBird (Zaheer et al., 2020); Mistral 7B (Jiang et al., 2023) | direct access to tokens outside the pattern |
| content-based sparse | the subset is chosen per query by a cheap scoring step | Reformer (Kitaev et al., 2020); Routing Transformer (Roy et al., 2021); NSA (Yuan et al., DeepSeek, 2025); MoBA (Lu et al., Moonshot, 2025); DSA (DeepSeek-V3.2, 2025) | some recall if the selector misses; complex kernels |
| low-rank / kernel approximations | approximate the softmax matrix by a product of thinner matrices | Linformer (Wang et al., 2020); Performer (Choromanski et al., 2021); Linear Transformers (Katharopoulos et al., 2020) | the sharpness of softmax |
| recurrent / state-space | compress the past into a fixed-size state updated per token | RetNet (Sun et al., 2023); Mamba (Gu & Dao, 2023); Mamba-2 (Dao & Gu, 2024); Gated DeltaNet (Yang et al., 2025) | exact recall of arbitrary past tokens |
| hybrids | mostly cheap layers, a few dense ones | Jamba (Lieber et al., 2024); Griffin (De et al., 2024); Gemma 2/3; Qwen3-Next; MiniMax-01 | a few dense layers still pay $O(T^2)$ |

The history is instructive. Between 2019 and 2021 dozens of "efficient transformers" were proposed, mostly evaluated on encoders and synthetic benchmarks. Few were adopted for large decoders, for two reasons: FlashAttention made dense attention fast enough for the context lengths of the time, and many approximations lost quality on exactly the tasks LLMs care about (in-context retrieval and copying). What survived in production is simple local windows (cheap, kernel-friendly, predictable), and, more recently, learned content-based sparsity and gated linear recurrences, almost always **mixed with some full-attention layers**. The hybrid ratio is the main design knob today, and what the best ratio is remains an open empirical question.

### The capability at stake: retrieval from context

The reason dense attention is hard to replace is that it can retrieve any past token exactly, in one step, by content. Many in-context abilities rest on that: copying a name from the prompt, following a variable through code, answering from a retrieved document. Arora et al. (2023, "Zoology") showed that the quality gap between attention and efficient alternatives on real language is largely explained by **associative recall**, and Jelassi et al. (2024, "Repeat After Me: Transformers are Better than State Space Models at Copying") proved and measured that fixed-state models cannot copy long strings the way attention can. The evaluation at the end of this notebook is a small version of that test.
''')
nb.code(SETUP)
nb.code(r'''
from llm_edu_suite.layers import (causal_mask, sliding_window_mask, longformer_mask, precompute_freqs_cis,
                             SlidingWindowAttention, LongformerAttention, BlockSparseAttention, LinearAttention, GroupedQueryAttention)
T = 20
show_mask(causal_mask(T), "dense causal")
show_mask(sliding_window_mask(T, 5), "sliding window, w=5")
show_mask(longformer_mask(T, 5, [0, 10]), "Longformer, w=5, global={0,10}")
''')
nb.md(r'''
## Sliding window: receptive field grows with depth

With `L` layers, information can travel `L·w` tokens (token *i* in layer 2 sees *i−w*, which in layer 1 saw *i−2w*…), like a CNN's receptive field. Exact retrieval of a far token degrades, which is why global layers are interleaved. Two bonus properties: the KV-cache can be a ring buffer of size `w`, and no single attention step ever spans more than `w` positions, which helps when running past the training length (whether the model really generalizes also depends on how positions are encoded and on what information gets relayed across layers).

A boolean mask in SDPA still computes the full T×T scores and discards — the real saving needs a kernel that skips blocks (**FlexAttention**, PyTorch ≥ 2.5) or the ring-buffer cache at inference.

### Sliding-window attention in detail

Formally, query $i$ may attend to key $j$ only if $i - w < j \le i$. Each row of the attention matrix has at most $w$ non-zero entries, so cost per layer is $O(T\,w\,C)$ instead of $O(T^2 C)$, and the KV-cache needs only the last $w$ positions per layer: memory is $O(w)$ regardless of how long generation runs.

**Stacking.** After $L$ layers a token can be influenced by tokens up to $L(w-1)$ positions back, the same receptive-field argument as for stacked convolutions. The catch is that distant information has to be *relayed*: it is summarized into intermediate tokens at each hop, and those intermediate tokens have their own work to do. The theoretical receptive field says what is reachable, not what is preserved; exact recall of a specific distant token is much harder than with direct access. This is why production models do not use sliding windows alone.

**Where it is used.** Mistral 7B (Jiang et al., 2023) used a 4,096-token window in every layer. Gemma 2 alternated local and global layers one to one, and Gemma 3 moved to five local layers per global one with a short local window, precisely to cut the KV-cache at long context while keeping some layers with full reach. The pattern is attractive because it is simple, works with standard kernels that skip masked blocks, and its quality cost is predictable.

**Failure modes.** Two are worth knowing. First, retrieval of facts beyond the window degrades, sometimes silently, so long-context evaluations (needle-in-a-haystack, RULER) are essential. Second, if a model trained with dense attention is run with a window at inference time by evicting old cache entries, quality collapses once the first tokens fall out, because trained models park a lot of attention on the initial tokens ("attention sinks", Xiao et al., 2023, "Efficient Streaming Language Models with Attention Sinks"). Keeping a handful of initial tokens permanently in the cache fixes most of the damage.

### Longformer: window plus global tokens

Longformer (Beltagy, Peters & Cohan, 2020) adds a small set $G$ of **global** positions to the window. A global token attends to every position, and every position attends to it:

$$\text{allowed}(i, j) = \big(|i - j| < w\big) \ \lor\ i \in G \ \lor\ j \in G \quad (\text{intersected with the causal mask for a decoder})$$

The global tokens act as hubs: any two tokens are at most two hops apart through them, regardless of distance. Cost is $O(T(w + |G|))$. BigBird (Zaheer et al., 2020) added a few random connections per query and proved that window + global + random sparse attention keeps the theoretical expressiveness of full attention (universal approximation and Turing completeness).

The weakness is that **someone has to choose the global positions**. In Longformer they were task-specific (the `[CLS]` token for classification, the question tokens for QA). That works for encoders fine-tuned on one task, but a general-purpose decoder does not know in advance which tokens will matter. This is the gap content-based selection fills: let the model choose its "global" tokens per query, from the data.
''')
nb.code(r'''
C, H = 128, 8
freqs = precompute_freqs_cis(C // H, 1024)
x = torch.randn(2, 64, C)
swa = SlidingWindowAttention(C, H, window=8)
print("SWA out:", tuple(swa(x, freqs).shape), "| receptive field with 32 layers and w=4096:", f"{32*4096:,} tokens")

# FlexAttention expresses the same window as a mask *function* that gets compiled into a block-skipping kernel on GPU. On CPU we run the reference implementation.
try:
    from torch.nn.attention.flex_attention import flex_attention, create_block_mask
    def window_mod(b, h, q_idx, kv_idx):
        return (kv_idx <= q_idx) & (q_idx - kv_idx < 8)
    bm = create_block_mask(window_mod, B=None, H=None, Q_LEN=4096, KV_LEN=4096, device="cpu")
    print("FlexAttention block mask built; sparsity (fraction of blocks skipped):", f"{bm.sparsity():.1f}%")
except Exception as e:
    print("FlexAttention unavailable here:", type(e).__name__)
''')
nb.md(r'''
## Block-sparse attention chosen by content (NSA-style, simplified)

1. Group keys in blocks of `block` tokens. 2. A cheap indexer scores each block per query (here `q · mean(k_block)`). 3. Attend only the top-`k` blocks + a local window, causally. The pattern is **dynamic** — different per query and per head — and block-structured so GPUs can gather efficiently. DeepSeek-V3.2's DSA goes one step finer: a lightweight indexer picks the top ~2k individual *tokens* out of a 128k context, and DeepSeek reports benchmark quality roughly on par with its dense predecessor (V3.1-Terminus).

### Content-based sparse attention in detail

The idea is to approximate softmax attention by keeping only its largest terms. Softmax weights are typically concentrated: for a given query, a small number of keys receive most of the mass. If we can find those keys cheaply, the rest can be skipped with little error. The design questions are how to find them, at what granularity, and how to train the selector.

**Finding them.** Early attempts hashed queries and keys so that similar vectors land in the same bucket (Reformer, Kitaev et al., 2020, with locality-sensitive hashing) or clustered them (Routing Transformer, Roy et al., 2021). Current designs use a learned, cheap scorer. Our simplified version scores block $b$ for query $q$ as

$$s_b = q^\top \bar{k}_b, \qquad \bar{k}_b = \frac{1}{|b|}\sum_{j \in b} k_j$$

and keeps the top-$k$ blocks plus the local window. Cost per query is $O(T/\text{block})$ for scoring plus $O((k \cdot \text{block} + w) D_h)$ for the attention itself. Scoring is still linear in $T$ per query, so the overall cost is still quadratic, but with a constant divided by the block size; the expensive part has become sparse.

**Granularity.** Selecting whole blocks of contiguous keys is not just a simplification: GPUs load memory in contiguous tiles, and a block-sparse pattern lets a kernel skip whole tiles. Selecting arbitrary individual tokens is more precise but needs gather operations that are harder to make fast.

**Training the selector.** Top-$k$ is not differentiable, so the selector must learn some other way. **NSA** (Native Sparse Attention, Yuan et al., 2025) runs three branches per query: attention over compressed block summaries, attention over the top-$k$ selected blocks (chosen using the compressed branch's scores, so the selection is guided by a quantity that does receive gradients), and a sliding window. A learned gate mixes the three. It is trained sparse from the start ("natively"), so the model learns to work with its own selection. **DSA** (DeepSeek Sparse Attention, in DeepSeek-V3.2, 2025) instead uses a separate small "lightning indexer" with few heads and low precision to score preceding tokens, selects the top 2,048 *tokens* (not blocks) per query, and was trained by first distilling the indexer from the dense model's attention, then continuing training with sparse attention. **MoBA** (Moonshot, 2025) applies a mixture-of-experts style router over key blocks.

**Trade-offs.** Content-based sparsity keeps direct, one-hop access to any distant token, which local windows lack, so it degrades much less on retrieval. The costs are engineering (custom kernels, an extra scorer, extra hyperparameters) and a residual risk: if the selector misses the relevant block, that information is simply not seen in this layer.
''')
nb.code(r'''
bsa = BlockSparseAttention(C, H, block=4, top_blocks=2, local_window=4)
y = bsa(torch.randn(1, 24, C), freqs)
show_mask(bsa.last_mask[0, 0], "block-sparse: head 0 (local diagonal + 2 content-selected blocks per query)")
show_mask(bsa.last_mask[0, 1], "block-sparse: head 1 — same input, different selection")
''')
nb.md(r'''
## Linear attention: a recurrent state instead of a cache

Replace softmax with a positive feature map φ:

$$o_t = \frac{\phi(q_t)^\top S_t}{\phi(q_t)^\top z_t}, \qquad S_t = S_{t-1} + \phi(k_t)v_t^\top,\quad z_t = z_{t-1} + \phi(k_t)$$

`S_t` is `Dh × Dh` — constant in `T`. Training uses the parallel (cumsum) form; inference uses `step()` with O(1) memory. We check both forms agree. The price: no sharp "exact lookup", hence hybrids.

### Linear attention in detail

Write causal softmax attention for one query as a similarity-weighted average:

$$o_t = \frac{\sum_{j \le t} \text{sim}(q_t, k_j)\, v_j}{\sum_{j \le t} \text{sim}(q_t, k_j)}, \qquad \text{sim}(q,k) = \exp\!\big(q^\top k / \sqrt{D_h}\big)$$

Katharopoulos et al. (2020, "Transformers are RNNs") replaced the exponential with a kernel that factorizes, $\text{sim}(q,k) = \phi(q)^\top \phi(k)$, for a positive feature map $\phi$ (they used $\text{elu}(x) + 1$). Then $\phi(q_t)$ can be pulled out of the sum:

$$o_t = \frac{\phi(q_t)^\top \sum_{j \le t} \phi(k_j) v_j^\top}{\phi(q_t)^\top \sum_{j \le t} \phi(k_j)} = \frac{\phi(q_t)^\top S_t}{\phi(q_t)^\top z_t}$$

This is just associativity of matrix products, $(\phi(Q)\phi(K)^\top)V = \phi(Q)(\phi(K)^\top V)$: computing the right-hand side first never forms the $T \times T$ matrix. Cost drops from $O(T^2 D_h)$ to $O(T D_h^2)$, and at inference the model is an RNN whose state $(S_t, z_t)$ has fixed size. Performer (Choromanski et al., 2021) took a different route to the same structure, using random features whose dot product approximates the softmax kernel in expectation.

**Why it falls short of softmax.** The state $S_t$ is a $D_h \times D_h$ matrix storing a sum of outer products $\phi(k_j) v_j^\top$; reading it with $\phi(q)$ retrieves $\sum_j (\phi(q)^\top \phi(k_j))\, v_j$. This is a classical linear associative memory, and it has two limits. Its *capacity* is bounded by its size: once more than about $D_h$ key-value pairs are stored, their keys cannot all be near-orthogonal and retrievals interfere. And plain linear attention only *adds*: old associations are never removed, so the state becomes noisier as the sequence grows. Softmax avoids both, because it keeps every key separately (the KV-cache) and its exponential can isolate a single match.

**What modern recurrent layers add.** The line of work that made linear-time layers competitive again fixes exactly these two problems:

| update rule | form | example models |
|---|---|---|
| additive (vanilla linear attention) | $S_t = S_{t-1} + k_t v_t^\top$ | Katharopoulos et al., 2020 |
| decay / gating | $S_t = \alpha_t S_{t-1} + k_t v_t^\top$, with $\alpha_t$ fixed or input-dependent | RetNet, GLA, Mamba-2 |
| delta rule | $S_t = S_{t-1} + \beta_t\, k_t \big(v_t - S_{t-1}^\top k_t\big)^\top$ | DeltaNet (Schlag et al., 2021; Yang et al., 2024) |
| gated delta rule | decay plus delta rule | Gated DeltaNet (Yang et al., 2025), used in Qwen3-Next; Kimi Delta Attention in Kimi Linear |

Gating lets the model forget; the delta rule *overwrites* the value stored under a key instead of piling on, which uses the limited capacity much better. Training uses a chunkwise-parallel form (parallel within chunks, recurrent across them) so it runs efficiently on GPUs. Even so, a fixed-size state cannot store an arbitrarily long sequence verbatim, which is the theoretical reason hybrids keep a few softmax layers.
''')
nb.code(r'''
lin = LinearAttention(C, H)
x = torch.randn(2, 10, C)
y_par = lin(x)
state = lin.init_state(2)
outs = []
for t in range(10):
    o, state = lin.step(x[:, t:t+1], state)
    outs.append(o)
y_rec = torch.cat(outs, 1)
print("parallel == recurrent:", torch.allclose(y_par, y_rec, atol=1e-4), f"(max diff {(y_par - y_rec).abs().max():.1e})")
print(f"state size per layer: {state[0].numel() + state[1].numel():,} values (constant) vs softmax KV-cache at T=100k: {2 * C * 100_000:,}")
''')
nb.md(r'''
## Evaluation: quality vs cost on a copy-heavy task

Which variant should you use? Measure. We build a corpus where **long-range exact recall matters**: a random span of 12 tokens, 30 random filler tokens, then the same span again. Only the repeated span is predictable, and only by looking back ~42 tokens.

**Metric.** Perplexity over the whole stream would be useless here: 44 of every 55 tokens are random noise that no model can predict, and they drown the signal. Instead we measure **copy accuracy**: on fresh held-out sequences, the fraction of repeated-span tokens (2nd to 12th) the model predicts exactly. This is the "measure the thing you care about" rule in action.

### Induction heads: the mechanism behind copying

How does a transformer copy a span it saw earlier? Elhage et al. (2021, "A Mathematical Framework for Transformer Circuits") and Olsson et al. (2022, "In-context Learning and Induction Heads") identified a two-layer circuit that does it, and the copy task here is designed to require it.

Suppose the sequence contains `... A B ... A` and we want to predict `B`.

```
position:      p     p+1                    t
token:         A      B       ......        A    -> predict B

layer 1  (previous-token head):  at p+1, write "the token before me was A"
layer 2  (induction head):       query from t:  "who was preceded by A?"
                                 matches key at p+1, copies its value: B
```

1. A **previous-token head** in an early layer attends from each position to the one right before it and writes that token's identity into the residual stream. After it, position $p+1$ carries both "I am B" and "I was preceded by A".
2. An **induction head** in a later layer forms its query from the current token `A` and its keys from the "preceded by" information, so it attends to position $p+1$, and its value/output circuit copies "B" into the prediction.

Because the key of the second head depends on the output of the first ("K-composition"), the circuit needs at least two layers; a single attention layer cannot implement it. Olsson et al. found that induction heads form abruptly during training, at the same moment as a visible bump in the loss curve and a jump in in-context learning ability, which is why the training length matters in the experiment below.

**What each variant needs to run this circuit.** The induction head must attend from the second span to the first, about 42 positions back, and pick out *one* exact position by content.

- Dense attention can do this directly.
- A sliding window with $w = 16$ cannot make that 42-token jump in any single layer; the only path is relaying through intermediate filler tokens, which the circuit above does not use.
- Content-based block selection can make the jump if its scorer picks the right block.
- Linear attention can attend anywhere in principle, but only through its fixed-size state, which blurs exact positions together.
- A hybrid keeps at least one layer that can make the jump.

Real-world versions of this test, at much larger scale, are needle-in-a-haystack retrieval and benchmarks like RULER (Hsieh et al., 2024), which add multi-hop tracing and aggregation tasks.
''')
nb.code(r'''
from llm_edu_suite.model import ModelConfig, TransformerLM
from llm_edu_suite.data import train_val_split
from llm_edu_suite.training import TrainConfig, train_lm

SPAN, GAP = 12, 30

def copy_task_corpus(n_tokens=120_000, vocab=512, seed=0):
    # A random span of SPAN tokens, GAP filler tokens, then the same span again: only long-range attention can predict the repeat.
    g = torch.Generator().manual_seed(seed)
    seq = []
    while len(seq) < n_tokens:
        s = torch.randint(1, vocab // 2, (SPAN,), generator=g).tolist()
        filler = torch.randint(vocab // 2, vocab, (GAP,), generator=g).tolist()
        seq += s + filler + s + [0]
    return torch.tensor(seq[:n_tokens])

def copy_accuracy(model, n=256, seed=1):
    # Fresh held-out sequences [0, span, filler, span]; we score the repeated tokens 2..SPAN (the first one has nothing to copy yet).
    g = torch.Generator().manual_seed(seed)
    s = torch.randint(1, 256, (n, SPAN), generator=g)
    f = torch.randint(256, 512, (n, GAP), generator=g)
    x = torch.cat([torch.zeros(n, 1, dtype=torch.long), s, f, s], 1).to(device)
    model.eval()
    with torch.no_grad():
        pred = model(x)[0].argmax(-1)
    start = 1 + SPAN + GAP                                          # index of the first repeated token
    return (pred[:, start:start + SPAN - 1] == x[:, start + 1:start + SPAN]).float().mean().item()

train_data, val_data = train_val_split(copy_task_corpus())

def train_variant(steps, **kw):
    seed_everything(0)
    model = TransformerLM(ModelConfig(vocab_size=512, d_model=128, n_layers=4, n_heads=8, n_kv_heads=2, **kw)).to(device)
    res = train_lm(model, train_data, val_data, TrainConfig(steps=steps, batch_size=32, block_size=96, max_lr=3e-3, warmup=30,
                                                            eval_every=steps, log=False), device)
    return model, res.wall_time

configs = {
    "dense GQA":      dict(attn="gqa"),
    "SWA w=16":       dict(attn="swa", window=16),
    "block-sparse":   dict(attn="blocksparse", block=8, top_blocks=2, window=8),
    "linear":         dict(attn="linear"),
    "hybrid LLLG":    dict(attn="swa", window=16, hybrid_pattern="LLLG"),
}
rows = []
for name, kw in configs.items():
    model, secs = train_variant(1500, **kw)
    rows.append((name, copy_accuracy(model), secs))
    print(f"{name:14s} copy accuracy {rows[-1][1]:6.1%}   ({secs:5.1f}s for 1500 steps)")

short, _ = train_variant(300, attn="gqa")
print(f"\ndense GQA after only 300 steps: copy accuracy {copy_accuracy(short):.1%}")
''')
nb.code(r'''
fig, ax = plt.subplots(1, 2, figsize=(12, 3.8))
names = [r[0] for r in rows]
ax[0].bar(names, [100 * r[1] for r in rows], color="tab:green")
for i, r in enumerate(rows):
    ax[0].text(i, 100 * r[1] + 2, f"{100 * r[1]:.0f}%", ha="center")
ax[0].set_ylim(0, 110); ax[0].set_ylabel("copy accuracy (%)"); ax[0].set_title("exact long-range recall (1500 steps)")
ax[1].bar(names, [r[2] for r in rows]); ax[1].set_ylabel("train seconds"); ax[1].set_title("cost (mask-based implementations)")
for a in ax: a.tick_params(axis="x", rotation=20); a.grid(alpha=.3, axis="y")
plt.tight_layout(); plt.show()
''')
nb.md(r'''
**Takeaways**
- **Induction heads appear suddenly.** After 300 steps even the dense model copies essentially nothing; by 1500 steps it is near-perfect. This "phase transition" is the induction-head mechanism described by Olsson et al. (Anthropic, 2022): two attention layers cooperate to find the previous occurrence of the current token and copy what followed it. Train too briefly and you would wrongly conclude the architecture cannot do the task.
- **Who can see far enough decides who can copy.** Dense attention sees everything. Block-sparse attention selects the relevant distant blocks by content and gets most of the way. SWA with `w=16` cannot attend across the 30-token gap in one step: with 4 layers its receptive field (about 64 tokens) does reach that far, but only by relaying information hop by hop, which blurs the exact token an induction head needs to look up, and linear attention compresses the past into a fixed-size state that cannot hold an exact 12-token sequence. The hybrid (one global layer out of four) recovers part of the ability — the argument behind Gemma-3/Qwen3-Next-style hybrids, which in practice use more global layers and far longer training.
- Timing here reflects *mask-based* implementations; the asymptotic savings appear only with real kernels (FlexAttention, FlashAttention, Triton) at long sequence lengths.
- Decision rule: dense + GQA by default; SWA/hybrid when context ≥ 32k or KV-cache memory binds; linear/hybrid for ≥ 256k; learned block-sparse when you can afford the kernels — and always test exact long-range recall on your own data (needle-in-a-haystack, RULER).

## Results

| question | measured | lesson |
|---|---|---|
| how far can a sliding window see? | 32 layers × w=4096 → receptive field 131,072 tokens | depth relays information, but only hop by hop |
| how sparse is a window mask? | FlexAttention block mask skips 93.8 % of blocks | the savings come from skipping blocks in the kernel |
| is linear attention a real RNN? | parallel == recurrent (max diff 4.8e-07); state 4,352 values vs 25,600,000 for a softmax cache at T=100k | constant memory, at the price of a lossy state |
| exact recall 42 tokens back (1500 steps) | copy accuracy: dense 99.9 %, block-sparse 88.7 %, hybrid LLLG 48.4 %, SWA w=16 0.6 %, linear 0.3 % | only layers that can jump there by content can copy |
| does training length matter? | dense after 300 steps: 0.3 %; after 1500: 99.9 % | induction heads appear suddenly; short runs mislead |
| cost here | 29.5–38.9 s for 1500 steps, all variants | with mask-based code there are no savings; they need real kernels |

## In production

**What long-context models do today.**
- Hybrids are the common answer: mostly local or linear layers with some global ones (Gemma 3 interleaves sliding-window and global layers; Qwen3-Next and several other recent models mix linear-attention and full-attention layers).
- Content-selected sparse attention (DeepSeek's NSA and DSA line) for very long contexts.
- Dense GQA attention with FlashAttention remains the default up to tens of thousands of tokens.

**Tools.** FlashAttention, FlexAttention (`torch.nn.attention.flex_attention`), Triton kernels, and serving engines (vLLM, SGLang) that support sliding-window and hybrid caches.

**Evaluate before trusting a context length.**
- Advertised context is not usable context. Test with needle-in-a-haystack, RULER or long-document QA on your own data.
- For many applications, retrieval (notebook 18) is cheaper and more reliable than a very long prompt.

**Failure modes.**
- Assuming a deep sliding-window stack can recall exact tokens because its receptive field is large enough.
- Linear-attention models that are fluent but forget exact names, numbers and code.
- Concluding an architecture "cannot" do a task after too short a run.
- Measuring speed with reference implementations instead of the kernels you will deploy.
''')
nb.md(r'''
**Next:** `05_ffn_moe.ipynb`.
''')
nb.save(notebook_path("04_efficient_attention.ipynb"))
print("04_efficient_attention.ipynb written")
