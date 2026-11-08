"""Build notebooks 01-06, covering architecture, pre-training and inference.

Run it with: python tools/build_notebooks_part1.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
from nb_utils import NB, notebook_path

OUT = os.path.join(os.path.dirname(__file__), "..", "notebooks")

SETUP = r'''
import os, sys, math, time, tempfile
ROOT = os.path.abspath(".")
while not os.path.isdir(os.path.join(ROOT, "llmsuite")) and os.path.dirname(ROOT) != ROOT:
    ROOT = os.path.dirname(ROOT)                    # notebooks live in notebooks/<topic>/, so climb up to the repo root
sys.path.insert(0, ROOT)                            # so that `import llmsuite` works from the notebook
import numpy as np
import matplotlib.pyplot as plt
import torch, torch.nn as nn, torch.nn.functional as F
from llmsuite.utils import seed_everything, get_device, count_params, human, show_mask
seed_everything(0)
device = get_device()
print("torch", torch.__version__, "| device", device)
'''

# Notebook 01 - Tokenization
nb = NB()
nb.md(r'''
# 01 · Tokenization: from text to integers

**What you will learn**
- Why LLMs operate on sub-word tokens rather than characters or words
- How byte-level BPE is trained, how it encodes/decodes, and why it never produces `<UNK>`
- How vocabulary size trades off sequence length against embedding size
- How to *evaluate* a tokenizer: compression ratio, fertility, round-trip correctness

**Source:** `llmsuite/tokenizer.py` (pure Python, ~120 lines).

| Granularity | Vocab | Sequence length | Problem |
|---|---|---|---|
| bytes | 256 | very long | attention is O(T²); the model wastes capacity spelling |
| words | 100k–1M+ | short | unknown words → `<UNK>`; terrible for code and multilingual text |
| **sub-words (BPE)** | 32k–256k | medium | the standard: frequent words = 1 token, rare words = a few pieces |
''')
nb.code(SETUP)
nb.code(r'''
from llmsuite.tokenizer import ByteBPETokenizer, compression_ratio
from llmsuite.data import template_text_corpus

corpus = template_text_corpus(n_sentences=3000)      # offline English-like text; swap in any .txt you like
print(corpus[:300], "...")
print(f"\ncorpus: {len(corpus):,} chars")
''')
nb.md(r'''
## Training BPE

Start with 256 byte tokens. Repeatedly find the most frequent adjacent pair and merge it into a new token.
Each merge is a rule `(a, b) -> new_id`; encoding replays the rules in order.
''')
nb.code(r'''
tok = ByteBPETokenizer()
t0 = time.time()
tok.train(corpus, vocab_size=600, verbose=False)
print(f"trained {len(tok.merges)} merges in {time.time() - t0:.1f}s; vocab size {tok.vocab_size} (incl. {len(tok.special)} special tokens)")

# The first merges are the most frequent byte pairs; later merges build whole words.
for (a, b), new_id in list(tok.merges.items())[:8]:
    print(f"merge {new_id}: {tok.vocab[a]!r} + {tok.vocab[b]!r} -> {tok.vocab[new_id]!r}")
print("...")
for (a, b), new_id in list(tok.merges.items())[-5:]:
    print(f"merge {new_id}: {tok.vocab[a]!r} + {tok.vocab[b]!r} -> {tok.vocab[new_id]!r}")
''')
nb.code(r'''
sentence = "The transformer predicts the next token efficiently."
ids = tok.encode(sentence)
print("ids   :", ids)
print("pieces:", [tok.decode([i]) for i in ids])
print("round-trip exact:", tok.decode(ids) == sentence)
print(f"{len(sentence.encode())} bytes -> {len(ids)} tokens  ({compression_ratio(tok, sentence):.2f} bytes/token)")

# Unseen text still works: it just falls back to smaller pieces, down to raw bytes, so there is never an <UNK>.
weird = "Zyxwvut 🦙 ¿Qué tal? def f(x): return x**2"
print("\nunseen text pieces:", [tok.decode([i]) for i in tok.encode(weird)])
print("round-trip exact:", tok.decode(tok.encode(weird)) == weird)
''')
nb.md(r'''
## Special tokens and persistence

Special tokens (`<|bos|>`, `<|eos|>`, `<|pad|>`) get the last ids so they can never be produced by a merge.
A tokenizer is frozen *before* pretraining — changing it later means re-learning the embedding table.
''')
nb.code(r'''
print("special:", tok.special)
print("with bos/eos:", tok.encode("hello", add_bos=True, add_eos=True))
tok.save(os.path.join(tempfile.gettempdir(), "bpe.json"))
tok2 = ByteBPETokenizer.load(os.path.join(tempfile.gettempdir(), "bpe.json"))
assert tok2.encode(sentence) == ids and tok2.decode(ids) == sentence
print("save/load round-trip OK")
''')
nb.md(r'''
## Evaluation: how good is a tokenizer?

1. **Round-trip correctness** — `decode(encode(x)) == x` for every input (a hard requirement).
2. **Compression** (bytes per token) on *your* domain — higher is better: shorter sequences = cheaper attention and more effective context.
3. **Fertility** (tokens per word) — a tokenizer trained on English is inefficient on Spanish or code.
4. **Vocabulary size trade-off** — bigger vocab compresses more but makes the embedding/LM-head matrices (`V × C`) heavier.

Below: compression as a function of vocabulary size, and fertility on in-domain vs out-of-domain text.
''')
nb.code(r'''
import matplotlib.pyplot as plt

eval_text_in = template_text_corpus(n_sentences=200, seed=123)             # same distribution, but sentences the tokenizer never saw
eval_text_out = ("def attention(q, k, v):\n    return softmax(q @ k.T / sqrt(d)) @ v\n" * 20
                 + "El modelo de lenguaje predice el siguiente token. " * 20)   # code and Spanish, which the tokenizer never saw

sizes = [300, 400, 600, 900, 1300]
ratios_in, ratios_out = [], []
for V in sizes:
    t = ByteBPETokenizer().train(corpus, vocab_size=V)
    ratios_in.append(compression_ratio(t, eval_text_in))
    ratios_out.append(compression_ratio(t, eval_text_out))
    assert t.decode(t.encode(eval_text_out)) == eval_text_out      # decoding must give back the exact text at every size

plt.plot(sizes, ratios_in, "o-", label="in-domain (English templates)")
plt.plot(sizes, ratios_out, "s-", label="out-of-domain (code + Spanish)")
plt.xlabel("vocab size"); plt.ylabel("bytes per token"); plt.title("Compression vs vocabulary size"); plt.legend(); plt.grid(alpha=.3)
plt.show()

def fertility(t, text):
    return len(t.encode(text)) / max(1, len(text.split()))
print(f"fertility (tokens/word) — in-domain: {fertility(tok, eval_text_in):.2f}   out-of-domain: {fertility(tok, eval_text_out):.2f}")
''')
nb.md(r'''
**Takeaways**
- In-domain compression improves quickly with vocabulary size and then saturates; out-of-domain text barely benefits — the merges learned are the *wrong* merges for it. This is why Llama 3 (128k vocab) and Gemma (256k) train tokenizers on multilingual + code mixes.
- Real tokenizers add a regex *pre-tokenizer* (no merges across spaces/punctuation), digit splitting (helps arithmetic), and are implemented in Rust (`tokenizers`, `tiktoken`). The algorithm is the same one you just ran.

**Next:** `02_attention_rope_norm.ipynb` — what the model does with these ids.
''')
nb.save(notebook_path("01_tokenization.ipynb"))

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

**Source:** `llmsuite/layers.py`.
''')
nb.code(SETUP)
nb.md(r'''
## 1. RMSNorm

LayerNorm re-centers and re-scales each token vector; RMSNorm only re-scales:

$$\text{RMSNorm}(x) = \gamma \odot \frac{x}{\sqrt{\tfrac{1}{C}\sum_i x_i^2 + \epsilon}}$$

Scale invariance is what stabilizes training; centering adds nothing measurable (Zhang & Sennrich 2019). One reduction less, no bias, equally good → the standard in Llama / Mistral / DeepSeek / Gemma.
''')
nb.code(r'''
from llmsuite.layers import RMSNorm

x = torch.randn(2, 5, 16) * 7 + 3                      # deliberately large scale and offset
ln, rms = nn.LayerNorm(16), RMSNorm(16)
for name, y in [("input", x), ("LayerNorm", ln(x)), ("RMSNorm", rms(x))]:
    print(f"{name:10s} mean {y.mean():6.2f}   rms {y.pow(2).mean().sqrt():5.2f}")
print("matches torch.nn.RMSNorm:", torch.allclose(rms(x), nn.RMSNorm(16)(x), atol=1e-5))
print("scale invariance: RMSNorm(3x) == RMSNorm(x):", torch.allclose(rms(3 * x), rms(x), atol=1e-5))
''')
nb.md(r'''
## 2. RoPE — rotary position embeddings

Attention is permutation-invariant, so position must be injected. RoPE rotates the query at position *m* by angle *mθ* and the key at *n* by *nθ*. Because rotations compose by adding angles:

$$(R(m\theta)q)^\top (R(n\theta)k) = q^\top R((n-m)\theta)\,k$$

**The score depends only on the distance n − m.** Each head's `Dh` dims are split into `Dh/2` pairs with frequencies $\theta_i = \text{base}^{-2i/D_h}$: fast "hands" for local resolution, slow ones for long range.
''')
nb.code(r'''
from llmsuite.layers import precompute_freqs_cis, apply_rope

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

During generation every past token's **k** and **v** are cached per layer. For Llama-2-70B (80 layers, 64 heads × 128 dims) that is 2.6 MB per token in fp16: with 4k tokens of context, **10 GB per sequence**. The cache, not compute, limits how many users a GPU serves.

| | k/v heads | cache per token per layer | used by |
|---|---|---|---|
| MHA | H | 2·H·Dh | GPT-2/3 |
| MQA | 1 | 2·Dh | PaLM, Falcon |
| **GQA** | Hkv (H/4–H/8) | 2·Hkv·Dh | Llama 2/3, Mistral, Qwen |
| **MLA** | latent | r + Dr (512 + 64) | DeepSeek V2/V3/R1 |

`GroupedQueryAttention` implements all of the first three via `n_kv_heads`; `MultiHeadLatentAttention` compresses k and v of *all* heads into one low-rank latent and caches that.
''')
nb.code(r'''
from llmsuite.layers import GroupedQueryAttention, MultiHeadLatentAttention

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
''')
nb.code(r'''
from llmsuite.model import ModelConfig, TransformerLM
from llmsuite.data import synthetic_structured_corpus, train_val_split
from llmsuite.training import TrainConfig, train_lm
from llmsuite.evaluation import perplexity

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

# Notebook 03 - Efficient attention
nb = NB()
nb.md(r'''
# 03 · Efficient attention for long contexts

Dense attention costs O(T²). This notebook implements and compares the main alternatives, all with the
same interface `forward(x, freqs_cis) -> (B, T, C)` so they drop into the same model:

| variant | pattern | cost | used by |
|---|---|---|---|
| dense causal | everything in the past | O(T²) | default; FlashAttention makes it memory-efficient |
| sliding window (SWA) | last `w` tokens | O(T·w) | Mistral, Gemma local layers, gpt-oss |
| Longformer | window + global tokens | O(T·(w+g)) | 2020 encoders; ancestor of hybrids |
| block-sparse (NSA/DSA) | top-k blocks chosen *by content* + local | O(T·k·b) | DeepSeek-V3.2, Kimi |
| linear | recurrent state, no softmax | O(T) | hybrids: Qwen3-Next, Kimi Linear, MiniMax |
| hybrid | mostly cheap layers + few dense | ~O(T) | Gemma 3 (5:1), Qwen3-Next (3:1) |

**Source:** `llmsuite/layers.py`.
''')
nb.code(SETUP)
nb.code(r'''
from llmsuite.layers import (causal_mask, sliding_window_mask, longformer_mask, precompute_freqs_cis,
                             SlidingWindowAttention, LongformerAttention, BlockSparseAttention, LinearAttention, GroupedQueryAttention)
T = 20
show_mask(causal_mask(T), "dense causal")
show_mask(sliding_window_mask(T, 5), "sliding window, w=5")
show_mask(longformer_mask(T, 5, [0, 10]), "Longformer, w=5, global={0,10}")
''')
nb.md(r'''
## Sliding window: receptive field grows with depth

With `L` layers, information can travel `L·w` tokens (token *i* in layer 2 sees *i−w*, which in layer 1 saw *i−2w*…), like a CNN's receptive field. Exact retrieval of a far token degrades, which is why global layers are interleaved. Two bonus properties: the KV-cache can be a ring buffer of size `w`, and the model generalizes to lengths never seen in training (no token ever sees more than `w` positions).

A boolean mask in SDPA still computes the full T×T scores and discards — the real saving needs a kernel that skips blocks (**FlexAttention**, PyTorch ≥ 2.5) or the ring-buffer cache at inference.
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
## Block-sparse attention chosen by content (NSA / DSA simplified)

1. Group keys in blocks of `block` tokens. 2. A cheap indexer scores each block per query (here `q · mean(k_block)`). 3. Attend only the top-`k` blocks + a local window, causally. The pattern is **dynamic** — different per query and per head — and block-structured so GPUs can gather efficiently. DeepSeek-V3.2 attends ~2k of 128k tokens this way with no measurable loss.
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
''')
nb.code(r'''
from llmsuite.model import ModelConfig, TransformerLM
from llmsuite.data import train_val_split
from llmsuite.training import TrainConfig, train_lm

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
- **Who can see far enough decides who can copy.** Dense attention sees everything. Block-sparse attention selects the relevant distant blocks by content and gets most of the way. SWA with `w=16` cannot see across the 30-token gap at all, and linear attention compresses the past into a fixed-size state that cannot hold an exact 12-token sequence. The hybrid (one global layer out of four) recovers part of the ability — the argument behind Gemma-3/Qwen3-Next-style hybrids, which in practice use more global layers and far longer training.
- Timing here reflects *mask-based* implementations; the asymptotic savings appear only with real kernels (FlexAttention, FlashAttention, Triton) at long sequence lengths.
- Decision rule: dense + GQA by default; SWA/hybrid when context ≥ 32k or KV-cache memory binds; linear/hybrid for ≥ 256k; learned block-sparse when you can afford the kernels — and always test exact long-range recall on your own data (needle-in-a-haystack, RULER).

''')
nb.md(r'''
**Next:** `04_ffn_moe.ipynb`.
''')
nb.save(notebook_path("03_efficient_attention.ipynb"))

# Notebook 04 - Feed-forward networks and MoE
nb = NB()
nb.md(r'''
# 04 · Feed-forward networks and Mixture of Experts

**What you will learn**
- The FFN is where ~2/3 of parameters (and most factual knowledge) live; SwiGLU vs GELU
- MoE: decouple parameters from compute with a router that picks top-k experts per token
- The load-balancing problem: auxiliary loss (Switch/Mixtral) vs bias-balanced routing (DeepSeek-V3)
- Fine-grained + shared experts (DeepSeekMoE)
- Evaluation: total vs active parameters, expert utilization, quality vs a dense model at equal *active* compute

**Source:** `llmsuite/layers.py` (`SwiGLU`, `MoE`, `AuxFreeRouter`), `llmsuite/model.py`.
''')
nb.code(SETUP)
nb.md(r'''
## SwiGLU
$$\text{FFN}(x) = W_{down}\big(\text{SiLU}(W_{gate}x) \odot W_{up}x\big)$$
A gated unit: one branch decides *how much* passes, the other *what*. Three matrices instead of two, so `d_ff ≈ 8/3·C` keeps the parameter count (Llama 3 8B: 4096 → 14336).
''')
nb.code(r'''
from llmsuite.layers import SwiGLU, swiglu_hidden_dim
C = 128
class FFN_GELU(nn.Module):
    def __init__(self, C, d_ff):
        super().__init__(); self.w1 = nn.Linear(C, d_ff, bias=False); self.w2 = nn.Linear(d_ff, C, bias=False)
    def forward(self, x): return self.w2(F.gelu(self.w1(x)))

d_ff = swiglu_hidden_dim(C, multiple_of=8)
print(f"GELU FFN d_ff={4*C}: {count_params(FFN_GELU(C, 4*C)):,} params | SwiGLU d_ff={d_ff}: {count_params(SwiGLU(C, d_ff)):,} params")
print("Llama-3-8B rule:", swiglu_hidden_dim(4096, multiple_of=1024, ffn_dim_multiplier=1.3))
''')
nb.md(r'''
## MoE: router + experts

```
probs   = softmax(W_router x)                 # one score per expert
top_k   = best k experts, weights renormalized
output  = Σ_{e∈top_k} w_e · Expert_e(x)       # only k experts run
```
Mixtral 8x7B: 47B total / 13B active. DeepSeek-V3: 671B / 37B (256 experts, k=8, 1 shared).

**Load balancing.** Without it the router collapses onto a few experts. Two fixes:
- **Auxiliary loss** $\mathcal{L}_{aux} = E\sum_e f_e P_e$ (fraction of tokens × mean router prob per expert), minimal when uniform; added with coefficient ≈ 0.01.
- **Aux-free (DeepSeek-V3):** a per-expert bias added to the scores *only for selection*, nudged up for under-used experts and down for over-used ones after each batch. No distortion of the language objective.
''')
nb.code(r'''
from llmsuite.layers import MoE, AuxFreeRouter

moe = MoE(C, d_ff=128, n_experts=8, top_k=2, n_shared=1, routing="softmax")
x = torch.randn(4, 32, C)
y, aux = moe(x)
# f_e counts routed *slots* per token, so sum_e f_e = top_k and a perfectly balanced router scores exactly top_k (2.0 here).
print("out", tuple(y.shape), f"| aux loss {aux.item():.3f} (perfectly uniform = top_k = {moe.k:.1f}; larger = imbalanced)")
print("slots per expert / tokens:", [round(v, 2) for v in moe.last_load.tolist()], f"(ideal top_k/E = {2/8:.2f} each)")
n_total = count_params(moe); n_expert = count_params(moe.experts[0]); n_active = n_total - 6 * n_expert
print(f"params total {n_total:,} | active per token {n_active:,} | ratio {n_total / n_active:.2f}x")
''')
nb.md(r'''
### Aux-free balancing in action
We force a badly imbalanced router (experts 0 and 1 receive almost every token) and watch the per-expert bias equalize the load — **without any gradient** on the router weights. After each batch the bias of over-used experts goes down by `bias_lr` and the bias of under-used ones goes up (`bias -= lr * sign(load - mean)`).

The update size matters: the scores are sigmoids in [0, 1], so a large `bias_lr` (e.g. 0.1) overshoots and the load *oscillates* between two bad states; DeepSeek-V3 uses a small value (0.001 at scale). We plot the imbalance ratio `max load / mean load` (1.0 = perfect).
''')
nb.code(r'''
torch.manual_seed(0)
xf = torch.randn(1024, C) + 0.5                       # the inputs all share a common direction...
fig, ax = plt.subplots(1, 2, figsize=(11, 3.5))
for bias_lr in (0.1, 0.01):
    router = AuxFreeRouter(C, n_experts=8, top_k=2, bias_lr=bias_lr).train()
    with torch.no_grad():
        router.weight.weight[0] += 0.05; router.weight.weight[1] += 0.03   # ...and we nudge experts 0 and 1 to point along it
    imbalance, loads = [], []
    for step in range(80):
        idx, w, _ = router(xf)
        load = F.one_hot(idx, 8).sum((0, 1)).float()
        imbalance.append((load.max() / load.mean()).item()); loads.append(load)
    ax[0].plot(imbalance, label=f"bias_lr={bias_lr}")
    print(f"bias_lr={bias_lr:<5}: step 0 load {loads[0].int().tolist()}  ->  step 79 load {loads[-1].int().tolist()}"
          f"  (imbalance {imbalance[0]:.2f}x -> {imbalance[-1]:.2f}x)")
ax[0].axhline(1, color="gray", ls="--", lw=1); ax[0].set_xlabel("router update step"); ax[0].set_ylabel("max load / mean load")
ax[0].set_title("aux-free balancing: imbalance over time"); ax[0].legend(); ax[0].grid(alpha=.3)
ax[1].bar(range(8), loads[0].tolist(), alpha=.5, label="step 0"); ax[1].bar(range(8), loads[-1].tolist(), alpha=.8, label="step 79 (bias_lr=0.01)")
ax[1].axhline(1024 * 2 / 8, color="gray", ls="--", lw=1, label="uniform"); ax[1].set_xlabel("expert"); ax[1].set_ylabel("tokens routed")
ax[1].set_title("load per expert"); ax[1].legend(); plt.tight_layout(); plt.show()
''')
nb.md(r'''
## Evaluation: MoE vs dense at equal *active* compute

The fair comparison is not "same total parameters" but "same FLOPs per token". We train (a) a dense model, (b) a MoE with the same active parameters but 4× total, (c) the same MoE with aux-free routing, for the same steps, and compare validation perplexity and expert utilization. On a toy corpus the MoE advantage is small or absent — MoE pays off with large data and compute; the notebook shows **how to measure**, not that MoE always wins.
''')
nb.code(r'''
from llmsuite.model import ModelConfig, TransformerLM
from llmsuite.data import synthetic_structured_corpus, train_val_split
from llmsuite.training import TrainConfig, train_lm
from llmsuite.evaluation import perplexity

data = synthetic_structured_corpus(n_tokens=150_000, vocab=512, n_words=120)
train_data, val_data = train_val_split(data)
tcfg = TrainConfig(steps=300, batch_size=32, block_size=64, max_lr=3e-3, warmup=30, eval_every=300, log=False)

runs = {
    "dense":          ModelConfig(vocab_size=512, d_model=128, n_layers=4, n_heads=8, n_kv_heads=2, d_ff=256),
    "MoE 4x top-2":   ModelConfig(vocab_size=512, d_model=128, n_layers=4, n_heads=8, n_kv_heads=2, d_ff=128, moe=True, n_experts=4, top_k=2),
    "MoE aux-free":   ModelConfig(vocab_size=512, d_model=128, n_layers=4, n_heads=8, n_kv_heads=2, d_ff=128, moe=True, n_experts=4, top_k=2, moe_routing="auxfree"),
}
for name, cfg in runs.items():
    seed_everything(0)
    model = TransformerLM(cfg)
    res = train_lm(model, train_data, val_data, tcfg, device)
    ppl = perplexity(model, val_data, block_size=64, n_batches=20, device=device)
    loads = [[round(v, 2) for v in b.ffn.last_load.tolist()] for b in model.blocks if isinstance(b.ffn, MoE)]
    print(f"{name:14s} total {human(model.num_params()):>7}  active {human(model.num_active_params()):>7}  val ppl {ppl:7.2f}  {res.wall_time:4.1f}s")
    if loads:
        print(f"{'':14s} expert load (last batch), layer 0: {loads[0]}")
''')
nb.md(r'''
**Takeaways**
- Watch the *expert load* row: values near `[0.5, 0.5, 0.5, 0.5]` (k/E = 2/4) mean balanced routing; a dead expert shows as `0.0`.
- MoE's wall time is higher here because the expert loop is Python; grouped GEMM kernels remove that overhead in practice.
- Use MoE when compute-bound at scale with lots of data and expert-parallel infrastructure; otherwise dense. For fine-tuning, dense is far simpler.

**Next:** `05_pretraining.ipynb` — the full training pipeline end to end.
''')
nb.save(notebook_path("04_ffn_moe.ipynb"))

# Notebook 05 - Pre-training
nb = NB()
nb.md(r'''
# 05 · Pretraining a small LM end to end

**Pipeline:** raw text → tokenizer → token tensor → model → training loop → curves → perplexity → generation.

Everything here is what a real pretraining run does, minus multi-GPU distribution and a streaming data loader:
- next-token cross-entropy (all positions in parallel thanks to the causal mask)
- AdamW with decay on matrices only, β₂=0.95
- warmup + cosine (or WSD) schedule, gradient clipping, mixed precision on GPU
- validation perplexity, checkpoints, loss/LR/grad-norm curves (your training health dashboard)

**Source:** `llmsuite/training.py`, `llmsuite/model.py`, `llmsuite/data.py`.
''')
nb.code(SETUP)
nb.md(r'''
## 1. Text → tokens
We use a generated English-like corpus so the notebook runs offline. To use real text, download e.g. tiny Shakespeare
(`curl -O https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt`) and set `TEXT_PATH`.
''')
nb.code(r'''
from llmsuite.tokenizer import ByteBPETokenizer
from llmsuite.data import template_text_corpus, load_text_file, train_val_split

TEXT_PATH = None    # e.g. os.path.join(ROOT, "data", "input.txt")
text = load_text_file(TEXT_PATH) if TEXT_PATH else template_text_corpus(n_sentences=20_000, seed=1)
print(f"{len(text):,} characters")

tok = ByteBPETokenizer().train(text[:300_000], vocab_size=1024)
ids = torch.tensor(tok.encode(text), dtype=torch.long)
train_data, val_data = train_val_split(ids, val_frac=0.1)
print(f"{len(ids):,} tokens ({len(text)/len(ids):.2f} chars/token) | train {len(train_data):,} | val {len(val_data):,}")
''')
nb.md(r'''
## 2. Model and sanity checks
Initial loss must be ≈ ln(V): an untrained model predicts uniformly. Much higher → bad init (logits too large); lower → information leak (broken causal mask).
''')
nb.code(r'''
from llmsuite.model import ModelConfig, TransformerLM
from llmsuite.data import get_batch

cfg = ModelConfig(vocab_size=tok.vocab_size, max_seq_len=256, d_model=192, n_layers=4, n_heads=6, n_kv_heads=2)
model = TransformerLM(cfg).to(device)
x, y = get_batch(train_data, 128, 8, device)
with torch.no_grad():
    _, loss0 = model(x, y)
print(f"params {human(model.num_params())} | initial loss {loss0.item():.3f} vs ln(V) = {math.log(cfg.vocab_size):.3f}")

# Causal-mask leak test: changing a future token must not change the logits at earlier positions.
x2 = x.clone(); x2[:, -1] = (x2[:, -1] + 1) % cfg.vocab_size
with torch.no_grad():
    l1, _ = model(x); l2, _ = model(x2)
print("no future leak:", torch.allclose(l1[:, :-1], l2[:, :-1], atol=1e-5))
''')
nb.md(r'''
## 3. Train
`TrainConfig` collects the knobs. On CPU this takes a few minutes; shrink `steps` or the model if needed.
''')
nb.code(r'''
from llmsuite.training import TrainConfig, train_lm, plot_training

tcfg = TrainConfig(steps=600, batch_size=32, block_size=128, max_lr=3e-3, warmup=60, schedule="cosine",
                   weight_decay=0.1, grad_clip=1.0, eval_every=100, checkpoint_path=os.path.join(tempfile.gettempdir(), "pretrain_ckpt.pt"))
res = train_lm(model, train_data, val_data, tcfg, device)
plot_training(res, "pretraining"); import matplotlib.pyplot as plt; plt.show()
''')
nb.md(r'''
**Reading the dashboard**: loss should fall monotonically (noisy per step is fine); grad norm should stay O(1) — a spike with a loss jump is instability (lower LR, longer warmup, add QK-norm: `ModelConfig(qk_norm=True)`); the LR curve confirms warmup/decay happened as intended.
''')
nb.md(r'''
## 4. Generate
''')
nb.code(r'''
prompt = torch.tensor([tok.encode("The model")], device=device)
for name, kw in {"greedy": dict(temperature=0.0), "T=0.8 top-k 40": dict(temperature=0.8, top_k=40), "T=1.0 top-p 0.9": dict(temperature=1.0, top_p=0.9)}.items():
    out = model.generate(prompt, 40, eos_id=tok.eos_id, **kw)
    print(f"{name:16s}| {tok.decode(out[0].tolist())!r}")
''')
nb.md(r'''
## 5. Evaluation

1. **Perplexity** on held-out text — the primary pretraining metric.
2. **How good is "good"? The irreducible loss.** Our corpus is generated by a known process: each sentence picks a subject, verb, object and tail uniformly from 10 options each, so it carries exactly $4\ln 10 = 9.21$ nats of information. Divided by the number of tokens per sentence, that is the **lowest loss any model can reach** (the $E$ term of the Chinchilla scaling law $L(N,D) = E + A/N^\alpha + B/D^\beta$). Real text has an unknown, much higher floor — that is why real LMs keep improving with scale while ours saturates.
3. **Schedule ablation** — cosine vs WSD at equal steps.
4. **Scaling sanity** — a bigger model should learn faster / reach lower loss at the same steps if data is not the bottleneck.

Because our model already sits at the floor after ~100 steps, comparing *final* numbers would show nothing; we compare **learning curves at a short budget** instead, where the differences live.
''')
nb.code(r'''
from llmsuite.evaluation import perplexity
n_sentences = 20_000
floor_loss = 4 * math.log(10) * n_sentences / len(ids)          # true information content, in nats per token
val_ppl = perplexity(model, val_data, block_size=128, n_batches=20, device=device)
print(f"tokens per sentence      : {len(ids) / n_sentences:.2f}")
print(f"irreducible loss (floor) : {floor_loss:.3f} nats/token  -> perplexity floor {math.exp(floor_loss):.2f}")
print(f"our model                : val perplexity {val_ppl:.2f}  ({100 * (math.log(val_ppl) / floor_loss - 1):+.1f}% above the floor in loss)")
''')
nb.code(r'''
def curve(cfg_, sched="cosine", steps=150):
    seed_everything(0)
    m = TransformerLM(cfg_)
    r = train_lm(m, train_data, val_data, TrainConfig(steps=steps, batch_size=32, block_size=128, max_lr=3e-3, warmup=15,
                                                      schedule=sched, eval_every=10, eval_batches=5, log=False), device)
    return m, r

fig, ax = plt.subplots(1, 2, figsize=(12, 4))
for sched in ("cosine", "wsd"):
    m, r = curve(cfg, sched)
    s_, v_ = zip(*r.val_losses); ax[0].plot(s_, v_, "o-", ms=3, label=f"{sched} (final {v_[-1]:.3f})")
    print(f"schedule {sched:6s}: val loss @50 {dict(r.val_losses)[50]:.3f} | final {v_[-1]:.3f}")
ax[0].set_title("schedule ablation (same model, 150 steps)")

for d_model, n_layers in [(64, 2), (192, 4)]:
    c = ModelConfig(vocab_size=tok.vocab_size, max_seq_len=256, d_model=d_model, n_layers=n_layers, n_heads=4 if d_model == 64 else 6, n_kv_heads=2)
    m, r = curve(c)
    s_, v_ = zip(*r.val_losses); ax[1].plot(s_, v_, "o-", ms=3, label=f"{human(m.num_params())} params (final {v_[-1]:.3f})")
    print(f"size {human(m.num_params()):>7}: val loss @50 {dict(r.val_losses)[50]:.3f} | final {v_[-1]:.3f}")
ax[1].set_title("scaling sanity (cosine, 150 steps)")
for a in ax:
    a.axhline(floor_loss, color="red", ls="--", lw=1, label=f"irreducible loss {floor_loss:.3f}")
    a.set_ylim(floor_loss - 0.1, 3.0); a.set_xlabel("step"); a.set_ylabel("val loss (nats/token)"); a.grid(alpha=.3); a.legend()
plt.tight_layout(); plt.show()
''')
nb.md(r'''
**What to read from these plots**
- Every curve converges to the red line: once a model reaches the irreducible loss, more parameters or a better schedule cannot help — the only way to go lower is *more information in the data*.
- The bigger model gets there in fewer steps (higher sample efficiency) — that is the scaling behaviour you see in real pretraining, where the floor is far away.
- Cosine and WSD end up practically identical here. WSD keeps the LR high for longer and decays only at the end; its practical advantage is operational: you can extend a run from the stable phase (or branch several decays from one checkpoint) without restarting the schedule — which is why several recent open models use it.
''')
nb.md(r'''
## 6. Save / load
The checkpoint stores the config and weights; `TransformerLM.load` rebuilds the exact model. The tokenizer is saved alongside — a model is useless without *its* tokenizer.
''')
nb.code(r'''
model.save(os.path.join(tempfile.gettempdir(), "mini_lm.pt")); tok.save(os.path.join(tempfile.gettempdir(), "mini_lm_tokenizer.json"))
m2 = TransformerLM.load(os.path.join(tempfile.gettempdir(), "mini_lm.pt")).to(device); t2 = ByteBPETokenizer.load(os.path.join(tempfile.gettempdir(), "mini_lm_tokenizer.json"))
with torch.no_grad():
    a, _ = model(x); b, _ = m2(x)
print("reloaded model identical:", torch.allclose(a, b, atol=1e-6))
''')
nb.md(r'''
**From here to a real LLM:** the same loop wrapped in FSDP/DDP (`torchtitan` is the reference), a memory-mapped token file instead of a tensor in RAM, 100–2000 tokens per parameter of deduplicated, filtered data, bf16 + `torch.compile`, and checkpoints every few hundred steps. Read the Llama 3 and DeepSeek-V3 reports for the recipes.

**Next:** `06_inference_serving.ipynb`.
''')
nb.save(notebook_path("05_pretraining.ipynb"))

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

**Source:** `llmsuite/model.py` (`generate`, `sample_next_token`), `llmsuite/inference.py`.
''')
nb.code(SETUP)
nb.code(r'''
# Train a small target model and an even smaller draft model on the same data. Each takes a few seconds.
from llmsuite.model import ModelConfig, TransformerLM
from llmsuite.data import synthetic_structured_corpus, train_val_split
from llmsuite.training import TrainConfig, train_lm

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
from llmsuite.model import next_token_distribution

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
from llmsuite.inference import kv_cache_memory
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
from llmsuite.inference import speculative_decode

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
from llmsuite.inference import batched_generate
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
from llmsuite.evaluation import benchmark_generation
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

print("part 1 notebooks written")
