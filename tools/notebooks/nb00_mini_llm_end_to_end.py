"""Notebook 00 - A complete mini LLM, end to end

Builds 00_mini_llm_end_to_end.ipynb. Run: python tools/build_notebooks.py 00
"""
import os
from _common import *  # noqa: F401,F403

# Notebook 00 - A complete mini LLM, end to end
nb = NB()
script_cells: list[str] = []


def code(src: str, script: bool = True):
    """Add a code cell; script=True also exports it to scripts/mini_llm.py."""
    nb.code(src)
    if script:
        script_cells.append(src.strip("\n"))


nb.md(r'''
# 00 · A complete mini LLM in one notebook

Everything needed to build a GPT-style language model, **in one place and with no project imports** — only PyTorch:

1. **Data** — a text corpus (built-in generator, or any `.txt` you provide)
2. **Tokenizer** — byte-level BPE, trained on the corpus
3. **Model** — a modern decoder-only transformer: token embeddings, **RoPE**, **RMSNorm**, causal multi-head attention, **SwiGLU** feed-forward, tied output head (the Llama recipe, miniaturized)
4. **Training** — AdamW with weight-decay groups, warmup + cosine LR, gradient clipping, train/val curves
5. **Evaluation** — perplexity, and the theoretical loss floor of the data
6. **Generation** — temperature, top-k and top-p sampling
7. **Save / load** — a self-contained checkpoint

The other notebooks take each piece apart in depth (01 tokenization → 02 attention → 04 FFN → 05 pre-training → 06 inference); this one shows how they **fit together**. The same code is exported to `scripts/mini_llm.py`: `python scripts/mini_llm.py` trains the model from the command line.

**Runtime:** about a minute on a laptop GPU (RTX 3060), 3–5 minutes on a CPU. The code picks the GPU automatically when PyTorch sees one.
''')
nb.md(r'''
## Background: what a language model is, end to end

A causal language model is a probability distribution over token sequences, built from the chain rule of probability:

$$p(x_1, \dots, x_T) = \prod_{t=1}^{T} p(x_t \mid x_1, \dots, x_{t-1})$$

The network never models whole sentences at once. Its only job is, given a prefix, to output a distribution over the next token. Training maximizes the likelihood of real text, which is the same as minimizing the average cross-entropy of the true next token:

$$\mathcal{L}(\theta) = -\frac{1}{T}\sum_{t=1}^{T} \log p_\theta(x_t \mid x_{<t})$$

This objective is **self-supervised**: the targets are the input shifted by one position, so any text is training data and no labels are needed. That is the property that let language models scale to trillions of tokens. Everything in this notebook exists to serve this one objective:

```
 raw text ─► tokenizer ─► ids ─► transformer ─► logits (T × V) ─► softmax ─► p(next token)
                                                                       │
             training:   cross-entropy against the ids shifted by one ◄┤
             generation: sample one token, append it, run again      ◄┘
```

| stage | what it does | the key idea | goes deeper in |
|---|---|---|---|
| tokenizer | text to integers and back | byte-level BPE: no unknown tokens, frequent strings get one id | 01 |
| embedding + blocks | turn ids into contextual vectors | attention mixes information across positions, the FFN transforms each position | 02, 03, 04 |
| LM head | vector to $V$ logits | a linear layer, here sharing weights with the embedding | 02 |
| training | fit $\theta$ | cross-entropy, AdamW, LR schedule | 05 |
| generation | turn the distribution into text | repeated sampling, with a KV-cache in practice | 06 |

### Why this architecture

Before transformers, language models were n-gram counts and then recurrent networks (LSTMs), which read text one token at a time. Recurrence makes training sequential and makes it hard for gradients to connect distant tokens. The Transformer (Vaswani et al., 2017, "Attention Is All You Need") replaced recurrence with attention, where every position looks at every earlier position directly. Thanks to the causal mask, a single forward pass computes all $T$ next-token predictions in parallel during training ("teacher forcing"); generation is still sequential, one token per step.

The decoder-only variant was popularized by the GPT series (Radford et al., 2018 and 2019; Brown et al., 2020, "Language Models are Few-Shot Learners"). It won over encoder-decoder and encoder-only designs for general-purpose models because one simple objective covers every task that can be phrased as text continuation, and because its performance improves predictably with scale (Kaplan et al., 2020; Hoffmann et al., 2022, the "Chinchilla" paper).

The model below is not GPT-2, though. It follows the recipe consolidated by Llama (Touvron et al., 2023), which replaced several GPT-2 components:

| component | GPT-2 (2019) | modern recipe (this notebook) | why it changed |
|---|---|---|---|
| position | learned absolute embeddings | RoPE (Su et al., 2021) | relative positions, no length-limited table |
| normalization | LayerNorm | RMSNorm (Zhang & Sennrich, 2019) | cheaper, equally stable |
| FFN | GELU MLP | SwiGLU (Shazeer, 2020) | better quality at the same parameter count |
| biases | in linear layers | mostly removed | simpler, no measurable benefit at scale |

Each of these is explained in the notebook listed above. Here the aim is to see the whole loop work once, with nothing hidden.
''')
nb.md(r'''
## 0. Configuration
All the knobs in one place. Doubling `d_model`/`n_layers` or `steps` gives better text at the cost of time.
''')
code(r'''
import math, os, re, time, random, json
from collections import Counter
import torch
import torch.nn as nn
import torch.nn.functional as F

CFG = dict(
    text_path=None,          # e.g. "input.txt" (tiny Shakespeare); None uses the built-in synthetic corpus
    vocab_size=512,          # 256 raw bytes plus the learned merges
    block_size=128,          # context length in tokens
    d_model=192, n_layers=4, n_heads=6,
    dropout=0.0,
    batch_size=32, steps=800, max_lr=3e-3, min_lr=3e-4, warmup=80, weight_decay=0.1, grad_clip=1.0,
    eval_every=100, eval_batches=20, seed=0,
)
torch.manual_seed(CFG["seed"]); random.seed(CFG["seed"])
device = "cuda" if torch.cuda.is_available() else "cpu"
# On an NVIDIA GPU we train in bfloat16 mixed precision (Ampere or newer) and let matmuls use TF32: same results, much faster.
use_bf16 = device == "cuda" and torch.cuda.is_bf16_supported()
torch.set_float32_matmul_precision("high")
print("torch", torch.__version__, "| device", device, f"({torch.cuda.get_device_name(0)})" if device == "cuda" else "", "| bf16 autocast:", use_bf16)
''')
nb.md(r'''
## 1. Data

To stay offline we generate English-like text from a small grammar: each sentence picks a subject, verb, object and place at random. Because we know exactly how it is generated, we can compute the **best loss any model could reach** (section 5) — a luxury you never have with real text. Point `text_path` to any `.txt` file to train on real data instead, e.g. tiny Shakespeare:
`curl -O https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt`.
''')
code(r'''
SUBJECTS = ["the old sailor", "a young fox", "the tired queen", "my neighbour", "the little robot", "a clever student",
            "the quiet painter", "our teacher"]
VERBS = ["found", "painted", "carried", "lost", "built", "followed", "remembered", "admired"]
OBJECTS = ["a golden key", "the broken clock", "a red umbrella", "the secret map", "an empty bottle", "the wooden boat",
           "a strange letter", "the blue lantern"]
PLACES = ["near the river", "in the garden", "under the bridge", "at the market", "on the hill", "behind the castle",
          "in the library", "by the sea"]
N_SENTENCES = 30_000

def synthetic_corpus(n, seed=0):
    rng = random.Random(seed)
    sents = [f"{rng.choice(SUBJECTS)} {rng.choice(VERBS)} {rng.choice(OBJECTS)} {rng.choice(PLACES)}." for _ in range(n)]
    return " ".join(s[0].upper() + s[1:] for s in sents)

if CFG["text_path"] and os.path.exists(CFG["text_path"]):
    text = open(CFG["text_path"], encoding="utf-8").read()
else:
    text = synthetic_corpus(N_SENTENCES)
print(f"{len(text):,} characters | sample: {text[:160]!r}")
''')
nb.md(r'''
## 2. Tokenizer: byte-level BPE

Start from the 256 possible bytes (so *any* text can be encoded — no unknown tokens), then repeatedly **merge the most frequent adjacent pair** into a new token. Frequent words end up as single tokens, rare words as a few pieces.

If the corpus has few distinct words (ours has ~40), training stops early: once every word is a single token there is nothing left to merge. With real text the merges go on until `vocab_size` (32k–256k in production models).

Speed trick used by real implementations: split the text into words first (a regex "pre-tokenizer") and count each *unique* word once, so a merge never crosses a word boundary and each step costs O(unique words) instead of O(text).
''')
code(r'''
class BPETokenizer:
    PATTERN = re.compile(r" ?[A-Za-z]+| ?[0-9]+| ?[^A-Za-z0-9\s]+|\s+")      # a simplified GPT-2-style pre-tokenizer

    def __init__(self):
        self.merges: dict[tuple[int, int], int] = {}                        # maps (a, b) to the new token id, kept in merge order
        self.vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}

    @staticmethod
    def _merge(ids, pair, new_id):
        out, i = [], 0
        while i < len(ids):
            if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
                out.append(new_id); i += 2
            else:
                out.append(ids[i]); i += 1
        return out

    def train(self, text, vocab_size):
        words = Counter(self.PATTERN.findall(text))                          # each unique pre-token once, with its count
        seqs = {w: list(w.encode("utf-8")) for w in words}
        for new_id in range(256, vocab_size):
            pairs = Counter()
            for w, ids in seqs.items():
                for p in zip(ids, ids[1:]):
                    pairs[p] += words[w]
            if not pairs:
                break
            best = max(pairs, key=pairs.get)
            self.merges[best] = new_id
            self.vocab[new_id] = self.vocab[best[0]] + self.vocab[best[1]]
            seqs = {w: self._merge(ids, best, new_id) for w, ids in seqs.items()}
        return self

    def encode(self, text):
        out = []
        for w in self.PATTERN.findall(text):
            ids = list(w.encode("utf-8"))
            while len(ids) > 1:                                              # replay the merges in the order they were learned
                pair = min(zip(ids, ids[1:]), key=lambda p: self.merges.get(p, float("inf")))
                if pair not in self.merges:
                    break
                ids = self._merge(ids, pair, self.merges[pair])
            out.extend(ids)
        return out

    def decode(self, ids):
        return b"".join(self.vocab[i] for i in ids).decode("utf-8", errors="replace")

t0 = time.time()
tok = BPETokenizer().train(text[:200_000], CFG["vocab_size"])
ids = tok.encode(text)
V = len(tok.vocab)                                   # the real vocabulary size; training stops early once there is nothing left to merge
print(f"trained {len(tok.merges)} merges in {time.time() - t0:.1f}s -> vocabulary {V} | {len(ids):,} tokens | {len(text) / len(ids):.2f} chars/token")
print("round-trip OK:", tok.decode(tok.encode("The old sailor found a golden key.")) == "The old sailor found a golden key.")
print("last merges:", [tok.vocab[i].decode() for i in range(V - 6, V)])
print("'The old sailor found a golden key.' ->", [tok.vocab[i].decode() for i in tok.encode("The old sailor found a golden key.")])
''')
code(r'''
data = torch.tensor(ids, dtype=torch.long)
n_val = len(data) // 10
train_data, val_data = data[:-n_val], data[-n_val:]                       # keep the last 10% aside for validation

def get_batch(split):
    """Random windows of block_size tokens; the target is the same window shifted by one (next-token prediction)."""
    d = train_data if split == "train" else val_data
    ix = torch.randint(len(d) - CFG["block_size"] - 1, (CFG["batch_size"],))
    x = torch.stack([d[i : i + CFG["block_size"]] for i in ix])
    y = torch.stack([d[i + 1 : i + 1 + CFG["block_size"]] for i in ix])
    return x.to(device), y.to(device)

x, y = get_batch("train")
print("x:", x.shape, "| first 10 inputs ", x[0, :10].tolist(), "\n   first 10 targets", y[0, :10].tolist())
''')
nb.md(r'''
## 3. The model

```
tokens ─► embedding ─► [ RMSNorm ─► causal self-attention (RoPE) ─► + ] ─► [ RMSNorm ─► SwiGLU FFN ─► + ] ─► ... ─► RMSNorm ─► lm_head ─► logits
                         └──────────── residual ─────────────┘             └──────── residual ──────┘        × n_layers
```

- **RMSNorm** — rescales each vector to unit RMS (cheaper than LayerNorm, same stability). *Pre-norm*: applied before each sub-layer.
- **RoPE** — rotates query/key dimension pairs by an angle proportional to the position, so `q·k` depends only on the *relative* distance.
- **Causal attention** — each position attends only to itself and the past (`is_causal=True`); `scaled_dot_product_attention` picks a fused kernel (FlashAttention on GPU).
- **SwiGLU** — `W_down(SiLU(W_gate x) ⊙ W_up x)`, the gated FFN used by Llama, Qwen, Mistral.
- **Tied embeddings** — the output head reuses the input embedding matrix (fewer parameters, works well for small models).
''')
code(r'''
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps, self.weight = eps, nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return self.weight * x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)


def rope_tables(head_dim, max_len, base=10_000.0):
    """cos/sin tables: pair i rotates with frequency base^(-2i/d) — fast pairs encode local order, slow pairs long range."""
    inv_freq = 1.0 / base ** (torch.arange(0, head_dim, 2).float() / head_dim)
    angles = torch.outer(torch.arange(max_len).float(), inv_freq)            # (T, head_dim/2)
    return angles.cos(), angles.sin()


def apply_rope(x, cos, sin):
    """x: (B, H, T, D). Rotate each (even, odd) pair of dimensions by its position-dependent angle."""
    x1, x2 = x[..., 0::2], x[..., 1::2]
    cos, sin = cos[: x.shape[-2]], sin[: x.shape[-2]]
    out = torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
    return out.flatten(-2)


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout):
        super().__init__()
        self.h, self.dh = n_heads, d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = dropout

    def forward(self, x, cos, sin):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=-1)
        q, k, v = (t.view(B, T, self.h, self.dh).transpose(1, 2) for t in (q, k, v))   # (B, H, T, Dh)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=self.dropout if self.training else 0.0)
        return self.proj(y.transpose(1, 2).reshape(B, T, C))


class SwiGLU(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        hidden = 64 * ((int(8 * d_model / 3) + 63) // 64)                     # Llama's 8/3 * d rule, rounded up to a multiple of 64
        self.w_gate = nn.Linear(d_model, hidden, bias=False)
        self.w_up = nn.Linear(d_model, hidden, bias=False)
        self.w_down = nn.Linear(hidden, d_model, bias=False)

    def forward(self, x):
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


class Block(nn.Module):
    def __init__(self, d_model, n_heads, dropout):
        super().__init__()
        self.norm1, self.attn = RMSNorm(d_model), CausalSelfAttention(d_model, n_heads, dropout)
        self.norm2, self.ffn = RMSNorm(d_model), SwiGLU(d_model)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.norm1(x), cos, sin)                          # pre-norm, then add back the residual
        return x + self.ffn(self.norm2(x))


class MiniGPT(nn.Module):
    def __init__(self, vocab_size, block_size, d_model, n_layers, n_heads, dropout=0.0):
        super().__init__()
        self.block_size = block_size
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList(Block(d_model, n_heads, dropout) for _ in range(n_layers))
        self.norm = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight                            # tie the output head to the input embedding
        cos, sin = rope_tables(d_model // n_heads, block_size)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        self.apply(self._init)
        for name, p in self.named_parameters():                              # as in GPT-2, shrink residual projections by 1/sqrt(2L) so deep stacks start stable
            if name.endswith(("proj.weight", "w_down.weight")):
                nn.init.normal_(p, std=0.02 / math.sqrt(2 * n_layers))

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, idx, targets=None):
        x = self.tok_emb(idx)
        for blk in self.blocks:
            x = blk(x, self.cos, self.sin)
        logits = self.lm_head(self.norm(x))
        loss = None if targets is None else F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

model = MiniGPT(V, CFG["block_size"], CFG["d_model"], CFG["n_layers"], CFG["n_heads"], CFG["dropout"]).to(device)
n_params = sum(p.numel() for p in model.parameters())
with torch.no_grad():
    _, loss0 = model(x, y)
print(f"{n_params / 1e6:.2f}M parameters | initial loss {loss0.item():.3f} (uniform guess = ln V = {math.log(V):.3f})")
''')
nb.md(r'''
**Sanity check:** an untrained model should be ≈ uniform, so the initial loss must be close to `ln(V)`. Much higher means a bad initialization; much lower would mean the model can see the answer (a broken causal mask).

## 4. Training

- **AdamW** with weight decay on matrices only (not on norms/embedding gains — decaying those just hurts).
- **Warmup + cosine decay** of the learning rate: large steps early are unstable, small steps late fine-tune.
- **Gradient clipping** at norm 1.0 protects against rare loss spikes.
- We evaluate on held-out data every `eval_every` steps — the gap between train and val loss reveals overfitting.

**The optimizer in brief.** Adam (Kingma & Ba, 2015) keeps running averages of each parameter's gradient and squared gradient, and divides one by the square root of the other, so every parameter gets its own effective step size. AdamW (Loshchilov & Hutter, 2019, "Decoupled Weight Decay Regularization") applies weight decay directly to the weights instead of folding it into the gradient, where Adam's rescaling would distort it; this is the default optimizer for transformers. Warmup exists because Adam's squared-gradient estimate is unreliable for the first steps, and large updates on a random network can push it into a bad region. Cosine decay (from Loshchilov & Hutter's SGDR, 2017) lowers the step size gradually so the end of training settles into a minimum rather than bouncing around it. Notebook 05 compares schedules and covers scaling laws.
''')
code(r'''
decay = [p for n, p in model.named_parameters() if p.dim() >= 2]
no_decay = [p for n, p in model.named_parameters() if p.dim() < 2]
opt = torch.optim.AdamW([{"params": decay, "weight_decay": CFG["weight_decay"]}, {"params": no_decay, "weight_decay": 0.0}],
                        lr=CFG["max_lr"], betas=(0.9, 0.95))

def lr_at(step):
    if step < CFG["warmup"]:
        return CFG["max_lr"] * (step + 1) / CFG["warmup"]
    progress = (step - CFG["warmup"]) / max(1, CFG["steps"] - CFG["warmup"])
    return CFG["min_lr"] + 0.5 * (CFG["max_lr"] - CFG["min_lr"]) * (1 + math.cos(math.pi * progress))

@torch.no_grad()
def estimate_loss(split):
    model.eval()
    losses = [model(*get_batch(split))[1].item() for _ in range(CFG["eval_batches"])]
    model.train()
    return sum(losses) / len(losses)

history = {"step": [], "train": [], "val": [], "lr": []}
t0 = time.time()
model.train()
for step in range(CFG["steps"]):
    for g in opt.param_groups:
        g["lr"] = lr_at(step)
    xb, yb = get_batch("train")
    with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=use_bf16):
        _, loss = model(xb, yb)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), CFG["grad_clip"])
    opt.step()
    if step % CFG["eval_every"] == 0 or step == CFG["steps"] - 1:
        tr, va = estimate_loss("train"), estimate_loss("val")
        for k, v in zip(history, (step, tr, va, lr_at(step))):
            history[k].append(v)
        print(f"step {step:4d} | train {tr:.3f} | val {va:.3f} | ppl {math.exp(va):6.2f} | lr {lr_at(step):.1e} | {time.time() - t0:5.0f}s")
''')
code(r'''
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 2, figsize=(12, 3.8))
ax[0].plot(history["step"], history["train"], "o-", label="train")
ax[0].plot(history["step"], history["val"], "o-", label="validation")
ax[0].set_xlabel("step"); ax[0].set_ylabel("loss (nats/token)"); ax[0].set_title("learning curves"); ax[0].grid(alpha=.3); ax[0].legend()
ax[1].plot(range(CFG["steps"]), [lr_at(s) for s in range(CFG["steps"])])
ax[1].set_xlabel("step"); ax[1].set_title("learning rate: warmup + cosine"); ax[1].grid(alpha=.3)
plt.tight_layout(); plt.show()
''', script=False)
nb.md(r'''
## 5. Evaluation: how good is it, really?

**Perplexity** = `exp(loss)`: the model is as uncertain as if it were choosing uniformly among that many tokens.

For the synthetic corpus we can do better than "lower is better": each sentence makes 4 independent choices among 8 options, i.e. carries exactly $4 \ln 8 = 8.32$ nats of information. Spread over the tokens of a sentence, that is the **irreducible loss** — no model, however large, can go below it. The gap between our validation loss and this floor is what is left to learn. (With real text the floor is unknown and much higher, which is why real LMs keep improving with scale.)
''')
code(r'''
val_loss = estimate_loss("val")
print(f"validation loss {val_loss:.3f} | perplexity {math.exp(val_loss):.2f}")
if not (CFG["text_path"] and os.path.exists(CFG["text_path"])):
    floor = 4 * math.log(8) * N_SENTENCES / len(ids)                   # nats per token
    print(f"irreducible loss of the corpus {floor:.3f} (perplexity {math.exp(floor):.2f}) -> "
          f"our model is {100 * (val_loss / floor - 1):.1f}% above the floor")
''')
nb.md(r'''
## 6. Generation

Generation is a loop: run the model on the context, take the logits of the **last** position, turn them into a distribution, sample a token, append it, repeat.

- `temperature` < 1 sharpens the distribution (safer), > 1 flattens it (more creative / more mistakes)
- `top_k` keeps only the k most likely tokens; `top_p` keeps the smallest set whose probability reaches p

(For speed, real implementations keep a **KV-cache** so each step only processes the new token — see notebook 06.)

**Decoding strategies.** Given logits $z$, temperature $\tau$ defines $p_i \propto \exp(z_i / \tau)$: $\tau \to 0$ approaches greedy decoding (always the most likely token), $\tau = 1$ is the model's own distribution. Greedy and beam search maximize likelihood, which suits tasks with one right answer (translation, extraction), but for open-ended text they tend to produce bland, repetitive loops. Pure sampling has the opposite problem: the long tail of individually unlikely tokens adds up to a real chance of picking nonsense. Truncation fixes the tail: top-k (Fan et al., 2018) keeps a fixed number of candidates, and top-p or nucleus sampling (Holtzman et al., 2020, "The Curious Case of Neural Text Degeneration") keeps a variable number that adapts to how confident the model is. Note that none of this changes the model; it only changes how we read its distribution.
''')
code(r'''
@torch.no_grad()
def generate(prompt, max_new_tokens=60, temperature=0.8, top_k=None, top_p=None):
    model.eval()
    idx = torch.tensor([tok.encode(prompt)], dtype=torch.long, device=device)
    for _ in range(max_new_tokens):
        logits, _ = model(idx[:, -CFG["block_size"]:])                     # the model only sees the last block_size tokens
        logits = logits[:, -1, :] / max(temperature, 1e-6)
        if top_k is not None:
            kth = logits.topk(top_k).values[:, -1:]
            logits = logits.masked_fill(logits < kth, float("-inf"))
        probs = logits.softmax(-1)
        if top_p is not None:                                              # nucleus sampling: keep tokens until their cumulative probability reaches p
            sp, si = probs.sort(descending=True)
            keep = (sp.cumsum(-1) - sp) < top_p
            probs = torch.zeros_like(probs).scatter(-1, si, sp * keep)
            probs = probs / probs.sum(-1, keepdim=True)
        nxt = torch.multinomial(probs, 1) if temperature > 0 else logits.argmax(-1, keepdim=True)
        idx = torch.cat([idx, nxt], dim=1)
    return tok.decode(idx[0].tolist())

torch.manual_seed(1)
for name, kw in {"greedy": dict(temperature=0), "T=0.8, top-k 40": dict(temperature=0.8, top_k=40),
                 "T=1.0, top-p 0.9": dict(temperature=1.0, top_p=0.9)}.items():
    print(f"[{name}]\n  {generate('The old sailor', **kw)}\n")
''')
nb.md(r'''
Check the samples: sentences should be **grammatical** in the corpus's grammar (subject → verb → object → place → period → new capitalized sentence). That structure was never programmed into the model — it was learned from next-token prediction alone.
''')
code(r'''
# How often does the model produce a sentence that is valid under the corpus grammar?
valid = re.compile(r"^(?:%s) (?:%s) (?:%s) (?:%s)\.$" % tuple("|".join(map(re.escape, xs)) for xs in (SUBJECTS, VERBS, OBJECTS, PLACES)), re.I)
torch.manual_seed(0)
samples = " ".join(generate("The", max_new_tokens=60, temperature=0.8) for _ in range(10))
sents = [s.strip() + "." for s in samples.split(".")[1:-1] if s.strip()]          # drop the first sentence (it depends on the prompt) and the last (it gets cut off)
ok = sum(bool(valid.match(s)) for s in sents)
print(f"grammatical sentences: {ok}/{len(sents)} = {100 * ok / max(1, len(sents)):.0f}%")
''', script=False)
nb.md(r'''
## 7. Save and load

A checkpoint must contain everything needed to rebuild the model: the **config**, the **weights** and the **tokenizer** (a model is useless without the exact tokenizer it was trained with).
''')
code(r'''
import tempfile
ckpt_path = os.path.join(tempfile.gettempdir(), "mini_gpt.pt")
torch.save({"config": {**CFG, "vocab_size": V}, "state_dict": model.state_dict(),
            "tokenizer": {"merges": [[a, b, i] for (a, b), i in tok.merges.items()]}}, ckpt_path)

ckpt = torch.load(ckpt_path, map_location=device)
c = ckpt["config"]
model2 = MiniGPT(c["vocab_size"], c["block_size"], c["d_model"], c["n_layers"], c["n_heads"]).to(device)
model2.load_state_dict(ckpt["state_dict"])
tok2 = BPETokenizer()
for a, b, i in ckpt["tokenizer"]["merges"]:
    tok2.merges[(a, b)] = i; tok2.vocab[i] = tok2.vocab[a] + tok2.vocab[b]
xb, _ = get_batch("val")
model.eval(); model2.eval()
with torch.no_grad():
    same = torch.allclose(model(xb)[0], model2(xb)[0])
print(f"saved to {ckpt_path} ({os.path.getsize(ckpt_path) / 1e6:.1f} MB) | reloaded model identical: {same} | "
      f"tokenizer identical: {tok2.encode(text[:500]) == tok.encode(text[:500])}")
''')
nb.md(r'''
## Results

| question | measured | lesson |
|---|---|---|
| does the tokenizer work? | 199 merges → vocabulary 455, 5.01 characters per token, exact round-trip | a tiny BPE already compresses its own domain well |
| is the untrained model sane? | 1.86M parameters, initial loss 6.144 vs ln V = 6.120 | start near ln V or something is wrong with the init |
| how far does training get? | 800 steps in about 19 s on a laptop RTX 3060: validation loss 6.025 → 0.778 (perplexity 2.18) | most of the learning happens in the first 100 steps |
| how good is "good"? | irreducible loss of the corpus 0.774 (perplexity 2.17): the model is 0.6 % above it | a known floor turns "lower is better" into "how much is left" |
| did it learn the grammar? | 40 of 49 sampled sentences (82 %) valid under the corpus grammar | structure emerges from next-token prediction alone |
| greedy vs sampling | greedy repeats "in the garden" in every sentence; sampling varies all slots | greedy is deterministic and collapses to the most likely pattern |
| save and reload | 7.4 MB checkpoint; reloaded model and tokenizer identical | a checkpoint is config + weights + tokenizer |

## In production

**The same pipeline, scaled up.**
- Every step here (tokenizer, decoder-only transformer with RoPE, RMSNorm and SwiGLU, AdamW with warmup and cosine decay, sampling) is what open models such as Llama, Qwen, Mistral and Gemma use. What changes is scale: trillions of tokens, billions of parameters, thousands of GPUs.
- Few teams pre-train from scratch. Most start from an open-weights checkpoint and fine-tune it (notebooks 07–08), or call a hosted model through an API.

**Tools.**
- Tokenizers: Hugging Face `tokenizers`, `tiktoken`, SentencePiece.
- Training: PyTorch with `torchtitan`, Megatron-LM or DeepSpeed for multi-GPU runs; nanoGPT-style repositories for learning and small experiments.
- Checkpoints: `safetensors` files plus a config and tokenizer files, the layout the Hugging Face Hub expects.

**Failure modes.**
- Loading a checkpoint with a different tokenizer than the one it was trained with: no error, just bad output.
- Judging a model by training loss alone; validation must be held out by document, after deduplication.
- No floor or baseline to compare against, so nobody knows whether a loss of 0.78 is good.
- Greedy decoding in open-ended generation, which loops.
''')
nb.md(r'''
## Where to go next

| You want to understand… | Notebook |
|---|---|
| how BPE vocabulary size trades off sequence length | 01 · tokenization |
| why RoPE works, GQA/MLA and the KV-cache | 02 · attention |
| long-context attention variants | 03 · efficient attention |
| Mixture-of-Experts | 04 · FFN & MoE |
| LR schedules, scaling, the training dashboard | 05 · pre-training |
| fast generation: KV-cache, speculative decoding, batching | 06 · inference |
| turning this into an assistant: SFT, LoRA, DPO, GRPO | 07, 08 |
| making it smaller: INT8 / INT4 / NF4 | 09 · quantization |

**Exercises:** (1) set `text_path` to tiny Shakespeare and train for 3000 steps; (2) replace `SwiGLU` with a plain GELU MLP of the same size and compare the final loss; (3) remove RoPE and watch what happens to the grammar; (4) add a KV-cache to `generate` and measure the speed-up.
''')
nb.save(notebook_path("00_mini_llm_end_to_end.ipynb"))

header = '''#!/usr/bin/env python
"""
A complete mini LLM in one file. It trains a BPE tokenizer, builds a small GPT (RoPE, RMSNorm, SwiGLU),
trains and evaluates it, generates some text and saves a checkpoint. The only dependency is PyTorch.

This file is generated from notebooks/00_start_here/00_mini_llm_end_to_end.ipynb by tools/notebooks/nb00_mini_llm_end_to_end.py,
so edit the generator rather than this file. Run it with `python scripts/mini_llm.py`; tweak CFG below to change
the model size, the number of steps or the training data.
"""
'''
with open(os.path.join(ROOT, "scripts", "mini_llm.py"), "w", encoding="utf-8") as f:
    f.write(header + "\n\n".join(script_cells).replace("print(", "print(").rstrip() + "\n")
print("notebook 00 + scripts/mini_llm.py written")
print("00_mini_llm_end_to_end.ipynb written")
