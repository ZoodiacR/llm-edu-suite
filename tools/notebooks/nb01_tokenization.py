"""Notebook 01 - Tokenization

Builds 01_tokenization.ipynb. Run: python tools/build_notebooks.py 01
"""
from _common import *  # noqa: F401,F403

# Notebook 01 - Tokenization
nb = NB()
nb.md(r'''
# 01 · Tokenization: from text to integers

**What you will learn**
- Why LLMs operate on sub-word tokens rather than characters or words
- How byte-level BPE is trained, how it encodes/decodes, and why it never produces `<UNK>`
- How vocabulary size trades off sequence length against embedding size
- How to *evaluate* a tokenizer: compression ratio, fertility, round-trip correctness

**Source:** `llm_edu_suite/tokenizer.py` (pure Python, ~120 lines).

| Granularity | Vocab | Sequence length | Problem |
|---|---|---|---|
| bytes | 256 | very long | attention is O(T²); the model wastes capacity spelling |
| words | 100k–1M+ | short | unknown words → `<UNK>`; terrible for code and multilingual text |
| **sub-words (BPE)** | 32k–256k | medium | the standard: frequent words = 1 token, rare words = a few pieces |
''')
nb.code(SETUP)
nb.code(r'''
from llm_edu_suite.tokenizer import ByteBPETokenizer, compression_ratio
from llm_edu_suite.data import template_text_corpus

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
print("01_tokenization.ipynb written")
