# Theoretical foundations

🇪🇸 [Versión en español](../es/fundamentos.md) · [← Back to the index](README.md)

> The theory behind every technique and pattern in the course: the problem it solves, the core idea (with its equations), why it works, what it competes with, why it is what the industry uses today and what it replaced, and how it fails. Read it before each notebook or alongside it. Short definitions live in the [glossary](glossary.md); they are linked here, not repeated.

**How to read this page.** It follows the topics of the [topic map](README.md#topic-map), grouped by theory rather than by notebook number. Each section ends with the notebook that implements it and an **In the course** line saying what is measured there. The numbers quoted are the ones each notebook reports, measured with small models (SmolLM2, `qwen2.5:3b`) and small evaluation sets, so treat them as large-effect signals, not benchmarks. Where a point is still argued in the literature, the text says so.

## Contents

1. [Inside the model](#1-inside-the-model)
   - [1.0 The big picture](#10-the-big-picture-from-text-to-the-next-token) · [1.1 Tokenization](#11-tokenization) · [1.2 The transformer block](#12-the-transformer-block-and-the-residual-stream) · [1.3 Attention, RoPE and normalization](#13-attention-rope-and-normalization) · [1.4 Efficient attention](#14-efficient-attention-for-long-contexts) · [1.5 FFN and Mixture of Experts](#15-ffn-and-mixture-of-experts) · [1.6 Encoders and encoder-decoders](#16-encoders-and-encoder-decoders)
2. [Pre-training and inference](#2-pre-training-and-inference)
   - [2.0 Pre-training data](#20-pre-training-data) · [2.1 The objective](#21-the-objective-next-token-prediction) · [2.2 Optimizer and schedule](#22-optimizer-and-learning-rate-schedule) · [2.3 Scaling laws](#23-scaling-laws) · [2.3b Systems](#23b-systems-memory-precision-kernels-and-parallelism) · [2.4 Decoding](#24-decoding-strategies) · [2.5 KV-cache](#25-prefill-decode-and-the-kv-cache) · [2.6 Batching and serving](#26-batching-and-serving) · [2.7 Speculative decoding](#27-speculative-decoding) · [2.8 Quantization](#28-quantization) · [2.8b Serving under load](#28b-serving-under-load)
3. [Post-training](#3-post-training)
   - [3.1 SFT](#31-supervised-fine-tuning) · [3.2 LoRA](#32-lora-and-parameter-efficient-fine-tuning) · [3.3 Distillation](#33-distillation) · [3.3b Model merging](#33b-model-merging) · [3.4 RLHF, DPO and GRPO](#34-from-rlhf-to-dpo-and-grpo) · [3.5 Reasoning and test-time compute](#35-reasoning-and-test-time-compute)
4. [Applications](#4-applications)
   - [4.0 Embeddings and rerankers](#40-embeddings-and-rerankers) · [4.0b Clustering and topic modeling](#40b-clustering-and-topic-modeling) · [4.0c Multimodal models](#40c-multimodal-models-clip-and-vision-language-models) · [4.1 RAG](#41-retrieval-augmented-generation) · [4.2 Retrieval metrics](#42-retrieval-metrics) · [4.3 RAGAS and LLM-as-judge](#43-ragas-and-llm-as-judge) · [4.4 Application patterns](#44-application-patterns) · [4.4b Prompts as programs (DSPy)](#44b-prompts-as-programs-dspy-and-prompt-optimization) · [4.5 Recommendation](#45-recommender-systems) · [4.5b Drift and online learning](#45b-drift-online-learning-and-continual-updates)
5. [Agents and MCP](#5-agents-and-mcp)
   - [5.1 Tool calling](#51-tool-calling) · [5.2 Constrained decoding](#52-constrained-decoding-and-structured-output) · [5.3 ReAct, plan-and-execute and reflection](#53-react-plan-and-execute-and-reflection) · [5.4 State graphs](#54-langgraph-agents-as-state-machines) · [5.5 Memory](#55-memory) · [5.6 Limits and human-in-the-loop](#56-limits-and-human-in-the-loop) · [5.7 Multi-agent teams](#57-multi-agent-teams-supervisor-and-handoffs) · [5.8 Code execution and sandboxes](#58-code-execution-and-sandboxes) · [5.9 MCP](#59-model-context-protocol)
6. [LLMOps](#6-llmops)
   - [6.1 Traces and spans](#61-traces-spans-and-opentelemetry-genai) · [6.2 Offline and online evaluation](#62-offline-and-online-evaluation) · [6.3 Percentiles and SLOs](#63-percentiles-and-slos) · [6.4 PII and sampling](#64-pii-and-trace-sampling) · [6.5 Statistics of small eval sets](#65-statistics-of-small-evaluation-sets) · [6.5b Evaluation and tracing frameworks](#65b-evaluation-and-tracing-frameworks) · [6.6 Safety and guardrails](#66-safety-refusals-jailbreaks-and-guard-models)

---

## 1. Inside the model

Terms for this part: [glossary, Tokenization and architecture](glossary.md#tokenization-and-architecture).

A causal language model is a probability distribution over token sequences, factorized by the chain rule:

$$p(x_1, \dots, x_T) = \prod_{t=1}^{T} p(x_t \mid x_{<t})$$

The network never models whole sentences. Given a prefix, it returns a distribution over the next token. Everything in this part (tokenizer, attention, FFN) exists to compute that one factor well.

```
text ─► tokenizer ─► ids ─► embedding ─► L blocks ─► norm ─► lm_head ─► logits (T × V) ─► softmax
```

> **Entry notebooks:** [00 · Transformers, the big picture](../../notebooks/00_start_here/00_transformers_big_picture.ipynb) first ([section 1.0](#10-the-big-picture-from-text-to-the-next-token)), then [01 · A complete mini LLM](../../notebooks/00_start_here/01_mini_llm_end_to_end.ipynb) · **In the course (01):** the whole model in one notebook; the initial loss must be close to $\ln V$, and with the synthetic corpus the irreducible loss is known exactly ($4 \ln 8 = 8.32$ nats per sentence).

### 1.0 The big picture: from text to the next token

**Problem.** Every model in this course, from SmolLM2-135M to the 7B models served by Ollama and the frontier models behind APIs, does one thing: it reads a sequence of token ids and returns a probability for every possible next token. Then one token is picked, appended, and the loop repeats. The sections below take each piece apart (tokenizer, attention, FFN, decoding), but studied one at a time they hide how the pieces fit together, and several facts that matter in practice (what a token costs, why the answer appears only in the last layers, where the parameters and the memory go) only show up when you follow one sentence through a real model.

**The pipeline, with real sizes.** For SmolLM2-135M (30 blocks, hidden size $C = 576$, 9 query heads sharing 3 key/value heads, vocabulary $V = 49{,}152$):

```
"The capital of France is" ─► 5 ids ─► 5 vectors of 576 ─► 30 × (attention + FFN) ─► last vector ─► 49,152 logits ─► softmax ─► " the", " Paris", ...
```

Three ideas carry everything:

- **Tokens are ids.** The model never sees letters, only indices into the fixed vocabulary of its tokenizer ([section 1.1](#11-tokenization)). The leading space is part of the token (`' capital'`), and how many tokens a text needs depends on its language, its digits and how rare its words are. Cost, latency and context length are all counted in tokens, so the same content can be cheaper in one language than in another.
- **Meaning is a direction in space.** The first layer is a table $E \in \mathbb{R}^{V \times C}$, one row per id, and looking up a row *is* the embedding. Training to predict the next token pushes tokens used in similar contexts toward similar rows, so nearest neighbours by cosine similarity make sense (Monday → Tuesday) without anyone defining a word. A similarity means something only against a baseline: learned token embeddings are **anisotropic**, packed into a narrow cone around a shared direction (Gao et al., 2019), so even two random tokens are far from orthogonal.
- **Attention is a learned lookup.** In every block, each token builds a query, compares it with the keys of the tokens before it and takes a weighted average of their values ([section 1.3](#13-attention-rope-and-normalization)). The causal mask lets a token look only backwards, which is what lets the model generate left to right. Every head has its own weights, so **heads specialize**: one head may resolve what a pronoun refers to while most heads do other jobs, and an average over heads hides it.

**The residual stream and the logit lens.** Each block *adds* its output to a running vector per token, $x_\ell = x_{\ell-1} + \mathrm{Attn}_\ell(\cdot) + \mathrm{FFN}_\ell(\cdot)$ ([section 1.2](#12-the-transformer-block-and-the-residual-stream)), and only the last one is turned into logits, $z = W_U\,\mathrm{Norm}(x_L)$. Because every layer writes into the same space, the final norm and head can be applied to an intermediate vector to read what the model "would say" at that depth:

$$z_\ell = W_U\,\mathrm{Norm}(x_\ell), \qquad \ell = 1, \dots, L$$

That is the **logit lens** (nostalgebraist, 2020). Following the rank of the right token layer by layer shows the answer being assembled through depth and committed in the last few layers. It is a rough reading: the head was trained only on the last layer, and intermediate layers live in a somewhat different basis, so in a small model the middle layers look wrong when they are only "in another language". The **tuned lens** (Belrose et al., 2023) trains one small affine map per layer for a fairer reading. A lens or an attention map *suggests* where a behaviour appears; ablating a head or patching an activation *confirms* it, which is how facts were located and edited inside GPT models (ROME; Meng et al., 2022).

**Attention sinks.** A softmax must put its weight somewhere. When a head has nothing useful to attend to, it parks the weight on the first token, whatever that token is, and the first token ends up receiving a large share of all attention: an **attention sink**. Xiao et al. (2024) described it in several open model families and turned it into StreamingLLM: a sliding-window KV-cache that keeps the first few tokens stays stable over long streams, while one that evicts them falls apart ([section 1.4](#14-efficient-attention-for-long-contexts)). Some recent models add explicit sink tokens or gated attention, so the parking spot is designed instead of accidental.

**The output is a distribution, then a choice.** The head returns $V$ logits for the last position, and softmax with temperature $T$ turns them into probabilities. The entropy $H = -\sum_i p_i \ln p_i$ says how spread they are, from 0 (one certain token) to $\ln V$ (10.8 nats for a uniform distribution over this vocabulary). Temperature divides the logits: below 1 it sharpens the distribution, above 1 it flattens it, and it never changes the ranking ([section 2.4](#24-decoding-strategies)). A base model's most likely token is the most likely *continuation*, not the answer to a question: after "The capital of France is", ' the' ("...is the capital of...") can beat ' Paris'. Knowing a fact and saying it next are different things. Greedy decoding of a base model drifts into bland loops (Holtzman et al., 2020), which is why real systems sample with truncation, penalize repetition and instruction-tune the model so the likeliest continuation is a direct answer (part 3). Generation is the loop *predict, pick, append*, and the **KV-cache** keeps the keys and values of earlier tokens so each step computes only the newest one ([section 2.5](#25-prefill-decode-and-the-kv-cache)).

**Where the parameters live.** With hidden size $C$, FFN size $F$, $L$ layers, $H$ query heads of size $D_h = C/H$, $H_{kv}$ key/value heads and vocabulary $V$, the main counts are

$$\underbrace{L\,\big(2C^2 + 2\,C\,D_h H_{kv}\big)}_{\text{attention: } Q, O \text{ and } K, V} \;+\; \underbrace{3\,L\,C\,F}_{\text{SwiGLU FFN}} \;+\; \underbrace{V C\ (\times 2 \text{ if the head is not tied})}_{\text{embeddings and head}}$$

Norms add almost nothing. The feed-forward layers are most of the model, and much of its factual knowledge is stored there (Geva et al., 2021), which is why Mixture of Experts splits exactly those layers ([section 1.5](#15-ffn-and-mixture-of-experts)) and LoRA often targets them alongside attention. Grouped-query attention (Ainslie et al., 2023) shrinks the attention term by sharing keys and values among query heads. The vocabulary term depends only on $V$ and $C$, so it weighs a lot in small models and little in large ones. At inference, memory is these parameters times the bytes per parameter **plus the KV-cache**, which grows with context length × layers × key/value heads.

**Production uses.** Count tokens with the deployed model's tokenizer on your real traffic, not words: non-English text, code and numbers can cost 1.5–3× more, and tokenizers differ a lot in how evenly they treat languages (Petrov et al., 2023), so pick one suited to your languages. Treat decoding settings as product settings (temperature 0 or low for extraction, classification and tool calls; moderate, with top-p or min-p, for writing; never greedy on a base model for open-ended text) and log them with every request. Size the hardware from parameters × bytes plus the KV-cache: a 7B model is ~15 GB in bf16 and ~4.5 GB at 4 bits ([section 2.8](#28-quantization)). Use the logit lens, attention maps and probes to locate where a behaviour appears, then confirm it with an intervention.

**Failure modes.** Assuming one word is one token; reading a single attention head, or the average of all of them, as "what the model thinks" (attention weights are not explanations by themselves; Jain and Wallace, 2019); treating the top-1 token as the model's knowledge; greedy decoding of a base model; a tokenizer that does not match the model exactly.

**Key papers.** Vaswani et al. (2017), *Attention Is All You Need*; Elhage et al. (2021), *A Mathematical Framework for Transformer Circuits* (the residual-stream view); nostalgebraist (2020), *interpreting GPT: the logit lens*; Belrose et al. (2023), *Eliciting Latent Predictions from Transformers with the Tuned Lens*; Xiao et al. (2024), *Efficient Streaming Language Models with Attention Sinks*; Geva et al. (2021), *Transformer Feed-Forward Layers Are Key-Value Memories*; Meng et al. (2022), *Locating and Editing Factual Associations in GPT* (ROME); Gao et al. (2019), *Representation Degeneration Problem in Training Natural Language Generation Models*; Jain and Wallace (2019), *Attention is not Explanation*; Petrov et al. (2023), *Language Model Tokenizers Introduce Unfairness Between Languages*; Ainslie et al. (2023), *GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints*; Allal et al. (2025), *SmolLM2*.

> **Notebook:** [00 · Transformers, the big picture](../../notebooks/00_start_here/00_transformers_big_picture.ipynb) (read first, before 01) · **In the course:** on SmolLM2-135M, "The capital of France is" is 5 tokens, Spanish needs 1.73× the tokens of English for the same sentences and `123456789` is 9 tokens; in the embedding table Monday → Tuesday has a cosine of 0.92 and seven → eight 0.89, against 0.445 on average for random pairs; averaged over heads, "it" puts between 0.00 and 0.09 of its attention on "animal" in every layer, while layer 13, head 8 puts 0.73 on "animal" and 0.02 on "street", and 61 % of all attention lands on the first token; with the logit lens the right token sits at rank 546 to 20,984 of 49,152 at layers 10 and 20, enters the top 10 at layers 25–30 and becomes top-1 only at layers 28–30, and Paris ends at rank 2, behind ' the'; at $T = 1$ ' the' gets 26 % and ' Paris' 9 %, with an entropy of 0.93 / 4.41 / 7.80 nats at $T$ = 0.5 / 1 / 1.5, and greedy decoding goes in circles (*"The capital of France is the capital of the country. The capital of France"*); the feed-forward layers hold 59.2 % of the parameters, the embedding shared with the head 21.0 % and attention 19.7 %, against 75 % / 14 % / 11 % (feed-forward / embeddings / attention) for Qwen2.5-7B by the same formula from its published config.

### 1.1 Tokenization

**Problem.** A transformer does not read text. Its first layer is an embedding table $E \in \mathbb{R}^{V \times C}$ and its last layer produces $V$ logits. Before any training, you must fix an `encode` function from text to integers and its inverse. That choice sets the sequence length (and therefore the cost of attention, which is $O(n^2)$), how many parameters sit in the embedding and output head, and which languages or domains are expensive.

**Core idea.** Two forces pull in opposite directions. Short sequences make training and inference cheaper; a small, well-trained vocabulary gives well-estimated embeddings. Whole words fall into Zipf's long tail (many words seen a handful of times) and produce `<UNK>`; bytes make sequences 4 to 5 times longer. Subwords are the compromise. **BPE** (Sennrich et al., 2016) starts from base symbols and repeats, $M$ times, the merge of the most frequent adjacent pair:

$$(a^*, b^*) = \arg\max_{(a,b)} \; c(a,b), \qquad c(a,b) = \left|\lbrace i : s_i = a,\ s_{i+1} = b \rbrace\right|$$

Each merge shortens the corpus by exactly $c(a^*, b^*)$ symbols, so BPE is a greedy compressor: not optimal, but cheap and deterministic. Encoding new text replays the merges in the order they were learned.

**Why byte-level BPE is the default.** GPT-2 moved the base alphabet to the 256 UTF-8 byte values: any string can be encoded and decoding is exact, with no `<UNK>`. The merges then learn by themselves which byte sequences (accents, CJK, emoji) deserve a token. The price is small: a rare character costs 2 to 4 tokens, and streaming output must buffer incomplete UTF-8 sequences. tiktoken, Llama 3 and Qwen are byte-level BPE; SentencePiece (BPE or Unigram on raw text, with byte fallback) remains common in T5, Llama 1-2 and Gemma.

| Method | How the vocabulary is built | Segmentation | Used by |
|---|---|---|---|
| BPE | bottom-up, merge the most frequent pair | replay merges by rank | GPT, Llama, Mistral, Qwen |
| WordPiece | bottom-up, merge the pair that most increases likelihood | greedy longest match from the left | BERT |
| Unigram LM | top-down, prune pieces from a large candidate set | most probable segmentation (Viterbi) | T5, ALBERT |

**Details that matter.** A *pre-tokenizer* (a regex) separates letters, digits and whitespace before BPE: it speeds training up (each unique word is counted once) and prevents accidental tokens like `"dog."`. Special tokens (BOS, EOS, PAD, the role markers of a chat template) get ids no merge can produce, and a production tokenizer must not turn a user-typed `<|eos|>` string into the control token. A larger vocabulary compresses more but costs $V \cdot C$ parameters per matrix; the trend toward larger vocabularies (Llama 3 at 128k, Gemma at 256k) is driven by multilingual and code coverage.

**Failure modes.** Language unfairness (a tokenizer trained mostly on English charges more tokens for the same content in other languages; Petrov et al., 2023); character-level blindness (counting the letters in `"strawberry"`); numbers split inconsistently; under-trained tokens that trigger odd behaviour (`SolidGoldMagikarp`); a prompt ending in a space. How much better compression means better downstream quality is debated: the only definitive test is training models with each tokenizer. Tokenizer-free models (ByT5, MEGABYTE, Byte Latent Transformer) are active research and have not displaced BPE.

**Key papers.** Sennrich, Haddow and Birch (2016); Radford et al. (2019, GPT-2); Kudo (2018); Kudo and Richardson (2018, SentencePiece); Rust et al. (2021); Petrov et al. (2023).

> **Notebook:** [02 · Tokenization](../../notebooks/01_architecture/02_tokenization.ipynb) · **In the course:** BPE in pure Python with round-trip, compression and fertility checks; in-domain compression rises fast with vocabulary size and saturates, while out-of-domain text barely benefits.

### 1.2 The transformer block and the residual stream

**Problem.** Before the Transformer, language models were n-grams and then RNNs/LSTMs, which read one token at a time: training is sequential and gradients connect distant tokens poorly.

**Core idea.** The Transformer (Vaswani et al., 2017) replaced recurrence with attention, and the *decoder-only* variant with a causal mask computes all $T$ predictions of a sequence in one pass (teacher forcing). A modern model stacks $L$ identical pre-norm blocks:

$$h = x + \mathrm{Attn}\big(\mathrm{Norm}(x)\big), \qquad y = h + \mathrm{FFN}\big(\mathrm{Norm}(h)\big)$$

**Attention** is the only place where positions exchange information (between tokens); the **FFN** transforms each position on its own (within a token). The **residual stream** $x$ is the running sum every sub-layer reads from and writes into.

**What the industry uses today, and what it replaced.** The recipe consolidated by Llama (Touvron et al., 2023) changed several GPT-2 components:

| Component | GPT-2 (2019) | Current recipe | Why it changed |
|---|---|---|---|
| Position | learned absolute embeddings | RoPE | relative positions, no length-limited table |
| Normalization | LayerNorm | pre-norm RMSNorm | cheaper, equally stable |
| FFN | GELU MLP | SwiGLU | lower perplexity at the same parameter count |
| Biases | in linear layers | mostly removed | simpler, no measurable benefit at scale |
| Output head | — | often tied to the embedding | saves $V \cdot C$ parameters in small models |

Decoder-only won over encoder-decoder and encoder-only designs for general models because one objective (continue the text) covers every task that can be phrased as text, and it improves predictably with scale.

**Hardware.** During training and *prefill*, $T$ positions are processed at once: large matrix multiplications, limited by compute. During *decode*, one token is processed per step: every weight and the whole KV-cache must be read to produce one token, limited by memory bandwidth. Much of the architectural evolution (GQA, MLA) targets this second regime.

> **Notebook:** [01 · A complete mini LLM](../../notebooks/00_start_here/01_mini_llm_end_to_end.ipynb) and [03 · Attention, RoPE and normalization](../../notebooks/01_architecture/03_attention_rope_norm.ipynb) · **In the course:** the whole block is written by hand, and the model learns the corpus grammar from next-token prediction alone.

### 1.3 Attention, RoPE and normalization

#### Scaled, multi-head attention

**Core idea.** Each token projects its state into a query $q = W_Q x$, a key $k = W_K x$ and a value $v = W_V x$:

$$\mathrm{Attn}(Q,K,V) = \mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{D_h}} + M\right)V$$

It is a soft dictionary lookup: instead of returning the value of the one matching key, it returns a blend of all values weighted by how well each key matches, and the softmax makes the blend differentiable. The causal mask $M$ puts $-\infty$ above the diagonal so the future gets exactly zero weight.

**The intuition behind $\sqrt{D_h}$.** If the components of $q$ and $k$ have mean 0 and variance 1, then $q^\top k$ has variance $D_h$. With $D_h = 128$ the logits would have a standard deviation around 11, the softmax would saturate on a single key and the gradient toward the others would vanish. Dividing by $\sqrt{D_h}$ restores unit variance at initialization.

**Multiple heads.** One softmax gives one weighting per query; $H$ heads of size $D_h = C/H$ let a layer track several relationships at once (previous token, subject, matching bracket). The cost is $O(T^2 C)$ per layer. **FlashAttention** (Dao et al., 2022-2024) computes *exactly* the same result without materializing the $T \times T$ matrix, using tiles and an online softmax: memory drops to $O(T)$ and speed rises because slow memory traffic is avoided, but FLOPs stay quadratic.

#### Normalization: from post-norm LayerNorm to pre-norm RMSNorm

Without normalization the residual stream grows layer by layer and the scale each sub-layer sees drifts during training. **RMSNorm** (Zhang and Sennrich, 2019) only rescales:

$$\mathrm{RMSNorm}(x) = \gamma \odot \frac{x}{\sqrt{\tfrac{1}{C}\sum_i x_i^2 + \epsilon}}$$

Its authors' hypothesis (rescaling matters, re-centering does not) has held in practice at scale; it saves a reduction in an operation that is memory-bound. *Where* the norm sits matters more than *which* norm. With **post-norm** (original Transformer, BERT), gradients near the output are large at initialization, which is why careful warmup was needed (Xiong et al., 2020). **Pre-norm** leaves an un-normalized identity path from input to output and trains stably with less tuning. Variants are still being studied: norms before and after each sub-layer (Gemma 2) and **QK-norm**, which normalizes $q$ and $k$ so attention logits do not grow (OLMo 2, Gemma 3). One criticism of pre-norm is that later layers contribute relatively less; which variant is best remains open.

#### RoPE: relative position by rotation

**Problem.** Without a mask, attention is permutation-equivariant: "dog bites man" and "man bites dog" would look the same.

**Core idea.** RoPE (Su et al., 2021) splits each head into $D_h/2$ pairs and rotates the query at position $m$ by $m\theta_i$ and the key at position $n$ by $n\theta_i$, with $\theta_i = \text{base}^{-2i/D_h}$. Rotations compose by adding angles:

$$\big(R(m\theta)q\big)^\top \big(R(n\theta)k\big) = q^\top R\big((n-m)\theta\big)\,k$$

The score depends only on the distance. Rotations preserve norms; the first pairs rotate fast (local resolution) and the last ones barely move (almost position-free channels).

| Approach | Relative | Extra parameters | Drawback |
|---|---|---|---|
| Sinusoidal absolute (2017) | indirectly | none | mixes position into content; weak extrapolation |
| Learned absolute (GPT-2, BERT) | no | $T_{\max} \times C$ | cannot go past $T_{\max}$ |
| Relative bias (T5) | yes | few | touches the $T \times T$ score matrix, complicating fused kernels |
| ALiBi | yes | none | strong built-in recency bias |
| RoPE | yes | none | needs a scaling rule to extend context |

RoPE became the default because it is relative, adds no parameters and acts on $q$ and $k$ *before* attention, so FlashAttention runs unchanged and a key is rotated once and cached forever. **Context extension:** beyond the training length the slow pairs reach angles never seen and perplexity blows up. Position Interpolation, "NTK-aware" scaling, YaRN and Llama 3.1 scaling all edit the angle table: they stretch the low-frequency pairs and leave the high-frequency ones (which already cycled many times during training) alone. Whether position encoding is always needed in causal decoders is debated (NoPE, Kazemnejad et al., 2023).

#### The KV-cache and the MHA → GQA → MLA family

During generation, past keys and values never change, so they are stored. Per sequence:

$$\text{bytes} = 2 \times L \times H_{kv} \times D_h \times T \times b$$

Every decode step reads the whole cache, so its size sets both the maximum batch (memory) and the per-token latency at long context (bandwidth).

| Variant | Cache per token per layer | Quality (as reported) | Complexity |
|---|---|---|---|
| MHA | $2 H D_h$ | baseline | simple |
| MQA (Shazeer, 2019) | $2 D_h$ | lower, less stable | simple |
| GQA (Ainslie et al., 2023) | $2 H_{kv} D_h$ | close to MHA | simple, works with every kernel |
| MLA (DeepSeek-V2, 2024) | $d_c + d^R$ (512 + 64) | on par with or better than MHA in DeepSeek's ablations | extra projections, weight absorption, decoupled RoPE |

GQA is the standard for dense open models because it cuts the cache 4 to 8 times with a minimal change, and an MHA checkpoint can be converted by mean-pooling heads and "uptraining". MLA caches one latent vector $c_t = W^{DKV} x_t$ per token and reconstructs keys and values from it; **absorption** folds $W^{UK}$ into the query so attention runs directly against the latents. RoPE would break absorption, hence **decoupled RoPE** (a small, head-shared part of the key that carries position). Whether MLA's quality edge over GQA holds outside DeepSeek's setup is not settled.

**Failure modes.** An off-by-one `start_pos` (quality degrades with no error), rotating cached keys twice, forgetting the causal mask during a multi-token prefill, caching MLA's up-projected keys instead of the latent. And *attention sinks* (Xiao et al., 2023): models park a lot of attention on the first tokens, so evicting them from the cache breaks the model.

**Key papers.** Vaswani et al. (2017); Ba et al. (2016); Zhang and Sennrich (2019); Xiong et al. (2020); Su et al. (2021); Shazeer (2019); Ainslie et al. (2023); DeepSeek-AI (2024, DeepSeek-V2); Dao et al. (2022).

**Two RoPE conventions.** The rotated pairs can be adjacent dimensions $(2i, 2i+1)$ (Meta's original Llama and this course) or dimensions half a head apart $(i, i + d_h/2)$ (*rotate_half*, from GPT-NeoX and Hugging Face). Since a dot product does not change when both vectors' dimensions are permuted the same way, $q^\top k = (Pq)^\top (Pk)$, permuting the rows of $W_q$ and $W_k$ inside each head converts one into the other. Forgetting it raises no error: the model runs, only worse. Notebook 10 measures it while loading SmolLM2 into our code.

> **Notebook:** [03 · Attention, RoPE and normalization](../../notebooks/01_architecture/03_attention_rope_norm.ipynb) · **In the course:** numerical checks that RoPE depends only on $n-m$, that hand-written attention matches the fused kernel, and that incremental decoding with a cache equals the full forward pass; with a Llama-2-70B shape, MHA would need 2.6 MB per token (10 GB per sequence at 4k context), and GQA with 8 KV heads divides that by 8.

### 1.4 Efficient attention for long contexts

**Problem.** Per token and layer, projections and FFN cost on the order of $24C^2$ FLOPs, and attention mixing about $4TC$. Attention dominates once $T$ is about an order of magnitude larger than $C$ (tens of thousands of tokens for $C = 4096$). Inference has a second wall: each generated token reads a KV-cache that grows with $T$. FlashAttention solves the $O(T^2)$ memory, not the $O(T^2)$ compute nor the cache size.

**Families of solutions.**

| Family | Idea | Examples | What you give up |
|---|---|---|---|
| Fixed sparse patterns | local window, strided positions, global tokens | Sparse Transformer, Longformer, BigBird, Mistral 7B | direct access outside the pattern |
| Content-based sparse | a cheap scorer picks which blocks or tokens to attend | Reformer, Routing Transformer, NSA, MoBA, DSA | some recall if the selector misses; complex kernels |
| Low-rank / kernel | approximate softmax with thinner matrices | Linformer, Performer, Linear Transformers | the sharpness of softmax |
| Recurrent / state-space | compress the past into a fixed-size state | RetNet, Mamba, Mamba-2, Gated DeltaNet | exact recall of arbitrary tokens |
| Hybrids | mostly cheap layers, a few dense ones | Jamba, Griffin, Gemma 2/3, Qwen3-Next | the dense layers still pay $O(T^2)$ |

**Sliding window.** Query $i$ may attend to $j$ if $i - w < j \le i$: cost $O(TwC)$, and the cache is a ring buffer of size $O(w)$. After $L$ layers the receptive field is $L(w-1)$, but distant information arrives *relayed* and blurred; the receptive field says what is reachable, not what is preserved.

**Linear attention.** If similarity factorizes, $\mathrm{sim}(q,k) = \phi(q)^\top\phi(k)$, associativity of matrix products avoids the $T \times T$ matrix and the model becomes an RNN with a fixed-size state:

$$o_t = \frac{\phi(q_t)^\top S_t}{\phi(q_t)^\top z_t}, \qquad S_t = S_{t-1} + \phi(k_t)\,v_t^\top$$

$S_t$ is a linear associative memory: its capacity is bounded (about $D_h$ associations before they interfere) and the additive version never forgets. Modern recurrent layers fix exactly that: *gating* adds forgetting ($S_t = \alpha_t S_{t-1} + k_t v_t^\top$) and the *delta rule* overwrites the value stored under a key ($S_t = S_{t-1} + \beta_t k_t (v_t - S_{t-1}^\top k_t)^\top$), the basis of Gated DeltaNet in Qwen3-Next.

**Why dense attention is hard to replace.** It can retrieve any past token exactly, in one step, by content. Copying a name from the prompt, following a variable through code, answering from a retrieved document all rest on it. The quality gap between attention and its alternatives is largely explained by *associative recall* (Arora et al., 2023), and fixed-state models cannot copy long strings the way attention can (Jelassi et al., 2024). The mechanism that implements copying is the **induction head** (Olsson et al., 2022): a previous-token head in an early layer plus an induction head in a later layer that asks "who was preceded by A?" and copies what followed.

**What survived in production.** Between 2019 and 2021 dozens of "efficient transformers" were proposed; few reached large decoders, because FlashAttention made dense attention fast enough and many approximations lost exactly on in-context retrieval. What is used today is local windows, learned content-based sparsity and gated linear recurrences, almost always **mixed with some full-attention layers**. The hybrid ratio is the main design knob, and the best ratio is an open empirical question.

**Failure modes.** Recall beyond the window that degrades silently (hence needle-in-a-haystack and RULER); evicting the first tokens from the cache without keeping the *attention sinks*; a sparse selector that misses the relevant block.

> **Notebook:** [04 · Efficient attention](../../notebooks/01_architecture/04_efficient_attention.ipynb) · **In the course:** a copy task across ~42 tokens shows the induction-head phase transition (at 300 steps the dense model copies essentially nothing, by 1500 it is near-perfect), that SWA with $w=16$ and linear attention cannot copy, and that a hybrid with one global layer out of four recovers part of the ability.

### 1.5 FFN and Mixture of Experts

**The FFN.** With the original $d_{ff} = 4C$, attention has about $4C^2$ parameters per layer and the FFN $8C^2$: roughly two thirds of the non-embedding parameters sit in the FFN. A useful reading is the **key-value memory** view (Geva et al., 2021):

$$\mathrm{FFN}(x) = \sum_{i=1}^{d_{ff}} \sigma(k_i^\top x)\, v_i$$

Each row of $W_1$ is a pattern matched against the hidden state, and each column of $W_2$ a vector written into the residual stream. "Knowledge lives in the FFN" is a good working approximation (ROME locates and edits facts in mid-layer MLPs), not a law: other studies find facts spread across components.

**SwiGLU.** The current standard adds a gate:

$$\mathrm{FFN}(x) = W_{down}\big(\mathrm{SiLU}(W_{gate}x) \odot W_{up}x\big)$$

Each hidden unit is the product of two different linear features, a multiplicative interaction that a linear map plus a pointwise nonlinearity cannot express directly, and the linear branch never saturates. The evidence is empirical and consistent across labs (Shazeer, 2020, who credits the success to "divine benevolence"). With three matrices, $d_{ff} = \tfrac{8}{3}C$ matches the parameter budget; Llama 3 8B deliberately widens it ($d_{ff} = 14336 = 3.5C$).

**MoE: the problem it solves.** In a dense model, doubling the parameters doubles the FLOPs per token. An MoE layer replaces the FFN with $E$ experts and a router that picks $k \ll E$:

$$g = \mathrm{softmax}(W_r x), \qquad \mathcal{T} = \mathrm{TopK}(g, k), \qquad y = \sum_{e \in \mathcal{T}} \frac{g_e}{\sum_{j \in \mathcal{T}} g_j}\, \mathrm{FFN}_e(x)$$

Top-k is not differentiable, so the only gradient the router gets flows through the gate values of the selected experts. The economics involve three separate budgets:

| Quantity | Dense FFN | MoE with $E$ experts, top-$k$ |
|---|---|---|
| Parameters in memory | $P$ | about $E \cdot P_e$ |
| FLOPs per token | proportional to $P$ | proportional to $k \cdot P_e$ (active parameters) |
| Communication | nothing extra | all-to-all when experts live on different GPUs |

MoE buys **capacity** at fixed compute and pays in **memory** and **communication** (Mixtral 8x7B needs memory for ~47B parameters although each token uses ~13B). That is why it shines in large training runs and distributed serving, and much less on one consumer GPU.

**Load balancing, the central difficulty.** Routing is rich-get-richer: an expert chosen slightly more early on gets more gradient, improves and is chosen even more. With a fixed capacity per expert, overflow tokens are dropped. Two fixes:

- **Auxiliary loss** (Shazeer 2017, simplified in Switch): $\mathcal{L}_{aux} = \alpha E \sum_e f_e P_e$, with $f_e$ the fraction of tokens routed (no gradient) and $P_e$ the mean router probability (with gradient). It competes with the language-modeling loss: a large $\alpha$ hurts quality, a small one does not balance.
- **Auxiliary-loss-free** (Wang et al., 2024; DeepSeek-V3): a per-expert bias used only for *selection*, $\mathcal{T} = \mathrm{TopK}(s_e + b_e, k)$, updated without gradient as $b_e \leftarrow b_e - \gamma\,\mathrm{sign}(\text{load}_e - \overline{\text{load}})$. It works like a market price, or the integral term of a controller. $\gamma$ must be small (0.001 in DeepSeek-V3) or the loads oscillate; DeepSeek-V3 also keeps a very small balance loss as a safety net.

**Fine-grained and shared experts** (DeepSeekMoE): splitting each expert into $m$ smaller ones and routing to $mk$ multiplies the possible combinations; one or a few always-on experts absorb common knowledge. Experts do not specialize along human topics ("a biology expert"): in Mixtral, routing correlates more with syntax and token type.

**Failure modes.** Router collapse and dead experts; instability from growing router logits (z-loss, router in fp32); token dropping (MegaBlocks avoids fixed capacity); overfitting when fine-tuning on little data (ST-MoE); serving throughput that depends on the traffic mix.

**Key papers.** Shazeer et al. (2017); Lepikhin et al. (2020, GShard); Fedus et al. (2021, Switch); Zoph et al. (2022, ST-MoE); Jiang et al. (2024, Mixtral); Dai et al. (2024, DeepSeekMoE); Wang et al. (2024); DeepSeek-AI (2024, V3); Shazeer (2020); Geva et al. (2021).

> **Notebook:** [05 · FFN and MoE](../../notebooks/01_architecture/05_ffn_moe.ipynb) · **In the course:** a badly imbalanced router is equalized by the aux-free bias with no gradient, while a large `bias_lr` (0.1) oscillates; dense vs MoE at equal active compute shows a small or absent MoE advantage on a toy corpus, which is the expected outcome.

### 1.6 Encoders and encoder-decoders

**Problem.** Everything above describes a decoder-only model, the design of GPT, Llama and Qwen. But the original transformer (Vaswani et al., 2017) had two halves, and each became a family of its own. To understand a text that is already complete (classify it, extract entities, turn it into an embedding) nothing has to be generated, and reading only to the left wastes half the context. To turn one sequence into a different one (translate, transcribe audio) it pays to read the whole input once and write the output separately.

**The difference is the mask.** All three families stack the same block (attention, FFN, residuals, normalization). What changes is which positions each token may attend to:

| Family | Attention | Training objective | Well-known members | Used today for |
|---|---|---|---|---|
| encoder-only | bidirectional: every token sees every other | fill in hidden tokens | BERT, RoBERTa, DeBERTa, ModernBERT | classification, entities, embeddings, rerankers, safety filters |
| decoder-only | causal: a token sees only the past | predict the next token | GPT, Llama, Qwen, SmolLM2 | generation, chat, agents |
| encoder-decoder | bidirectional encoder + causal decoder with cross-attention | rebuild corrupted spans, or map input to output | T5, BART, Flan-T5, Whisper | translation, speech recognition, structured rewriting |

In code, an encoder is the same attention layer called without the causal mask (`is_causal=False`); in this repository, `ModelConfig(causal=False)`.

**Masked language modeling (BERT).** A bidirectional model cannot be trained to predict the next token: it would be looking at it. BERT (Devlin et al., 2019) hides a random subset $\mathcal{M}$ of 15 % of the positions and asks the model to recover them from the rest:

$$\mathcal{L}_{\text{MLM}} = -\sum_{i \in \mathcal{M}} \log p_\theta\big(x_i \mid x_{\setminus \mathcal{M}}\big)$$

Of the chosen positions, 80 % are replaced by `[MASK]`, 10 % by a random token and 10 % are left unchanged. The mix exists because `[MASK]` never appears when the model is used later: if it only ever saw `[MASK]` at the positions it is graded on, it would not learn useful representations of the real tokens. RoBERTa (Liu et al., 2019) showed that BERT was under-trained: more data, dynamic masking and no next-sentence task.

**The cost of MLM, and ELECTRA.** A causal LM learns from every position of every sequence; MLM only from 15 %. ELECTRA (Clark et al., 2020) fixes this with *replaced token detection*: a small generator corrupts some tokens and the encoder (the discriminator $D$) classifies **every** position as original or replaced,

$$\mathcal{L}_{\text{disc}} = -\sum_{t=1}^{T} \Big[\mathbb{1}(\tilde x_t = x_t)\log D(\tilde x, t) + \mathbb{1}(\tilde x_t \ne x_t)\log\big(1 - D(\tilde x, t)\big)\Big]$$

so every position gives signal and, at equal compute, the resulting encoder is better. DeBERTa-v3 uses this objective.

**Span corruption (T5).** T5 (Raffel et al., 2020) replaces contiguous spans of the input with sentinels (`<extra_id_0>`, `<extra_id_1>`, ...) and the decoder has to write the missing spans, each preceded by its sentinel:

$$\mathcal{L}_{\text{span}} = -\sum_{j=1}^{|y|} \log p_\theta\big(y_j \mid y_{<j}, \tilde x\big)$$

where $\tilde x$ is the corrupted input and $y$ the sequence of sentinels and spans. The target is short (only what is missing), so pre-training is cheap, and the text-to-text format lets every task be treated the same way.

**Classification head.** For a task with a fixed label set, the sentence is summarized in one vector: the state of the special first token `[CLS]`, or the mean of all token states (*mean pooling*, using the attention mask so padding is not counted). A linear layer on top is fine-tuned together with the whole encoder:

$$p(y \mid x) = \mathrm{softmax}\big(W\, h_{\text{[CLS]}}(x) + b\big)$$

This is the recipe behind `AutoModelForSequenceClassification`. One forward pass, no generation; the output is a distribution over the labels and never an answer outside them. If the encoder is frozen and only the linear layer is trained, the result is a **linear probe**: it measures how much task-relevant information the representations already hold.

**ModernBERT.** BERT-base (2018) stayed the default encoder for six years. ModernBERT (Warner et al., 2024) rebuilds it with what decoders learned in the meantime: RoPE instead of learned positions, GeGLU feed-forward layers, pre-normalization, global attention layers alternating with 128-token sliding-window layers, *unpadding* (no compute spent on padding), an 8,192-token context and pre-training on 2 trillion tokens that include code. The base version has 150M parameters.

**Cross-attention.** In an encoder-decoder, the encoder reads the input $X$ once, bidirectionally, and produces the states $H_X$. The decoder generates the output $Y$ causally and, in every layer, adds between self-attention and the FFN a cross-attention layer whose queries come from the output being written and whose keys and values come from the encoder:

$$\mathrm{CrossAttn}(Y, X) = \mathrm{softmax}\!\Big(\frac{(Y W_Q)(H_X W_K)^\top}{\sqrt{d_h}}\Big) H_X W_V$$

It has no causal mask: every output token may read the whole input. The keys and values of $H_X$ are computed once per input and reused at every decoding step. Flan-T5 (Chung et al., 2022) is T5 with instruction tuning.

**Why decoder-only won, and where the others survive.** A decoder-only model has a single objective that turns any text into training signal (every position is graded), scales predictably and learns new tasks from the prompt (*in-context learning*) without a task head. For a general assistant that has to generate, it won. But:

- when the task is a **fixed set of labels** and some labelled data exists, a small fine-tuned encoder is usually as accurate as an LLM, or more, at a fraction of the cost and latency;
- **embedding models and rerankers** (cross-encoders) are encoders, and so are many safety filters (Meta's Prompt Guard is a fine-tuned mDeBERTa);
- when input and output are different sequences, as in **translation and speech** (Whisper, NLLB), encoder-decoders remain competitive. Whether they are better at equal compute is still debated; Google's T5Gemma (2025) revisits the question.

**Failure modes.** An encoder does not generate: there is no natural way to write left to right with it. The representations of an encoder pre-trained only with MLM are not built for similarity: without contrastive training, their mean is a poor sentence embedding. With few labels per class, a fine-tuned encoder falls short. Fine-tuned models are over-confident on out-of-distribution text (calibrate with temperature scaling on validation), and labels drift when new kinds of requests are forced into old classes (add an "other" class or a confidence threshold). In attention maps, *attention sinks* soak up probability and can hide the real alignment. And near-duplicates landing in both train and test inflate accuracy.

**Key papers.** Vaswani et al. (2017); Devlin et al. (2019, BERT); Liu et al. (2019, RoBERTa); Clark et al. (2020, ELECTRA); Raffel et al. (2020, T5); He et al. (2021, DeBERTa); Chung et al. (2022, Flan); Warner et al. (2024, ModernBERT); Casanueva et al. (2020, Banking77).

> **Notebook:** [06 · Encoders and encoder-decoders](../../notebooks/01_architecture/06_encoders_and_encoder_decoders.ipynb) (read after 05) · **In the course:** at the same size and 800 steps, the decoder recovers the first token of a hidden word 2.9 % of the time and the encoder that sees both sides 100 % (the same encoder with the right side cut off falls to 2.5 %); on Banking77 (77 intents), fine-tuned ModernBERT-base reaches 93.0 % test accuracy (69.6 % with 8 examples per intent, 89.3 % with 32), `nomic-embed-text` + logistic regression 93.3 % and frozen ModernBERT 89.2 %, while `qwen2.5:3b` gets 48.9 % zero-shot and 61.9 % few-shot with constrained decoding, and 15.2 % with function calling (181 of 231 outputs invalid); latency per message is 18 ms (embeddings + regression), 40 ms (encoder) and ~250 ms (LLM); Flan-T5-small splits its 77M into a 19M encoder, a 25M decoder with cross-attention and a 16M shared embedding, and its last-layer cross-attention is dominated by two attention sinks.

---

## 2. Pre-training and inference

Terms for this part: [glossary, Pre-training data](glossary.md#pre-training-data), [Training and post-training](glossary.md#training-and-post-training) and [Inference and systems](glossary.md#inference-and-systems).

### 2.0 Pre-training data

**Problem.** A pre-training corpus starts as web crawls (Common Crawl: billions of pages per snapshot), and most of what they contain is not worth learning from: menus, templates, spam, listings, pages copied from other pages. The open models that caught up with closed ones in 2023–2024 did it largely through better data, not new architectures (FineWeb, DCLM, Llama 3's data pipeline). Data work is most of the work.

**The pipeline.** The crawl is reduced by one to two orders of magnitude:

```
crawl (WARC) -> text extraction -> language ID -> URL / safety filters -> quality filters
             -> deduplication (exact, near-duplicate) -> decontamination -> mixing -> tokenization
```

FineWeb (Penedo et al., 2024) documents every step with ablations: a filter was kept only if a small model trained on the filtered data scored better on benchmarks. That is the reference method: filters are chosen by their effect on a trained model, not by eye.

**Why duplicates hurt.** Crawls are highly redundant (mirrored sites, templates, syndicated articles, boilerplate); Lee et al. (2022) found single sequences repeated tens of thousands of times in C4. Duplicates **waste compute** (the model pays again for text it has already seen), **cause memorization** (a sequence seen many times is reproduced verbatim, a privacy and copyright risk; Carlini et al., 2021, 2023) and **leak evaluations** (a benchmark copied onto the web ends up in the training set and inflates scores). Exact deduplication (a hash of the normalized text) is linear and catches mirrors, but a single changed character defeats it; everything else needs near-duplicate detection.

**Jaccard and MinHash.** A document $A$ is represented by its set of **shingles** (for example, word 5-grams), and the overlap of two documents is their Jaccard similarity:

$$J(A, B) = \frac{|A \cap B|}{|A \cup B|}$$

Comparing all pairs is $O(N^2)$, impossible for billions of documents. **MinHash** (Broder, 1997) applies a random hash function $h$ to every shingle and keeps the minimum. For a random permutation,

$$\Pr\big[\min h(A) = \min h(B)\big] = J(A, B)$$

because the minimum over $A \cup B$ is equally likely to be any of its elements, and both minima coincide exactly when that element is in $A \cap B$. With $k$ independent hash functions, the share of matching minima in two **signatures** of length $k$ estimates $J$ with standard error

$$\mathrm{SE}(\hat J) = \sqrt{\frac{J(1-J)}{k}}$$

In practice, permutations are replaced by affine functions $h_i(x) = (a_i x + b_i) \bmod p$ with a large prime $p$ (a Mersenne prime makes the modulo cheap).

**LSH: finding the candidate pairs.** Signatures are small, but comparing all of them is still quadratic. **Locality-sensitive hashing** splits each signature into $b$ bands of $r$ rows and puts documents into buckets by each band; two documents become candidates if they match *exactly* on any band:

$$\Pr[\text{candidate}] = 1 - \big(1 - J^r\big)^b$$

This is an S-shaped curve with a soft threshold near

$$J^* \approx \left(\frac{1}{b}\right)^{1/r}$$

Pairs above it are almost always found and pairs below almost never. With $b = 16$ and $r = 8$ (128 hash functions), $J^* \approx 0.71$. More rows per band sharpen the curve and raise the threshold; more bands lower it at the price of more candidates. Candidates are verified (estimated Jaccard above the threshold) and accepted pairs are grouped with union-find, so one document per cluster survives. Choosing $b$ and $r$ is not a technical detail: the threshold defines what a "duplicate" is. FineWeb uses 5-grams and a threshold near 0.75, which removes templated copies and light rewrites but keeps documents that merely quote each other. `datatrove`, `text-dedup` and Spark implementations do exactly this at the scale of billions of documents.

**Heuristic filters.** They remove what is obviously not prose. The **Gopher** rules (Rae et al., 2021) bound the number of words and the mean word length, limit symbols, bullet lines and ellipses, require common English stop words and penalize repeated lines and n-grams. The **C4** rules (Raffel et al., 2020) keep lines that end in punctuation and drop pages with code-like braces or "lorem ipsum". They are cheap and interpretable, and recording which rule removes each document makes them auditable.

**Model-based filters.** They score each document with a small classifier trained to recognise good text. **FineWeb-Edu** trained a regressor on a BERT encoder with 450,000 documents rated by Llama-3-70B for educational value (0 to 5); keeping documents scored 3 or more gave large gains on knowledge benchmarks at the same token budget. **DCLM** (Li et al., 2024) did the same with a fastText classifier. The trade-off: every filter also removes some good text, and a classifier encodes the taste of whoever labelled its data (it favours textbook prose over conversation or code). Rules and classifiers measure different things: rules remove what is not text; a classifier judges what the text is worth.

**Decontamination.** If test questions are in the training data, benchmark scores measure memory, not ability. The standard check (GPT-3, Llama) indexes long n-grams (8 to 13 tokens) of every benchmark and flags training documents that share them. It is cheap and catches verbatim copies; paraphrased contamination needs embedding similarity or is simply not caught. And nobody has to copy the test set on purpose: the web repeats itself, and the same text exists in both places.

**Synthetic data and model collapse.** LLM-written text already makes up a large part of post-training data and a growing part of pre-training (the Phi models' "textbooks", Cosmopedia). It is controllable and cheap, but a model's samples are less diverse than human text, and training repeatedly on model outputs narrows the distribution: the tails go first and variety drops generation after generation (***model collapse***, Shumailov et al., 2024). Synthetic data needs the same deduplication and filtering as web data, plus explicit diversity measures (lexical, such as distinct bigrams, and embedding-based), and it is evaluated by training on it: a quality classifier that rewards what *looks* educational rewards precisely the text written to look that way.

**Failure modes.** A threshold that silently removes a whole domain (code, other languages or dialects the classifier sees as "low quality"). Over-aggressive dedup that removes legitimate repetition (licences that must be present, common phrases). Benchmark contamination discovered after publication. Synthetic data that inflates the very metrics used to judge it. A site change that floods the corpus with boilerplate without anyone noticing: that is why you monitor the share of each source and filter over time, duplicate rates per source, the classifier's score distribution, and a random sample read by a person at every release.

**Key papers.** Broder (1997), *On the resemblance and containment of documents*; Rae et al. (2021, Gopher); Raffel et al. (2020, T5 / C4); Carlini et al. (2021, 2023); Lee et al. (2022), *Deduplicating Training Data Makes Language Models Better*; Penedo et al. (2024), *The FineWeb Datasets*; Li et al. (2024, DataComp-LM / DCLM); Shumailov et al. (2024), *AI models collapse when trained on recursively generated data*.

> **Notebook:** [07 · Data engineering](../../notebooks/02_pretraining_and_inference/07_data_engineering.ipynb) (read before 08) · **In the course:** on 12,000 FineWeb documents and a simulated crawl of 13,500 with planted duplicates, exact hashing removes the 1,100 planted copies and nothing else (13,500 → 12,400); MinHash + LSH from scratch ($k = 128$, 16 × 8, threshold 0.7) accepts 1,633 pairs (→ 10,767) with 100 % recall on copies in another template and with 1 % of words replaced, 72.2 % at 3 % and 0 % at 10 %, on the theoretical curve and with no false positives; the Gopher and C4 rules remove 12.0 % (1,173 documents by C4's terminal-punctuation rule); the FineWeb-Edu classifier gives FineWeb after the rules a mean of 1.20 and 2.0 % scored ≥ 3, against 3.04 and 47.6 % for FineWeb-Edu; 13-gram decontamination finds 50 of 50 planted leaks plus 29 natural overlaps (syndicated news); two ~16M `TransformerLM`s trained on the same 12.3M tokens tie on held-out text (5.547 raw vs 5.537 clean, noise), but on the 43 documents repeated 10 times the raw model drops to 4.350 against 5.281 (memorization); the 60 synthetic `qwen2.5:3b` passages score 4.15 on the classifier against 3.14 for human text, with 0.60 distinct bigrams against 0.72.

### 2.1 The objective: next-token prediction

**Core idea.** Pre-training minimizes the average cross-entropy of the true next token:

$$\mathcal{L}(\theta) = -\frac{1}{T}\sum_{t=1}^{T} \log p_\theta(x_t \mid x_{<t}), \qquad \mathrm{PPL} = e^{\mathcal{L}}$$

It is self-supervised: the targets are the input shifted by one, so any text is training data. With the causal mask, a sequence of length $T$ gives $T$ examples in one pass. Because loss depends on the tokenizer, comparisons across vocabularies use *bits per byte*.

**Why the loss has a floor.** The expected cross-entropy decomposes as

$$H(p, q_\theta) = H(p) + D_{\mathrm{KL}}(p \,\Vert\, q_\theta) \;\ge\; H(p)$$

Training can only reduce the KL term; the entropy of the text is irreducible. With real text it is unknown; with a synthetic corpus it can be computed.

**What it replaced.** BERT (masked LM: only ~15 % of positions get a loss, no natural way to generate), T5 (span corruption, two stacks) and UL2 (a mixture of denoisers) lost to the causal LM as the default: every token is a target, training equals inference, and it scales with no special tricks. GPT-3 showed **in-context learning**, which settled it. Code models add *fill-in-the-middle*.

**Data decides most of the quality.** The usual pipeline (text extraction, language ID, heuristic filters, a quality classifier, exact and MinHash deduplication, PII filtering, benchmark decontamination, domain weights and a final *annealing* phase on high-quality data) is where open datasets improved most (RefinedWeb, FineWeb-Edu); [section 2.0](#20-pre-training-data) covers it in depth. Repeating data for a few epochs is nearly as good as fresh data (Muennighoff et al., 2023).

### 2.2 Optimizer and learning-rate schedule

**AdamW.** Almost every LLM is trained with

$$m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t, \qquad v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2$$

$$\theta_t = \theta_{t-1} - \eta_t\Big(\frac{\hat m_t}{\sqrt{\hat v_t} + \epsilon} + \lambda\,\theta_{t-1}\Big)$$

Dividing by $\sqrt{\hat v_t}$ gives each parameter its own step size, independent of the gradient scale (which differs by orders of magnitude between embeddings, attention and FFN). Decoupled weight decay is the "W": L2 inside Adam would be divided by $\sqrt{\hat v}$ and would regularize less exactly the parameters with large gradients. Decay applies to matrices only, never to norm gains or biases. $\beta_2 = 0.95$ (since GPT-3) averages over about 20 steps instead of 1000 and reacts faster to changes in gradient scale, which reduces loss spikes. Clipping the global norm at 1.0 guards against bad batches. Memory cost: about **16 bytes per parameter** in mixed precision (the ZeRO accounting), before activations. Alternatives (Adafactor, Lion, Sophia, Muon) exist; whether they beat a well-tuned AdamW at scale is debated.

**Schedule.** *Warmup* exists because early on the $v_t$ estimates rest on few steps, and a full-size update on a random network can push it into a bad region. The final decay lets parameters settle into the minimum instead of bouncing around it. Cosine:

$$\eta_t = \eta_{min} + \tfrac{1}{2}(\eta_{max} - \eta_{min})\Big(1 + \cos\big(\pi\,\tfrac{t - t_{w}}{T - t_{w}}\big)\Big)$$

forces you to fix $T$ in advance. **WSD** (warmup-stable-decay; MiniCPM, Hägele et al., 2024) keeps the LR constant and decays only at the end: you can continue from any stable checkpoint, or branch several short decays to evaluate different budgets. Its advantage is operational.

**Stability.** Loss spikes tied to large attention logits (QK-norm, z-loss, lower LR, and in the worst case restarting from a checkpoint and skipping batches, as PaLM did); divergence from too high a peak LR ($\mu$P helps transfer the LR across widths); data bugs (broken shuffle, mismatched tokenizer, validation text leaking into training).

### 2.3 Scaling laws

**Core idea.** Training $N$ parameters on $D$ tokens costs $C \approx 6ND$ FLOPs (2 per parameter per token forward, 4 backward). Kaplan et al. (2020) found loss falls as a power law in $N$, $D$ and $C$ over many orders of magnitude, which made it possible to plan large runs from small ones. Chinchilla (Hoffmann et al., 2022) fitted

$$L(N, D) = E + \frac{A}{N^{\alpha}} + \frac{B}{D^{\beta}}$$

and, minimizing at fixed $C$, concluded that parameters and data should grow in equal proportion, about **20 tokens per parameter** (their 70B model on 1.4T tokens beat the 280B Gopher at the same compute). Kaplan's earlier advice (grow the model faster than the data) was largely an artifact of not adjusting the schedule to the run length, and the Chinchilla fit has itself been re-examined (Besiroglu et al., 2024), so treat the exponents as approximate.

**Compute-optimal is not deployment-optimal.** A model served to millions of users is paid for mostly at inference, which scales with $N$. So labs **over-train** small models far beyond 20 tokens per parameter (Llama 3 8B, over 15T tokens, about 1,900 per parameter): the loss keeps improving, only more slowly per FLOP.

> **Notebook:** [08 · Pre-training](../../notebooks/02_pretraining_and_inference/08_pretraining.ipynb) · **In the course:** initial loss $\approx \ln V$ and a causal-mask leak test as sanity checks; the synthetic corpus has an exact floor ($4 \ln 10 = 9.21$ nats per sentence) that the models reach in ~100 steps, the bigger model gets there sooner, and cosine and WSD end up practically identical.

### 2.3b Systems: memory, precision, kernels and parallelism

**Problem.** Scaling laws say how much compute a run needs; they do not say whether it fits on the GPU, how fast it goes or how it is split across many. That is the systems side of LLM engineering, and almost every decision in it trades one resource (memory, compute, communication) for another.

**Compute accounting.** A forward pass costs about $2N$ FLOPs per token (one multiply and one add per weight) and the backward pass twice that, so training costs

$$C \approx 6\,N\,D \quad \text{FLOPs for } D \text{ tokens}$$

plus an attention term of about $12\,L\,d\,T$ FLOPs per token ($L$ layers of width $d$, context $T$), which only matters when $T$ is large compared with $d$ (Kaplan et al., 2020; Chowdhery et al., 2022). **Model FLOPs utilization** measures what share of the GPU's peak is useful work:

$$\mathrm{MFU} = \frac{\text{tokens/s} \times (6N + 12LdT)}{\text{peak FLOP/s}}$$

It counts only the model's FLOPs, not the ones recomputed by checkpointing (that would be *hardware FLOPs utilization*, HFU). Well-tuned large training runs reach 40–60 %.

**Memory accounting.** With Adam and mixed precision, each parameter costs (Rajbhandari et al., 2020):

$$\underbrace{2}_{\text{bf16 weights}} + \underbrace{2}_{\text{bf16 gradients}} + \underbrace{4 + 4 + 4}_{\text{fp32 master weights},\ m,\ v} = 16 \ \text{bytes} \quad\Rightarrow\quad 16\Psi \ \text{bytes for } \Psi \text{ parameters}$$

A 7B model needs about 112 GB before the first activation; in plain fp32 it is also 16 bytes (4 + 4 + 8). The **activations**, each layer's outputs kept for the backward pass, grow with batch × sequence × width × layers, plus a $T^2$ term if attention stores the score matrix (Korthikanti et al., 2022, give about $sbh(34 + 5as/h)$ bytes per layer in 16 bits, for sequence $s$, batch $b$, width $h$ and $a$ heads). For long sequences and large batches, they, not the parameters, decide what fits. An easy part to forget is the logits: batch × $T$ × $V$ numbers kept for the loss.

**Arithmetic intensity and the roofline.** Whether an operation is compute-bound or memory-bound depends on its **arithmetic intensity** $I$, FLOPs per byte moved. The *roofline* model bounds performance:

$$\text{attainable FLOP/s} = \min\big(\text{peak FLOP/s},\ I \times \text{memory bandwidth}\big)$$

An $n \times n$ matmul does $O(n^3)$ work on $O(n^2)$ data and runs near peak; element-wise operations, normalizations and softmax do $O(1)$ work per byte and are memory-bound. It is the same calculation that explains decoding in [section 2.5](#25-prefill-decode-and-the-kv-cache).

**Tricks that trade one resource for another.**

- **Mixed precision** (Micikevicius et al., 2018): compute in bf16 or fp16 on the tensor cores (several times the fp32 throughput) and keep fp32 master weights, so that small updates are not lost to rounding. bf16 has fp32's 8 exponent bits, so it needs no *loss scaling*; fp16, with 5 exponent bits, does, or small gradients underflow to zero. fp8 (Hopper onwards) adds per-tensor or per-block scales.
- **Activation checkpointing** (Chen et al., 2016): keep only each block's input and recompute its internals during the backward pass. It costs one extra forward pass (~30 % more compute) and leaves one tensor per block in memory plus the internals of a single block at a time; with a checkpoint every $\sqrt{L}$ layers, activation memory drops to $O(\sqrt{L})$. Selective recomputation (Korthikanti et al., 2022) recomputes only the large, cheap parts, such as attention.
- **FlashAttention** (Dao et al., 2022; Dao, 2023): standard attention writes the $T \times T$ matrix $S = QK^\top$ to HBM, reads it back for the softmax and reads it again to multiply by $V$; its cost is dominated by those bytes, not by the FLOPs. FlashAttention is **IO-aware**: it splits $Q$, $K$ and $V$ into tiles that fit in the chip's SRAM and keeps, for each row, a running maximum $m$ and sum $\ell$ that are corrected with every new tile $j$ (*online softmax*):

  $$m' = \max(m, \tilde m_j), \qquad \ell' = e^{m - m'}\,\ell + e^{\tilde m_j - m'}\,\tilde\ell_j, \qquad O' = \frac{e^{m - m'}\,\ell\,O + e^{\tilde m_j - m'}\,\tilde P_j V_j}{\ell'}$$

  The result is exact, memory is linear in $T$, and the backward pass recomputes $S$ tile by tile instead of storing it. FlashAttention-2 splits the work better across warps and FlashAttention-3 exploits Hopper's asynchrony; the *memory-efficient* kernel (Rabe and Staats, 2021; xformers) and cuDNN's fused attention apply the same idea. None of them reduces the $O(T^2)$ compute: that is [section 1.4](#14-efficient-attention-for-long-contexts).
- **Kernel fusion and CUDA graphs.** Every PyTorch operation launches a kernel with a fixed CPU cost of a few microseconds, and every element-wise operation reads and writes its whole tensor in HBM. `torch.compile` (TorchDynamo captures the graph; its default backend, `inductor`, generates Triton kernels) fuses chains of operations into one kernel that reads once and writes once. A **CUDA graph** records a sequence of kernels with their memory addresses and replays it with a single launch: it saves neither bytes nor FLOPs, it saves launches, and it needs static shapes. Inference servers use it for the decode step.

**Parallelism.**

| Strategy | What is split | Communication per step | Used for |
|---|---|---|---|
| data (DDP) | the batch; every GPU holds the whole model | all-reduce of the gradients | models that fit on one GPU |
| ZeRO-1 / 2 / 3, FSDP | optimizer states (1), + gradients (2), + weights (3) | reduce-scatter + all-gather | large models without changing their code |
| tensor (Megatron) | each matmul across GPUs | all-reduce inside every layer | very wide layers, within one node (fast links) |
| pipeline | the layers, in stages | activations between stages | very deep models, across nodes |
| sequence / context | the sequence | attention blocks | very long contexts |
| expert | the experts of a MoE layer | all-to-all token routing | MoE models ([section 1.5](#15-ffn-and-mixture-of-experts)) |

Data parallelism rests on an exact fact: the average of the gradients of $k$ shards of a batch is the gradient of the whole batch, so $k$ GPUs with batch $b$ train like one GPU with batch $kb$. The **ring all-reduce** runs a reduce-scatter and an all-gather of $k-1$ steps each, and in each step every GPU sends a chunk of $P/k$ bytes:

$$\text{bytes sent per GPU} = 2\,\frac{k-1}{k}\,P$$

almost independent of $k$; that is why DDP scales as long as the network keeps up, and it overlaps communication with the backward pass by grouping gradients into *buckets*. **ZeRO** (Rajbhandari et al., 2020) removes DDP's redundancy by sharding the states across the $N_d$ data-parallel GPUs:

$$\text{memory per GPU:}\quad \text{DDP } 16\Psi, \qquad \text{ZeRO-1 } 4\Psi + \frac{12\Psi}{N_d}, \qquad \text{ZeRO-2 } 2\Psi + \frac{14\Psi}{N_d}, \qquad \text{ZeRO-3 } \frac{16\Psi}{N_d}$$

ZeRO-1 and ZeRO-2 communicate as much as DDP (a reduce-scatter plus an all-gather); ZeRO-3 adds an all-gather of the weights before each layer's forward and backward pass, about 1.5 times DDP's volume. FSDP (Zhao et al., 2023) is PyTorch's implementation (*full shard* is ZeRO-3; in FSDP2, `fully_shard`). The largest models combine strategies ("3D/4D parallelism", Narayanan et al., 2021): tensor inside the node, pipeline and FSDP across nodes. A pipeline pays an idle "bubble" of $(p-1)/(m+p-1)$ of the time with $p$ stages and $m$ micro-batches (GPipe; 1F1B reduces its memory).

**What it replaced.** Before ZeRO, a model that did not fit on one GPU needed hand-written model parallelism (Megatron) or pipelines; before FlashAttention, long context was limited by $O(T^2)$ memory, which is why approximate attention variants multiplied. Today the default stack is bf16, fused attention, FSDP or ZeRO-3 and, at large scale, tensor parallelism inside the node (`torchtitan`, Megatron-LM / NeMo, DeepSpeed).

**Failure modes.** Out of memory from activations, not parameters, which shows up when the sequence or the batch grows. Silent fallbacks: a build without FlashAttention, or Windows paging GPU memory into system RAM; both show up as a sudden slowdown, not as an error. fp16 without loss scaling: gradients that underflow to zero. Communication-bound scaling: adding GPUs does not speed up the job because the network cannot keep up with ZeRO-3's all-gathers. Nondeterminism between runs from GPU kernels, which matters when comparing small differences.

**Key papers.** Chen et al. (2016), *Training Deep Nets with Sublinear Memory Cost*; Micikevicius et al. (2018), *Mixed Precision Training*; Shoeybi et al. (2019), *Megatron-LM*; Rajbhandari et al. (2020), *ZeRO*; Kaplan et al. (2020), *Scaling Laws for Neural Language Models*; Rabe and Staats (2021), *Self-attention Does Not Need O(n²) Memory*; Narayanan et al. (2021), *Efficient Large-Scale Language Model Training on GPU Clusters*; Dao et al. (2022), *FlashAttention*, and Dao (2023), *FlashAttention-2*; Korthikanti et al. (2022), *Reducing Activation Recomputation in Large Transformer Models*; Chowdhery et al. (2022, PaLM, MFU); Zhao et al. (2023), *PyTorch FSDP*.

> **Notebook:** [09 · Systems: memory, precision, kernels and parallelism](../../notebooks/02_pretraining_and_inference/09_systems_memory_and_parallelism.ipynb) (read after 08) · **In the course:** on a 6 GB RTX 3060 Laptop, with a 29.4M-parameter `TransformerLM`, the fp32 weights take 0.111 GB (0.110 predicted), the gradients 0.118 and Adam 0.219, but the activations go from 0.43 to 3.39 GB from batch 2 to 16; bf16 makes a step 2.2× faster and lowers the peak from 2.18 to 1.66 GB; checkpointing cuts the peak by 57 % (fp32) and 51 % (bf16) for ~22 % more time; naive attention takes 4.1 GB at 8,192 tokens and runs out of memory at 16,384, while the fused kernels (memory-efficient and cuDNN; FlashAttention-2 is not in the Windows wheels) need 0.22 GB at 16,384 and cuDNN takes 51.9 ms; the unfused *math* backend spills into Windows shared GPU memory (8.1 GB on a 6 GB card, 978 ms); the profiler puts 47 % of GPU time in matmuls, with 1,136 kernels per step; without Triton there is no `inductor`, but `backend="cudagraphs"` gives 4.1× (8.68 → 2.11 ms) with the same output; MFU is 32 % (7.38 TFLOP/s against a measured peak of 23.0, 36,608 tokens/s); real DDP with two processes (`gloo`, `FileStore`, `mp.spawn`) ends with the same weights as one process with the global batch (7.45e-9) and all-reduces 0.53 MB per step; and the calculator gives a 7B model 112 GB per GPU under DDP against 14 GB with FSDP on 8 GPUs (1.8 GB with 64), and a 70B model 140 GB with 8 and 17.5 GB with 64.

### 2.4 Decoding strategies

**Problem.** The model produces logits; something has to turn them into text. Mode-seeking methods (greedy, beam search) suit tasks with one right answer (translation, extraction), but on open-ended text they produce bland loops: maximizing likelihood gives text *more* predictable than human writing, and repetition reinforces itself (Holtzman et al., 2020). Pure sampling has the opposite problem: the long tail of unlikely tokens adds up to enough mass to derail the text.

**Core idea.** Temperature plus truncation:

$$p_i(T) = \frac{e^{z_i / T}}{\sum_j e^{z_j / T}}, \qquad V_p = \text{smallest } S \text{ with } \sum_{i \in S} p_i \ge p$$

Temperature never changes the ranking of tokens, only how much mass the top ones get. Top-k cuts to a fixed size; top-p (nucleus) cuts to a set that adapts to the model's confidence, which is why it replaced top-k in most APIs; min-p uses a threshold relative to the top token and is popular in open-source runtimes (how much it improves on top-p is still discussed).

| Strategy | Adapts to confidence | Typical use |
|---|---|---|
| greedy / beam | no | extraction, translation, deterministic evals |
| temperature | no | always combined with a truncation |
| top-k | no | GPT-2-era default |
| top-p | yes | chat (often $T \approx 0.7$, $p \approx 0.9$) |
| min-p | yes | open-source runtimes, creative writing |

**Nuances.** Repetition penalties also punish correct repetition (a variable name). Engines apply temperature, top-k and top-p in different orders, which changes the distribution. And $T = 0$ is not perfectly deterministic on a server: a matmul's result can depend on the batch it ran in. Structured output (section 5.2) is a decoding strategy too.

### 2.5 Prefill, decode and the KV-cache

**Core idea.** Generation has two phases with different physics. **Prefill** processes the prompt at once: each weight read is used $P$ times, it is compute-bound, and it sets the *time to first token* (TTFT). **Decode** produces one token per pass: each weight is read for a single vector (or $B$ with batch $B$), it is bandwidth-bound, and it sets the *time per output token* (TPOT). The arithmetic intensity of a decode step is

$$\text{arithmetic intensity} \approx \frac{2NB \text{ FLOPs}}{2N \text{ bytes}} = B \ \text{FLOPs per byte}$$

A datacenter GPU does on the order of a hundred or more FLOPs per byte read (the roofline "ridge point"). At $B = 1$ the GPU mostly waits on memory. That one calculation explains most of what follows: the batch-1 speed ceiling is roughly bandwidth divided by model size in bytes; batching is almost free until the batch approaches the ridge point; quantization speeds up decode by reducing bytes, not FLOPs; and speculative decoding uses the idle compute.

The **KV-cache** turns $O(N^2)$ projection work into $O(N)$ and is an *exact* optimization: with greedy decoding, cached and uncached paths must give the same tokens (barring near-ties flipped by rounding). Its size, not the weights', usually decides how many sequences fit on a GPU; hence GQA, MLA, windows and fp8 caches.

### 2.6 Batching and serving

| Technique | Problem it solves | Reference |
|---|---|---|
| Static batching | amortizes weight reads | — |
| Continuous batching | in static batching, finished sequences leave idle slots until the longest ends; here the scheduler adds and removes sequences every step | Orca (Yu et al., 2022) |
| PagedAttention | contiguous per-sequence buffers waste memory; fixed-size blocks with a block table, like virtual-memory pages | vLLM (Kwon et al., 2023) |
| Prefix caching | many prompts share a system prompt or few-shot block; reuse their KV blocks | SGLang (Zheng et al., 2024) |
| Chunked prefill | one long prompt's prefill stalls everyone in decode | Sarathi-Serve (Agrawal et al., 2024) |
| Prefill/decode disaggregation | the two phases want different hardware | DistServe (Zhong et al., 2024) |
| CUDA graphs | kernel launch overhead dominates small steps | — |

**The latency-throughput trade-off.** A request takes roughly $\text{TTFT} + N_{out} \times \text{TPOT}$. Raising the batch increases aggregate throughput almost for free while decode is memory-bound; once a step becomes compute-bound, every extra sequence slows everyone down. A serving system picks the largest batch that meets the p99 latency target ("goodput"). Benchmarks that report only peak throughput, or only single-request latency, hide that choice. Section [2.8b](#28b-serving-under-load) measures these trade-offs on a real server under load.

### 2.7 Speculative decoding

**Problem.** In decode, verifying $k+1$ positions costs about the same as generating one.

**Core idea.** A cheap draft model proposes $k$ tokens and the target verifies them in one pass (Leviathan et al., 2023; Chen et al., 2023):

```
draft:   proposes  d1 d2 d3 d4            (k = 4 cheap steps)
target:  one pass scores 1..5             accept d1, accept d2, reject d3 -> resample
output:  d1 d2 r3                         (3 tokens for 1 target pass)
```

Each proposed token $x \sim q$ is accepted with probability $\min(1, p(x)/q(x))$; at the first rejection a replacement is drawn from the residual $p'(x) \propto \max(0, p(x) - q(x))$. The output has **exactly** the target distribution:

$$P(x) = q(x)\min\Big(1, \frac{p(x)}{q(x)}\Big) + (1-\beta)\,p'(x) = \min(p, q) + \max(0, p - q) = p(x)$$

with $\beta = \sum_y \min(p(y), q(y)) = 1 - \mathrm{TV}(p,q)$. A bad draft makes the method slow, never wrong. With an independent per-token acceptance rate $\alpha$:

$$\mathbb{E}[\text{tokens per round}] = \frac{1 - \alpha^{k+1}}{1 - \alpha}$$

The real gain is smaller because the draft's $k$ passes are not free.

**When it does not help.** At large batch the idle compute disappears; at high temperature the two distributions disagree more; the draft must share the tokenizer. Modern variants (Medusa, EAGLE, prompt lookup, DeepSeek-V3's MTP head) remove the separate draft model but keep the same acceptance rule, which is what guarantees correctness.

> **Notebook:** [10 · Inference and serving](../../notebooks/02_pretraining_and_inference/10_inference_serving.ipynb) · **In the course:** entropy and truncation for each strategy at the position where the model hesitates (greedy repeats a single continuation 16 times), exact equality with and without the KV-cache, speculative decoding with $k = 4$ (ceiling of 5 tokens per round) plus a statistical check that the distribution does not change; on a toy model kernel launch overhead dominates, so the cache speed-up looks modest.

### 2.8 Quantization

**Problem.** A 7B model in bf16 needs 14 GB for its weights alone. Quantization addresses three different problems that are worth keeping apart: **memory** (bytes per weight; the saving is guaranteed), **bandwidth** (bytes read per token in decode; only with a kernel that works on the packed format) and **compute** (needs low-precision weights *and* activations, and hardware that supports them).

**Core idea.** With a symmetric per-group scale:

$$s = \frac{\max|w|}{2^{b-1}-1},\qquad q = \mathrm{round}(w/s),\qquad \hat w = s\,q, \qquad \mathbb{E}[(w-\hat w)^2] = \frac{s^2}{12}$$

Each extra bit halves $s$ and divides the error variance by four. What makes or breaks a scheme is $s$, set by the **largest value in the group**: one outlier inflates the step for everyone. That is why scales are per small block (32 to 128 weights), at $16/g$ extra bits per weight, and why NF4 places its 16 levels at quantiles of a normal distribution. Integers space levels evenly; floats (fp8 E4M3/E5M2) space them logarithmically, which absorbs part of the dynamic range.

**Why LLMs tolerate it, and where they do not.** They are overparameterized and trained with noise; rounding errors in a long dot product partly cancel. But activations are harder than weights (LLM.int8() found a few hidden dimensions with huge values beyond a few billion parameters), small models suffer more, and embeddings, the output head and the first and last blocks are more sensitive.

| Method | Data | Cost | Sweet spot |
|---|---|---|---|
| RTN (quanto, bitsandbytes) | no | seconds | 8-bit; 4-bit with small groups on large models |
| LLM.int8() | no | load time | 8-bit with outliers in fp16 |
| NF4 + double quantization | no | load time | 4-bit base for QLoRA |
| GPTQ | small calibration set | minutes to hours | 4-bit (and 3-bit) weight-only |
| AWQ | small calibration set | minutes | 4-bit weight-only |
| SmoothQuant / FP8 | activation scales | minutes | W8A8 on int8/fp8 hardware |
| QAT, BitNet | full training data | a training run | 2-bit and below |

**GPTQ** minimizes the error of each layer's *output*, $\min_{\hat W} \Vert WX - \hat W X\Vert_F^2$, quantizing column by column and compensating the error in the remaining columns with the inverse Hessian (the Optimal Brain Surgeon idea). **AWQ** finds the important channels by their *activation* magnitude and scales them before quantizing, using the identity $Wx = (W\,\mathrm{diag}(s))(\mathrm{diag}(s)^{-1}x)$. **SmoothQuant** uses the same identity the other way for W8A8, and QuaRot/SpinQuant rotate with orthogonal matrices to spread outliers.

**What the industry does.** FP8 weights and activations for serving on recent GPUs (close to lossless, native tensor cores); 4-bit weight-only for local and cost-sensitive serving (AWQ and GPTQ on servers, GGUF k-quants in llama.cpp and Ollama); FP8 in training too (DeepSeek-V3). At a fixed memory budget, a larger model at ~4 bits usually beats a smaller one at higher precision (Dettmers and Zettlemoyer, 2023).

**Measuring fidelity.** Perplexity only looks at the true token and can stay put while the model behaves differently. Compare against the original with per-position $D_{\mathrm{KL}}(p_{ref} \,\Vert\, p_q)$ (non-negative, cannot cancel out) and **top-1 agreement** (would greedy say the same thing?). A quantized model can match the original's accuracy while changing many individual answers (Dutta et al., 2024).

**Failure modes.** Outliers with per-tensor scales; RTN at 2-3 bits (collapse); specific skills that break first (math, code, long context, non-English languages); calibration on the wrong domain; a format with no fused kernel, which saves memory but runs *slower*.

> **Notebook:** [11 · Quantization](../../notebooks/02_pretraining_and_inference/11_quantization.ipynb) · **In the course:** with SmolLM2-360M and fp32 as the reference, bf16 alone gives KL 0.001 and 97.2 % top-1 agreement (the noise floor); int8 and float8 cut the weights from 0.72 to 0.41 GB at ~96 % agreement; int4 reaches 0.26 GB but drops to 73 % agreement and +29 % perplexity; int2 destroys the model. The quantized models were *slower* than bf16 (~24 → 17 and 10.5 tok/s) for lack of a fused kernel.

### 2.8b Serving under load

**Problem.** Sections 2.5–2.8 look at one generation at a time. An application sends many requests at once to an HTTP server, and what matters then is what each user waits for, what the hardware does with the traffic and what it costs. A server can look fast for one request and collapse under eight.

**The metrics.** A streamed response has two phases, and each has its own metric:

$$\text{e2e} \approx \text{TTFT} + \text{TPOT} \times (N_{out} - 1), \qquad \text{throughput} = \frac{\sum \text{output tokens}}{\text{wall time}}$$

**TTFT** (time to first token) includes the time in the queue, the prefill and the network: it is the "is it doing anything?" of the user. **TPOT** (time per output token, or inter-token latency) is the gap between tokens during decode, and $1/\text{TPOT}$ is one user's reading speed (20–30 tokens per second already outpaces reading). **Throughput** adds up every user and is what the bill depends on. Latency tails are long (section 6.3), so targets are **SLOs on percentiles** (TTFT p95 < 1 s, TPOT p95 < 50 ms, say), and the useful capacity is **goodput**: requests per second that meet the SLO, not raw tokens per second (Zhong et al., 2024).

**Why the two phases behave differently.** Prefill pushes the whole prompt through the model in one pass of large matmuls, so it is compute-bound; decode reads every weight for each new token, so it is memory-bound (section 2.5). Every serving optimization exploits that asymmetry: batching shares one weight read among many sequences, prefix caching skips prefill already done, quantization reduces the bytes read per token, speculative decoding (2.7) turns several decode steps into one verification. **Disaggregated serving** (DistServe; Mooncake) goes further and runs prefill and decode on different GPUs, because a long prefill in the same batch stalls everyone's decode and the two phases want different parallelism.

**Batching.** **Static batching** waits for a batch and runs it until the longest sequence ends, leaving finished slots idle. **Continuous batching** (Orca, Yu et al., 2022) schedules at the iteration level: sequences join and leave at every decode step, so the batch stays full. What limits the batch is then KV-cache memory, and **PagedAttention** (vLLM, Kwon et al., 2023) stores it in fixed-size blocks with a block table, like virtual-memory pages, so almost nothing is lost to fragmentation and blocks can be shared between sequences. llama.cpp, and Ollama on top of it, serves a fixed number of **slots** (`OLLAMA_NUM_PARALLEL`): each slot is one sequence in the batch with its own share of the KV-cache, and requests beyond the slots wait in a queue. Batching is a trade: total throughput goes up and each user's TPOT goes up with it.

**Queues and Little's law.** For any stable system, whatever the arrival distribution (Little, 1961):

$$L = \lambda W$$

requests in the system = arrival rate × time in the system. It turns sizing into arithmetic: at 2 requests per second and 3 s per request, 6 are in flight on average, so provision slots or replicas above that. And it explains the knee of every latency curve: once requests arrive faster than the server completes them, the queue grows without bound and TTFT with it. In the simplest queueing model (M/M/1), $W = 1/(\mu - \lambda)$, which explodes as the utilization $\lambda/\mu$ approaches 1. That is why capacity is planned on latency-against-load curves and kept well below saturation, and why autoscaling watches queue length or KV-cache usage rather than CPU.

**Prefix caching.** Many requests share a long prefix: the system prompt, the tool definitions, the few-shot examples, the document under discussion. Its KV-cache can be kept and reused instead of recomputed. vLLM's automatic prefix caching hashes KV blocks; SGLang's **RadixAttention** (Zheng et al., 2024) keeps cached prefixes in a radix tree shared by all requests, with LRU eviction; provider APIs sell it as **prompt caching**, with cached input tokens billed at a fraction of the price; llama.cpp reuses a slot's cache when the next prompt starts with the same tokens. The match is token by token from the start, so one changed token (a timestamp at the top) invalidates everything after it: **shared content first, the user's message last**.

**Weight formats: GGUF and k-quants.** GGUF is llama.cpp's single-file format: weights, tokenizer and metadata together. Its **k-quants** (Q4_K_M, Q5_K_M, Q6_K...) quantize weights in super-blocks of 256, split into small sub-blocks whose scales are themselves quantized, the per-group scales of section 2.8; the `_S`/`_M`/`_L` suffix says how many sensitive tensors are kept at a higher precision, so Q4_K_M averages a little under 5 bits per weight. Q8_0 is blocks of 32 int8 values with one fp16 scale; FP16 is unquantized. Because decode is memory-bound, fewer bytes means faster decode on top of less memory, provided the runtime has a kernel for the format (unlike notebook 11's quanto). The quality cost has to be measured on your own task.

**Gateways, cascades and routers.** Applications rarely call one model directly. A **gateway** (LiteLLM, Portkey, OpenRouter, cloud AI gateways) gives every caller one interface, usually OpenAI-compatible, and handles the cross-cutting concerns in one place: routing, retries with backoff, **fallbacks** to another model on errors or **timeouts**, **circuit breakers** that stop sending traffic to a failing backend for a while, rate limits, budgets, caching and logs. On top of it, cost-aware routing:

- a **cascade** (FrugalGPT, Chen, Zaharia and Zou, 2023) tries models in order of cost and stops when a scorer accepts the answer. Its expected cost is $c_{small} + c_{signal} + p_{esc} \cdot c_{big}$: it pays off when the escalation rate $p_{esc}$ is low **and** the signal is cheap;
- a **router** (RouteLLM, Ong et al., 2024) predicts before any call whether the strong model is needed, with a classifier trained on preference data, so each request costs one call.

The uncertainty signal can be agreement between samples (as in self-consistency, 3.5), token log-probabilities, a verifier or a trained classifier. Routing pays most when the price gap is large, a small or local model in front of a frontier API.

**What the industry does.** llama.cpp or Ollama for local, edge and single-user setups; vLLM, SGLang or TGI (and TensorRT-LLM for the highest throughput on NVIDIA) for multi-user serving on data-center GPUs, with continuous batching, PagedAttention, prefix caching across users, FP8 or AWQ weights and speculative decoding; all of them behind an OpenAI-compatible API and, often, a gateway. Load tests use realistic prompt and output lengths and arrival patterns (`vllm bench`, `genai-perf`, Locust, k6).

**Failure modes.** Parallel traffic sent to a single-slot server (concurrency becomes a queue); a prompt layout that defeats the cache; context sizes that silently reload the model; capacity planned on peak throughput instead of the latency knee; over-quantized models that pass a smoke test and fail on edge cases; timeouts shorter than a cold model load; fallbacks that hide a dead primary; routers whose signal costs more than it saves.

**Key papers.** Little (1961), *A Proof for the Queuing Formula: L = λW*; Yu et al. (2022), *Orca*; Kwon et al. (2023), *Efficient Memory Management for Large Language Model Serving with PagedAttention*; Chen, Zaharia and Zou (2023), *FrugalGPT*; Zheng et al. (2024), *SGLang: Efficient Execution of Structured Language Model Programs*; Zhong et al. (2024), *DistServe*; Ong et al. (2024), *RouteLLM*.

> **Notebook:** [12 · Serving under load](../../notebooks/02_pretraining_and_inference/12_serving_and_load.ipynb) (read after 10) · **In the course:** with `qwen2.5:3b` on Ollama and a 6 GB RTX 3060 Laptop, prefill runs at ~3,900 tokens/s and decode at ~110 (35×), and TTFT grows from 101 to 645 ms between 338 and 2,322 prompt tokens; with a shared system prompt, 99.1 % of the prompt comes from the cache after the first request and TTFT falls from 654 to 39 ms; with one slot, 8 requests in flight push TTFT p50 to 13.4 s while throughput stays flat (84 → 67 tok/s); four slots reach 171 tok/s (2×) with TTFT under 100 ms up to 4 in flight, at the price of a TPOT of 23 ms instead of 12.5; Little's law holds to the decimal (1.33 req/s × 3.00 s = 4.0); `qwen2.5:1.5b-instruct` takes 1.17 GB at 130 tok/s in Q4_K_M, 1.83 GB at 105 tok/s in Q8_0 and 3.28 GB at 64 tok/s in FP16, with 89.5 %, 91.0 % and 91.0 % on 200 ARC-Easy questions; and on 80 GSM8K problems a 3B → 7B cascade that escalates when two 3B answers disagree (16 % of the problems) reaches 85.0 % against 82.5 % for always-3B and 78.8 % for always-7B, but at 88 % of the 7B's GPU time, because the agreement check itself doubles the 3B's cost.

---

## 3. Post-training

Terms for this part: [glossary, Training and post-training](glossary.md#training-and-post-training) and [Reasoning and test-time compute](glossary.md#reasoning-and-test-time-compute).

### 3.1 Supervised fine-tuning

**Problem.** An instruct model already chats, but it does not know *your* task, output format or conventions.

**Core idea.** SFT is exactly the pre-training objective on a curated set of (prompt, ideal answer) pairs, applied only to the answer tokens:

$$\mathcal{L}_{\text{SFT}}(\theta) = -\,\mathbb{E}_{(x,y)\sim\mathcal{D}} \sum_{t=1}^{|y|} \log \pi_\theta(y_t \mid x, y_{<t})$$

**Loss masking** (labels set to `-100` on the prompt) turns the joint likelihood $\log \pi(x,y)$ into the conditional $\log \pi(y \mid x)$, which is what inference asks for; the loss includes the end-of-turn token so the model learns to stop. Training must use the model's exact **chat template**. Keeping the prompt loss can help with very short answers and little data (Shi et al., 2024); masking remains the default.

**Intuition and limits.** SFT is behaviour cloning: it imitates the demonstrations, mistakes included, and gets no signal that one wrong answer is worse than another. It also suffers from **exposure bias**: it trains conditioned on correct tokens and at inference conditions on its own, so an early mistake leads to states it never saw. It reliably teaches **form** (format, style, a label set, applying a rule the model half-knows) and is a poor way to add **new facts**: examples with unknown facts are learned slowly and increase hallucination (Gekhman et al., 2024). LIMA's "superficial alignment hypothesis" (a thousand good examples teach the format) is useful and debated.

| Approach | Changes | Good for | Weak at |
|---|---|---|---|
| Prompting / few-shot | nothing | fast iteration | long strict formats; every call pays for the examples |
| RAG | the context | changing facts, citations, access control | teaching a behaviour |
| SFT (full or LoRA) | the weights | formats, style, narrow skills | adding knowledge |
| Distillation | weights of a smaller model | cutting serving cost | inherits the teacher's errors |

In practice they are combined: a model tuned to your format, fed with retrieved context. The rule is to measure a **few-shot baseline** first: fine-tuning has to beat a good prompt.

### 3.2 LoRA and parameter-efficient fine-tuning

**Problem.** Full fine-tuning costs about 16 bytes per trainable parameter (112 GB for 7B) and one full model copy per task.

**Core idea.** Freeze $W$ and learn a low-rank update (Hu et al., 2021):

$$h = W x + \frac{\alpha}{r}\, B (A x), \qquad A\in\mathbb{R}^{r\times k},\ B\in\mathbb{R}^{d\times r},\ B=0 \text{ at init}$$

$B = 0$ makes the model start exactly as the base; $\alpha/r$ decouples the update size from the rank. A $4096 \times 4096$ matrix at $r = 16$ goes from 16.8 M trainable parameters to 131 k (a factor of 128), and gradients and Adam states shrink by the same factor. After training, `merge_and_unload()` folds $BA$ into $W$: **zero extra latency**. It saves optimizer and gradient memory, not activation memory.

**Why it works.** Fine-tuning has a low **intrinsic dimension** (Aghajanyan et al., 2021): optimizing a few hundred or thousand parameters projected into the full space is enough. Hu et al. found that $\Delta W$ amplifies a few directions already present in $W$. That is a finding about *narrow* tasks: with a distant target (a new language, continued pre-training) the needed update has higher rank and LoRA falls behind. Biderman et al. (2024) summarize it: "LoRA learns less and forgets less".

| | Full fine-tuning | LoRA | QLoRA |
|---|---|---|---|
| Trainable parameters | all | $r(d+k)$ per adapted matrix | same as LoRA |
| Frozen base in memory | — | 16-bit | 4-bit (NF4) |
| Quality on narrow tasks | reference | usually close | close, a bit more quantization noise |
| Artefact per task | full model | small adapter | small adapter |

LoRA displaced adapters (extra latency) and prefix/prompt tuning (longer effective sequence) because it adds no latency, matches full fine-tuning on many narrow tasks, and the whole ecosystem supports it (`peft`, TRL, vLLM serving several adapters on one base). Variants: rsLoRA ($\alpha/\sqrt{r}$), DoRA.

**QLoRA: the same LoRA on a 4-bit base.** LoRA freezes the base weights, but they still sit in memory in bf16: 2 bytes per parameter, about 14 GB for a 7B model before a single activation is stored. QLoRA (Dettmers et al., 2023) keeps those frozen weights in 4 bits and trains ordinary bf16 LoRA adapters on top:

$$h = \operatorname{dequant}(W_0^{\text{NF4}})\,x + \frac{\alpha}{r}\, B A x$$

The 4-bit weight is dequantized to bf16 block by block just for the matmul and thrown away; the gradient flows through the frozen layer to $A$ and $B$, which stay in bf16, so the optimizer never touches a 4-bit number. The paper's three ideas: **NF4** (16 levels at the quantiles of a normal distribution, with one absmax scale per block of 64 weights; pretrained weights are roughly normal, see section [2.8](#28-quantization)), **double quantization** (the per-block scales are themselves quantized to 8 bits, cutting the overhead from 0.5 to about 0.127 bits per parameter) and **paged optimizers** (optimizer state lives in CUDA unified memory and is paged to CPU RAM when memory spikes, instead of dying with an out-of-memory error). The price: every step is slower because it dequantizes on each forward and backward pass, and the adapter belongs to a *quantized* base; to merge it, reload the base in bf16 and apply it there. The rule of thumb: if the model fits in bf16 with room to train, use LoRA (faster, merges directly); if it does not, QLoRA. In `transformers` it is `BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)`, plus `prepare_model_for_kbit_training` and `optim="paged_adamw_8bit"`.

**Practical rules and failure modes.** Adapting all linear layers at $r = 8$-16 usually beats a large rank on `q_proj` and `v_proj` only; LoRA needs a learning rate about 10 times higher than full fine-tuning (an empirical finding); when comparing ranks, check whether $\alpha$ scales with $r$. The **new token** trap: with tied, frozen embeddings, a row initialized at the mean of the embeddings can never win the argmax, because $h \cdot \bar e$ is the average of the $h \cdot e_i$ and an average never exceeds the maximum; that row must be trained (`modules_to_save` or `trainable_token_indices`).

> **Notebook:** [13 · SFT, LoRA and distillation](../../notebooks/03_post_training/13_finetuning_sft_lora.ipynb) · **In the course:** booking extraction to JSON with SmolLM2-135M; few-shot reaches 17.5 % exact match, full SFT 35.0 % and LoRA (3.5 % of the parameters, a 20 MB adapter) 32.5 %; rank lifts exact match from 20 % ($r=2$) to 45 % ($r=64$); with frozen embeddings the new token is never emitted (it ranks around 24,000th), and with a trainable row it comes first at p ≈ 0.93; QLoRA r=16 on the NF4 base reaches 37.5 % exact match and 86.5 % of fields (LoRA r=16: 32.5 % and 86.5 %, two requests apart, which is noise), the weights in memory drop from 269 MB to 110 MB but the training peak barely moves (1.41 → 1.34 GB) because activations dominate at 135M; for a 7B model the frozen weights would go from 14 GB to about 3.6 GB.

### 3.3 Distillation

**Problem.** Labelled data is expensive, unlabelled prompts are cheap, and a big model that works is expensive to serve.

**Core idea.** A strong teacher transfers what it knows to a small student. Hinton, Vinyals and Dean (2015) observed that the teacher's output distribution carries more than its top answer ("dark knowledge": which mistakes are plausible) and exposed it by softening with a temperature:

$$\mathcal{L} = (1-\lambda)\,\mathrm{CE}(y, p_s) + \lambda\,\tau^2\,D_{\mathrm{KL}}\big(p_t^{\tau} \,\Vert\, p_s^{\tau}\big), \qquad p_i^\tau = \frac{e^{z_i/\tau}}{\sum_j e^{z_j/\tau}}$$

The $\tau^2$ factor compensates for the softened KL's gradient scaling as $1/\tau^2$. The direction of the KL matters: **forward** KL is mode-covering (the student spreads mass over everything the teacher might say), **reverse** KL is mode-seeking; MiniLLM argues for reverse KL in generation, and GKD trains on the student's own samples (on-policy), which addresses exposure bias.

| | Sequence-level (synthetic labels) | Logit (token-level) | On-policy (GKD, MiniLLM) |
|---|---|---|---|
| Needs from the teacher | text only | logits at every position | logits on student samples |
| Tokenizer | any | must match | must match |
| Cost | one generation per prompt | a teacher pass per step | student sampling + teacher pass per step |
| Main risk | copies the teacher's errors | memory for two models | slow, more moving parts |

**Why the sequence-level version dominates.** Kim and Rush (2016) showed that training on the teacher's decoded outputs works surprisingly well: dark knowledge is lost, but it works across tokenizers, with a teacher behind an API, and it is plain SFT. Alpaca, Vicuna and many reasoning distillations (DeepSeek-R1-Distill) were built this way.

**Failure modes.** The student is bounded by the teacher and inherits its systematic errors; a large capacity gap can make a very strong teacher a worse teacher (Mirzadeh et al., 2020); some API terms of use forbid training competing models on their outputs.

> **Notebook:** [13 · SFT, LoRA and distillation](../../notebooks/03_post_training/13_finetuning_sft_lora.ipynb) · **In the course:** a 360M teacher trained on the same 300 labels reaches 87.5 %, labels 600 new requests with 90 % exactly right, and the 135M student goes from 32.5 % to 55 % exact match (with the caveat that it also took more optimizer steps).

### 3.3b Model merging

**Problem.** After a while you have several fine-tunes of the same base: one per task, per customer or per experiment. You want one model, and you may not have the data or the GPU budget to train them together.

**Core idea.** Treat a fine-tune as a vector. Ilharco et al. (2023, *Editing Models with Task Arithmetic*) call the difference $\tau_t = \theta_t - \theta_0$ a **task vector** and showed that you can do arithmetic with it (**task arithmetic**):

$$\theta_{\text{merged}} = \theta_0 + \sum_t \lambda_t\, \tau_t$$

Adding two task vectors gives a model that does both tasks, often nearly as well as each specialist; subtracting one ($\lambda < 0$) removes a behaviour, which the paper used to make a model less toxic. With LoRA, the task vector of each adapted matrix is just $\Delta W = \frac{\alpha}{r} BA$. Merging is arithmetic only: no data, no gradient steps.

**Why it works at all.** Weights are not normally additive: average two networks trained from different random initializations and you get a broken model, because their neurons are permuted relative to each other. Fine-tunes of the *same* pretrained model are different. They stay in one basin of the loss landscape, joined by a straight path along which the loss stays low (**linear mode connectivity**, Frankle et al., 2020). Wortsman et al. (2022) used this for **model soups**: averaging fine-tunes of one task trained with different hyperparameters beat the best single run in accuracy and robustness, at no extra inference cost. For *different* tasks, addition works when the task vectors interfere little, and in a space with millions of coordinates per matrix two unrelated updates are nearly **orthogonal** (cosine similarity close to 0): adding one barely moves the model along the direction of the other.

**When they interfere: TIES and DARE.** When task vectors overlap, plain addition goes wrong in two ways: small, noisy coordinates add up, and coordinates where the tasks pull in opposite directions cancel out or overshoot.

- **TIES-Merging** (Yadav et al., 2023): *trim* each task vector to its largest-magnitude fraction (the *density*); *elect* a sign per coordinate, the one with the larger total magnitude; then a *disjoint merge*, averaging only the values that agree with that sign.
- **DARE** (Yu et al., 2024, *Language Models are Super Mario*): randomly *drop* a fraction $p$ of each task vector's coordinates and *rescale* the rest by $1/(1-p)$, so the expected update stays the same. Fine-tuning updates are so redundant that 90 % or more can be dropped. It is usually combined with TIES' sign election (`dare_ties`).
- **SLERP** interpolates two models along the sphere instead of a straight line, which keeps the norm of the weights; it only works for two models. **LoraHub** (Huang et al., 2023) learns the mixing weights $\lambda_t$ of many LoRA adapters from a few examples of a new task.

The standard tool for full models is **MergeKit** (Goddard et al., 2024), behind many of the merged open-weight models on public leaderboards; for adapters, `peft`'s `add_weighted_adapter`.

**The LoRA factor trap.** An adapter stores $B$ and $A$, not $\Delta W$, and

$$\big(B_1 + B_2\big)\big(A_1 + A_2\big) = B_1 A_1 + B_2 A_2 + \underbrace{B_1 A_2 + B_2 A_1}_{\text{cross terms}}$$

so summing the factors adds two terms neither task asked for. In `peft`, `combination_type="linear"` does exactly that (its docs call it a very rough approximation); `cat` concatenates the factors into an adapter of rank $r_1 + r_2$ whose $\Delta W$ is the exact weighted sum, that is, task arithmetic; `svd` forms the exact sum and compresses it back to a chosen rank (exact if `svd_rank` is at least the rank of the sum); `ties_svd` and `dare_ties_svd` apply TIES or DARE to the full $\Delta W$ and then an SVD.

| | Merging | Multi-task training |
|---|---|---|
| Needs | the fine-tuned weights or adapters, from the same base | every dataset at once |
| Cost | seconds of arithmetic | a training run over all the data |
| What you get | the behaviours side by side, each a little below its specialist | usually the best on every task, and skills can combine |
| Use it when | the adapters come from other teams or the Hub, the data is gone, there is no budget to retrain | you own all the data |

**Failure modes.** Merging fine-tunes of different bases (or an adapter into a base it was not trained on); merging LoRA factors with `linear`; using TIES or DARE where there is no interference, so trimming only throws signal away; expecting the merge to *combine* skills, when it only puts them next to each other; and evaluating the merge on one task, when it can silently break another. Run every task's evaluation on the merged model.

> **Notebook:** [13 · SFT, LoRA and distillation](../../notebooks/03_post_training/13_finetuning_sft_lora.ipynb) (section 8) · **In the course:** a second LoRA adapter learns to convert times to 24-hour format and is merged with the booking adapter. The two task vectors are nearly orthogonal (mean cosine +0.012 over 210 matrices, 49 % sign conflicts where both updates are large). Task arithmetic (`cat`) keeps booking at 32.5 % exact, the same as the booking adapter alone, and converts times at 90 % (specialist 97.5 %); `linear` gets 27.5 % / 87.5 %, `ties_svd` 27.5 % / 80 % and `dare_ties_svd` 30 % / 87.5 %. No merge carries the conversion skill into the booking `time` field. One multi-task adapter trained on both datasets (76 steps) reaches 55 % exact and 97.5 %, better than every merge on both tasks.

### 3.4 From RLHF to DPO and GRPO

**Problem.** SFT does not say *how bad* an answer is and only shows good behaviour in states taken from the data. Also, it is often easier to **compare** two answers than to **write** the ideal one.

**Classic RLHF** (Christiano et al., 2017; Stiennon et al., 2020; InstructGPT, 2022): SFT, then a reward model fitted on human comparisons through **Bradley-Terry**,

$$P(y_w \succ y_l \mid x) = \sigma\big(r(x, y_w) - r(x, y_l)\big)$$

and finally RL with PPO on the regularized objective:

$$\max_{\pi}\; \mathbb{E}_{x,\,y\sim\pi}\big[r_\phi(x,y)\big] \;-\; \beta\,D_{\mathrm{KL}}\big(\pi(\cdot|x)\,\Vert\,\pi_{\text{ref}}(\cdot|x)\big)$$

The KL matters as much as the reward: the reward model is only accurate near its data, and without the leash the policy finds outputs that score high for the wrong reasons (**reward hacking**; Gao et al., 2023, measured true quality rising and then falling). PPO needs four models at once (policy, reference, reward and a similarly sized critic), sampling in the loop and many sensitive hyperparameters: it works, but it is expensive and fragile.

**DPO: the reward model hides inside the policy.** Rafailov et al. (2023) noticed the objective above has a closed-form optimum, $\pi^*(y|x) \propto \pi_{\text{ref}}(y|x)\exp(r(x,y)/\beta)$, which can be inverted:

$$r(x,y) = \beta \log\frac{\pi^*(y\mid x)}{\pi_{\text{ref}}(y\mid x)} + \beta \log Z(x)$$

Substituting into Bradley-Terry, the intractable $\log Z(x)$ cancels in the difference (both answers share the prompt), leaving a classification loss on pairs:

$$\mathcal{L}_{\text{DPO}} = -\,\mathbb{E}\,\log\sigma\Big(\beta\log\frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta\log\frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)}\Big)$$

No sampling, no critic, no reward model: just a policy and a reference (with LoRA, the reference is the same network with the adapter off). The gradient weights wrongly ranked pairs more and **constrains only the difference** of log-probabilities. That causes the most instructive failure, **likelihood displacement**: when chosen and rejected are nearly identical, the cheapest way to widen the margin can be to push *both* down and send the mass to answers in neither set (Razin et al., 2024). The standard fix re-anchors the chosen answer with an SFT term: $\mathcal{L}_{\text{DPO}} + \lambda\,\mathrm{NLL}(y_w)$ (DPO+NLL, RPO). Other limits: it is offline (no exploration; iterated DPO closes part of the gap), it overfits deterministic preferences (IPO), it exploits length. Whether DPO matches a well-tuned PPO is **debated** (Xu et al., 2024 found PPO ahead on code); its popularity comes from simplicity.

**RL with verifiable rewards and GRPO.** In math or code the reward model is replaced by a program (tests, a numeric check, a schema): RLVR. By the policy gradient

$$\nabla_\theta\, \mathbb{E}_{y\sim\pi_\theta}[r] = \mathbb{E}\big[(r - b)\,\nabla_\theta \log\pi_\theta(y\mid x)\big]$$

any baseline $b$ reduces variance without bias. PPO learns $b$ with a critic; **GRPO** (Shao et al., 2024) estimates it from a group of $G$ samples of the same prompt:

$$A_i = \frac{r_i - \operatorname{mean}(r_1,\dots,r_G)}{\operatorname{std}(r_1,\dots,r_G)}$$

with PPO's clipped ratio and a KL to the reference. Consequences: no critic; a group where every answer gets the same reward contributes no gradient (so the fraction of zero-variance groups is the key diagnostic, and DAPO filters them out); and the normalization introduces biases (Dr. GRPO drops the division by length and by the standard deviation). DeepSeek-R1-Zero showed GRPO with simple correctness and format rewards producing long reasoning chains without supervised traces. How much RLVR *creates* new abilities versus *sharpening* ones the base model can already sample is an open question (Yue et al., 2025).

**Reward design** is most of the work. A dense reward creates variance in more groups, but every term is a behaviour you are paying for (Goodhart's law); verifiers have false positives and, with enough samples, the policy finds them.

| | RLHF with PPO | DPO | GRPO (RLVR) |
|---|---|---|---|
| Signal | learned reward model | comparisons, used directly | a program that checks |
| Models in memory | policy, reference, reward, critic | policy, reference | policy, reference (none if $\beta=0$) |
| Sampling during training | yes | no | yes, $G$ per prompt |
| Typical failure | reward hacking, instability | overfitting, likelihood displacement | a lenient checker, zero-variance groups |
| Used today for | chat alignment in large labs | open-model alignment, style, safety | reasoning: math, code, tool use |

**Key papers.** Christiano et al. (2017); Schulman et al. (2017, PPO); Ouyang et al. (2022); Rafailov et al. (2023); Azar et al. (2023, IPO); Shao et al. (2024, DeepSeekMath); DeepSeek-AI (2025, R1); Razin et al. (2024); Gao et al. (2023).

> **Notebook:** [14 · DPO and GRPO](../../notebooks/03_post_training/14_preference_rl_dpo_grpo.ipynb) · **In the course:** plain DPO wins on its own training metrics but drops exact match from 30 % to 7.5 % (the correct answer's log-probability falls from −1.91 to −5.63); DPO+NLL holds it (−1.91 → −2.01) and breaks nothing, though it only reaches 32.5 %; GRPO takes arithmetic from 5 % to 46 % (exact reward) and 42 % (dense) in 40 steps with no labels, and the dense reward cuts the share of zero-variance groups from 60-80 % to ~5-25 %.

### 3.5 Reasoning and test-time compute

**Problem.** Some problems need several dependent steps ("first the eggs left over, then what they sell for"), and a transformer does a fixed amount of computation per token: $L$ layers, whatever the question. If the steps do not fit in one forward pass, the model cannot get the answer right by producing it straight away, however much it knows.

**Chain of thought: tokens as serial computation.** Writing the intermediate steps as tokens turns the output into a scratchpad: every new token attends to the results written before it, so **more tokens buy more serial computation**. Wei et al. (2022) showed that a few worked examples in the prompt unlock multi-step reasoning in large models (few-shot CoT); Kojima et al. (2022), that "Let's think step by step" alone does much of it (zero-shot CoT). There is a theoretical side too: with intermediate steps, transformers can solve problems they provably cannot solve in a single pass (Feng et al., 2023; Merrill and Sabharwal, 2024). The written chain is not necessarily the computation that produced the answer: it can be **unfaithful** (Turpin et al., 2023), so reading it is not an explanation.

**Spending more compute at inference.** Once a model can reason, there are three ways to spend more compute on one problem:

- **Parallel sampling and self-consistency** (Wang et al., 2022; ICLR 2023). Sample $k$ reasoning paths at temperature $> 0$ and return the most frequent final answer: correct paths tend to agree, wrong ones scatter. Its accuracy is **maj@k**. The ceiling is **pass@k**, the probability that at least one of the $k$ samples is right, which is what a perfect verifier would reach. With $n$ samples of which $c$ are correct, the unbiased estimate is (Chen et al., 2021)

  $$\text{pass@}k = 1 - \binom{n-c}{k}\Big/\binom{n}{k}$$

  The gap between pass@k and maj@k is the most any verifier can add on top of voting.
- **Best-of-N with a verifier.** A second model scores each candidate and the best one is returned. An **outcome reward model** (ORM) scores the final answer; Cobbe et al. (2021), who introduced GSM8K for this, trained one and found that sampling many solutions and letting the verifier pick beat fine-tuning alone. A **process reward model** (PRM) scores every step; Lightman et al. (2023, *Let's Verify Step by Step*) found process supervision clearly better than outcome supervision on MATH. The catch is the verifier's **discrimination**: if it cannot tell right from wrong better than the vote, it adds cost and no accuracy, and with enough candidates the search finds the answers that fool it (*reward hacking* at inference time, the same Goodhart effect as in [section 3.4](#34-from-rlhf-to-dpo-and-grpo)). Programmatic verifiers (unit tests, a calculator, a solver, a schema) have none of these problems where they exist.
- **Sequential refinement.** Revise one answer several times (self-refine, the reflection of [section 5.3](#53-react-plan-and-execute-and-reflection)) or force the model to keep thinking (*budget forcing*, s1; Muennighoff et al., 2025).

Snell et al. (2024) showed that, for a fixed budget, spending compute at test time adaptively (more samples or revisions on harder prompts) can beat a model several times larger on problems of the right difficulty: **test-time compute is a scaling axis of its own**, next to parameters and data ([section 2.3](#23-scaling-laws)). It does not replace them: on the hardest problems, a better base model still wins.

**Reasoning models.** Instead of prompting for it, reasoning can be **trained**. DeepSeek-AI (2025, *DeepSeek-R1*) applied large-scale RL with verifiable rewards (GRPO, [section 3.4](#34-from-rlhf-to-dpo-and-grpo)) and got long chains of thought with self-checking and backtracking; OpenAI's o1 did something similar with an undisclosed recipe. The thinking is emitted as tokens (often between `<think>` tags) and paid for like any other output. **Distillation** transfers it cheaply: R1-Distill models are small Qwen and Llama models fine-tuned (plain SFT, [section 3.3](#33-distillation)) on R1's reasoning traces. Whether small distilled models reason robustly outside their training distribution is an open question. The known failure is **overthinking**: long, looping reasoning on easy problems, or reasoning that runs out of budget before it answers.

**The cost.** Reasoning is paid in output tokens, the slow and expensive ones ([section 2.5](#25-prefill-decode-and-the-kv-cache)): ten times the tokens is roughly ten times the latency and the price, and reasoning models have long tails. Two practical traps come with it: answers truncated by `max_tokens`, and answer-extraction errors that silently turn a right solution into a wrong score.

**Comparing strategies: Bradley-Terry and pairwise judges.** Many evaluations compare two systems on the same input instead of scoring each one alone. The **Bradley-Terry** model (Bradley and Terry, 1952), the same one behind RLHF reward models in [section 3.4](#34-from-rlhf-to-dpo-and-grpo), turns such outcomes into scores:

$$P(i \succ j) = \frac{e^{\beta_i}}{e^{\beta_i} + e^{\beta_j}} = \sigma(\beta_i - \beta_j)$$

with the $\beta$ fitted by maximum likelihood (Chatbot Arena does it with millions of human votes; Elo is an online approximation of the same idea), and intervals from bootstrapping the inputs. When the judge is an LLM, its biases have to be measured (Zheng et al., 2023, *Judging LLM-as-a-Judge*): **position bias** (preferring the first or the second answer regardless of content), verbosity bias and self-preference. The standard defence is to **judge every pair in both orders**, report how often the verdict survives the swap, and keep only consistent verdicts; and to validate the judge against ground truth before trusting it ([section 4.3](#43-ragas-and-llm-as-judge)).

| Strategy | Extra cost | Helps when | Typical failure |
|---|---|---|---|
| chain of thought | many more output tokens than a direct answer | multi-step problems | unfaithful or overlong reasoning |
| self-consistency (maj@k) | $k$× the chain of thought | answers that can be compared exactly | saturates fast; useless for open-ended text |
| best-of-N with a verifier | $k$× + verification | the verifier discriminates (tests, solvers, trained PRMs) | a weak or hackable verifier picks fluent wrong answers |
| reasoning model | long thinking, billed as output | hard problems with checkable answers | overthinking, truncation, cost tails |

**Key papers.** Wei et al. (2022), *Chain-of-Thought Prompting Elicits Reasoning in Large Language Models*; Kojima et al. (2022), *Large Language Models are Zero-Shot Reasoners*; Wang et al. (2022), *Self-Consistency Improves Chain of Thought Reasoning*; Cobbe et al. (2021), *Training Verifiers to Solve Math Word Problems* (GSM8K); Chen et al. (2021), *Evaluating Large Language Models Trained on Code* (pass@k); Lightman et al. (2023), *Let's Verify Step by Step*; Turpin et al. (2023), *Language Models Don't Always Say What They Think*; Snell et al. (2024), *Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters*; DeepSeek-AI (2025), *DeepSeek-R1*; Muennighoff et al. (2025), *s1: Simple test-time scaling*; Zheng et al. (2023), *Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena*; Bradley and Terry (1952).

> **Notebook:** [15 · Reasoning and test-time compute](../../notebooks/03_post_training/15_reasoning_test_time_compute.ipynb) (read after 14) · **In the course:** on 80 GSM8K problems with `qwen2.5:3b`, a direct answer gets 12.5 % and greedy chain of thought 85.0 % (246 output tokens against 3); self-consistency with 5 samples also gets 85.0 % (+0.0 points, paired interval [−6.2, +6.2]) for about 1,241 tokens per problem (an earlier run's 90.0 % was noise at n = 80), while pass@5 reaches 92.5 %, so a few right answers were in the minority; best-of-5 with the 7B judge as verifier scores 80.0 % on the first 40 problems, against 85.0 % for the vote, because it gives correct solutions 7.14 on average and wrong ones 6.32; DeepSeek-R1-Distill 1.5B reaches 63.7 % with 870 tokens per problem, and 16 of its 80 answers hit the 2,048-token limit; Bradley-Terry ranks self-consistency (+2.06) and chain of thought (+1.58) at the top with overlapping intervals, then R1-Distill (−0.25) and direct (−3.38); the pairwise 7B judge, asked in both orders, keeps its verdict on only 47 % of 30 pairs (53 % of its verdicts go to position A, so it is noise more than a fixed preference); and on Claude Haiku 5.5, `effort` mostly sets how often the model thinks (a thinking block in 4 % of answers at `low`, 100 % at `max`) while accuracy stays within noise (92.5–96.2 %) until `max` (98.8 %), at 174 to 674 output tokens per problem.

> **Notebook:** [31 · Capstone: a reasoning model from scratch (mini-R1)](../../notebooks/07_capstone/31_mini_r1_capstone.ipynb) (read last) · **In the course:** the whole R1 recipe on the course's 1.6M-parameter `TransformerLM`, pretrained on chains of operations modulo 10 where 10 % of the examples show the work: GRPO written by hand, straight from the base (R1-Zero), lifts accuracy from 69.4 % to 99.8 % (+30.4 points [+27.7, +33.1]) and the share of samples that reason from 11 % to 67 %, with no reasoning examples, and it learns to skip the steps on one-operation chains; but the base, made to reason, already scored 99.9 %, so RL brought out a skill pretraining had put there (as Yue et al., 2025 argue), a 300-example SFT cold start reaches the same 100.0 % far more cheaply, R1 and a same-budget SFT control tie at 100.0 %, and on chains of 8–10 operations, longer than any in training, every model stays at 14.8–22.6 %.

---

## 4. Applications

Terms for this part: [glossary, Embeddings and rerankers](glossary.md#embeddings-and-rerankers), [Clustering and topic modeling](glossary.md#clustering-and-topic-modeling), [Multimodal models](glossary.md#multimodal-models), [Applications: LangChain and RAG](glossary.md#applications-langchain-and-rag), [Recommender systems](glossary.md#recommender-systems) and [Drift and online learning](glossary.md#drift-and-online-learning).

### 4.0 Embeddings and rerankers

**Problem.** RAG, semantic search, recommendation and agent memory all start by finding the few documents, out of thousands or millions, that are relevant to a query. Reading every document with a large model is out of the question, so retrieval is a trade between how carefully a model reads and how much of the corpus it can afford to read.

**Three ways to score a query against a document.**

| Architecture | How it scores | Cost at query time | Quality |
|---|---|---|---|
| **bi-encoder** (embedding model) | query and document encoded *separately* into one vector each; score = cosine similarity | one forward pass for the query, then a vector search over precomputed document vectors | good |
| **cross-encoder** (reranker) | query and document fed *together* through one transformer; a head outputs a relevance score | one forward pass **per candidate** | best |
| **late interaction** (ColBERT) | one vector per token; score = sum over query tokens of the best-matching document token (MaxSim) | vector search over token vectors | in between |

The bi-encoder is fast because the expensive part, encoding the corpus, happens once, offline. The price is that query and document never see each other: everything the model knows about a document has to fit into one vector before it knows the question. A cross-encoder (Nogueira and Cho, 2019) lets every query token attend to every document token, the full attention of [section 1.3](#13-attention-rope-and-normalization), which makes it more accurate and too slow for a whole corpus. ColBERT (Khattab and Zaharia, 2020) keeps the token vectors and defers the interaction to a cheap MaxSim. Production search uses the **retrieve-then-rerank** funnel, as in recommendation ([section 4.5](#45-recommender-systems)): a cheap first stage (BM25, embeddings or both) returns 50–100 candidates and a cross-encoder reorders them. A reranker can only promote what the first stage found.

**How embedding models are trained: contrastive learning.** An embedding model is an encoder ([section 1.6](#16-encoders-and-encoder-decoders)) with a pooling step (the mean of the token vectors, or the `[CLS]` vector), as in Sentence-BERT (Reimers and Gurevych, 2019). It is trained so that a query lands near its relevant documents and far from the rest, with **InfoNCE** (van den Oord et al., 2018) and **in-batch negatives**: in a batch of $B$ (query, positive document) pairs, every other document in the batch is a negative for each query.

$$\mathcal{L} = -\frac{1}{B}\sum_{i=1}^{B} \log \frac{\exp(\mathrm{sim}(q_i, d_i)/\tau)}{\sum_{j=1}^{B} \exp(\mathrm{sim}(q_i, d_j)/\tau)}$$

It is a cross-entropy over the batch in which the "class" of query $i$ is its own document. Three details decide how well it works:

- **Temperature.** Cosine similarities live in $[-1, 1]$; dividing by a small $\tau$ (often 0.01–0.05) makes the softmax sharp enough to learn from. `sentence-transformers` calls this loss `MultipleNegativesRankingLoss` and writes it as a `scale` of $1/\tau = 20$.
- **Batch size.** More in-batch negatives make the task harder and the embeddings better, which is why embedding models are trained with batches of thousands (CLIP used 32,768) and why tricks such as GradCache exist.
- **Hard negatives.** Random negatives are easy. Documents that BM25 or an earlier model ranks high but that are *not* relevant are the cases that actually confuse the model; mining them is the main ingredient of DPR (Karpukhin et al., 2020) and of every modern recipe. The risk is mining unlabelled positives (false negatives), usually filtered with a reranker score.

Modern models (E5, Wang et al., 2022; BGE, GTE, nomic-embed, Nussbaum et al., 2024) are trained in two stages: weakly supervised contrastive pre-training on hundreds of millions of naturally paired texts (title–body, question–answer, query–click), then fine-tuning on smaller labelled sets with hard negatives. Many are **asymmetric** and expect a **task prefix**, such as `search_query:` / `search_document:` (nomic) or `query:` / `passage:` (E5), because a short question and a long passage are different kinds of text. Forgetting the prefix fails silently: nothing breaks, quality just drops.

**Making vectors cheaper: Matryoshka and quantization.** 100 million documents with 768 float32 dimensions take about 300 GB of RAM. Two independent ideas shrink that:

- **Matryoshka Representation Learning** (Kusupati et al., 2022) sums the contrastive loss over several prefix lengths (768, 512, 256, 128, 64), so the *first* $m$ dimensions of every vector are a good embedding on their own and vectors can be truncated after the fact. OpenAI's `text-embedding-3` (`dimensions=`) and nomic-embed v1.5 are trained this way. A model trained without it has no reason to put the important information first, and truncation hurts it much more.
- **Quantization** stores each dimension in int8 (4× smaller) or as a single sign bit (**binary**, 32× smaller, compared with the Hamming distance, which CPUs compute in one instruction). The standard recipe is to retrieve with the cheap codes and **rescore** the top candidates with more precise vectors, which recovers most of the lost quality. It combines with Matryoshka truncation, and it is what vector databases implement (binary or product quantization plus rescoring).

**How retrieval quality is measured.** The BEIR benchmark (Thakur et al., 2021), on which the retrieval part of MTEB (Muennighoff et al., 2023) is built, reports **nDCG@10**: a relevant document at rank $r$ earns $1/\log_2(r+1)$, and the sum is divided by the best possible score for that query ([section 4.2](#42-retrieval-metrics)). It rewards putting relevant documents first. **Recall@100**, the share of relevant documents in the top 100, is what a reranker has to work with. Leaderboards are a starting point, not an answer: MTEB averages dozens of tasks, scores on public sets can be inflated by training on them, and the number that decides is the one measured on *your* queries and documents, with paired intervals ([section 6.5](#65-statistics-of-small-evaluation-sets)).

| Lever | What it buys | What it costs | Typical failure |
|---|---|---|---|
| a modern embedding model | semantic matching, high recall | an encoder pass per query; the corpus re-encoded when the model changes | out-of-domain drops; missing prefixes |
| contrastive fine-tuning with hard negatives | large in-domain gains from a few hundred labels | labelling; a forgetting check | false negatives among the mined ones; overfitting |
| Matryoshka truncation | smaller vectors, as a dial | a little quality at full size | truncating a model not trained for it |
| binary / int8 + rescoring | 32× / 4× less memory | a second, small scoring step | skipping the rescoring |
| cross-encoder reranker | a careful read of the top candidates | one forward pass per candidate | the wrong domain; too deep a list |

**Key papers.** Reimers and Gurevych (2019), *Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks*; van den Oord, Li and Vinyals (2018), *Representation Learning with Contrastive Predictive Coding* (InfoNCE); Karpukhin et al. (2020), *Dense Passage Retrieval for Open-Domain Question Answering* (DPR); Khattab and Zaharia (2020), *ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT*; Nogueira and Cho (2019), *Passage Re-ranking with BERT*; Thakur et al. (2021), *BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models*; Muennighoff et al. (2023), *MTEB: Massive Text Embedding Benchmark*; Wang et al. (2022), *Text Embeddings by Weakly-Supervised Contrastive Pre-training* (E5); Kusupati et al. (2022), *Matryoshka Representation Learning*; Nussbaum et al. (2024), *Nomic Embed: Training a Reproducible Long Context Text Embedder*.

> **Notebook:** [17 · Embeddings and rerankers](../../notebooks/04_applications/17_embeddings_and_rerankers.ipynb) (read before 18) · **In the course:** on SciFact (5,183 abstracts, 300 test claims), BM25 reaches nDCG@10 0.652 and the classic all-MiniLM-L6-v2 0.648, while bge-small (0.706) and nomic-embed-text (0.708) beat it clearly, and forgetting nomic's prefixes costs only 0.006, within the noise; InfoNCE by hand equals `MultipleNegativesRankingLoss` (0.18594); fine-tuning MiniLM on 919 pairs lifts it to 0.723 with in-batch negatives and to 0.762 with one BM25 hard negative per pair, in 22 s, with no measurable forgetting on NFCorpus (within ±0.005); at 64 dimensions the Matryoshka model keeps 0.646 against 0.589 without the loss; binary codes with int8 rescoring score 0.760 against 0.762 for float32, at 48 bytes per document instead of 1,536; and on this task hybrid retrieval (0.735) and both general rerankers (ms-marco-MiniLM 0.692, bge-reranker-base 0.731 at best, 0.3–1.7 s per query) lose to the fine-tuned bi-encoder alone.

### 4.0b Clustering and topic modeling

**Problem.** A deployed LLM system produces piles of unlabeled text: user messages, tickets, failed agent runs, search queries, feedback. Before you can build an evaluation set, design a router or find what the system fails on, you need to know what is in there, and nobody has labelled it. Clustering groups similar texts; topic modeling also says what each group is about.

**The modern pipeline.** Since BERTopic (Grootendorst, 2022) the standard recipe has four steps, each replaceable:

| Step | Usual choice | Why |
|---|---|---|
| embed | a sentence-embedding model ([section 4.0](#40-embeddings-and-rerankers)) | similar meanings land close together |
| reduce | UMAP to ~5 dimensions | distances and densities become reliable again |
| cluster | HDBSCAN | finds the number of clusters and leaves outliers as noise |
| describe | c-TF-IDF keywords, then an LLM that names the cluster | a person can tell what each cluster is |

**Three families of clustering.**

- **Centroid-based: k-means** (MacQueen, 1967; k-means++ seeding, Arthur and Vassilvitskii, 2007) minimizes the distance of every point to the centroid of its cluster, $\sum_i \lVert x_i - \mu_{c(i)} \rVert^2$, alternating assignment and update. It is fast, but needs $k$ in advance, assumes round clusters of similar size and assigns every point, outliers included.
- **Hierarchical (agglomerative)** clustering starts with every point alone and repeatedly merges the closest pair (by single, complete, average or Ward linkage). The result is a tree that can be cut at any level, at a cost of $O(n^2)$ memory, which limits it to tens of thousands of points.
- **Density-based: DBSCAN and HDBSCAN** (Campello, Moulavi and Sander, 2013). Clusters are dense regions separated by sparse ones. HDBSCAN builds a hierarchy over every density level and keeps the most stable clusters, so it **finds the number of clusters itself**, accepts clusters of any shape and size, and labels points in sparse regions as **noise** instead of forcing them into a cluster. Its main knob, `min_cluster_size`, is the smallest group you are willing to call a topic.

**Why reduce first: the curse of dimensionality.** With 384–1,024 dimensions, distances concentrate: a point's nearest and farthest neighbours end up at similar distances, and density estimates (which HDBSCAN depends on) stop meaning much. **UMAP** (McInnes, Healy and Melville, 2018) builds a graph of each point's nearest neighbours and finds a low-dimensional layout that preserves that graph, keeping local neighbourhoods while discarding most of the noise in the other directions. Around 5 dimensions with `min_dist=0` is the usual setting for clustering; 2 dimensions are for pictures, since they lose too much structure. PCA is linear and keeps global variance, which is not what clustering needs; t-SNE is for plots only.

**Describing clusters: c-TF-IDF and LLM names.** **c-TF-IDF** (class-based TF-IDF) concatenates the documents of each cluster into one and weights each word $w$ in cluster $c$ by

$$W_{w,c} = \mathrm{tf}_{w,c} \cdot \log\left(1 + \frac{A}{f_w}\right)$$

where $\mathrm{tf}_{w,c}$ is the word's frequency in the cluster, $A$ the average number of words per cluster and $f_w$ the word's total frequency across clusters: frequent here, rare elsewhere. It needs no model and costs nothing. An **LLM** then turns the top keywords and a few example documents into a readable name, the step that made topic modeling practical. Before embeddings, topics came from **LDA** (Latent Dirichlet Allocation, Blei, Ng and Jordan, 2003): a probabilistic model in which each document is a mixture of topics and each topic a distribution over words, fitted on bag-of-words counts. It ignores word order and meaning beyond co-occurrence, works poorly on short texts such as chat messages, and leaves you to interpret word lists; it is still used on long documents and where an interpretable generative model matters.

**Evaluating clusters.**

- **With labels** (benchmarks, or a hand-labelled sample): the **adjusted Rand index** (ARI; Hubert and Arabie, 1985) counts pairs of points that are together or apart in both the clustering and the labels, corrected for chance, so 0 is random and 1 perfect. **Normalized mutual information** (NMI) measures how much knowing the cluster tells you about the label, between 0 and 1. **Purity** is the share of a cluster's points that carry its majority label; it is easy to read per cluster, but it rewards many tiny clusters.
- **Without labels** (the real situation): the **silhouette score** of a point is $s = (b - a) / \max(a, b)$, with $a$ its mean distance to its own cluster and $b$ to the nearest other cluster, averaged over points. It can rule out bad values of $k$, but real text rarely has a single right number of topics, because topics nest. In practice, a person reads a sample of every cluster.

**Production uses.** Understanding what users ask, with counts and examples, is the first analysis on any LLM product's logs. Clusters also give evaluation sets that cover rare intents (sample from each), a way to find failures (cluster the conversations with negative feedback or failed tool calls and fix the biggest cluster), intents and routes designed from data, and **emerging-topic detection**: flag messages far from everything the system handles (low similarity to every known intent), cluster the flags, name them and review the largest each week. BERTopic packages the pipeline with topics over time and hierarchical merging; cuML runs UMAP and HDBSCAN on the GPU for millions of documents, or a sample is clustered and the rest assigned to the nearest centroid.

**Failure modes.** Trusting LLM names without reading the examples; one giant cluster hiding several topics; an embedding model that does not understand the domain (codes, jargon, other languages); choosing $k$ by a metric instead of by the decision it serves; re-clustering from scratch each time, so cluster ids do not persist; personal data in the example messages shown on dashboards.

**Key papers.** MacQueen (1967), *Some Methods for Classification and Analysis of Multivariate Observations* (k-means); Arthur and Vassilvitskii (2007), *k-means++: The Advantages of Careful Seeding*; Campello, Moulavi and Sander (2013), *Density-Based Clustering Based on Hierarchical Density Estimates* (HDBSCAN); McInnes, Healy and Melville (2018), *UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction*; Blei, Ng and Jordan (2003), *Latent Dirichlet Allocation*; Grootendorst (2022), *BERTopic: Neural topic modeling with a class-based TF-IDF procedure*; Hubert and Arabie (1985), *Comparing Partitions* (ARI).

> **Notebook:** [21 · Clustering and topic modeling](../../notebooks/04_applications/21_clustering_topic_modeling.ipynb) (read after 17) · **In the course:** on Banking77's 3,080 test messages (77 intents, used only to measure), k-means with the true $k = 77$ reaches ARI 0.685 on a 5-d UMAP of bge-small embeddings, against 0.581 on the raw vectors and 0.576 with all-MiniLM-L6-v2 on UMAP; the silhouette score picks $k = 60$ and is flat between 50 and 100 (0.721–0.742); HDBSCAN with `min_cluster_size=20` finds 73 clusters on its own with ARI 0.683 and 5 % noise; the median cluster is 91 % one intent, but the largest (122 messages) is only 32 %; `qwen2.5:3b`'s names point to the majority intent for 58 % of clusters (42 of 73), many misses being right but broader; and with seven intents held out as unknown, a novelty flag at the 5th percentile of known-intent similarity reaches 53 % precision and 56 % recall, and clustering the flagged messages brings out all 7 unknown intents as clusters of their own.

### 4.0c Multimodal models: CLIP and vision-language models

**Problem.** Many inputs are not text: product photos, scanned invoices, screenshots, charts. Two families of models extend what the course built for text to images, and they do different jobs: **contrastive image-text encoders** (CLIP, SigLIP) give one vector per image and per text in a shared space, for classifying, searching and filtering; **vision-language models** (VLMs) generate text conditioned on images, for describing, answering questions and reading documents.

**Images as tokens: the Vision Transformer.** ViT (Dosovitskiy et al., 2021) cuts the image into fixed-size patches (16×16 or 32×32 pixels), flattens each one and projects it linearly to a vector (the **patch embedding**), adds position embeddings for the 2-D layout and runs the sequence through the same blocks as text ([section 1.2](#12-the-transformer-block-and-the-residual-stream)). A 224×224 image with 32×32 patches is $7 \times 7 = 49$ tokens, plus a class token whose final vector stands for the whole image. Without the inductive biases of a CNN, ViT needs a lot of data to train, which is exactly what web-scale image-text pairs provide.

**CLIP: a shared space learned from captions.** CLIP (Radford et al., 2021) trains an image encoder $f$ and a text encoder $g$ together on 400 million (image, caption) pairs, with the InfoNCE loss of [section 4.0](#40-embeddings-and-rerankers) applied in both directions: in a batch of $N$ pairs, image $i$ must be closer to caption $i$ than to the other $N-1$ captions, and caption $i$ closer to image $i$ than to the other images.

$$\mathcal{L} = \tfrac{1}{2}\big(\mathrm{CE}_{\text{image}\to\text{text}} + \mathrm{CE}_{\text{text}\to\text{image}}\big), \qquad s_{ij} = \frac{f(x_i)^\top g(t_j)}{\tau}$$

Nobody labels classes, yet the space supports **zero-shot classification**: write one sentence per class, embed the sentences and give each image the class of the most similar one. The wording is part of the model, as prompts are for LLMs. A bare class name is ambiguous ("boxer" the dog or the athlete) and unlike the captions CLIP saw; a template such as "a photo of a {}, a type of pet." fixes both, and **prompt ensembling**, averaging the text embeddings of several templates per class (CLIP's authors used 80), smooths out the quirks of any one wording. With a few labelled images per class, a **linear probe** (logistic regression on the frozen image embeddings, the image version of embeddings + classifier in [section 1.6](#16-encoders-and-encoder-decoders)) is the next step; whether it beats zero-shot depends on how well web captions already describe your categories.

**SigLIP.** The softmax in CLIP's loss normalizes over the whole batch, which needs very large batches and a costly all-gather across devices. SigLIP (Zhai et al., 2023) treats every (image, text) pair in the batch as an independent binary problem, matched or not:

$$\mathcal{L} = -\frac{1}{N}\sum_{i=1}^{N}\sum_{j=1}^{N} \log \sigma\big(z_{ij}\,(t\, \mathbf{x}_i^\top \mathbf{y}_j + b)\big), \qquad z_{ij} = \begin{cases}+1 & i = j\\ -1 & i \neq j\end{cases}$$

with a learned temperature $t$ and bias $b$. It trains better, also at smaller batches, and SigLIP encoders sit inside many current VLMs.

**VLMs: an image encoder wired into an LLM.** Two designs dominate:

| Design | How the image reaches the LLM | Examples | Cost |
|---|---|---|---|
| **projected visual tokens** | a ViT/SigLIP encoder produces patch vectors; a small **projector** (linear layer or MLP) maps them into the LLM's embedding space; they are inserted into the prompt as ordinary tokens | LLaVA (Liu et al., 2023), Qwen-VL, InternVL | image tokens take up context and prefill |
| **cross-attention** | extra cross-attention blocks inside the LLM's layers read the image features, as an encoder-decoder's decoder reads the encoder | Flamingo (Alayrac et al., 2022), Llama 3.2 Vision | images do not take up context; more new parameters |

The first design is the simpler and more common one. It is usually trained in two steps: align the projector on captions with the encoder and the LLM frozen, then instruction-tune on visual question answering.

**Dynamic resolution and what an image costs.** Early VLMs such as LLaVA resized every image to one fixed, small size, which blurs small text. Qwen2-VL and Qwen2.5-VL use **dynamic resolution**: the number of visual tokens grows with the number of pixels (one token per 28×28 area after merging neighbouring patches), and other models tile large images into crops. Detail is kept, and the bill grows with it: **an image is paid for in tokens**, which means prefill time, context length and, with a provider, money ([section 2.8b](#28b-serving-under-load)). How many tokens an image becomes depends on the serving stack as well as the model, so it has to be measured.

**Typographic attacks: prompt injection through pixels.** Goh et al. (2021) found that CLIP reads text in images so well that an apple labelled "iPod" is classified as an iPod. A VLM is an LLM reading its visual tokens, so text it can read in an image becomes part of its input and may be obeyed, like any untrusted text ([section 6.6](#66-safety-refusals-jailbreaks-and-guard-models)). A screenshot, a scanned document or a web page seen by a browsing agent can carry instructions. The defences are those for prompt injection: treat image content as data, do not give an agent that reads images secrets or dangerous tools, and test with adversarial images.

| Use | Family | Why |
|---|---|---|
| classification over a fixed label set, image search, deduplication, filtering | CLIP / SigLIP embeddings (+ a linear probe when labels exist) | milliseconds per image, a vector index, strong on common categories |
| captions, visual questions, open-ended answers, following instructions | VLM | generates text; no fixed label set |
| reading documents (invoices, forms, screenshots) | VLM with structured output, or OCR + layout model + LLM | fields a program can check; rule-based validation on top |

**Failure modes.** Fine-grained confusion between similar classes; hallucinated objects or fields; OCR errors on small or degraded text; detail lost to downscaling; text in the image taking over the answer; an evaluation set cleaner than production data.

**Key papers.** Dosovitskiy et al. (2021), *An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale*; Radford et al. (2021), *Learning Transferable Visual Models From Natural Language Supervision* (CLIP); Zhai et al. (2023), *Sigmoid Loss for Language Image Pre-Training* (SigLIP); Alayrac et al. (2022), *Flamingo: a Visual Language Model for Few-Shot Learning*; Liu et al. (2023), *Visual Instruction Tuning* (LLaVA); Bai et al. (2025), *Qwen2.5-VL Technical Report*; Goh et al. (2021), *Multimodal Neurons in Artificial Neural Networks*.

> **Notebook:** [22 · Multimodal models](../../notebooks/04_applications/22_multimodal.ipynb) (read after 17) · **In the course:** on 600 test images of Oxford-IIIT Pet (37 breeds), CLIP ViT-B/32 zero-shot reaches 80.0 % with bare breed names, 85.2 % with one template and 86.3 % with an ensemble of five, at 5.9 ms per image; a linear probe on its frozen embeddings needs all ~100 labelled images per breed to pass that (89.0 %; 79.5 % with 16, 34.3 % with one); `qwen2.5vl:3b` through Ollama, choosing from the 37 names, gets 72.0 % on 150 images, 11.3 points below CLIP on the same images, at 1.45 s and about 1,071 prompt tokens per image, and that count stays at ~1,080 whether the image is capped at 224, 448 or 896 px (Ollama resizes to a fixed budget), while capping at 224 px drops accuracy to 55 %; with a JSON schema it extracts all five fields of 40 synthetic invoices, clean and degraded, in about 2.3 s per document, a sign that the invoices are too easy; and with a wrong breed written in a corner of the image, CLIP follows the label 8 % of the time and the VLM 72 %.

### 4.1 Retrieval-augmented generation

**Problem.** A model's **parametric** knowledge is frozen, cannot be cited or deleted for one user, and is unreliable for rare facts: the model answers fluently whether or not it knows. **Non-parametric** knowledge (a document collection searched at query time) fixes that.

**Core idea.** Retrieve, then read. Roughly:

$$P(\text{correct answer}) \approx P(\text{evidence retrieved}) \times P(\text{correct answer} \mid \text{evidence in context})$$

The two factors fail for different reasons and are fixed with different tools (chunking, indexes and query processing for the first; prompting, model and context size for the second), so they are **evaluated separately**.

| Approach | Knowledge update | Citations | Access control |
|---|---|---|---|
| Rely on the model | retrain | no | no |
| Fine-tune on the documents | retrain; weak at injecting facts | no | no |
| RAG | re-index | yes | filter at retrieval time |
| Long context (paste everything) | edit the prompt | possible | per request |

The term comes from Lewis et al. (2020), who trained retriever and generator jointly; what the industry calls RAG today is simpler: an off-the-shelf embedding model, an index and an instruct LLM reading the passages, with no joint training. It won because each piece can be swapped and updated independently. Retrieval beats unsupervised fine-tuning for injecting facts (Ovadia et al., 2024); long context is a real alternative for small corpora, but every query pays for the whole context and models use the middle of it less reliably (*Lost in the Middle*, Liu et al., 2024).

**Sparse and dense retrieval.** **BM25** scores by the query terms present, weighted by rarity, with term-frequency saturation ($k_1$) and length normalization ($b$):

$$\text{BM25}(q, d) = \sum_{t \in q} \text{IDF}(t)\cdot\frac{f(t,d)\,(k_1+1)}{f(t,d) + k_1\,(1 - b + b\,|d|/\text{avgdl})}$$

It is exact on identifiers and needs no training, and it suffers from *vocabulary mismatch*. **Dense retrieval** (DPR, Karpukhin et al., 2020) uses a bi-encoder trained with the contrastive InfoNCE loss,

$$\mathcal{L} = -\log\frac{\exp(\mathrm{sim}(q, d^+)/\tau)}{\exp(\mathrm{sim}(q, d^+)/\tau) + \sum_j \exp(\mathrm{sim}(q, d^-_j)/\tau)}$$

which brings paraphrases together but blurs exact strings, names and numbers, and degrades out of domain (in BEIR, BM25 was a hard baseline). Only the bi-encoder lets documents be embedded ahead of time; the **cross-encoder** reads query and document together and is used to rerank the top 20 to 100; ColBERT (late interaction) sits in between; [section 4.0](#40-embeddings-and-rerankers) covers all three, and how they are trained, in detail.

**Hybrid with RRF, the industry default.** Because they fail on different queries, both are run and their rankings fused (not their scores, which live on incomparable scales):

$$\text{RRF}(d) = \sum_{r} w_r \cdot \frac{1}{c + \text{rank}_r(d)}, \qquad c = 60$$

RRF (Cormack et al., 2009) ignores magnitudes, needs no calibration and rewards agreement between retrievers. A tuned convex combination can beat it (Bruch et al., 2023), so "RRF always wins" is not settled; it is the safe default when you have no labels to tune with.

**Chunking.** The underlying tension: retrieval wants small single-topic units, generation wants enough context. Fixed-size, recursive by separators, structure-aware (headings), semantic, *small-to-big* (match the small chunk, show the parent section) or *contextual chunks* (prepend an LLM-written description). Chunk size is a hyperparameter you measure. To scale, approximate indexes (HNSW, IVF, product quantization), whose recall is a quality knob independent of the embedding model.

**Query transformations.** Multi-query/RAG-Fusion, HyDE (embed a hypothetical answer), query2doc, step-back, decomposition: all spend LLM calls before retrieval to attack vocabulary mismatch, and all risk added latency and *drift* in meaning.

**Failure modes.**

| Symptom | Likely cause | Fix |
|---|---|---|
| the answer is in the docs but not retrieved | badly sized chunks, wording mismatch | measure sizes, hybrid search, multi-query/HyDE |
| retrieved but ignored by the model | too much context, small model | fewer, better chunks, a reranker, require citations |
| facts needed from several places | single-shot retrieval | iterative or agentic retrieval |
| wrong on names, codes, versions | dense-only retrieval | add BM25 |

### 4.2 Retrieval metrics

For each query, $\text{rank}_q$ is the position of the first relevant result and $\text{rel}_q(i) \in \lbrace 0,1 \rbrace$:

$$\text{hit@}k = \frac{1}{|Q|}\sum_q \mathbb{1}[\text{rank}_q \le k], \qquad \text{MRR} = \frac{1}{|Q|}\sum_q \frac{1}{\text{rank}_q}$$

$$\text{nDCG@}k = \frac{\text{DCG@}k}{\text{IDCG@}k}, \qquad \text{DCG@}k = \sum_{i=1}^{k} \frac{\text{rel}_q(i)}{\log_2(i+1)}$$

hit@k equals recall@k when there is one relevant item. MRR rewards putting the evidence *first*, which matters in RAG because the model reads the top chunks most reliably and a smaller $k$ means a cheaper prompt. nDCG generalizes to several items with graded relevance (the BEIR standard). **File-level** evaluation saturates quickly; **evidence-level** evaluation (does the chunk contain the string that answers?) is stricter and closer to what the LLM needs. Labelling with literal strings is cheap and transparent but misses paraphrases; the alternative is an LLM judge (next section).

> **Notebook:** [18 · RAG with LangChain and RAGAS](../../notebooks/04_applications/18_rag.ipynb) · **In the course:** on the course's own docs, every retriever finds the right file (hit@4 = 1.00); at evidence level hybrid is best at the top of the ranking (hit@1 0.80, MRR 0.84 vs 0.73 for dense), BM25 reaches hit@4 0.95, multi-query does not help (MRR 0.82 for one extra call), and dense gets worse with bigger chunks (MRR 0.77 → 0.67).

### 4.3 RAGAS and LLM-as-judge

**Problem.** Good retrieval is necessary, not sufficient: the model can ignore the context, add facts or answer a different question. And free-text quality has no cheap deterministic metric.

**The RAGAS idea** (Es et al., 2023): break "is this answer good?" into checks a judge can perform reliably one at a time.

$$\text{faithfulness} = \frac{\left|\lbrace s \in S(a) : s \text{ inferable from } c_{1..K} \rbrace\right|}{|S(a)|}, \qquad \text{context recall} = \frac{\left|\lbrace s \in S(g) : s \text{ attributable to } c_{1..K} \rbrace\right|}{|S(g)|}$$

Faithfulness measures groundedness, not correctness (faithfully repeating a wrong passage scores 1). Answer relevancy generates questions from the answer and measures their similarity to the real one (and scores 0 if the answer is noncommittal). Context precision is an average precision over the ranking, like MRR but with a judge deciding relevance. Together they localize a failure: low recall points to retrieval; high recall with low faithfulness, to the generator. Faithfulness and relevancy need no reference (usable on live traffic); precision and recall do.

**Why the judge works.** Grading is usually easier than generating. Zheng et al. (2023) found strong judges agreeing with humans about as often as humans agree with each other, and documented biases:

| Bias | What happens | Mitigation |
|---|---|---|
| position | in comparisons, the first (or second) option wins | swap the order and average |
| verbosity | longer looks better | a rubric that rules out length |
| self-preference | favours its own model family | a judge from another family, or a panel |
| limited capability | accepts answers it could not produce | give it a reference; narrow rubric |
| prompt sensitivity | small wording changes move scores | freeze and version the judge prompt |
| scale drift | 1-10 scales used inconsistently | binary verdicts; G-Eval weights by token probabilities |

**The working rule:** an LLM judge is a measurement instrument that must be **validated**. Label a few dozen cases by hand and measure agreement with Cohen's kappa, $\kappa = (p_o - p_e)/(1 - p_e)$, which corrects for chance agreement; freeze judge and prompt while comparing versions; read differences more than absolute values; put the reasoning *before* the verdict in the structured output. A panel of small judges from different families can match one large judge (Verga et al., 2024); ARES corrects small judges with a human-labelled set through prediction-powered inference.

**Judge strength.** A judge cannot verify what it does not understand, and a weak judge does not only add noise: it can *invent* effects, differences between systems that are not there, because its errors are not spread evenly across configurations (a shorter answer, a different format or a different context is enough to make it fail more on one than on the other). Three practical consequences. The judge should be clearly stronger than the system under test and, ideally, not the same model that generates (which also brings self-preference). Every comparison gets an interval (paired bootstrap, section [6.5](#65-statistics-of-small-evaluation-sets)): with 20 questions, a 0.1 difference on a RAGAS metric often fits inside the noise. And changing the judge changes the measuring instrument: numbers from two different judges are not comparable, only differences measured with the same judge. A stronger judge costs time, so keep it for a fixed, small, representative set.

> **Notebook:** [18 · RAG with LangChain and RAGAS](../../notebooks/04_applications/18_rag.ipynb) and [28 · Observability and evaluation](../../notebooks/06_llmops/28_observability_llmops.ipynb) · **In the course:** with `qwen2.5:7b` as the judge (and `qwen2.5:3b` generating), the only clear difference between dense and hybrid retrieval is context precision, 0.79 → 0.92 (+0.13, 95 % CI [+0.03, +0.25]); faithfulness (0.87 → 0.87), answer relevancy (0.78 → 0.75) and context recall (0.95 → 0.95) do not move beyond their intervals, and scoring takes 1038 s. When the 3B model judged itself it reported faithfulness 0.38 → 0.54 and relevancy 0.44 → 0.30: effects the 7B judge does not confirm. In notebook 28 (still judged by the 3B model) the judge agrees with a deterministic check on 20 of 24 answers, and the disagreements show what each one misses.

### 4.4 Application patterns

**What they replaced.** Before instruction-tuned LLMs, each task had its own pipeline: NER plus regex for extraction, a classifier trained on thousands of tickets, an extractive summarizer, a keyword blocklist. In-context learning (GPT-3) and instruction tuning (FLAN, InstructGPT) let one model do all of that from a description. The pattern today: write the task as a prompt, constrain the output to a schema, measure on a small labelled set, and train a dedicated model only when volume or latency justifies it.

**Structured output.** A Pydantic model with good field descriptions, plus constrained decoding (section 5.2), removes parsing code and every invalid label. What remains are **semantic** errors that look valid: one wrong field in perfect JSON. A detail that changes results: depending on the method, the model may never *read* the field descriptions (if the schema only goes to the decoder), so repeat the rules in the system message.

**Few-shot.** Examples do not update weights; they are conditioning. Min et al. (2022) found that randomizing their labels hurt less than expected: format, label space and input distribution carry much of the benefit. Zhao et al. (2021) documented majority-label, recency and common-token biases. So the choice, balance and order of examples change predictions: examples are a prompt change, evaluated like any other. Alternatives for classification: embeddings plus logistic regression, or a small fine-tuned encoder (SetFit), once volume and label stability justify it.

**Map-reduce for long inputs.** Many servers **silently truncate** a prompt that exceeds the window. And even when it fits, quality drops before the limit (*Lost in the Middle*).

| Strategy | Calls | Parallel | Main risk |
|---|---|---|---|
| stuff (all in one prompt) | 1 | — | silent truncation, lost in the middle |
| map-reduce | $n+1$ (or a tree) | yes | each chunk summarized out of context; the reduce drops topics |
| refine (incremental) | $n$ | no | later content overwrites earlier content |

There is no clear winner on long texts (BooookScore, Chang et al., 2024); map-reduce wins on wall time and robustness when chunks are independent. Without a reference, evaluating a summary means measuring **coverage** (does it mention what it should?) and **faithfulness** (does it say only what the source says?).

**Caching.** Three different things share the name:

| Layer | What is reused | Key | Risk |
|---|---|---|---|
| exact response cache | the answer | prompt + model settings | stale answers; keys that omit settings |
| semantic cache | the answer | the prompt's embedding under a threshold | answering a different question (false hit) |
| prefix cache | the KV-cache of a prefix | the token prefix | none: the output is computed normally |

With hit rate $h$, $\mathbb{E}[t] = t_{\text{lookup}} + (1-h)\,t_{\text{model}}$. An exact cache pays at any $h$; a semantic one costs an embedding on *every* request and its threshold is a binary classifier with the usual precision-recall trade-off. Embeddings place same-topic texts close together, so two questions that differ by one word (weekdays/weekends, pro/basic plan) are close by construction: negation, numbers and names are the weak spot.

**Prompt injection.** There is no separate channel for data: SQL injection was solved by parameterized queries that send code and data separately; an LLM receives one token sequence and is trained to follow instructions wherever they appear. *Indirect* injection (inside retrieved content; Greshake et al., 2023) is the dangerous case for RAG and agents; it is LLM01 in the OWASP Top 10 for LLM applications.

| Layer | Examples | Limit |
|---|---|---|
| prompt design | delimiters, "data, not instructions", sandwich, spotlighting | a convention the model may ignore |
| model training | instruction hierarchy, StruQ | requires control of training |
| detection | LLM classifiers, Prompt Guard, Llama Guard | adaptive attackers, false positives |
| system design | least privilege, human confirmation, dual-LLM, CaMeL | constrains what the app can do |

The consensus: the first three layers reduce the probability of success; only the last bounds the consequences. The layers multiply, which is why none is optional.

**Batch economics.** $\text{cost} \approx N\,(t_{\text{in}}\,p_{\text{in}} + t_{\text{out}}\,p_{\text{out}})$; output tokens cost several times more because decode is sequential. Little's law, $L = \lambda W$, explains when raising concurrency raises throughput: only if the server can take more requests without each one taking proportionally longer (a hosted API, or an engine with continuous batching). A serious job needs retries with exponential backoff, `return_exceptions=True`, idempotent bookkeeping and, when latency does not matter, discounted asynchronous batch APIs.

> **Notebook:** [19 · Application patterns](../../notebooks/04_applications/19_modern_llm_applications.ipynb) · **In the course:** extraction with `json_schema` and no rules in the prompt gets every field right in 5 of 8 records, and 7 of 8 with the rules repeated; few-shot adds nothing to classification (20/20 both ways); "stuff" processed only ~2,050 of ~4,100 tokens with no error and covered 9 of 13 topics, map-reduce 11 of 13; the semantic cache needs a 0.94 threshold for zero false hits and then catches only 2 of 10 paraphrases; injection succeeds 9/10 with a naive prompt, 10/10 with delimiters alone and 4/10 with the sandwich, and an LLM classifier catches 10/10 with 0/10 false positives on its set.

### 4.4b Prompts as programs: DSPy and prompt optimization

**Problem.** Every prompt in the previous sections is written and tuned by hand. That is slow, the result is tied to one model (a prompt tuned for one model often transfers badly to the next), the wording is mixed with the parsing code, and nobody can say whether a better prompt exists. With several LM steps in a pipeline, tuning each one by hand against an end-to-end metric does not scale.

**Core idea: separate what a step does from how it is prompted.** DSPy (Khattab et al., 2023) treats the prompt as a *parameter* of a program, the way weights are parameters of a network:

- a **signature** declares what a step does as typed inputs and outputs (`message: str -> intent: Literal[...]`), with an optional docstring as the task description and nothing about wording;
- a **module** decides how to prompt for it: `Predict` asks for the outputs directly, `ChainOfThought` adds a `reasoning` field before them, `ReAct` runs a tool loop. Modules compose like PyTorch layers, so retrieve → rerank → answer is one program with several predictors;
- an **adapter** writes the actual prompt for a given model and parses the answer. The default `ChatAdapter` puts the field names, types and allowed values in the system message, asks for delimited fields and type-checks what comes back. Changing the model or the adapter changes the prompt without touching the program;
- each predictor holds an **instruction** and a list of **demos** (input/output examples placed in the prompt as earlier turns). Those are what optimizers change.

**Optimizers.** Given a program, a metric and a few hundred labelled examples, an optimizer searches for the instruction and demos that maximize the metric. From cheapest to most expensive:

| Optimizer | What it changes | How |
|---|---|---|
| `LabeledFewShot` | demos | $k$ random labelled examples, the same for every input |
| `KNNFewShot` | demos, **per input** | the $k$ training examples nearest to the input in an embedding space (Liu et al., 2021) |
| `BootstrapFewShot(WithRandomSearch)` | demos | runs the program on training data and keeps the traces that pass the metric, so demos can include intermediate steps such as reasoning; random search over several demo sets, scored on a validation set |
| `MIPROv2` | instruction and demos | a *prompt model* proposes instructions from a summary of the data and the program; Bayesian optimization (TPE; Bergstra et al., 2011) picks the instruction × demo combination, scored on minibatches (Opsahl-Ong et al., 2024) |
| `GEPA` | instruction | a *reflection model* reads failed runs with textual feedback from the metric and rewrites the instruction, keeping a Pareto front of candidates that win on different examples (Agrawal et al., 2025) |

They descend from automatic prompt engineering, where an LLM proposes prompts and a score selects them (APE, Zhou et al., 2023; OPRO, Yang et al., 2024). The output of an optimizer is data, instructions and demos, saved as JSON: an optimized prompt is a versioned artifact, reviewed like code and regenerated (not edited by hand) when the model, the data or the metric changes.

**The metric is the specification.** An optimizer maximizes exactly what the metric measures, on exactly the data it is given. Validation sets for prompt search are small, so the selection overfits: with 77 examples one answer is 1.3 points ([section 6.5](#65-statistics-of-small-evaluation-sets)), and the best of many candidates scored on such a set is optimistically biased: part of why it won is noise that will not repeat on new data. A better validation score is a hypothesis until a held-out test confirms it. Reflective optimizers add a risk of their own: an instruction written from a handful of failures tends to memorize them.

**Why the optimizers did not help on a 77-way lookup.** An optimizer searches *within* the space you give it, here one instruction and a few static demos. On a task like Banking77, the hard part is knowing which of 77 similar-sounding labels the bank uses for *this* message. The missing ingredient is per-input information (which labelled training messages look like this one), and no instruction can supply it, while a few static demos cover a few intents and pull the model toward those labels. Retrieving the nearest labelled examples per input does supply it. With thousands of labelled examples and a good embedding model, the task is mostly lookup, so a $k$-NN vote with no LLM at all is the baseline to beat ([section 4.0](#40-embeddings-and-rerankers)). Chain of thought does not help either: reasoning helps when the answer has to be *derived*, not *recalled*. Optimizers pay off on multi-step programs with a stable task description and real room for improvement in how each step is phrased.

**Cascades and routers.** When a cheap component is usually confident and usually right, let it answer and send only the rest to the LLM: the cascade pattern (FrugalGPT, Chen, Zaharia and Zou, 2023; [section 2.8b](#28b-serving-under-load)). Here the cheap component is the $k$-NN vote, and its agreement (how many of the $k$ neighbours share the majority label) is the confidence signal. The routing threshold is a hyperparameter, chosen on validation and never on the test, and the result is judged on accuracy, latency and LLM calls per request together.

**Production uses.** Declarative programs pay off when the pipeline has several LM steps, the model may change, and there is a metric with a few hundred labelled examples: re-optimizing for a new model is cheaper than rewriting prompts. Before optimizing the wording, fix the information flow (retrieve examples or documents per request) and measure a non-LLM baseline. Compare variants paired on the same examples. Optimization is paid once (hundreds of calls), inference forever.

**Failure modes.** A validation set too small to separate the candidates, or overlapping the training examples (a training message that retrieves itself as a demo inflates the score); a weak baseline; chain of thought by default; trusting the long instruction a reflective optimizer wrote; random static demos on a task with many labels; silent output-format changes when switching models (count the answers that do not parse).

**Key papers.** Khattab et al. (2023), *DSPy: Compiling Declarative Language Model Calls into Self-Improving Pipelines*; Opsahl-Ong et al. (2024), *Optimizing Instructions and Demonstrations for Multi-Stage Language Model Programs* (MIPROv2); Agrawal et al. (2025), *GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning*; Liu et al. (2021), *What Makes Good In-Context Examples for GPT-3?*; Zhou et al. (2023), *Large Language Models Are Human-Level Prompt Engineers* (APE); Yang et al. (2024), *Large Language Models as Optimizers* (OPRO); Bergstra et al. (2011), *Algorithms for Hyper-Parameter Optimization* (TPE); Chen, Zaharia and Zou (2023), *FrugalGPT: How to Use Large Language Models While Reducing Cost and Improving Performance*.

> **Notebook:** [20 · Prompt optimization with DSPy](../../notebooks/04_applications/20_dspy_prompt_optimization.ipynb) (read after 19) · **In the course:** on notebook 06's 231-message Banking77 sample with `qwen2.5:3b`, DSPy's zero-shot `Predict` reaches 0.558 [0.489, 0.623] against 0.489 for notebook 06's hand-written prompt, with no unparsable answer, and chain of thought 0.541 at more than twice the latency; 16 random demos drop it to 0.433, while the 4, 8 and 16 nearest training messages as demos lift it to 0.818, 0.848 and 0.870; the majority vote of the 8 nearest neighbours, with no LLM, reaches 0.913 at 0.02 s per message; BootstrapFewShotWithRandomSearch (0.558), MIPROv2 (0.554) and GEPA (0.554) stay within noise of the starting point after 5–9 minutes and 650–870 LM calls each, and GEPA's instruction scored 59.7 % on validation; GEPA on top of the kNN program kept the original instruction (0.870); a router that answers with the vote when at least 4 of the 8 neighbours agree (threshold chosen on validation) reaches 0.922 at 0.03 s per message, with 97 % of the messages never reaching the LLM, against 0.926 at 1.62 s for `qwen2.5:7b` with 16 kNN demos; and the optimized program is 2.4 KB of JSON that gives the same prediction on 30 of 30 messages after reloading.

### 4.5 Recommender systems

**Problem.** For a user $u$ in a context $c$, estimate which item $i$ is the "right" next thing (watched, played, bought, enjoyed) and show the top few. Three facts shape every real system. **Scale**: catalogs of millions of items and answers within tens of milliseconds, so an expensive model cannot run on every item. **Implicit, biased feedback**: we mostly see what people clicked or watched, not what they disliked, and only among what the system already showed them (exposure bias); popular items get more exposure and therefore more data (popularity bias). **The world changes**: new items arrive every day with no history (*cold start*), and tastes drift.

**The funnel.** The answer to scale, popularised by YouTube's paper (Covington, Adams and Sargin, 2016), is to split the problem into stages with different goals:

| Stage | Input → output | What it must do | Its own metric | Typical models |
|---|---|---|---|---|
| candidate generation | catalog → hundreds | not miss good items; cheap per item | recall@K of the candidate set | two-tower + ANN, sequential model, item-item co-occurrence, content, popular lists |
| ranking | hundreds → tens | order precisely; can afford many features | nDCG@10, hit@10 of the final list | boosted trees (LambdaMART), deep models (Wide & Deep, DCN, DLRM) |
| re-ranking | tens → the page | rules, diversity, exploration, explanations | what it trades: accuracy against coverage or diversity | MMR, calibration, bandits, business rules |

Each candidate source sees the user differently, so systems take the **union** of several. The most useful consequence for debugging: the ranker can only order what retrieval brought, and the recall of the candidate set is a ceiling no ranker can exceed. Improving retrieval and improving ranking are different projects, and they are measured separately.

**Sequential recommendation as language modelling.** If a user's history is a sentence whose words are item ids, predicting the next item is predicting the next token. SASRec (Kang and McAuley, 2018) is exactly that: a decoder-only transformer with causal attention whose vocabulary is the catalog; in the course, notebook 03's `TransformerLM` with `vocab_size = n_items + 1`. It works because taste is sequential (franchises, binges of one genre, moods), and attention weighs the history like a learned, context-dependent recency. The industry is heading this way: Meta's HSTU (Zhai et al., 2024) scales like an LLM, and Netflix describes an autoregressive foundation model over each member's interaction sequence (2025).

The training detail that matters most is the loss. The original SASRec used a binary loss with one sampled negative per positive. The **full softmax** over the catalog, the language-model loss,

$$\mathcal{L} = -\log \frac{\exp(s_{u,i^+})}{\sum_{j \in \mathcal{I}} \exp(s_{u,j})},$$

is much stronger with the same architecture: Klenitskiy and Vasilev (2023) go from hit@10 0.250 to 0.315 on MovieLens-1M. With a few thousand items it is cheap; with millions the denominator does not fit, and you use a **sampled softmax**: the denominator is estimated from a subset of negatives, often the other users' positives in the batch (*in-batch negatives*). But those negatives are not uniform: a popular item shows up as a negative far more often, and the model learns to under-score it. The **logQ correction** (Yi et al., 2019) subtracts from each logit the log of the probability that the item is sampled, $s_{u,j} - \log Q(j)$, which pulls the estimator back towards the full softmax.

**Two-tower and approximate search.** Two networks map the user and the item into the same space, and the score is the dot product, $s(u, i) = \langle f(u), g(i) \rangle$. Because the item tower does not depend on the user, every item vector is precomputed and put in an approximate nearest-neighbour (ANN) index: serving is computing one user vector and looking up the closest items, milliseconds over millions of items. The usual indexes are graphs (**HNSW**), partitions (IVF) and compression (product quantization), with FAISS or ScaNN; their recall against exact search is a quality knob of its own, independent of how good the model is. The design limit: a dot product cannot model fine interactions between user and item features (that is the ranker's job). For *cold start*, the item tower can use content (text, genre, year); if the id is also hidden at random during training (*id dropout*, as in DropoutNet, Volkovs et al., 2017), the tower learns to place an item from its content alone.

**Learning to rank.** The ranker scores each (user, candidate) pair from features and sorts. There are three ways to train it:

| Approach | Loss | Example |
|---|---|---|
| pointwise | predict each label independently (log-loss on "was this the next one?") | logistic regression, a binary GBDT |
| pairwise | prefer the positive to each negative: $\log(1 + e^{-(s_+ - s_-)})$ | RankNet |
| listwise | optimise the quality of the whole ordered list | LambdaMART, softmax / XE-NDCG |

**LambdaMART** (Burges, 2010) is gradient-boosted trees whose gradients are RankNet's pairwise gradients multiplied by how much nDCG would change if the two items were swapped:

$$\lambda_{ij} = \frac{-\sigma}{1 + e^{\sigma (s_i - s_j)}}\,\bigl|\Delta \text{nDCG}_{ij}\bigr|,$$

so the trees concentrate on the top of the list. It has been the default industrial ranker for over a decade (search engines, e-commerce, Kaggle recommender competitions) because trees take heterogeneous features (scores, counts, categories, ratios) without scaling, train in minutes and are easy to inspect; in LightGBM it is `objective="lambdarank"`. Two conditions decide whether a second-stage ranker adds anything over a strong first stage. **Many label groups from a period after** what the retrieval models saw: trained on a single positive per user, the best feature is the first stage's own score and the trees relearn "sort by SASRec" plus noise. The recipe of RecTools and of the top Kaggle OTTO solutions is to fit retrieval on the history minus the last $R$ events and turn each of those events into a ranker query. And **features the first stage cannot see**: explicit ratings, demographics, co-visitation, time since the previous event, recent trends. Without them the ranker is only cost.

**Cold start and exploration.** Nothing collaborative can recommend an item nobody has seen: its embedding row was never trained. Two different things solve it. **Content** places the new item (embedded text and metadata, towers with id dropout). **Exploration** gives it exposure so the system learns what it is worth, which is the exploration/exploitation dilemma: a system that only exploits never learns about anything new. The production tool is **bandits**: $\varepsilon$-greedy (explore at random a fraction of the time), UCB (score by the mean plus an uncertainty bonus, $\hat\mu_i + c\sqrt{\ln t / n_i}$) or Thompson sampling (sample from each item's posterior), and their contextual versions (LinUCB; BaRT on Spotify's home page, McInerney et al., 2018). The minimal version you can measure offline is a fixed **exploration slot**: the top positions for the ranking and one reserved for the best new item according to a content-aware model. Offline you only see its cost on known items and its reach on new ones; the benefit of learning shows online.

**Diversity, MMR and calibration.** A list of ten near-identical items is accurate and boring. **MMR** (Carbonell and Goldstein, 1998) builds the list greedily, each time choosing

$$\arg\max_{i \notin S} \; \lambda\,\text{rel}(i) - (1 - \lambda) \max_{j \in S} \text{sim}(i, j),$$

with $\lambda = 1$ the plain ranking and lower $\lambda$ more diversity. **Intra-list diversity** is measured as one minus the mean pairwise similarity of the list, $\text{ILD} = 1 - \frac{2}{k(k-1)} \sum_{i<j} \text{sim}(i, j)$. **Calibration** (Steck, 2018) asks for something else: that the genre mix of the list $q$ resembles the history's $p$, penalising $\mathrm{KL}(p \,\|\, q)$, so that someone who watches 70 % drama and 30 % comedy does not get ten dramas. A caveat: hit@10 rewards predicting *the* next item, so any change to the list can only look neutral or worse offline; the benefits (less fatigue, discovery, retention) show up in A/B tests over weeks, which is why this knob is tuned online.

**Offline evaluation protocol.** Four decisions separate an honest evaluation from a misleading one:

| Decision | Honest choice | What goes wrong otherwise |
|---|---|---|
| split | temporal: per-user *leave-last-out* (the last event is test, the one before validation) or a global time cut | a random split leaks the future into training |
| candidates for the metric | **full ranking** over the whole catalog | sampled metrics (the target against 100 random negatives) inflate the numbers and can flip which model looks best (Krichene and Rendle, 2020) |
| aggregation | by segment: known versus new items, users with short or long histories | a global average hides cold start completely |
| comparison | **paired bootstrap** over users, 95 % interval of the difference | one or two points of hit@10 with a few thousand users can be noise |

With a single relevant item, hit@k is the fraction of users whose target is in the top $k$, and nDCG@k is $1/\log_2(p+1)$ for a hit at position $p$ (1 at the top, 0.5 third, 0.29 tenth). A useful working rule: **a stage only enters the system if it beats the previous one on validation**, with an interval that excludes zero; test is looked at once, at the end. And offline predicts online only imperfectly: the final decision is an A/B test.

**Where LLMs help and where they do not.** Published evidence and this course point the same way:

| Use of the LLM | Verdict | Why |
|---|---|---|
| re-ordering a strong collaborative model's recommendations | not worth it with small models | the LLM does not see the co-watching signal the model encodes, only short descriptions; it costs seconds per user against milliseconds |
| enriching items offline (keywords, descriptions; KAR, Xi et al., 2023; LLMRec, Wei et al., 2024) | depends; measure it | works as a separate feature in trained models and with large LLMs; mixed into the item text it can dilute the signal |
| understanding natural-language requests (structured output → filters) | clearly helps | date ranges, negations and hard constraints are exactly what an embedding cannot enforce |
| explaining ("because you watched X and Y") | helps when grounded | with the history items that drove the recommendation it is checkable; without anchors it invents attributes |

The resulting division of labour: the LLM understands *what* is asked and talks about it; the user's history and the collaborative models decide *which*. The research line that tries to merge the two is generative retrieval with *semantic IDs* (TIGER, Rajput et al., 2023), where the model generates the item's identifier token by token.

> **Notebook:** [23 · Recommender systems](../../notebooks/04_applications/23_recommender_systems.ipynb) · **In the course:** on MovieLens-1M with full ranking, SASRec trained with the full softmax reaches hit@10 0.293 (ItemKNN 0.125, popularity 0.036) and never hits a new movie; the union of five sources retrieves the target for 68.4 % of users; LambdaMART with the RecTools/OTTO recipe lifts it to 0.307 (+1.4 points, 95 % CI [+0.7, +2.1]) after passing the validation gate the first attempt failed; one exploration slot takes hit@10 on new movies from 0.009 to 0.044 for 0.7 points overall; MMR at $\lambda = 0.8$ raises diversity from 0.659 to 0.719 for 0.4 points; the LLM keywords hurt content retrieval and add nothing as a ranker feature, but the LLM that turns requests into filters satisfies the constraints in 87 % of the top 5, against 60 % for embedding search.

### 4.5b Drift, online learning and continual updates

**Problem.** A model is trained on the past and used on the future. When the world moves (new users, new products, new questions, changed tastes) quality decays, usually silently: nothing raises an error, results just get a little worse every week. Two questions follow: how to keep the model up to date, and how to notice that it needs updating, often before labels arrive.

**What can drift.** Write the data as inputs $x$ and outcomes $y$, with $P(x, y) = P(x)\,P(y \mid x)$:

| Kind | What changes | Example |
|---|---|---|
| **covariate shift** | $P(x)$, with $P(y \mid x)$ fixed | users from a new country; a new kind of question to a chatbot |
| **label (prior) shift** | $P(y)$ | a blockbuster takes a large share of all views; a fraud wave |
| **concept drift** | $P(y \mid x)$ | the same history now leads to different choices, because tastes or the catalogue changed |

Drift can be **sudden** (a launch, an outage, a policy change), **gradual** or **recurring** (weekdays, seasons). Some of it is made by the model itself: a recommender decides what users see and therefore what they click, so the next training set reflects the last model's choices. That **feedback loop** narrows exposure and inflates the model's apparent accuracy unless exploration ([section 4.5](#45-recommender-systems)) or logged propensities break it.

**Keeping models fresh.** The options trade freshness against cost and risk:

- **periodic retraining** on all data so far (stable, slow to react) or on a **sliding window** that forgets the old (reacts faster, loses rare long-term patterns);
- **warm starting**: continue training from the last model on the new data, much cheaper than starting over, standard for large models;
- **online (incremental) learning**: update after every event or mini-batch, by SGD or with counts that decay exponentially ($c \leftarrow \gamma^{\Delta t} c + 1$) so recent events weigh more. It is the freshest option, but bad data goes live at once, and neural networks can suffer **catastrophic forgetting** of what they knew. This is the field of **continual learning** (Gama et al., 2014; Lu et al., 2018).

Whatever the strategy, a new model is a deployment: it is evaluated offline on the most recent data and rolled out gradually (a **canary** on a small share of traffic, then an A/B test), with the previous model ready for rollback.

**Evaluating on a stream: prequential.** A random train/test split leaks the future into training. On a stream, the honest protocol is **prequential** ("test-then-train", Dawid, 1984; Gama et al., 2014): each event is first predicted with the model as it is, and only then used for learning. Every prediction is out of sample, and plotting the score over time shows the model ageing, and how much each update strategy holds back the decay.

**Detecting drift without labels.** Labels often arrive late (a purchase, a churn) or never, so monitoring watches **inputs** and **outputs** as well:

- **PSI** (population stability index) between a reference distribution $q$ and the current one $p$ over bins $b$:

$$\mathrm{PSI} = \sum_b (p_b - q_b)\,\ln\frac{p_b}{q_b}$$

  a symmetrized KL divergence; the industry rule of thumb is < 0.1 stable, 0.1–0.25 moderate, > 0.25 a major shift. The **Kolmogorov-Smirnov** test (maximum gap between two empirical CDFs) and the chi-squared test do the same per feature with a p-value; **KL** and **Jensen-Shannon** divergences measure the gap between distributions directly.
- **For embeddings** (text, images), feature-by-feature tests do not work: there are hundreds of dimensions and no single one means anything. Three tests compare the two clouds of vectors. The **centroid shift** is the distance between their means: cheap, but blind to changes that keep the mean. The **maximum mean discrepancy** (MMD; Gretton et al., 2012) compares the samples through a kernel,

$$\mathrm{MMD}^2 = \mathbb{E}[k(x, x')] + \mathbb{E}[k(y, y')] - 2\,\mathbb{E}[k(x, y)]$$

  with a p-value from a permutation test (shuffle which window each point came from). The **classifier two-sample test** (Lopez-Paz and Oquab, 2017) trains a classifier to tell reference from current data: a cross-validated AUC well above 0.5 means they differ, and the classifier's errors show *where*. Rabanser, Günnemann and Lipton (2019) compared these on many shifts; reducing dimensions first and using two-sample tests worked best.

**What to monitor.** Three levels, from cheapest to most meaningful: **inputs** (feature and embedding drift, the share of never-seen items, users or intents), **outputs** (prediction distribution, confidence, refusal and fallback rates) and **outcomes** (online quality from delayed labels: clicks, purchases, ratings, task success, feedback). Only the last measures what you care about; input drift is an alarm to investigate, not proof of harm, and concept drift can happen with no input drift at all. Per-item novelty flags (as in [section 4.0b](#40b-clustering-and-topic-modeling)) catch a new topic long before it moves a whole window's distribution. Evidently, NannyML, WhyLabs and Arize implement these checks; for LLM applications they sit next to tracing ([section 6.1](#61-traces-spans-and-opentelemetry-genai)).

**For LLM applications.** The weights of a pinned model do not change, but everything around them does: what users ask, the documents in the RAG index, the tools' APIs, and the provider's model behind an API name. Retraining is replaced by re-running the evaluation set on every change, refreshing the index and the evaluation set from production traffic, and tracking new-intent clusters.

**Failure modes.** Watching only input distributions and missing concept drift; alarms nobody owns; retraining automatically on corrupted or poisoned data; feedback loops; windows too small to detect anything; silent upstream changes (a renamed field, a new tokenizer, a new model version behind the same API name).

**Key papers.** Gama, Žliobaitė, Bifet, Pechenizkiy and Bouchachia (2014), *A Survey on Concept Drift Adaptation*; Lu et al. (2018), *Learning under Concept Drift: A Review*; Gretton et al. (2012), *A Kernel Two-Sample Test*; Lopez-Paz and Oquab (2017), *Revisiting Classifier Two-Sample Tests*; Rabanser, Günnemann and Lipton (2019), *Failing Loudly: An Empirical Study of Methods for Detecting Dataset Shift*; Harper and Konstan (2015), *The MovieLens Datasets: History and Context*.

> **Notebook:** [24 · Online learning and drift](../../notebooks/04_applications/24_online_learning_drift.ipynb) (read after 23) · **In the course:** on MovieLens-1M replayed as 22 weekly windows after a training cut-off of 1 August 2000, a frozen co-visitation recommender falls from hit@10 0.184 to 0.160 while the weekly-retrained one holds at about 0.20 (0.207 against 0.175 over all weeks, +18 %); popularity barely ages, and its four update strategies (frozen, weekly, 4-week window, online with decay) end within 0.004 of each other; PSI on the item distribution stays at 0.01–0.03 every week, so the decay is concept drift that input monitoring misses, and a PSI > 0.25 retraining trigger never fires; on Banking77 with bge-small embeddings, all three detectors keep false alarms at 5 %, but with 400 messages per window a shift where 20 % of messages come from new intents is caught only 25–30 % of the time, while at 40 % MMD and the classifier test catch it every time.

---

## 5. Agents and MCP

Terms for this part: [glossary, Agents and LangGraph](glossary.md#agents-and-langgraph) and [MCP](glossary.md#mcp).

### 5.1 Tool calling

**Problem.** A pretrained model has four limits no prompt removes: a knowledge cut-off, no access to your private data, unreliable exact symbolic work (arithmetic, dates, counting), and no ability to *act*. Tools address all four, provided the model can express a request a program can execute. The real difficulty is the bridge: the model emits tokens, a program needs a function name and typed arguments.

**Core idea.** Each tool is described by a name, a description and a **JSON Schema** for its arguments. The model replies with text or with a structured call `{name, args, id}`; your code validates, runs, and returns a tool message carrying the same `id`. The model never executes anything. The schema plays two roles: *documentation* for the model (the only thing it knows about the tool) and a *validator* for your code. That is why a vague description produces bad calls just as a vague prompt produces bad answers.

**What it replaced.** MRKL and ReAct (2022) asked the model for `Action:` lines parsed with string matching, which broke whenever the model added a word. Toolformer and Gorilla trained API use. Since 2023 the industry has used **native** tool calling: the format is part of the model's training (Qwen2.5 writes calls as JSON inside `<tool_call>` tags) and the serving layer turns it into structured data. Training teaches four skills that fail independently: **deciding** whether a tool is needed, **selecting** which one, **filling** the arguments (classic slot filling) and **using** the result.

**An agent is a loop with a cost structure.** The model is a stateless function from a list of messages to a new message, so every call re-sends the whole conversation and the schemas. With a fixed prompt of $P$ tokens and $g$ tokens added per step, $k$ steps cost roughly

$$\text{input tokens} \approx kP + g\,\frac{k(k-1)}{2}$$

linear in the fixed prompt and quadratic in the history. The number of steps, not the length of the answer, dominates cost and latency, and every tool you add costs tokens on every request whether or not it is used.

**Errors as observations.** Models find their own mistakes poorly without help (Huang et al., 2024), but fix them reasonably once an external checker points at them (CRITIC). A schema validator is exactly such a checker, so the error message is part of the design. Validation errors go back to the model; execution errors only by choice, because they can leak paths, SQL or secrets.

**Evaluation.** Evaluate the **trajectory**, not just the final text: tool-selection accuracy (including "none" for chit-chat), argument validity, task success, latency and tokens. That is the decomposition of the Berkeley Function-Calling Leaderboard (AST and execution checks, plus a "do not call" category); τ-bench adds *pass^k*, the probability of succeeding on all $k$ repeats, which exposes inconsistency.

| Failure | Mitigation |
|---|---|
| wrong decision (answers from memory, or calls a tool for chit-chat) | clear descriptions, system prompt, evaluation |
| wrong tool | fewer tools per prompt, distinct names, routing |
| bad arguments | schema validation, error sent back, normalize inside the tool |
| loops | step and spend limits |
| injection through results | treat output as data, confirm actions |
| excessive agency | least privilege, read-only by default |

> **Notebook:** [16 · Tools and structured output](../../notebooks/04_applications/16_agents_tools_structured_output.ipynb) · **In the course:** the model computed `(120 - 45.5) / 2` "in its head" without the calculator (right number, wrong trajectory), used `search_docs` for chit-chat and skipped a step on a multi-step request; the schemas cost about 390 prompt tokens per call before any conversation, and with 14 cases each one is worth ~7 points.

### 5.2 Constrained decoding and structured output

**Three levels of guarantee.**

| Level | Mechanism | What is guaranteed |
|---|---|---|
| prompting | "answer in JSON with keys..." | nothing; the model usually complies |
| JSON mode | the decoder only allows tokens that keep the JSON valid | syntactically valid JSON, any keys |
| schema-constrained decoding | only tokens that keep the output valid *for your schema* | right keys, types and enum values |

**Core idea.** The schema is compiled into a grammar (a regular expression, or a context-free grammar for nesting), and at every step a mask is applied to the logits:

$$p(y_t \mid y_{<t}) \;\propto\; \exp\big(z_{t,y_t} + m_t(y_t)\big), \qquad m_t(v) = \begin{cases} 0 & \text{if } y_{<t}\,v \text{ is a valid prefix} \\ -\infty & \text{otherwise} \end{cases}$$

Whatever is sampled is, by construction, a prefix of a valid document. The hard part is efficiency: the grammar works on characters, the model on tokens that span several symbols, and the vocabulary has ~150,000 entries. Outlines (Willard and Louf, 2023) precomputes the allowed tokens per state of a finite-state machine; XGrammar (Dong et al., 2024) extends this to context-free grammars; llama.cpp (under Ollama) does it with GBNF. Some engines are even *faster* with a grammar because they emit forced tokens without running the model.

**What it does not guarantee.** *Content* (a perfectly typed field can be wrong or `null`); an unbiased distribution (the mask is applied token by token and pushes the model down paths it considered unlikely; Park et al., 2024); the same reasoning quality (Tam et al., 2024 saw drops under strict formats; others attribute it to the prompt; **debated**). The practice both sides agree on: put a free-text `reasoning` field *before* the answer fields, because keys are generated in order. Business rules ("at most 12 words") are beyond any decoder: validate in your code, retry *with the error* (a blind retry at temperature 0 repeats itself), cap the attempts and keep a fallback.

> **Notebook:** [16 · Tools and structured output](../../notebooks/04_applications/16_agents_tools_structured_output.ipynb) · **In the course:** all three `with_structured_output` methods return a valid `Ticket` for all 8 emails, but `json_schema` returns `order_id = null` for every email that mentions an order while `function_calling` and `json_mode` extract it: the shape is guaranteed, the content is not; retrying with the error shrinks a summary from 22 to 13 words without reaching the 12-word limit.

### 5.3 ReAct, plan-and-execute and reflection

**What makes a system "agentic".** The useful definition is about control flow: a system is agentic to the degree that **a model decides what happens next**. It is a spectrum: single call, fixed workflow, router, tool loop, autonomous agent with its own sub-goals. Each step buys flexibility and pays in latency, cost, variance and testability. The industry advice (Anthropic, *Building effective agents*, 2024) is to use the simplest design that works and add autonomy only where an evaluation shows it helps.

**ReAct.** Before it there were two separate lines: chain-of-thought (reasons with what is in the weights; an early error propagates) and act-only agents (call tools with no place to decompose the goal). ReAct (Yao et al., 2022) enlarges the action space with *thoughts*, actions that change only the model's own context. The loop feeds back both ways: reasoning improves the choice of action, and each observation grounds the next thought in real data. Native tool calling keeps the loop and changes the surface: the `Action` line is a structured `tool_calls` field, and the thought is visible text, hidden reasoning tokens, or nothing.

**Plan-and-execute.** ReAct decides step by step with the whole history: the context grows and the model can lose the goal. Separating planner and executor (Plan-and-Solve; ReWOO, which writes the full plan with placeholders for results; LLMCompiler, which treats it as a dependency graph and parallelizes):

| | ReAct | Plan-and-execute |
|---|---|---|
| Model calls | one per step, fewer with parallel calls | planner + at least one per step + writer |
| Adapts to surprises | at every step | only with a re-planning step |
| Context per call | grows with the whole history | each executor sees its step |
| Inspectability | the trace, after the fact | the plan can be shown, edited or approved first |
| Typical failure | skips a step, loses the goal | redundant steps, information lost in hand-offs |

Neither is better in general.

**Reflection and the generation-verification gap.** Self-Refine (the same model generates, critiques, revises), Reflexion (turns a failure signal into a written lesson prepended to the next attempt: RL where the "weights" are words) and CRITIC (checks with tools). The pattern that holds across this literature: checking is often much easier than producing, so a cheap reliable verifier plus a few retries lifts a generator that is only sometimes right. When the verifier is the same model judging itself with no outside information, the gain often vanishes or reverses (Huang et al., 2024). Expected cost: $1 + \sum_{r \ge 2} P(\text{round } r \text{ is reached})$ generations per task; reflection is cheap when most drafts pass, which is exactly when it helps least.

**Compounding errors.** If each step succeeds with probability $p$, a $k$-step task succeeds with $p^k$; $0.95^{10} \approx 0.60$. Long runs need very reliable steps or checkpoints where errors are caught.

**Multi-agent.** Smaller prompts, fewer tools per decision, parallel work; but more calls, and hand-offs where context is lost. Practitioners disagree publicly (Anthropic described a 2025 multi-agent research system that beat a single agent at a much higher token cost; Cognition argued the same year for single-threaded agents). The disagreement is mostly about task type. The least controversial form is a **router in front of specialists**, and a router is a **classifier**: labelled set, accuracy, confusion matrix; a router wrong 1 time in 10 caps the system near 90 % even with perfect specialists. Teams that split one request across several agents are in [5.7](#57-multi-agent-teams-supervisor-and-handoffs).

> **Notebook:** [25 · Agentic AI with LangGraph](../../notebooks/05_agents_and_mcp/25_agentic_ai.ipynb) · **In the course:** with `qwen2.5:3b`, the ReAct agent beats plan-and-execute with about a third of the model calls (plan-and-execute has higher tool recall and lower precision); reflection with unit tests repaired nothing in one run (4/8 before and after) and one task in others; the router scores 67-75 % depending on the run.

### 5.4 LangGraph: agents as state machines

**Problem.** In 2023 the most visible agents (AutoGPT, BabyAGI) were open-ended loops: runs that never ended, repeated actions, drifting goals, unpredictable costs, no way to test a part or resume after a crash.

**Core idea.** Put control flow back into code and give the model control only at specific points. An application is a **graph** over a typed state: nodes read the state and return updates; edges (fixed or conditional) pick the next node; a per-key **reducer** combines updates (for example, `add_messages` appends instead of overwriting). The runtime follows the **Pregel / bulk synchronous parallel** model (Malewicz et al., 2010): in each *super-step* all triggered nodes run, their updates are merged through the reducers (which is what makes parallel branches safe), and the edges choose the next super-step.

**The properties it buys.** Explicit transitions testable with fake models; visible loops where limits go; **checkpoints** (state only changes between nodes, so it is saved after each step: resume, pause for a human as long as needed, "time travel"); and mixed autonomy (a scripted branch next to a free ReAct loop). `create_agent` is a two-node graph (`model`, `tools`) with one conditional edge that encodes ReAct's entire stopping rule: if the last message has tool calls, go to `tools`; otherwise, end.

> **Notebook:** [25 · Agentic AI with LangGraph](../../notebooks/05_agents_and_mcp/25_agentic_ai.ipynb) · **In the course:** a triage graph with plain functions to show the machinery, then the ReAct agent, plan-and-execute, reflection and a router with subgraphs, all tested with fake models in `tests/test_agents_langgraph.py`, which also runs real attack code against the sandbox of section 11.

### 5.5 Memory

Agent memory is described with categories from cognitive science (CoALA, Sumers et al., 2023):

| Kind | What it holds | In LangGraph |
|---|---|---|
| working | the current context window | the `messages` in the state |
| episodic | what happened in past interactions | checkpoints of other threads, or summaries |
| semantic | facts about the user and the world | the store, searched by embedding |
| procedural | how to do things | the prompt and the code itself |

Short-term memory is the conversation thread saved by a checkpointer under a `thread_id`; it grows each turn and every call re-sends it, hence summarization or trimming. Long-term memory is a namespaced store with semantic search. Generative Agents scores memories by recency, importance and relevance; MemGPT pages memory in and out like an operating system. The underlying constraint is that context is limited, expensive and used unevenly: **long-term memory is a retrieval problem**, with all of retrieval's failure modes, and it is measured as one (recall). The hard half is *writing*: what is worth keeping, deduplication, updates, privacy; a wrong or injected fact is retrieved in every later conversation.

### 5.6 Limits and human-in-the-loop

**Why agents do not stop.** In ReAct the only exit is the model choosing not to call a tool, and nothing in its training guarantees that: it may read a result as incomplete, retry with the same arguments, or keep searching because one more search always looks useful. Termination is imposed from outside, the way a server imposes timeouts. Layers, coarse to fine: the recursion limit (super-steps; a backstop that raises), model and tool call limits (the budget in the units you pay for, ending with a message), and a time or cost limit via middleware. A general reliability principle: the first line of defence degrades gracefully, the last cannot be bypassed. Every run ends with a **stop reason** you count and alert on.

**Approval gates in code.** Read-only tools run freely; tools that move money, send email or delete data wait for a person. A system prompt saying "ask before refunding" is an instruction the model may ignore, or be talked out of by an injected document; an `interrupt` in the graph is code that the path to the tool physically goes through. That is **least agency**: the model proposes, a deterministic control decides. Pausing requires **persistence**: a person may answer in seconds or days, from another machine, so state is written to storage (the same idea as Temporal or Step Functions). A crucial detail: on resume, LangGraph re-runs the interrupted node from its first line, so any side effect before the `interrupt` runs twice; it belongs in a later node or must be idempotent.

> **Notebook:** [25 · Agentic AI with LangGraph](../../notebooks/05_agents_and_mcp/25_agentic_ai.ipynb) · **In the course:** a fake model that requests the same tool forever stops only at the recursion limit (with an exception) without middleware, and with middleware ends with a message and a reason; the installed version's default recursion limit is 10,007, which is why every run passes its own; refund approval is decided by a policy (up to 40 dollars) outside the model.

### 5.7 Multi-agent teams: supervisor and handoffs

**Problem.** A router (5.3) sends each request to *one* specialist. Many requests need two or more ("what is the total of my order with express shipping?" needs billing and the shipping policy), and a single agent with every tool and every instruction gets a long prompt and more wrong tool choices.

**Two designs.** In a **supervisor** design (also called *agents as tools*, or *orchestrator-workers* in Anthropic's *Building effective agents*, 2024) a supervisor agent calls specialist agents as if they were tools: it writes a request, gets back the worker's final answer, may call again, and writes the reply itself. Control always returns to it. In a **handoff** design (the *swarm* style: OpenAI's Swarm and Agents SDK, `langgraph-swarm`) the active agent *transfers the conversation* to another agent, which takes over and may transfer again; whoever finishes answers. In LangGraph a worker-as-tool is a function that runs an agent and returns its last message, and a handoff is a tool that returns `Command(goto="billing", graph=Command.PARENT)`: the tool runs inside the agent's own graph, so the jump has to happen one level up, in the parent graph that holds all the agents.

| | supervisor | handoffs |
|---|---|---|
| Who holds control | the supervisor, always | whichever agent is active |
| What a specialist sees | only the request the supervisor wrote | the conversation it is handed |
| Who writes the answer | the supervisor | the last agent |
| Typical failure | the request or the reply loses a detail | an agent transfers too early, or the next one misreads its turn |
| Cost | supervisor calls + every worker's calls | one agent at a time, plus the transfers |

**Where the industry disagrees.** Anthropic's multi-agent research system (2025) uses a lead agent that plans and launches subagents in parallel, each with its own context window, and reported a large gain over a single agent on broad research questions, at about 15 times the tokens of a chat; much of the gain came from being able to spend more tokens in parallel. Cognition's *Don't Build Multi-Agents* (2025) argues the opposite for tightly coupled work such as coding: every action carries implicit decisions, subagents that do not see each other's full context make conflicting ones, and a single agent with a continuous context (compressed when it gets long) is more reliable. Both positions fit together once the task type is named: split work that is **parallel and loosely coupled**, keep **sequential, tightly coupled** work in one agent.

**Context engineering at the boundary.** Every hand-over is a lossy compression of context: a worker knows only what it is told, and the supervisor knows only what the worker returns. What goes across the boundary is part of the design: a self-contained request (ids, numbers and constraints spelled out), the earlier answers the next agent needs, and nothing of the transfer machinery that it could misread as "the job is done". A team adds model calls and failure points, so it has to earn its place on the same evaluation set as the single-agent baseline.

> **Notebook:** [25 · Agentic AI with LangGraph](../../notebooks/05_agents_and_mcp/25_agentic_ai.ipynb), section 10 · **In the course:** on the ten tasks of the ReAct evaluation, with `qwen2.5:3b`, the supervisor solved 80 % (5.3 model calls per task) against 70 % for the single agent (2.2 calls), a tie in quality at more than twice the cost, and handoffs solved 30 % (3.4 calls). The first handoff version failed because the specialist read "Transferred to the billing specialist" as the end of the job and answered with nothing; hiding the transfer messages and adding a "your turn" note helped, but agents still transferred before doing their part.

### 5.8 Code execution and sandboxes

**Why let the model write code.** Language models are poor calculators and good programmers. Program-Aided Language models (PAL, Gao et al., 2022) and Program of Thoughts (Chen et al., 2022) have the model write a short program for the reasoning and let an interpreter do the arithmetic, and both beat chain of thought on math word problems. CodeAct (Wang et al., 2024) turns this into an agent design: the agent's *actions* are Python snippets instead of JSON tool calls, so one action can loop, branch, combine several tools and reuse variables, and in their experiments code actions solved more tasks in fewer turns. Code interpreters (ChatGPT's data analysis, Claude's code execution tool, data-analysis agents) are this idea as a product.

**Execution feedback.** Running the code gives the model a signal it cannot get from its own text: a traceback, an empty output, a printed number. A loop that returns that signal and lets the model try again fixes crashes and wrong API calls. It does not fix code that runs and computes the wrong thing, because nothing signals the error; that needs a verifier (intermediate outputs, invariants, tests, a review step), as with reflection in 5.3.

**The agent-computer interface.** SWE-agent (Yang et al., 2024) showed that the same model solves many more software-engineering tasks when its tools are designed for a model rather than for a human: a file viewer that shows a window of lines, an editor that rejects edits that do not parse, short and explicit feedback, and a clear message when a command succeeds without output instead of an empty string. The authors call this the **agent-computer interface** (ACI), by analogy with human-computer interfaces. A model that gets an empty result tends to invent one, so the wording of a tool's error and empty-output messages is part of the agent's design.

**Model-written code is untrusted code.** The model can be manipulated by anything it reads (indirect prompt injection, [4.4](#44-application-patterns)), so its code has to be treated like code from a stranger. The threat model and the layer that contains each risk:

| Risk | Example | Containment |
|---|---|---|
| reading secrets | `open(".env")`, `os.environ` | no secrets in the sandbox: a scrubbed environment, nothing mounted |
| damaging files | `os.remove`, writes outside the work directory | an ephemeral work directory, a read-only root filesystem |
| exfiltration, attacks | HTTP requests, raw sockets | no network, or an egress allowlist |
| escaping the process | `subprocess`, `os.system`, native calls | kernel isolation: a user-space kernel or a microVM |
| running forever, exhausting memory | `while True`, a huge allocation | wall-clock timeout, CPU and memory quotas (cgroups) |
| flooding the context | printing a 1 GB file | truncated output |

**Audit hooks are not a security boundary.** PEP 578 (Python 3.8) added runtime audit hooks: the interpreter raises an event on `open`, socket calls, process creation, `ctypes` loads and more, and a hook installed with `sys.addaudithook` can veto it by raising an exception. They were designed for logging and detection, and the PEP and CPython's documentation say they are not a sandbox: code that reaches native code (`ctypes`, a C extension, a memory-corruption bug) bypasses them, because the hooks only see what the interpreter reports. Inside a separate process with a timeout and a scrubbed environment they make a good teaching tool and a tripwire, not an isolation layer.

**What production uses.** Isolation has to come from below Python, from the operating system or a hypervisor, in rising order of strength: a **container** with no network, a read-only filesystem and no secrets (`docker run --network none --read-only`), which still shares the host kernel, so a kernel bug can be an escape; **gVisor**, a user-space kernel that intercepts the container's system calls, so code never talks to the host kernel directly (Modal's sandboxes run on it); a **microVM** such as **Firecracker**, the KVM-based virtual machine monitor behind AWS Lambda, which boots a minimal VM in a fraction of a second and gives each run its own kernel (E2B's sandboxes are Firecracker microVMs); and **WebAssembly** (Pyodide runs CPython compiled to Wasm), where the code can only reach the capabilities the host explicitly grants. Managed code-execution tools from model providers run the code in this kind of isolation on their side. On top of any of them: quotas on CPU, memory, time and output, and logs of every execution.

**Bound the damage.** Prompts and filters lower the rate of successful injections; they do not bring it to zero. The design rule is to assume the model will eventually be talked into running the wrong code and to make sure that code has nothing to reach. In particular, never give one agent all three of access to private data, exposure to untrusted content and a way to send data out (what Simon Willison calls the *lethal trifecta*, 2025).

> **Notebook:** [25 · Agentic AI with LangGraph](../../notebooks/05_agents_and_mcp/25_agentic_ai.ipynb), section 11 · **In the course:** a subprocess sandbox (`python -I`, a fresh work directory, a scrubbed environment, a timeout, output truncated to 2,000 characters and an audit hook) contains 8 of 9 attack probes; the ninth, a native call through `ctypes`, gets through on purpose, because numpy needs `ctypes` on Windows. A pandas agent answers 0/10 exact questions about a 600-order table with `qwen2.5:3b`, 5/10 with `qwen2.5:7b` and one execution, and 6/10 with up to four executions and their feedback; the four misses ran without errors and printed wrong numbers. With an injected instruction in a data file, 0 of the 2 secret values in the notebook's process were reachable from the sandbox. The sandbox is tested in `tests/test_agents_langgraph.py`.

### 5.9 Model Context Protocol

**Problem.** Several AI applications (assistant, IDE plugin, support bot, internal agent) and many systems to reach (wiki, tickets, CRM, databases). If each application writes its own connector for each system, the work grows as $N \times M$, each connector is maintained separately and behaves slightly differently.

**Core idea.** The classic answer to an $N \times M$ problem is a **standard interface in the middle**: $N + M$ pieces of work instead of $N \times M$, like ODBC/JDBC for databases or the Language Server Protocol (2016) for editors and languages, from which MCP explicitly borrows its structure: JSON-RPC 2.0, a stateful session that starts with `initialize` and capability negotiation, a stdio subprocess as the local transport.

```
HOST (Claude Desktop, an IDE, your LangGraph agent)   <- trust boundary: policy lives here
  model <-> host logic (approvals, context assembly)
     client A ──stdio──► server A (files)
     client B ──stdio──► server B (git)
     client C ──Streamable HTTP──► server C (remote SaaS)
```

**One client per server** is an isolation choice: a server never sees the others or the full conversation. The three primitives have different owners, and that assignment is the design: **tools** are model-controlled (where risk concentrates and approvals belong), **resources** application-controlled (the host decides what enters the context), **prompts** user-controlled (explicit templates, like slash commands). On the client side, *sampling* lets a server use the host's model without its own API key, *elicitation* asks the user for input, and *roots* tell it which folders it may work in. The version is negotiated by date and undeclared capabilities must not be used, which lets the protocol add features without breaking older implementations.

**Why not just OpenAPI and function calling.** A fair question that is still debated (2023's ChatGPT plugins did exactly that and were retired in 2024). MCP adds a local transport with no ports or authentication, a session with negotiated capabilities, two-way communication (notifications, progress, server-to-client requests) and primitives designed for models. The counter-arguments are real too: another protocol to secure and operate, OAuth for remote servers, and for one application calling one well-documented API, a direct call is simpler. Use it when the same capability should reach several hosts or you want the existing ecosystem. Adoption is what gives a protocol its value: Anthropic released it in November 2024, OpenAI, Google and Microsoft added support in 2025, and in December 2025 it moved to the Agentic AI Foundation under the Linux Foundation.

**Two error channels.** A *tool error* is a normal result with `isError: true` and goes to the **model** as an observation it can correct; a *protocol error* is a JSON-RPC `error` object with a numeric code, is a **client** bug and never reaches the model. Implementations differ in the details, so test the real server with **contract tests**.

**Cost.** Transport is cheap (milliseconds next to seconds of inference); process start-up and the handshake are the expensive part, so sessions are reused. The real cost is **context**: every connected tool sends its name, description and schema on every call, and too many tools degrade selection; hosts answer with allowlists, per-task subsets or tool search.

**Security: servers are untrusted code and untrusted text.**

| Failure | Description | Control |
|---|---|---|
| tool poisoning | instructions hidden in a tool *description* | review servers, pin versions, scan descriptions |
| rug pull | a server changes a description after approval | re-confirm changes, pin versions |
| shadowing | one server redefines or instructs about another's tools | namespace tools per server |
| indirect injection | instructions inside returned data | treat results as data, confirm writes |
| confused deputy | forwarding a token issued for something else | accept only your own tokens (RFC 8707); passthrough forbidden |
| the "lethal trifecta" | private data + untrusted content + an outbound channel (Willison, 2025) | remove at least one of the three |

Annotations (`readOnlyHint`, `destructiveHint`) are *hints*: a malicious server can lie. The defences have to sit outside the model: least privilege, confirmation in code for anything not read-only, and a design in which a successful injection cannot do much harm.

**Client features in practice.** The three server-to-client requests only reach a client that declared the capability, and each one moves a decision to where it belongs. **Sampling** (`sampling/createMessage`) lets a server ask for a completion from the host's model: the server never holds a model or an API key, and the host can show the request to the user, change the model, cap the tokens or refuse (the spec recommends keeping a human able to review both the request and the result). **Elicitation** (`elicitation/create`, added in spec 2025-06-18) lets a server ask the *user* for structured input described by a small JSON schema, with three answers, *accept*, *decline* and *cancel*; it is the protocol-level version of an approval gate, because the model can be talked into calling a destructive tool but cannot answer the form for the person. Servers must not use it to ask for passwords or other secrets. **Roots** (`roots/list`) tell a server which folders or repositories it may work in. They are advisory: a cooperative server stays inside them, a malicious one can ignore them, so they scope well-behaved servers and do not replace operating-system isolation ([5.8](#58-code-execution-and-sandboxes)).

**Authorization for remote servers.** A stdio server runs as the user and needs no authentication (credentials come from its environment). A Streamable HTTP server is an endpoint anyone can reach, and since spec 2025-03-26 (reworked in 2025-06-18) MCP uses standard OAuth 2.1 with the MCP server as a **resource server** and a separate authorization server issuing the tokens:

1. A request without a valid token gets `401 Unauthorized` with a `WWW-Authenticate: Bearer ... resource_metadata="..."` header.
2. That URL serves the server's **protected-resource metadata** (RFC 9728, under `/.well-known/oauth-protected-resource`): which authorization servers it trusts and which scopes it supports.
3. The client discovers the authorization server's own metadata (RFC 8414), registers if needed (dynamic client registration, RFC 7591), and gets a token with the authorization-code flow and PKCE. It names the MCP server as the intended audience with a **resource indicator** (RFC 8707).
4. Every request carries `Authorization: Bearer <token>`, and the server validates it every time: signature and expiry for a JWT (or an introspection call), the scopes, and the **audience**, so a token issued for another service is rejected.

Two rules prevent the classic confused-deputy failures. **No token passthrough**: a server must not accept tokens that were not issued for it, and must not forward the client's token to the APIs it calls downstream; it gets its own credentials for those. **Validate `Origin`**: a Streamable HTTP server must check the `Origin` header of incoming requests, and a local one should bind to `127.0.0.1`, otherwise a web page in the user's browser can reach it through DNS rebinding.

> **Notebook:** [26 · MCP with the official SDK](../../notebooks/05_agents_and_mcp/26_mcp_model_context_protocol.ipynb) · **In the course:** all 14 conformance checks pass; `search_docs` gets 6/6 top-1 on an easy set; the agent's answer contains the key fact in 6 of 6 and cites the right file in 4 of 6; a `tools/call` costs ~1.6 ms in memory, ~3.0 ms over stdio and ~5.2 ms over Streamable HTTP with token validation, against ~2.4 s per call with the adapter that opens a new session every time. Section 9 shows sampling (two summaries from the host's model, every request visible to the host), elicitation (decline, accept and cancel all respected) and roots (the repository root refused), and runs `scripts/mcp_server.py --http --require-token`: 401 with `WWW-Authenticate` without a token or with a wrong one, protected-resource metadata naming the authorization server and the `docs:read` scope, and the same tools with the token.

---

## 6. LLMOps

Terms for this part: [glossary, LLMOps and LangSmith](glossary.md#llmops-and-langsmith) and [Safety and guardrails](glossary.md#safety-and-guardrails).

### 6.1 Traces, spans and OpenTelemetry GenAI

**Problem.** Classical monitoring assumes a request succeeds or fails and that success is correct. LLM applications break that: failures are **semantic** (a hallucination returns HTTP 200), behaviour is non-deterministic and input-dependent, one request is many steps (retrieval, several model calls, tools), cost scales with tokens that change every release, and quality is a distribution, not a bit. LLMOps joins two traditions: **distributed tracing** from SRE (what happened, where the time went) and **evaluation** from ML (how good the outputs are).

**Core idea.** The trace/span model comes from Dapper (Sigelman et al., 2010): a **trace** records one request as a tree of timed **spans**, each with a parent. The key mechanism is **context propagation**: the parent passes the trace id and its own span id so the child attaches to the right place, even across threads, processes or services. Dapper led to Zipkin, OpenTracing and OpenCensus, which merged in 2019 into **OpenTelemetry**. A LangSmith *run* is a span with LLM-specific fields (messages, tokens, `run_type`); Langfuse and Arize Phoenix use very similar models. Uploads go in batches from a background thread: tracing adds almost no latency, and a script that exits right away must flush.

**OTel GenAI semantic conventions.** A trace is only portable if everyone names things the same way: `gen_ai.operation.name`, `gen_ai.provider.name` (formerly `gen_ai.system`), `gen_ai.request.model`, `gen_ai.usage.input_tokens`... They were marked *in development* and names have changed between versions, so check the version before building dashboards. Their value is strategic: instrumenting with them lets you send traces to LangSmith, Langfuse, Phoenix, Datadog or Grafana without code changes.

**Signals.** Traces (structure and timing), metrics (cheap aggregates to alert on), logs (events) and, specific to LLMs, **feedback**: a score attached to a trace after the fact (users, heuristics, judges, reviewers). Feedback turns traces into a quality signal and is the bridge between monitoring and evaluation.

### 6.2 Offline and online evaluation

| | Offline | Online |
|---|---|---|
| When | before a change ships | continuously, in production |
| Data | a fixed, labelled dataset | real requests, no reference |
| Evaluators | reference-based (exact match, key facts, judge with reference) | reference-free (groundedness, format, judge, user feedback) |
| Question | is B better than A on the cases we care about? | is production healthy today, and which traces should a human read? |
| Risk | the dataset stops representing traffic | cheap signals are noisy; expensive ones only run on a sample |

They feed each other: bad production traces, once reviewed and labelled, join the dataset, so it keeps tracking what users actually ask. That loop is the core workflow of LLMOps.

Online, without references, you pick from a **pyramid** of cost against fidelity: deterministic checks on 100 % of traffic (format, length, lexical overlap), small models (NLI, embedding similarity) on most of it, an LLM judge on a sample, implicit signals and user thumbs (free but sparse and biased toward unhappy users), and human review of the worst traces. Cheap heuristics are for **ranking which traces to open**, not for grading. Offline, exact match measures phrasing, not correctness, on free text; key-fact checks are deterministic but blind to unanticipated phrasing; an LLM judge covers the rest once validated (section 4.3).

> **Notebook:** [28 · Observability and evaluation](../../notebooks/06_llmops/28_observability_llmops.ipynb) · **In the course:** the RAG version lifts judged correctness from 0.17 (closed book) to 0.83 and key-fact coverage from 0.17 to 1.00, while exact match (0.00 vs 0.58) only measures how much wording is copied; the groundedness heuristic (mean 0.81) surfaced two real bugs in the order-tool prompt that the offline set could not see.

### 6.3 Percentiles and SLOs

**Why percentiles.** Latency is never normal: bounded below, with a long right tail. For an LLM call,

$$T \;\approx\; T_{\text{queue}} + T_{\text{prefill}} + n_{\text{out}} \cdot T_{\text{per token}}$$

and because output length varies a lot and queues come in bursts, the tail is long by construction. A few slow requests drag the mean and it says nothing about them; p50 is the typical experience, p95 "1 request in 20 is slower than this", p99 the tail that trips timeouts. The tail weighs more than its share suggests (Dean and Barroso, 2013, *The Tail at Scale*): if one action needs 100 calls and each is slow 1 % of the time, the action is slow with probability $1 - 0.99^{100} \approx 63\%$, and agents have exactly that structure. Two consequences: percentiles need many samples (the p95 of 18 requests is decided by one or two of them), and they **cannot be averaged** (the mean of hourly p95s is not the daily p95), which is why systems store histograms or mergeable sketches (HDR, t-digest, DDSketch).

**SLI, SLO, SLA and error budget** (Google's SRE book, 2016). An SLI is a measurement (a ratio of good events), an SLO the target over a window, an SLA a contract with consequences. A 99 % objective explicitly *allows* 1 % bad events: while budget remains, risky changes ship; once it is spent, reliability comes first. Alerts fire on **burn rate** over a short and a long window together, so a brief spike pages no one and a sustained one does. LLMs add their own SLIs: TTFT, tokens or cost per request, the share passing a groundedness or format check, refusal rate, and the share of runs that end by hitting a step limit.

> **Notebook:** [28 · Observability and evaluation](../../notebooks/06_llmops/28_observability_llmops.ipynb) · **In the course:** 18 requests with an 11.1 % error rate (two unknown orders; the tool fails 2 of 6 calls) and 1605 input against 270 output tokens, so the retrieved context, not the answer, is the cost lever; p50/p95 of 0.22 s / 2.8 s on a free GPU and much more on a shared one, so the latency SLO passed in some executions and failed in others: with 18 requests the p95 is essentially the second-slowest request.

### 6.4 PII and trace sampling

**PII.** Traces are a copy of user data with a different purpose (debugging, evaluation) and audience (engineers, vendors), and the GDPR treats them like any copy: data minimization, purpose limitation and storage limitation.

| Technique | What it does | What it keeps |
|---|---|---|
| drop (`hide_inputs` / `hide_outputs`) | the payload is never sent | timing, tokens, errors |
| redact / mask | detected values replaced by a type label (`<email>`) | the surrounding text, so traces stay readable |
| pseudonymize | a consistent token or keyed hash per value | grouping by user without knowing who; legally still personal data |

Detection is the hard part: regexes are fast and precise for fixed structures (emails, card numbers with Luhn, phones, prefixed API keys) and blind to everything else; NER models (Presidio combines both) find names and addresses at the cost of latency and errors. Recall is never 100 %, one more reason to limit who reads traces and how long they are kept. And decide *where* to redact: in the tracing client it protects the trace store, not the model provider, which still receives the original text.

**Sampling.** At high traffic you do not need every trace.

| | Head-based | Tail-based |
|---|---|---|
| Decided | when the trace starts | after it finishes |
| Based on | the trace id or a random draw | anything: errors, latency, feedback |
| Cost | negligible | every span buffered until the trace ends (a collector) |
| Keeps all errors? | no | yes, if the policy says so |

The decision is per whole trace (half a trace leaves trees with holes). Uniform sampling keeps aggregate metrics unbiased (counts are recovered by dividing by the rate); rare events are what suffer. The common compromise is a small fraction of everything plus every error and every negative-feedback trace, which needs tail-based sampling or a separate logging path.

> **Notebook:** [28 · Observability and evaluation](../../notebooks/06_llmops/28_observability_llmops.ipynb) · **In the course:** the anonymizer replaces the email, phone, card and API key in the root run and every child span, and lets a name like "Ana" through; at a 0.25 rate over 40 traces, 7, 12 and 14 were kept in different executions (theoretical standard deviation ~2.7).

### 6.5 Statistics of small evaluation sets

The whole course evaluates on sets of 8 to 40 items, and that has consequences worth being able to compute.

**Standard error of a proportion.** With $n$ examples and a measured rate $\hat p$:

$$\mathrm{SE} \approx \sqrt{\frac{\hat p(1-\hat p)}{n}} \;\le\; \frac{0.5}{\sqrt{n}}$$

With $n = 12$ the 95 % interval spans roughly ±0.28; with $n = 100$, ±0.10. In a set of 40 each item is worth 2.5 points; in a set of 20, 5. A dozen examples can show a large effect (a capability one version has and the other lacks), not a difference of a few points.

**Paired design.** Most of the variation in a score comes from how hard each example is; comparing two versions on the **same** examples cancels it. For binary scores, McNemar's test looks only at the **discordant pairs** (one right, the other wrong); pairs both get right or both get wrong say nothing about which is better. For continuous scores, the **paired bootstrap**: resample the per-question differences with replacement (say 2,000 times) and take the 2.5 and 97.5 percentiles; if the interval excludes 0, the difference is unlikely to be noise.

**Other sources of variance.** At temperature 0 outputs can still change between runs (GPU kernels are not bit-for-bit deterministic, batching); an LLM judge flips verdicts on borderline cases; *pass^k* exposes inconsistency a single run hides. Useful habits: repeat runs and report the spread, always read the per-example table (the examples that change verdict between versions are where the explanation is), and put cost and latency in the same table as quality.

**What goes into the dataset** matters as much as its size: common intents, known edge cases, adversarial inputs and every production failure already fixed (a regression set). Version it, so each experiment is tied to the exact examples, and keep it out of prompt tuning, or the scores will measure memorization of the test set.

> **Notebook:** [28 · Observability and evaluation](../../notebooks/06_llmops/28_observability_llmops.ipynb) and [18 · RAG with LangChain and RAGAS](../../notebooks/04_applications/18_rag.ipynb) · **In the course:** bootstrap intervals and the paired hybrid-minus-dense difference on RAGAS (notebook 18), a paired bootstrap over 6,040 users for every comparison between recommender stages (notebook 23), the ±0.28 calculation for $n = 12$ in notebook 28, and in every application notebook a reminder of what one item is worth before reading a difference.

### 6.5b Evaluation and tracing frameworks

**Problem.** A hand-written evaluation loop and a hosted tracer are fine for learning, and both stop scaling in a team. Loops written by different people make different choices (prompt, decoding, answer extraction, what counts as an error) that are rarely recorded, so their numbers cannot be compared across people or months. A hosted tracer is a dependency, a cost, a quota and a place your users' data goes. Frameworks standardize both, and open protocols keep you from being tied to one vendor.

**An evaluation framework separates the parts of an evaluation,** so each one can be swapped and is recorded:

| Part | What it is | Examples |
|---|---|---|
| dataset | samples with input, target and metadata | 80 GSM8K problems |
| solver | what is done to each sample | a system message, a generation, chain of thought, tool use, an agent loop in a sandbox |
| scorer | how an output is graded, with its metrics | exact match, a regex, a numeric match, a model grader; accuracy and its standard error |
| log | every sample with its transcript, scores, tokens and timings, plus the run settings | one file you can reopen, diff and share |
| epochs | each sample run several times and reduced (mean, majority) | `--epochs 3` |

There are two families. **Benchmark harnesses** for model checkpoints (EleutherAI's lm-evaluation-harness, which powers leaderboards; HELM, Liang et al., 2022) run fixed tasks, often scored from log-probabilities. **System evaluation frameworks** (Inspect AI from the UK AI Security Institute, OpenAI Evals, promptfoo, DeepEval) evaluate prompts, tools, agents and sandboxes against any API. Either way, a framework is adopted only after it reproduces a number you already trust.

**The scorer is part of the metric.** The same generations graded by two answer extractors give two accuracies. A "last number" rule and a numeric match that picks another number out of a LaTeX `\boxed{}` answer disagree exactly where the output format departs from what the rule expects. Small implementation details of this kind are a main reason published numbers for the same model disagree (Biderman et al., 2024). A score is reported with its scorer, and the scorer is pinned like code.

**Epochs and variance.** For a sampled model, one run is one draw. With $n$ questions, a per-question probability of being right $\mu_i$ and $K$ samples per question, the variance of the mean score splits in two (Miller, 2024):

$$\mathrm{Var}(\hat{\mu}) \approx \frac{1}{n}\left(\mathrm{Var}_i(\mu_i) + \frac{\mathbb{E}_i\big[\mu_i(1-\mu_i)\big]}{K}\right)$$

The first term is *which questions you drew*, and only more questions reduce it. The second is *which sample you drew for each question*, and epochs ($K > 1$) reduce it. When most questions are always right or always wrong, $\mu_i(1-\mu_i) \approx 0$ for them and the first term dominates: about $\sqrt{0.85 \cdot 0.15 / 80} \approx 4$ points of standard error for 85 % on 80 problems ([section 6.5](#65-statistics-of-small-evaluation-sets)). So report the mean over epochs, its spread and a confidence interval, and compare systems paired on the same questions.

**Tracing on OpenTelemetry.** The trace and span model of [section 6.1](#61-traces-spans-and-opentelemetry-genai) becomes portable through three pieces:

- **semantic conventions** for attribute names: OpenTelemetry's `gen_ai.*`, and **OpenInference**, the set used by Phoenix and others (`llm.model_name`, `llm.token_count.prompt`, `input.value`, `output.value`, span kinds `LLM`, `CHAIN`, `RETRIEVER`, `TOOL`...);
- **instrumentors**, which create the spans for a library automatically (LangChain, LlamaIndex, the OpenAI SDK) with no change to the application code;
- **exporters**, which ship the spans over OTLP (HTTP or gRPC), in batches from a background thread, to any collector: Phoenix, Langfuse, LangSmith, Datadog or any OTel backend. Changing backend is a configuration change, not a rewrite.

**Annotations close the loop.** A score or a label attached to a span after the fact (by code, an LLM judge or a person) turns the trace store into an evaluation store: failed traces can be filtered, read, labelled and added to the offline dataset ([section 6.2](#62-offline-and-online-evaluation)). The production loop is trace, score, look at the failures, fix.

**Overhead.** Instrumentation runs in the request path, but it records a few timestamps and attributes per span and exports them asynchronously, so it costs milliseconds against LLM calls that take seconds. The reasons to sample traces are storage and privacy ([section 6.4](#64-pii-and-trace-sampling)), not speed. Traces store prompts and outputs verbatim, so a trace store is sensitive data.

**Choosing.** Self-hosted and open (Phoenix runs as one process with SQLite; Langfuse and Opik need Docker and a database) when data cannot leave your infrastructure; managed (LangSmith, Arize, Datadog LLM Observability) when you want no operations work. Both sides speak OpenTelemetry, so the instrumentation code is the same.

**Production uses.** Evaluations as code in the repository (dataset + solver + scorer), run in CI on every prompt, model or retrieval change, with the logs kept as artifacts, the scorer pinned, confidence intervals reported and epochs for anything sampled. Tracing instrumented at the framework level and exported in batches to a backend you control when prompts contain user data, with evaluations attached to the traces (code checks, LLM judges, user feedback) and alerts on latency, error rate and score drift ([section 4.5b](#45b-drift-online-learning-and-continual-updates)).

**Failure modes.** An evaluation that silently changes its scorer; comparing numbers from different answer extractors; a single sampled run reported as "the" accuracy; traces that capture secrets or personal data; dashboards nobody looks at; evaluation sets that leak into prompts or few-shot examples.

**Key papers.** Liang et al. (2022), *Holistic Evaluation of Language Models* (HELM); Biderman et al. (2024), *Lessons from the Trenches on Reproducible Evaluation of Language Models* (lm-evaluation-harness); Miller (2024), *Adding Error Bars to Evals: A Statistical Approach to Language Model Evaluations*; Cobbe et al. (2021), *Training Verifiers to Solve Math Word Problems* (GSM8K); Shankar et al. (2024), *Who Validates the Validators? Aligning LLM-Assisted Evaluation of LLM Outputs with Human Preferences*; Sigelman et al. (2010), *Dapper, a Large-Scale Distributed Systems Tracing Infrastructure*; and, as references rather than papers, the Inspect AI documentation, the OpenInference specification and OpenTelemetry's GenAI semantic conventions.

> **Notebook:** [29 · Evaluation and tracing frameworks](../../notebooks/06_llmops/29_eval_and_tracing_frameworks.ipynb) (read after 28) · **In the course:** as an Inspect AI task, notebook 15's 80 GSM8K problems with its last-number rule give 85.0 % [77.5, 92.5] with `qwen2.5:3b`, exactly notebook 15's hand-written result, while Inspect's built-in `match(numeric=True)` scores the same generations at 80.0 % (−5.0 points [−10.0, −1.2]; the 4 disagreements all end in `\boxed{}`); at temperature 0.7, three epochs give 80.0 %, 82.5 % and 81.2 % (mean 81.2 %), with 80 % of the problems right in all three, 17.5 % in none and only 2 problems changing between epochs; in Phoenix, each of 20 traced LangChain requests is a trace of 5 spans, the LLM span takes p50 3,585 ms and p95 4,641 ms and every other step under 1 ms, with 97 prompt and 242 completion tokens per call; 20 code-scored annotations posted on the root spans read back as 80 %; and with a fake, instant model, tracing adds 2.5 ms per request (3.52 against 1.02 ms), 0.07 % of one real LLM call.

### 6.6 Safety: refusals, jailbreaks and guard models

**Problem.** Observability tells you what the system did; safety is about what it should and should not do. A model that answers everything is a liability, and one that refuses whenever it sees an alarming word is useless. Safety therefore has **two error rates**, harmful requests answered and safe requests refused, and every defence moves both.

**Where refusals come from.** A pretrained model has no notion of harm: it continues text. Refusing is taught in post-training ([section 3.4](#34-from-rlhf-to-dpo-and-grpo)): SFT on examples of polite refusals, then preference tuning (RLHF, DPO) that makes refusing a harmful request the preferred answer. Constitutional AI (Bai et al., 2022) replaced most of the human harm labels with a model that critiques and revises its own answers against a written list of principles (RLAIF). Whatever the recipe, the result is a **learned preference, not a rule**: it generalizes from the training distribution and can fail outside it.

**Why safety training fails.** Wei, Haghtalab and Steinhardt (2023) traced almost every jailbreak to two mechanisms:

- **Competing objectives.** The model was trained both to follow instructions and to be harmless, and a prompt can set one against the other. *Prefix injection* asks for an answer that opens affirmatively; *refusal suppression* forbids the words refusals are made of. Once the first tokens comply, continuing is the likely text.
- **Mismatched generalization.** Capability training covered far more situations than safety training. A request wrapped in fiction or role-play, translated into a low-resource language or encoded (base64, ciphers) falls outside what refusal training saw, while the model can still do the task.

Hand-written templates are only the start. **GCG** (Zou et al., 2023) optimizes an adversarial suffix with gradients from open models, and the suffixes transfer to closed ones. **PAIR** (Chao et al., 2023) needs no gradients: an attacker LLM rewrites the prompt in a loop, guided by the target's replies and a judge, and often succeeds within twenty queries. Both are *adaptive*, which is the attacker a deployed system actually faces.

**The other failure: over-refusal.** Refusal training also teaches shortcuts: "kill", "shoot" or "poison" become triggers whatever the context. **XSTest** (Röttger et al., 2024) measures this *exaggerated safety* with 250 safe prompts built to trip the shortcuts (homonyms such as killing a process, figurative language, safe contexts such as a crime novel, privacy questions about fictional characters) and 200 unsafe contrast prompts. Over-refusal is not a safe default: it costs users, pushes them to rephrase until something gets through, and hides real refusals in noise.

**Guard models.** Instead of relying only on the model's own refusals, production systems put classifiers in front of it (on the prompt) and behind it (on the reply). **Llama Guard** (Inan et al., 2023) is an LLM fine-tuned to answer `safe` or `unsafe` plus a hazard category; Llama Guard 3 uses the MLCommons taxonomy **S1–S14** (violent crimes, non-violent crimes, sex crimes, child exploitation, defamation, specialized advice, privacy, intellectual property, weapons of mass destruction, hate, self-harm, sexual content, elections and code-interpreter abuse). A guard is cheap (a 1B model labels a prompt in tens of milliseconds) and independent of the model it protects, so templates written to manipulate the target model do not manipulate it, and one guard covers every model behind it. It is also a classifier with its own false positives and misses, strongest on the hazards its training data emphasizes, and it can be attacked like any model.

| Option | Kind | Notes |
|---|---|---|
| Llama Guard 3 (1B, 8B), ShieldGemma, Granite Guardian | open guard models | self-hosted; the categories and the threshold can be adapted to your policy |
| Prompt Guard and similar small encoders | open classifiers | aimed at jailbreak and injection patterns rather than harm categories ([section 1.6](#16-encoders-and-encoder-decoders)) |
| OpenAI moderation endpoint, Azure AI Content Safety, Amazon Bedrock Guardrails | managed services | no model to run; the policy is the vendor's, with some configuration |
| NVIDIA NeMo Guardrails, Guardrails AI | frameworks | wire checks, rules and dialogue flows into an application around any of the above |

**How safety is scored.**

- **Refusal rate** on harmful prompts (higher is safer) and **over-refusal rate** on safe ones (lower is better). Always report both.
- **Attack success rate (ASR)**: the share of harmful requests that get a *harmful* reply. Not refusing is not the same as complying usefully: a jailbroken reply can be vague, wrong or empty. **StrongREJECT** (Souly et al., 2024) showed that "did it refuse?" scoring overstates jailbreak success, and benchmarks score harmfulness with a judge instead: HarmBench's classifier (Mazeika et al., 2024), JailbreakBench's judge (Chao et al., 2024), StrongREJECT's rubric.
- **Judges must be validated.** A refusal or harm judge is a model with its own error rate; measure it against human labels before trusting the rates it produces ([section 4.3](#43-ragas-and-llm-as-judge)). And keep it **independent of the defences**: scoring the output of a guarded system with the same guard reports zero attack success by construction.

**Defence in depth.** No single layer is enough, so production stacks several that fail in different ways:

| Layer | Stops | Costs | Typical failure |
|---|---|---|---|
| model's own safety training | direct harmful requests | nothing extra | jailbreaks; over-refusal |
| safety system prompt | a little of both | some helpfulness | a competing instruction later in the prompt wins |
| input guard | most harmful requests, disguised or not | false positives; tens of ms | mild phrasings; adaptive attacks against the guard |
| output guard | harmful content that got through | latency; false positives on factual answers | needs an independent judge to be evaluated |
| application controls (auth, rate limits, no dangerous tools near untrusted input, abuse logging) | damage, whatever the model does | engineering | treating any prompt as a security control |

Every layer adds false refusals, and they add up; the stack is only as well understood as the judge used to measure it. Prompt injection, the attack on *applications* through the data they read, is the subject of [section 4.4](#44-application-patterns) and [5.8](#58-code-execution-and-sandboxes).

**Key papers.** Bai et al. (2022), *Constitutional AI: Harmlessness from AI Feedback*; Wei, Haghtalab and Steinhardt (2023), *Jailbroken: How Does LLM Safety Training Fail?*; Zou et al. (2023), *Universal and Transferable Adversarial Attacks on Aligned Language Models* (GCG); Chao et al. (2023), *Jailbreaking Black Box Large Language Models in Twenty Queries* (PAIR); Chao et al. (2024), *JailbreakBench*; Mazeika et al. (2024), *HarmBench*; Souly et al. (2024), *A StrongREJECT for Empty Jailbreaks*; Röttger et al. (2024), *XSTest: A Test Suite for Identifying Exaggerated Safety Behaviours in Large Language Models*; Inan et al. (2023), *Llama Guard: LLM-based Input-Output Safeguard for Human-AI Conversations*.

> **Notebook:** [30 · Safety and guardrails](../../notebooks/06_llmops/30_safety_guardrails.ipynb) (read after 28) · **In the course:** a keyword refusal judge agrees with XSTest's human labels on 89.7 % of 900 replies, a 7B LLM judge on only 70.0 %; asked directly, `qwen2.5:3b` and `qwen2.5:7b` refuse 89 % and 92 % of JailbreakBench's 100 harmful behaviours, while over-refusing 17 % and 5 % of safe XSTest prompts; against the 3B model, prefix injection reaches 64 % attack success, refusal suppression 24 % and fiction framing 12 %, and 70 % of the behaviours fall to at least one; Llama Guard 3 1B as an input filter flags 95–100 % of the harmful prompts, disguised or not, but blocks 8 % of safe XSTest prompts and misses 27 % of its unsafe ones; a safety system prompt cuts prefix injection only to 44 % and raises over-refusal to 21 %; with the guard, attack success drops to 1 % (optimistic, because the same guard is the judge) at the price of 21–25 % false refusals.

---

[← Back to the index](README.md) · [Glossary](glossary.md) · [Notebook guide](notebooks_guide.md)
