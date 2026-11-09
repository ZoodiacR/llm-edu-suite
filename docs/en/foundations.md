# Theoretical foundations

🇪🇸 [Versión en español](../es/fundamentos.md) · [← Back to the index](README.md)

> The theory behind every technique and pattern in the course: the problem it solves, the core idea (with its equations), why it works, what it competes with, why it is what the industry uses today and what it replaced, and how it fails. Read it before each notebook or alongside it. Short definitions live in the [glossary](glossary.md); they are linked here, not repeated.

**How to read this page.** It follows the same order as the [topic map](README.md#topic-map). Each section ends with the notebook that implements it and an **In the course** line saying what is measured there. The numbers quoted are the ones each notebook reports, measured with small models (SmolLM2, `qwen2.5:3b`) and small evaluation sets, so treat them as large-effect signals, not benchmarks. Where a point is still argued in the literature, the text says so.

## Contents

1. [Inside the model](#1-inside-the-model)
   - [1.1 Tokenization](#11-tokenization) · [1.2 The transformer block](#12-the-transformer-block-and-the-residual-stream) · [1.3 Attention, RoPE and normalization](#13-attention-rope-and-normalization) · [1.4 Efficient attention](#14-efficient-attention-for-long-contexts) · [1.5 FFN and Mixture of Experts](#15-ffn-and-mixture-of-experts)
2. [Pre-training and inference](#2-pre-training-and-inference)
   - [2.1 The objective](#21-the-objective-next-token-prediction) · [2.2 Optimizer and schedule](#22-optimizer-and-learning-rate-schedule) · [2.3 Scaling laws](#23-scaling-laws) · [2.4 Decoding](#24-decoding-strategies) · [2.5 KV-cache](#25-prefill-decode-and-the-kv-cache) · [2.6 Batching and serving](#26-batching-and-serving) · [2.7 Speculative decoding](#27-speculative-decoding) · [2.8 Quantization](#28-quantization)
3. [Post-training](#3-post-training)
   - [3.1 SFT](#31-supervised-fine-tuning) · [3.2 LoRA](#32-lora-and-parameter-efficient-fine-tuning) · [3.3 Distillation](#33-distillation) · [3.4 RLHF, DPO and GRPO](#34-from-rlhf-to-dpo-and-grpo)
4. [Applications](#4-applications)
   - [4.1 RAG](#41-retrieval-augmented-generation) · [4.2 Retrieval metrics](#42-retrieval-metrics) · [4.3 RAGAS and LLM-as-judge](#43-ragas-and-llm-as-judge) · [4.4 Application patterns](#44-application-patterns) · [4.5 Recommendation](#45-recommender-systems)
5. [Agents and MCP](#5-agents-and-mcp)
   - [5.1 Tool calling](#51-tool-calling) · [5.2 Constrained decoding](#52-constrained-decoding-and-structured-output) · [5.3 ReAct, plan-and-execute and reflection](#53-react-plan-and-execute-and-reflection) · [5.4 State graphs](#54-langgraph-agents-as-state-machines) · [5.5 Memory](#55-memory) · [5.6 Limits and human-in-the-loop](#56-limits-and-human-in-the-loop) · [5.7 MCP](#57-model-context-protocol)
6. [LLMOps](#6-llmops)
   - [6.1 Traces and spans](#61-traces-spans-and-opentelemetry-genai) · [6.2 Offline and online evaluation](#62-offline-and-online-evaluation) · [6.3 Percentiles and SLOs](#63-percentiles-and-slos) · [6.4 PII and sampling](#64-pii-and-trace-sampling) · [6.5 Statistics of small eval sets](#65-statistics-of-small-evaluation-sets)

---

## 1. Inside the model

Terms for this part: [glossary, Tokenization and architecture](glossary.md#tokenization-and-architecture).

A causal language model is a probability distribution over token sequences, factorized by the chain rule:

$$p(x_1, \dots, x_T) = \prod_{t=1}^{T} p(x_t \mid x_{<t})$$

The network never models whole sentences. Given a prefix, it returns a distribution over the next token. Everything in this part (tokenizer, attention, FFN) exists to compute that one factor well.

```
text ─► tokenizer ─► ids ─► embedding ─► L blocks ─► norm ─► lm_head ─► logits (T × V) ─► softmax
```

> **Entry notebook:** [00 · A complete mini LLM](../../notebooks/00_start_here/00_mini_llm_end_to_end.ipynb) · **In the course:** the whole model in one notebook; the initial loss must be close to $\ln V$, and with the synthetic corpus the irreducible loss is known exactly ($4 \ln 8 = 8.32$ nats per sentence).

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

> **Notebook:** [01 · Tokenization](../../notebooks/01_architecture/01_tokenization.ipynb) · **In the course:** BPE in pure Python with round-trip, compression and fertility checks; in-domain compression rises fast with vocabulary size and saturates, while out-of-domain text barely benefits.

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

> **Notebook:** [00 · A complete mini LLM](../../notebooks/00_start_here/00_mini_llm_end_to_end.ipynb) and [02 · Attention, RoPE and normalization](../../notebooks/01_architecture/02_attention_rope_norm.ipynb) · **In the course:** the whole block is written by hand, and the model learns the corpus grammar from next-token prediction alone.

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

> **Notebook:** [02 · Attention, RoPE and normalization](../../notebooks/01_architecture/02_attention_rope_norm.ipynb) · **In the course:** numerical checks that RoPE depends only on $n-m$, that hand-written attention matches the fused kernel, and that incremental decoding with a cache equals the full forward pass; with a Llama-2-70B shape, MHA would need 2.6 MB per token (10 GB per sequence at 4k context), and GQA with 8 KV heads divides that by 8.

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

> **Notebook:** [03 · Efficient attention](../../notebooks/01_architecture/03_efficient_attention.ipynb) · **In the course:** a copy task across ~42 tokens shows the induction-head phase transition (at 300 steps the dense model copies essentially nothing, by 1500 it is near-perfect), that SWA with $w=16$ and linear attention cannot copy, and that a hybrid with one global layer out of four recovers part of the ability.

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

> **Notebook:** [04 · FFN and MoE](../../notebooks/01_architecture/04_ffn_moe.ipynb) · **In the course:** a badly imbalanced router is equalized by the aux-free bias with no gradient, while a large `bias_lr` (0.1) oscillates; dense vs MoE at equal active compute shows a small or absent MoE advantage on a toy corpus, which is the expected outcome.

---

## 2. Pre-training and inference

Terms for this part: [glossary, Training and post-training](glossary.md#training-and-post-training) and [Inference and systems](glossary.md#inference-and-systems).

### 2.1 The objective: next-token prediction

**Core idea.** Pre-training minimizes the average cross-entropy of the true next token:

$$\mathcal{L}(\theta) = -\frac{1}{T}\sum_{t=1}^{T} \log p_\theta(x_t \mid x_{<t}), \qquad \mathrm{PPL} = e^{\mathcal{L}}$$

It is self-supervised: the targets are the input shifted by one, so any text is training data. With the causal mask, a sequence of length $T$ gives $T$ examples in one pass. Because loss depends on the tokenizer, comparisons across vocabularies use *bits per byte*.

**Why the loss has a floor.** The expected cross-entropy decomposes as

$$H(p, q_\theta) = H(p) + D_{\mathrm{KL}}(p \,\Vert\, q_\theta) \;\ge\; H(p)$$

Training can only reduce the KL term; the entropy of the text is irreducible. With real text it is unknown; with a synthetic corpus it can be computed.

**What it replaced.** BERT (masked LM: only ~15 % of positions get a loss, no natural way to generate), T5 (span corruption, two stacks) and UL2 (a mixture of denoisers) lost to the causal LM as the default: every token is a target, training equals inference, and it scales with no special tricks. GPT-3 showed **in-context learning**, which settled it. Code models add *fill-in-the-middle*.

**Data decides most of the quality.** The usual pipeline (text extraction, language ID, heuristic filters, a quality classifier, exact and MinHash deduplication, PII filtering, benchmark decontamination, domain weights and a final *annealing* phase on high-quality data) is where open datasets improved most (RefinedWeb, FineWeb-Edu). Repeating data for a few epochs is nearly as good as fresh data (Muennighoff et al., 2023).

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

> **Notebook:** [05 · Pre-training](../../notebooks/02_pretraining_and_inference/05_pretraining.ipynb) · **In the course:** initial loss $\approx \ln V$ and a causal-mask leak test as sanity checks; the synthetic corpus has an exact floor ($4 \ln 10 = 9.21$ nats per sentence) that the models reach in ~100 steps, the bigger model gets there sooner, and cosine and WSD end up practically identical.

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

**The latency-throughput trade-off.** A request takes roughly $\text{TTFT} + N_{out} \times \text{TPOT}$. Raising the batch increases aggregate throughput almost for free while decode is memory-bound; once a step becomes compute-bound, every extra sequence slows everyone down. A serving system picks the largest batch that meets the p99 latency target ("goodput"). Benchmarks that report only peak throughput, or only single-request latency, hide that choice.

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

> **Notebook:** [06 · Inference and serving](../../notebooks/02_pretraining_and_inference/06_inference_serving.ipynb) · **In the course:** entropy and truncation for each strategy at the position where the model hesitates (greedy repeats a single continuation 16 times), exact equality with and without the KV-cache, speculative decoding with $k = 4$ (ceiling of 5 tokens per round) plus a statistical check that the distribution does not change; on a toy model kernel launch overhead dominates, so the cache speed-up looks modest.

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

> **Notebook:** [09 · Quantization](../../notebooks/02_pretraining_and_inference/09_quantization.ipynb) · **In the course:** with SmolLM2-360M and fp32 as the reference, bf16 alone gives KL 0.001 and 97.2 % top-1 agreement (the noise floor); int8 and float8 cut the weights from 0.72 to 0.41 GB at ~96 % agreement; int4 reaches 0.26 GB but drops to 73 % agreement and +29 % perplexity; int2 destroys the model. The quantized models were *slower* than bf16 (~24 → 17 and 10.5 tok/s) for lack of a fused kernel.

---

## 3. Post-training

Terms for this part: [glossary, Training and post-training](glossary.md#training-and-post-training).

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

**Practical rules and failure modes.** Adapting all linear layers at $r = 8$-16 usually beats a large rank on `q_proj` and `v_proj` only; LoRA needs a learning rate about 10 times higher than full fine-tuning (an empirical finding); when comparing ranks, check whether $\alpha$ scales with $r$. The **new token** trap: with tied, frozen embeddings, a row initialized at the mean of the embeddings can never win the argmax, because $h \cdot \bar e$ is the average of the $h \cdot e_i$ and an average never exceeds the maximum; that row must be trained (`modules_to_save` or `trainable_token_indices`).

> **Notebook:** [07 · SFT, LoRA and distillation](../../notebooks/03_post_training/07_finetuning_sft_lora.ipynb) · **In the course:** booking extraction to JSON with SmolLM2-135M; few-shot reaches 17.5 % exact match, full SFT 35.0 % and LoRA (3.5 % of the parameters, a 20 MB adapter) 32.5 %; rank lifts exact match from 20 % ($r=2$) to 45 % ($r=64$); with frozen embeddings the new token is never emitted (it ranks around 24,000th), and with a trainable row it comes first at p ≈ 0.93.

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

> **Notebook:** [07 · SFT, LoRA and distillation](../../notebooks/03_post_training/07_finetuning_sft_lora.ipynb) · **In the course:** a 360M teacher trained on the same 300 labels reaches 87.5 %, labels 600 new requests with 90 % exactly right, and the 135M student goes from 32.5 % to 55 % exact match (with the caveat that it also took more optimizer steps).

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

> **Notebook:** [08 · DPO and GRPO](../../notebooks/03_post_training/08_preference_rl_dpo_grpo.ipynb) · **In the course:** plain DPO wins on its own training metrics but drops exact match from 30 % to 7.5 % (the correct answer's log-probability falls from −1.91 to −5.63); DPO+NLL holds it (−1.91 → −2.01) and breaks nothing, though it only reaches 32.5 %; GRPO takes arithmetic from 5 % to 46 % (exact reward) and 42 % (dense) in 40 steps with no labels, and the dense reward cuts the share of zero-variance groups from 60-80 % to ~5-25 %.

---

## 4. Applications

Terms for this part: [glossary, Applications: LangChain and RAG](glossary.md#applications-langchain-and-rag).

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

which brings paraphrases together but blurs exact strings, names and numbers, and degrades out of domain (in BEIR, BM25 was a hard baseline). Only the bi-encoder lets documents be embedded ahead of time; the **cross-encoder** reads query and document together and is used to rerank the top 20 to 100; ColBERT (late interaction) sits in between.

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

> **Notebook:** [10 · RAG with LangChain and RAGAS](../../notebooks/04_applications/10_rag.ipynb) · **In the course:** on the course's own docs, every retriever finds the right file (hit@4 = 1.00); at evidence level hybrid is best at the top of the ranking (hit@1 0.80, MRR 0.84 vs 0.73 for dense), BM25 reaches hit@4 0.95, multi-query does not help (MRR 0.82 for one extra call), and dense gets worse with bigger chunks (MRR 0.77 → 0.67).

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

> **Notebook:** [10 · RAG with LangChain and RAGAS](../../notebooks/04_applications/10_rag.ipynb) and [15 · Observability and evaluation](../../notebooks/06_llmops/15_observability_llmops.ipynb) · **In the course:** with hybrid retrieval, context precision rises from 0.76 to 0.84, context recall from 0.57 to 0.70 and faithfulness from 0.38 to 0.54; the local 3B judge gives relevancy 0 to correct answers and the evaluation takes ~440 s; in notebook 15 the judge agrees with a deterministic check on 20 of 24 answers, and the disagreements show what each one misses.

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

> **Notebook:** [13 · Application patterns](../../notebooks/04_applications/13_modern_llm_applications.ipynb) · **In the course:** extraction with `json_schema` and no rules in the prompt gets every field right in 5 of 8 records, and 7 of 8 with the rules repeated; few-shot adds nothing to classification (20/20 both ways); "stuff" processed only ~2,050 of ~4,100 tokens with no error and covered 9 of 13 topics, map-reduce 11 of 13; the semantic cache needs a 0.94 threshold for zero false hits and then catches only 2 of 10 paraphrases; injection succeeds 9/10 with a naive prompt, 10/10 with delimiters alone and 4/10 with the sandwich, and an LLM classifier catches 10/10 with 0/10 false positives on its set.

### 4.5 Recommender systems

The recommender section is coming with the v3 recommender: the notebook is being rebuilt around an LLM and a multi-stage funnel, and its theory will be written once the new notebook is measured.

> **Notebook:** [12 · Recommender systems](../../notebooks/04_applications/12_recommender_systems.ipynb)

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

> **Notebook:** [11 · Tools and structured output](../../notebooks/05_agents_and_mcp/11_agents_tools_structured_output.ipynb) · **In the course:** the model computed `(120 - 45.5) / 2` "in its head" without the calculator (right number, wrong trajectory), used `search_docs` for chit-chat and skipped a step on a multi-step request; the schemas cost about 390 prompt tokens per call before any conversation, and with 14 cases each one is worth ~7 points.

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

> **Notebook:** [11 · Tools and structured output](../../notebooks/05_agents_and_mcp/11_agents_tools_structured_output.ipynb) · **In the course:** all three `with_structured_output` methods return a valid `Ticket` for all 8 emails, but `json_schema` returns `order_id = null` for every email that mentions an order while `function_calling` and `json_mode` extract it: the shape is guaranteed, the content is not; retrying with the error shrinks a summary from 22 to 13 words without reaching the 12-word limit.

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

**Multi-agent.** Smaller prompts, fewer tools per decision, parallel work; but more calls, and hand-offs where context is lost. Practitioners disagree publicly (Anthropic described a 2025 multi-agent research system that beat a single agent at a much higher token cost; Cognition argued the same year for single-threaded agents). The disagreement is mostly about task type. The least controversial form is a **router in front of specialists**, and a router is a **classifier**: labelled set, accuracy, confusion matrix; a router wrong 1 time in 10 caps the system near 90 % even with perfect specialists.

> **Notebook:** [14 · Agentic AI with LangGraph](../../notebooks/05_agents_and_mcp/14_agentic_ai.ipynb) · **In the course:** with `qwen2.5:3b`, the ReAct agent beats plan-and-execute with about a third of the model calls (plan-and-execute has higher tool recall and lower precision); reflection with unit tests repaired nothing in one run (4/8 before and after) and one task in others; the router scores 67-75 % depending on the run.

### 5.4 LangGraph: agents as state machines

**Problem.** In 2023 the most visible agents (AutoGPT, BabyAGI) were open-ended loops: runs that never ended, repeated actions, drifting goals, unpredictable costs, no way to test a part or resume after a crash.

**Core idea.** Put control flow back into code and give the model control only at specific points. An application is a **graph** over a typed state: nodes read the state and return updates; edges (fixed or conditional) pick the next node; a per-key **reducer** combines updates (for example, `add_messages` appends instead of overwriting). The runtime follows the **Pregel / bulk synchronous parallel** model (Malewicz et al., 2010): in each *super-step* all triggered nodes run, their updates are merged through the reducers (which is what makes parallel branches safe), and the edges choose the next super-step.

**The properties it buys.** Explicit transitions testable with fake models; visible loops where limits go; **checkpoints** (state only changes between nodes, so it is saved after each step: resume, pause for a human as long as needed, "time travel"); and mixed autonomy (a scripted branch next to a free ReAct loop). `create_agent` is a two-node graph (`model`, `tools`) with one conditional edge that encodes ReAct's entire stopping rule: if the last message has tool calls, go to `tools`; otherwise, end.

> **Notebook:** [14 · Agentic AI with LangGraph](../../notebooks/05_agents_and_mcp/14_agentic_ai.ipynb) · **In the course:** a triage graph with plain functions to show the machinery, then the ReAct agent, plan-and-execute, reflection and a router with subgraphs, all tested with fake models in `tests/test_agents_langgraph.py`.

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

> **Notebook:** [14 · Agentic AI with LangGraph](../../notebooks/05_agents_and_mcp/14_agentic_ai.ipynb) · **In the course:** a fake model that requests the same tool forever stops only at the recursion limit (with an exception) without middleware, and with middleware ends with a message and a reason; the installed version's default recursion limit is 10,007, which is why every run passes its own; refund approval is decided by a policy (up to 40 dollars) outside the model.

### 5.7 Model Context Protocol

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

> **Notebook:** [16 · MCP with the official SDK](../../notebooks/05_agents_and_mcp/16_mcp_model_context_protocol.ipynb) · **In the course:** all 14 conformance checks pass; `search_docs` gets 6/6 top-1 on an easy set; the agent's answer contains the key fact in 6 of 6 and cites the right file in 5 of 6; a `tools/call` costs ~1.4 ms in memory and ~2.7 ms over stdio, against 2-3 s per call with the adapter that opens a new session every time.

---

## 6. LLMOps

Terms for this part: [glossary, LLMOps and LangSmith](glossary.md#llmops-and-langsmith).

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

> **Notebook:** [15 · Observability and evaluation](../../notebooks/06_llmops/15_observability_llmops.ipynb) · **In the course:** the RAG version lifts judged correctness from 0.17 (closed book) to 0.83 and key-fact coverage from 0.17 to 1.00, while exact match (0.00 vs 0.58) only measures how much wording is copied; the groundedness heuristic (mean 0.81) surfaced two real bugs in the order-tool prompt that the offline set could not see.

### 6.3 Percentiles and SLOs

**Why percentiles.** Latency is never normal: bounded below, with a long right tail. For an LLM call,

$$T \;\approx\; T_{\text{queue}} + T_{\text{prefill}} + n_{\text{out}} \cdot T_{\text{per token}}$$

and because output length varies a lot and queues come in bursts, the tail is long by construction. A few slow requests drag the mean and it says nothing about them; p50 is the typical experience, p95 "1 request in 20 is slower than this", p99 the tail that trips timeouts. The tail weighs more than its share suggests (Dean and Barroso, 2013, *The Tail at Scale*): if one action needs 100 calls and each is slow 1 % of the time, the action is slow with probability $1 - 0.99^{100} \approx 63\%$, and agents have exactly that structure. Two consequences: percentiles need many samples (the p95 of 18 requests is decided by one or two of them), and they **cannot be averaged** (the mean of hourly p95s is not the daily p95), which is why systems store histograms or mergeable sketches (HDR, t-digest, DDSketch).

**SLI, SLO, SLA and error budget** (Google's SRE book, 2016). An SLI is a measurement (a ratio of good events), an SLO the target over a window, an SLA a contract with consequences. A 99 % objective explicitly *allows* 1 % bad events: while budget remains, risky changes ship; once it is spent, reliability comes first. Alerts fire on **burn rate** over a short and a long window together, so a brief spike pages no one and a sustained one does. LLMs add their own SLIs: TTFT, tokens or cost per request, the share passing a groundedness or format check, refusal rate, and the share of runs that end by hitting a step limit.

> **Notebook:** [15 · Observability and evaluation](../../notebooks/06_llmops/15_observability_llmops.ipynb) · **In the course:** 18 requests with an 11.1 % error rate (two unknown orders; the tool fails 2 of 6 calls) and 1605 input against 270 output tokens, so the retrieved context, not the answer, is the cost lever; p50/p95 of 0.22 s / 2.8 s on a free GPU and much more on a shared one, so the latency SLO passed in some executions and failed in others: with 18 requests the p95 is essentially the second-slowest request.

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

> **Notebook:** [15 · Observability and evaluation](../../notebooks/06_llmops/15_observability_llmops.ipynb) · **In the course:** the anonymizer replaces the email, phone, card and API key in the root run and every child span, and lets a name like "Ana" through; at a 0.25 rate over 40 traces, 7, 12 and 14 were kept in different executions (theoretical standard deviation ~2.7).

### 6.5 Statistics of small evaluation sets

The whole course evaluates on sets of 8 to 40 items, and that has consequences worth being able to compute.

**Standard error of a proportion.** With $n$ examples and a measured rate $\hat p$:

$$\mathrm{SE} \approx \sqrt{\frac{\hat p(1-\hat p)}{n}} \;\le\; \frac{0.5}{\sqrt{n}}$$

With $n = 12$ the 95 % interval spans roughly ±0.28; with $n = 100$, ±0.10. In a set of 40 each item is worth 2.5 points; in a set of 20, 5. A dozen examples can show a large effect (a capability one version has and the other lacks), not a difference of a few points.

**Paired design.** Most of the variation in a score comes from how hard each example is; comparing two versions on the **same** examples cancels it. For binary scores, McNemar's test looks only at the **discordant pairs** (one right, the other wrong); pairs both get right or both get wrong say nothing about which is better. For continuous scores, the **paired bootstrap**: resample the per-question differences with replacement (say 2,000 times) and take the 2.5 and 97.5 percentiles; if the interval excludes 0, the difference is unlikely to be noise.

**Other sources of variance.** At temperature 0 outputs can still change between runs (GPU kernels are not bit-for-bit deterministic, batching); an LLM judge flips verdicts on borderline cases; *pass^k* exposes inconsistency a single run hides. Useful habits: repeat runs and report the spread, always read the per-example table (the examples that change verdict between versions are where the explanation is), and put cost and latency in the same table as quality.

**What goes into the dataset** matters as much as its size: common intents, known edge cases, adversarial inputs and every production failure already fixed (a regression set). Version it, so each experiment is tied to the exact examples, and keep it out of prompt tuning, or the scores will measure memorization of the test set.

> **Notebook:** [15 · Observability and evaluation](../../notebooks/06_llmops/15_observability_llmops.ipynb) and [10 · RAG with LangChain and RAGAS](../../notebooks/04_applications/10_rag.ipynb) · **In the course:** bootstrap intervals and the paired hybrid-minus-dense difference on RAGAS (notebook 10), the ±0.28 calculation for $n = 12$ in notebook 15, and in every application notebook a reminder of what one item is worth before reading a difference.

---

[← Back to the index](README.md) · [Glossary](glossary.md) · [Notebook guide](notebooks_guide.md)
