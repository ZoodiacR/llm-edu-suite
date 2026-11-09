🇪🇸 [Versión en español](../es/glosario.md)

# Glossary

[← Back to the index](README.md)

Short definitions of the terms used across the suite. In brackets, the module, library or notebook (NB) where each one appears. The last column gives the term used in the Spanish docs, where it differs from the English one (many terms are kept in English in Spanish too, because that is how they appear in papers, libraries and job postings).

## Tokenization and architecture

| Term | Meaning | Spanish docs |
|---|---|---|
| **Token** | The smallest unit the model processes: a word piece, a frequent word or a single byte. | token |
| **BPE (Byte-Pair Encoding)** | Builds the vocabulary by repeatedly merging the most frequent pair of symbols. The *byte-level* variant starts from the 256 bytes, so it never produces `<UNK>`. (`tokenizer.py`, NB 01) | |
| **Fertility** | Average number of tokens per word. Higher = the tokenizer splits text more (typical for under-represented languages). | |
| **Compression ratio** | Bytes of text per token. Higher = shorter sequences = cheaper inference. | relación de compresión |
| **Embedding** | A learned vector that represents a token (or a document, or an item in a recommender). | |
| **Decoder-only / causal LM** | A transformer that predicts the next token and can only look backwards (causal mask). The architecture of GPT, Llama, Qwen, DeepSeek, SmolLM2. (`model.py`) | |
| **Self-attention** | Each position computes a weighted average of the others: `softmax(QKᵀ/√d)·V`. Cost O(T²) in the sequence length T. | autoatención |
| **MHA / GQA / MQA** | *Multi-Head*, *Grouped-Query* and *Multi-Query Attention*. GQA shares each K/V head across several Q heads to shrink the KV-cache. (`layers.py`, NB 02) | |
| **MLA (Multi-head Latent Attention)** | DeepSeek-V2/V3 technique: compresses K and V into a small latent vector, which is the only thing kept in the cache. | |
| **RoPE (Rotary Position Embedding)** | Encodes position by rotating pairs of dimensions of Q and K, so the QKᵀ product depends only on the relative distance. Two conventions choose the pairs: interleaved (2i, 2i+1), used by Meta's original Llama and by this course, and *rotate_half* (i, i+d/2), used by Hugging Face; they are equivalent once the rows of W_q and W_k are permuted. | |
| **Checkpoint / safetensors** | A published model is code (the architecture) + `config.json` (hyperparameters) + weights. `safetensors` is the standard weight format: a name → tensor dictionary that loads without running code and can be memory-mapped. | checkpoint / safetensors |
| **RMSNorm** | Normalization that only divides by the root mean square (no mean subtraction). Cheaper than LayerNorm; standard in Llama. | |
| **Pre-norm / sandwich norm / QK-norm** | Where the normalizations go: before each sub-block, before and after it, or on Q and K to stabilize training. | |
| **SwiGLU** | Gated FFN: `W_down(SiLU(W_gate·x) ⊙ W_up·x)`. Standard in modern LLMs. | |
| **MoE (Mixture of Experts)** | Replaces the FFN with N experts and a *router* that activates only k of them per token: many total parameters, few active ones. (NB 04) | mezcla de expertos |
| **Aux-loss vs aux-free balancing** | Two ways to stop the router from always picking the same experts: an auxiliary loss, or a per-expert bias adjusted without gradients (DeepSeek-V3). | balanceo con / sin pérdida auxiliar |
| **Shared expert** | An expert that always processes every token, on top of the routed ones. | experto compartido |
| **Sliding-window / Longformer / block-sparse / linear attention** | Efficient attention variants for long contexts: local window, window + global tokens, content-selected blocks, and O(T) linear attention. (NB 03) | |
| **Attention sink** | A position (often the first token or the end-of-sequence token) where a layer parks attention probability when it has nothing specific to read. It must be kept when trimming the KV-cache, and in attention maps it hides the real alignment. (NB 03, NB 17) | attention sink |
| **Encoder-only** | A transformer with bidirectional attention (no causal mask): every token sees every other. It does not generate; it gives each position a representation that depends on the whole input. BERT, RoBERTa, DeBERTa, ModernBERT. Used for classification, entities, embeddings and rerankers. In the course, `ModelConfig(causal=False)`. (NB 17) | encoder-only |
| **Encoder-decoder** | A transformer with both halves of the original: a bidirectional encoder reads the input once and a causal decoder writes the output while reading the encoder states through *cross-attention*. T5, BART, Flan-T5, Whisper. Strong for translation and speech. (NB 17) | encoder-decoder |
| **MLM (masked language modeling)** | BERT's pre-training objective: 15 % of the positions are hidden (80 % `[MASK]`, 10 % a random token, 10 % unchanged) and the model recovers them from both sides. Only the hidden positions are graded; ELECTRA uses all of them with *replaced token detection*. (NB 17) | modelado de lenguaje enmascarado |
| **Cross-attention** | Attention whose queries come from one sequence (the output the decoder is writing) and whose keys and values come from another (the encoder states). In T5, each decoder block is self-attention → cross-attention → FFN. (NB 17) | cross-attention |
| **[CLS] / pooling** | Two ways to summarize a sentence in one vector from an encoder: the state of the special first token `[CLS]`, or the mean of all token states (*mean pooling*, ignoring padding through the attention mask). (NB 17) | [CLS] / pooling |
| **Classification head** | A linear layer on the sentence vector that gives a distribution over a fixed label set, `softmax(W·h + b)`. `AutoModelForSequenceClassification` adds it randomly initialised and it is fine-tuned together with the encoder. (NB 17) | cabeza de clasificación |
| **ModernBERT** | A 2024 encoder (Warner et al.) that applies what decoders learned to BERT: RoPE, GeGLU, pre-norm, global layers alternating with 128-token windows, *unpadding* and an 8,192-token context. The base version (150M) reaches 93.0 % on Banking77 when fine-tuned. (NB 17) | ModernBERT |
| **Linear probe** | A linear classifier (for example logistic regression) trained on frozen representations. It measures how much task-relevant information they already hold; in NB 17, frozen ModernBERT gives 89.2 % and `nomic-embed-text` 93.3 % on Banking77. | linear probe |

## Training and post-training

| Term | Meaning | Spanish docs |
|---|---|---|
| **Pre-training** | Training from scratch on next-token prediction over a large corpus. (NB 05) | preentrenamiento |
| **Perplexity** | `exp(mean loss)`. Intuition: how many tokens the model is "hesitating" between. Lower is better. | perplejidad |
| **Warmup / cosine / WSD** | Learning-rate schedules: initial ramp-up; cosine decay; or *Warmup-Stable-Decay* (constant, then a final drop). | calentamiento |
| **AdamW / weight decay** | The standard optimizer. Weight decay is not applied to norms or biases (parameter groups). | |
| **Gradient accumulation / clipping / AMP** | Simulate large batches, cap the gradient norm, and use mixed precision (bf16/fp16). | acumulación / recorte de gradiente, precisión mixta |
| **Gradient checkpointing** | Recomputing activations during the backward pass instead of storing them: less memory, about 30 % more compute. On by default in `scripts/finetune.py`. | |
| **SFT (Supervised Fine-Tuning)** | Tuning the model on instruction→response pairs, usually rendered with the model's **chat template**. (NB 07) | ajuste fino supervisado |
| **Completion-only loss / loss masking** | Computing the SFT loss only on the response tokens: the prompt tokens get label `-100`, so the cross-entropy ignores them. TRL does it automatically for *prompt-completion* datasets. (NB 07) | enmascarado de pérdida |
| **SFTTrainer** | TRL's trainer for SFT: takes a conversational or prompt-completion dataset, applies the chat template, masks the prompt, and accepts a `peft_config` to train LoRA adapters. Configured with `SFTConfig`. (`scripts/finetune.py`, NB 07) | |
| **LoRA** | Freezes W and learns a low-rank correction `W + (α/r)·B·A`. With B=0 at init, the model starts identical to the base; after training it can be *merged* (`merge_and_unload` in `peft`) at no inference cost. (NB 07) | |
| **LoRA `modules_to_save`** | `peft` option that trains (and saves) full copies of chosen modules next to the adapters, typically the embedding and `lm_head`. Needed when fine-tuning introduces tokens the base model never learned, such as new chat-role tokens. (NB 07) | |
| **QLoRA** | LoRA on a base model whose frozen weights are stored in 4 bits (NF4); the adapters stay in bf16 and every matmul dequantizes the weight first. It saves weight memory (a 7B model goes from ~14 GB to ~3.6 GB), not activation memory, and each step is a little slower. In `transformers`, `BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", ...)` + `prepare_model_for_kbit_training`. (NB 07) | QLoRA |
| **Double quantization** | Quantizing the per-block scales too (to 8 bits): the overhead drops from 0.5 to about 0.127 bits per parameter. One of QLoRA's three ideas (`bnb_4bit_use_double_quant=True`). (NB 07) | doble cuantización |
| **Paged optimizer** | Optimizer state lives in CUDA unified memory and is paged to CPU RAM when memory spikes, instead of failing with out-of-memory (`optim="paged_adamw_8bit"`). (NB 07) | optimizador paginado |
| **Distillation** | A *student* model learns to imitate a *teacher's* distribution (KL between logits). (NB 07) | destilación |
| **RLHF / preference tuning** | Aligning the model with human preferences: chosen vs rejected response. (NB 08) | alineamiento con preferencias |
| **DPO (Direct Preference Optimization)** | Optimizes preferences directly, with no reward model: `−log σ(β·[(log π−log π_ref)(y_w) − (log π−log π_ref)(y_l)])`. Equals ln 2 at init. | |
| **DPOTrainer** | TRL's trainer for DPO and its variants (`loss_type`). With a `peft_config`, the reference model is the same network with the adapter switched off, so no second copy is kept in memory. Configured with `DPOConfig` (`beta`, `loss_type`, `loss_weights`). (`scripts/finetune.py`, NB 08) | |
| **Likelihood displacement / DPO + NLL (RPO)** | DPO can widen the margin by pushing down the probability of *both* the chosen and the rejected answer. Adding the NLL (SFT loss) of the chosen answer counters it: in TRL, `loss_type=["sigmoid", "sft"]`. | |
| **GRPO (Group Relative Policy Optimization)** | RL from DeepSeekMath/R1: samples a group of responses per prompt and uses the reward normalized within the group as the advantage (no *critic*). A group where every sample gets the same reward gives no gradient. | |
| **GRPOTrainer** | TRL's trainer for GRPO: takes a prompt dataset and one or more Python **reward functions** `(completions, **columns) -> list[float]`, samples `num_generations` completions per prompt and optimizes the policy with a KL penalty (`beta`). Configured with `GRPOConfig`. (`scripts/finetune.py`, NB 08) | |
| **Verifiable reward** | A reward a program can compute (is the answer correct?), with no human or model judge. A *dense* reward gives partial credit, so groups are less likely to tie. | recompensa verificable |
| **KL penalty** | A term that keeps the policy from drifting too far from the reference model. | penalización KL |

## Inference and systems

| Term | Meaning | Spanish docs |
|---|---|---|
| **KV-cache** | Storing K and V of already-processed tokens so they are not recomputed at every generation step. Its size is often the memory bottleneck. (NB 02, 06) | |
| **Prefill / decode** | The phase that processes the prompt (parallel) and the phase that generates token by token (sequential, memory-bound). | |
| **TTFT / TPOT / throughput** | *Time To First Token*, time per output token, and total tokens per second. | |
| **Greedy / temperature / top-k / top-p** | Sampling strategies: the most likely token; flatten or sharpen the distribution; keep the k most likely; or keep the nucleus of cumulative probability p. `model.next_token_distribution` shows the resulting distribution. | muestreo |
| **Speculative decoding** | A small model (*draft*) proposes several tokens and the large one verifies them in a single pass. With the right acceptance rule, the output distribution is **identical** to the large model's. (`inference.py`, NB 06) | decodificación especulativa |
| **Continuous batching / PagedAttention** | vLLM techniques for serving many requests at once by managing the KV-cache as paged memory. | |
| **Quantization (INT8 / INT4 / NF4)** | Storing weights with fewer bits. *absmax* scales by the maximum absolute value; per-block scaling reduces the error; NF4 uses levels that are optimal for normally distributed weights. (NB 09) | cuantización |
| **optimum-quanto** | Hugging Face library that quantizes a PyTorch / `transformers` model in place: `quantize(model, weights=qint8 / qint4 / qfloat8)` swaps the linear layers for quantized ones and `freeze(model)` stores the quantized weights. Works on CPU and GPU. (NB 09) | |
| **NF4 (NormalFloat4)** | A 4-bit format with 16 levels placed at the quantiles of a normal distribution, with one absmax scale per block of 64 weights. Since pretrained weights are roughly normal, it loses less precision than int4 with evenly spaced levels. It is QLoRA's format (`bitsandbytes`). (NB 07, 09) | NF4 |
| **GGUF / Q4_K_M** | The file format and block-wise 4-bit quantization scheme that llama.cpp and Ollama use. `qwen2.5:3b` in Ollama is a Q4_K_M GGUF of about 1.9 GB. | |
| **Ollama** | A local server for open models: `ollama pull` downloads a model, and the server (port 11434) offers a native API and an OpenAI-compatible one (`/v1`). The course's default provider for notebooks 10–16 (`qwen2.5:3b`, `nomic-embed-text`), used through `ChatOllama` / `OllamaEmbeddings` from `langchain-ollama`. | |

## Applications: LangChain and RAG

| Term | Meaning | Spanish docs |
|---|---|---|
| **RAG (Retrieval-Augmented Generation)** | Retrieving relevant chunks of your documents and passing them to the LLM as context. (NB 10, `scripts/rag_demo.py`) | |
| **Chunking** | Splitting documents into (overlapping) pieces before indexing them, e.g. with LangChain's `RecursiveCharacterTextSplitter`. | troceado |
| **Runnable** | LangChain's common interface: every component (prompt, model, parser, retriever, function wrapped in `RunnableLambda`) has `invoke`, `batch` and `stream` (plus async versions), so they can be composed and traced the same way. | |
| **LCEL (LangChain Expression Language)** | Composing Runnables with `\|` (`prompt \| llm \| StrOutputParser()`). `RunnableParallel` runs branches on the same input, `RunnablePassthrough` forwards it unchanged, `.assign(...)` adds keys to a dict. | |
| **Vector store** | A database of embeddings that returns the nearest neighbours of a query vector: `FAISS` and `InMemoryVectorStore` in the course; pgvector, Qdrant or Weaviate in production. | almacén vectorial |
| **Retriever** | A Runnable that maps a query string to a list of `Document`s. `vector_store.as_retriever(search_kwargs={"k": 4})`, `BM25Retriever`, `EnsembleRetriever` and `MultiQueryRetriever` are all retrievers. | recuperador |
| **Dense retrieval / BM25 / hybrid** | Search by embeddings (semantic), by terms (lexical), or both combined. | búsqueda densa / léxica / híbrida |
| **RRF (Reciprocal Rank Fusion)** | Fusing rankings by summing `1/(60 + rank)`. It uses only ranks, not scores, so it needs no calibration. | |
| **EnsembleRetriever** | LangChain retriever (in `langchain_classic` in LangChain 1.x) that runs several retrievers and fuses their rankings with **weighted RRF**. The course combines BM25 and FAISS with weights 0.5 / 0.5. | |
| **Reranker / cross-encoder** | A second, more expensive stage that rescores the top candidates of a fast retriever by reading query and document together. An LLM can play the role, but it is not always worth it: in recommendation, a small LLM re-ordering a collaborative model's list made it worse (NB 12 in v2), and the current version uses a LambdaMART ranker. | reordenador |
| **recall@k / hit@k / MRR / nDCG** | Retrieval metrics: is the right document in the top-k?, inverse rank of the first correct one, and discounted cumulative gain by position. `rag_demo.py` reports them at file level and at *evidence* level (does the chunk actually contain the answer?). | |
| **Faithfulness / groundedness** | Whether the answer is supported by the retrieved context (rather than made up). | fidelidad |
| **RAGAS** | Library for evaluating RAG with an LLM as judge. Its four core metrics: **faithfulness** (are the answer's claims supported by the retrieved context?), **answer relevancy** (does the answer address the question? measured by generating questions back from the answer and comparing embeddings), **context precision** (are the relevant chunks ranked high?) and **context recall** (does the retrieved context cover the reference answer?). The last two need a reference answer. (NB 10, NB 13, `rag_demo.py --ragas`) | |
| **LLM-as-judge** | Using an LLM to score or compare answers (pairwise, with a rubric). Position and length biases must be controlled, and the judge must be validated before it is trusted. | LLM como juez |
| **Judge strength** | How capable the judge is relative to the system it grades. A weak judge does not only add noise: it can invent differences between systems. In NB 10, the 3B model grading itself reported faithfulness 0.38 → 0.54 between dense and hybrid retrieval; with `qwen2.5:7b` as the judge, 0.87 for both. Rule: a judge stronger than the system, kept fixed while comparing versions (`config.judge_model()`, `JUDGE_MODEL`). | fuerza del juez |
| **Structured output** | Forcing outputs that satisfy a schema. In LangChain, `llm.with_structured_output(PydanticModel)` uses the provider's tool calling or JSON mode and returns a validated object. (NB 11) | salida estructurada |
| **Constrained decoding** | The server turns the schema into a grammar and, at every step, masks the tokens that would break it, so the output is always valid (in Ollama, `method="json_schema"`). It guarantees the shape, not the content. With a large enum it is required: in NB 17, with 77 intents, `function_calling` gave 181 invalid outputs out of 231 and `json_schema` none; in NB 11, on a small schema, function calling was the more reliable method. (NB 11, NB 13, NB 17) | decodificación restringida |
| **Tool calling / function calling** | The model returns a structured call (`name` + JSON arguments) that your code executes. In LangChain a tool is a function decorated with `@tool`, and `llm.bind_tools(...)` exposes it to the model. (NB 11) | llamada a herramientas |
| **Agent loop** | Model → tool → observation → model loop, with a step limit. `create_agent` from `langchain.agents` builds it on LangGraph. | bucle de agente |
| **Middleware (LangChain agents)** | Hooks that wrap the agent's model and tool calls to add behaviour without rewriting the loop: `ModelCallLimitMiddleware` / `ToolCallLimitMiddleware` (budgets), `HumanInTheLoopMiddleware` (approval before a tool runs), `ToolErrorMiddleware` (turn tool exceptions into messages), `SummarizationMiddleware` (compress long histories), or your own `AgentMiddleware`. (NB 11, 14, 16) | |
| **Human-in-the-loop / confirmation gate** | Asking for approval before running actions with side effects (payments, deletions, sending messages). | compuerta de confirmación |
| **Prompt injection** | Malicious text inside the data (a web page, an email, a document, a tool result) that tries to give orders to the model. (NB 11, 13, 16) | inyección de prompts |
| **Semantic cache** | Reusing answers to *similar* questions (by embedding similarity). Risk: *false hits*. (NB 13) | caché semántica |

## Recommender systems

| Term | Meaning | Spanish docs |
|---|---|---|
| **Recommendation funnel** | The staged architecture of an industrial recommender: **candidate generation** (millions to hundreds, cheap, measured by recall), **ranking** (hundreds to tens, many features, measured by nDCG) and **re-ranking** (rules, diversity, exploration, explanations). (NB 12) | embudo de recomendación |
| **Candidate generation / retrieval** | The first stage of the funnel: several cheap sources (sequential model, two-tower + ANN, co-occurrence, content, popular lists) whose **union** forms the candidate set. Its recall is the ranker's ceiling: what is not retrieved cannot be ranked. (NB 12) | generación de candidatos |
| **Sequential recommender (SASRec)** | Recommendation as "next-item prediction" with a causal transformer over the user's history; the vocabulary is the catalog. It trains better with the full softmax than with sampled negatives; at the scale of millions, **sampled softmax with the logQ correction** (subtract from each logit the log of its sampling probability). (NB 12) | recomendador secuencial |
| **Two-tower model** | Two networks that map user and item into the same space; the score is their dot product. Because the item tower does not depend on the user, its vectors are precomputed and served from an ANN index. With **id dropout** (hiding the id at random during training) it learns to place new items from their content. (NB 12) | modelo de dos torres |
| **ANN / HNSW** | *Approximate nearest neighbours*: indexes that return almost the same neighbours as exact search in far less time. HNSW is the hierarchical-graph one (FAISS, ScaNN, vector databases); its recall against exact search is a quality knob of its own. (NB 10, 12) | búsqueda aproximada de vecinos |
| **Learning to rank** | Training a model to order items: **pointwise** (predict each label on its own), **pairwise** (prefer the positive to each negative, RankNet) or **listwise** (optimise the quality of the whole list). (NB 12) | aprendizaje de ranking |
| **LambdaMART** | Gradient-boosted trees whose gradients are RankNet's pairwise gradients weighted by how much nDCG would change if the two items were swapped. The default industrial ranker; in LightGBM, `objective="lambdarank"`. It only adds something over a strong first stage when it has many queries from a later period and features that stage cannot see. (NB 12) | LambdaMART |
| **Cold start** | Recommending items (or to users) with no interaction history. Collaborative models cannot: the item's row was never trained. Content (text, metadata) places the item and exploration gives it exposure. (NB 12) | arranque en frío |
| **Exploration slot / bandit** | Reserving positions in the list for new or uncertain items, to learn what they are worth (the exploration/exploitation dilemma). *Bandits* ($\varepsilon$-greedy, UCB, Thompson sampling, contextual ones like LinUCB) decide it adaptively; a fixed slot is the minimal version you can measure offline. (NB 12) | ranura de exploración |
| **MMR (Maximal Marginal Relevance)** | Build the list greedily, each time picking $\lambda \cdot$ relevance $- (1-\lambda) \cdot$ maximum similarity to what is already chosen. With $\lambda = 1$ it is the plain ranking; lower means more diversity. (NB 12) | relevancia marginal máxima |
| **Intra-list diversity** | One minus the mean pairwise similarity of the items in a list (in NB 12, genre Jaccard). Offline it can only look neutral or worse for accuracy; its benefit is measured online. **Calibration** (Steck, 2018) also asks the list's genre mix to resemble the history's. | diversidad intra-lista |
| **Leave-last-out / leave-last-two** | A per-user time split: the last event is the test target (and the one before, the validation target); everything earlier is history. A random split would leak the future. (`recsys.temporal_split`, NB 12) | dejar fuera el último |
| **Full ranking vs sampled metrics** | Ranking the target against **the whole** catalog or against ~100 random negatives. Sampled metrics inflate the numbers and can flip which model looks best (Krichene and Rendle, 2020). (`recsys.target_ranks`, NB 12) | ranking completo frente a métricas muestreadas |

## Agents and LangGraph

| Term | Meaning | Spanish docs |
|---|---|---|
| **Workflow vs agent** | In a workflow the code fixes the control flow and the LLM fills in steps; in an agent the LLM chooses the next step itself, in a loop. Start with the simplest one that works. (NB 14) | workflow frente a agente |
| **StateGraph** | LangGraph's builder for a graph whose nodes read and update a shared, typed **state** (a `TypedDict`, or `MessagesState` for chat). `builder.compile()` turns it into a Runnable. (NB 14) | |
| **Node** | A function `state -> partial state update` in a LangGraph graph: an LLM call, a retrieval, a tool, plain Python. | nodo |
| **Edge / conditional edge** | An edge fixes the next node (`add_edge("a", "b")`, from `START` or to `END`). A conditional edge (`add_conditional_edges`) calls a function of the state that names the next node: routing, loops, retries. | arista / arista condicional |
| **Checkpointer** | Saves the graph state after every step (`InMemorySaver` in the course; SQLite or Postgres in production). It gives a graph short-term memory, lets it resume after a failure or a pause, and enables time travel. | |
| **Thread (`thread_id`)** | The key under which a checkpointer stores one conversation: `config={"configurable": {"thread_id": "..."}}`. Same thread = same memory. | hilo |
| **Store** | LangGraph's long-term memory: a key-value store organised in namespaces (e.g. `("users", user_id)`), shared across threads, with optional semantic search when given an embedding model (`InMemoryStore`). | |
| **Interrupt** | `interrupt(value)` pauses a graph inside a node and returns `value` to the caller; the run resumes with `Command(resume=...)`. Needs a checkpointer. The basis of human-in-the-loop in LangGraph. | interrupción |
| **`recursion_limit` / agent budget** | Hard limits on one agent run: LangGraph's `recursion_limit` caps the number of steps, and the call-limit middlewares cap model and tool calls. When one is hit the run stops instead of looping. | presupuesto del agente |
| **Plan-and-execute** | Pattern that first asks the LLM for a numbered plan, then runs each step with a tool-using agent and finally synthesizes the answer. Inspectable and approvable. (NB 14) | |
| **Reflection / verifier** | Draft → verify → revise loop. The verifier checks the draft and returns feedback; code (tests, schema validators) is far more reliable than an LLM critic. (NB 14) | reflexión / verificador |
| **Router / supervisor** | A node that sends each request to a specialist (a sub-agent or a chain), by an LLM classification or by embedding similarity. (NB 13, 14) | |
| **Short-term / long-term memory** | The conversation of the current thread (checkpointer, possibly summarized) vs facts that survive across threads (store, retrieved by similarity). | memoria a corto / largo plazo |
| **Trajectory evaluation** | Scoring the path an agent took (which tools, in which order, how many extra calls), not only its final answer. (NB 11) | evaluación de trayectorias |

## LLMOps and LangSmith

| Term | Meaning | Spanish docs |
|---|---|---|
| **LangSmith** | LangChain's platform for tracing, monitoring and evaluating LLM apps. With `LANGSMITH_TRACING=true` and an API key, every LangChain / LangGraph call, and every function decorated with `@traceable`, is recorded. Optional in the course. (NB 15) | |
| **Run** | One unit of work recorded by LangSmith: an LLM call, a retriever, a tool, a chain, a graph node. Each run has inputs, outputs, timing, token counts, errors, tags, metadata and a parent run. | ejecución |
| **Trace** | The tree of runs produced by one top-level request, from the root run down to every nested call. The span tree of OpenTelemetry. | traza |
| **Project** | The LangSmith container that traces are sent to (`LANGSMITH_PROJECT`, or `setup(project=...)` per notebook). Dashboards and filters work per project. | proyecto |
| **Dataset** | A versioned set of examples (inputs + reference outputs) stored in LangSmith for offline evaluation. | |
| **Experiment** | One scored run of your app over a dataset (`evaluate(target, data=..., evaluators=[...])`). Comparing two experiments compares two versions of the app on the same examples. | experimento |
| **Evaluator** | A function that scores a run, with the signature `(inputs, outputs, reference_outputs) -> score`: a heuristic, a code check, or an LLM-as-judge. Used offline in experiments and online on production traces. | evaluador |
| **Feedback** | A score or comment attached to a run (`client.create_feedback(run_id, key, score=...)`): user thumbs, an evaluator's grade, a heuristic. The online-evaluation signal. | |
| **OTel GenAI semantic conventions** | Standard OpenTelemetry attribute names for LLM telemetry (`gen_ai.request.model`, `gen_ai.usage.input_tokens`, ...), so traces can go to any backend. LangSmith can ingest and export OpenTelemetry. | convenciones semánticas GenAI |
| **p50 / p95 / p99** | Latency percentiles: the value below which 50 / 95 / 99 % of requests fall. Reported instead of the mean, because the tail is what users notice. | |
| **SLO (Service-Level Objective)** | A target for a service metric, e.g. "p95 < 2 s, error rate < 1 %". Violations trigger alerts. (NB 15) | objetivo de nivel de servicio |
| **PII redaction** | Replacing personal data (e-mails, phone and card numbers, API keys) with placeholders before prompts and responses reach logs or traces. (NB 15) | redacción de PII |
| **Head sampling** | Deciding at the start of a trace whether to keep it as a whole (in LangSmith, `LANGSMITH_TRACING_SAMPLING_RATE=0.1` keeps 10 % of traces). *Tail sampling* decides at the end, e.g. to keep every error. | |
| **Bootstrap confidence interval** | Resample the per-example scores with replacement (say 2,000 times), recompute the mean each time and take the 2.5 % and 97.5 % quantiles: a 95 % interval without assuming any distribution. With 20 questions it comes out wide, and that is information. (`evaluation.bootstrap_ci`, NB 10, 12) | intervalo de confianza bootstrap |
| **Paired bootstrap** | The same resampling applied to the per-example differences between two systems evaluated on the same examples (questions, users). Pairing cancels the variation from how hard each example is; if the interval of the difference excludes 0, the difference is unlikely to be noise. (`evaluation.paired_bootstrap_diff`, NB 10, 12) | bootstrap pareado |

## MCP

| Term | Meaning | Spanish docs |
|---|---|---|
| **MCP (Model Context Protocol)** | Open protocol that standardizes how LLM applications get tools, data and prompts from external programs, over JSON-RPC 2.0. (NB 16, `scripts/mcp_server.py`) | |
| **MCP host / client / server** | The host is the user-facing app (Claude Code, Claude Desktop, an IDE, your agent); it runs one client per server; the server is the program that exposes capabilities. | |
| **Tool / resource / prompt (MCP)** | The three server primitives: tools the model calls, read-only resources the application loads into context (fixed URIs or URI templates like `docs://{name}`), and prompt templates the user picks. | |
| **Tool annotations** | Hints a server attaches to a tool (`readOnlyHint`, `destructiveHint`, `idempotentHint`, `openWorldHint`) so the host can decide, for example, to ask for confirmation before a destructive call. They are hints, not guarantees. | anotaciones de herramientas |
| **FastMCP** | The high-level server API of the official MCP Python SDK (`mcp.server.fastmcp.FastMCP` in 1.x): `@server.tool()`, `@server.resource(uri)` and `@server.prompt()` turn functions into primitives, deriving the JSON Schema from type hints and the description from the docstring; `server.run()` serves stdio, `server.run(transport="streamable-http")` HTTP. | |
| **MCP adapters (`langchain-mcp-adapters`)** | Library that turns the tools of one or more MCP servers into LangChain tools: `MultiServerMCPClient({...}).get_tools()`, ready for `create_agent`. `tool_name_prefix` namespaces tools from different servers. (NB 16) | adaptadores MCP |
| **JSON-RPC request vs notification** | A request has an `id` and always gets a response (`result` or `error`); a notification has no `id` and never gets one (e.g. `notifications/initialized`). | petición / notificación |
| **Protocol error vs tool error (`isError`)** | A JSON-RPC `error` means the request itself was wrong (unknown method, bad params); a result with `isError: true` means the tool ran and failed, so the model can see it and retry. | error de protocolo / de herramienta |
| **stdio transport** | The host launches the server as a subprocess and exchanges one JSON message per line on stdin/stdout. stdout is reserved for the protocol: logs go to stderr. | transporte stdio |
| **Streamable HTTP transport** | The server listens on one HTTP endpoint (e.g. `/mcp`); the client POSTs JSON-RPC messages and the server answers with JSON or an event stream, keeping a session with the `mcp-session-id` header. For remote or shared servers. | transporte Streamable HTTP |
| **Tool poisoning** | Attack where a malicious MCP server hides instructions in a tool's description (read by the model, rarely by the user) or changes it after approval. Mitigations: allowlists, namespacing, reviewing descriptions, confirmation for risky tools. | envenenamiento de herramientas |
