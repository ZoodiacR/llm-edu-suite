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
| **RoPE (Rotary Position Embedding)** | Encodes position by rotating pairs of dimensions of Q and K, so the QKᵀ product depends only on the relative distance. | |
| **RMSNorm** | Normalization that only divides by the root mean square (no mean subtraction). Cheaper than LayerNorm; standard in Llama. | |
| **Pre-norm / sandwich norm / QK-norm** | Where the normalizations go: before each sub-block, before and after it, or on Q and K to stabilize training. | |
| **SwiGLU** | Gated FFN: `W_down(SiLU(W_gate·x) ⊙ W_up·x)`. Standard in modern LLMs. | |
| **MoE (Mixture of Experts)** | Replaces the FFN with N experts and a *router* that activates only k of them per token: many total parameters, few active ones. (NB 04) | mezcla de expertos |
| **Aux-loss vs aux-free balancing** | Two ways to stop the router from always picking the same experts: an auxiliary loss, or a per-expert bias adjusted without gradients (DeepSeek-V3). | balanceo con / sin pérdida auxiliar |
| **Shared expert** | An expert that always processes every token, on top of the routed ones. | experto compartido |
| **Sliding-window / Longformer / block-sparse / linear attention** | Efficient attention variants for long contexts: local window, window + global tokens, content-selected blocks, and O(T) linear attention. (NB 03) | |

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
| **QLoRA** | LoRA on top of a base model quantized to 4 bits (NF4). | |
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
| **Reranker / cross-encoder** | A second, more expensive stage that rescores the top candidates of a fast retriever by reading query and document together. An LLM can play the role (NB 12 reranks recommendations with a structured-output call). | reordenador |
| **recall@k / hit@k / MRR / nDCG** | Retrieval metrics: is the right document in the top-k?, inverse rank of the first correct one, and discounted cumulative gain by position. `rag_demo.py` reports them at file level and at *evidence* level (does the chunk actually contain the answer?). | |
| **Faithfulness / groundedness** | Whether the answer is supported by the retrieved context (rather than made up). | fidelidad |
| **RAGAS** | Library for evaluating RAG with an LLM as judge. Its four core metrics: **faithfulness** (are the answer's claims supported by the retrieved context?), **answer relevancy** (does the answer address the question? measured by generating questions back from the answer and comparing embeddings), **context precision** (are the relevant chunks ranked high?) and **context recall** (does the retrieved context cover the reference answer?). The last two need a reference answer. (NB 10, NB 13, `rag_demo.py --ragas`) | |
| **LLM-as-judge** | Using an LLM to score or compare answers (pairwise, with a rubric). Position and length biases must be controlled, and a small judge is noisy. | |
| **Structured output** | Forcing outputs that satisfy a schema. In LangChain, `llm.with_structured_output(PydanticModel)` uses the provider's tool calling or JSON mode and returns a validated object. (NB 11) | salida estructurada |
| **Tool calling / function calling** | The model returns a structured call (`name` + JSON arguments) that your code executes. In LangChain a tool is a function decorated with `@tool`, and `llm.bind_tools(...)` exposes it to the model. (NB 11) | llamada a herramientas |
| **Agent loop** | Model → tool → observation → model loop, with a step limit. `create_agent` from `langchain.agents` builds it on LangGraph. | bucle de agente |
| **Middleware (LangChain agents)** | Hooks that wrap the agent's model and tool calls to add behaviour without rewriting the loop: `ModelCallLimitMiddleware` / `ToolCallLimitMiddleware` (budgets), `HumanInTheLoopMiddleware` (approval before a tool runs), `ToolErrorMiddleware` (turn tool exceptions into messages), `SummarizationMiddleware` (compress long histories), or your own `AgentMiddleware`. (NB 11, 14, 16) | |
| **Human-in-the-loop / confirmation gate** | Asking for approval before running actions with side effects (payments, deletions, sending messages). | compuerta de confirmación |
| **Prompt injection** | Malicious text inside the data (a web page, an email, a document, a tool result) that tries to give orders to the model. (NB 11, 13, 16) | inyección de prompts |
| **Semantic cache** | Reusing answers to *similar* questions (by embedding similarity). Risk: *false hits*. (NB 13) | caché semántica |
| **Sequential recommender (SASRec)** | Recommendation as "next-item prediction" with a causal transformer over the user's history. (`recsys.py`, NB 12) | recomendador secuencial |

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
