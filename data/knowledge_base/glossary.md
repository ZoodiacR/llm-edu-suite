🇪🇸 [Versión en español](../es/glosario.md)

# Glossary

[← Back to the index](README.md)

Short definitions of the terms used across the suite. In brackets, the module or notebook (NB) where each one is implemented. The last column gives the term used in the Spanish docs, where it differs from the English one (many terms are kept in English in Spanish too, because that is how they appear in papers, libraries and job postings).

## Tokenization and architecture

| Term | Meaning | Spanish docs |
|---|---|---|
| **Token** | The smallest unit the model processes: a word piece, a frequent word or a single byte. | token |
| **BPE (Byte-Pair Encoding)** | Builds the vocabulary by repeatedly merging the most frequent pair of symbols. The *byte-level* variant starts from the 256 bytes, so it never produces `<UNK>`. (`tokenizer.py`, NB 01) | |
| **Fertility** | Average number of tokens per word. Higher = the tokenizer splits text more (typical for under-represented languages). | |
| **Compression ratio** | Bytes of text per token. Higher = shorter sequences = cheaper inference. | relación de compresión |
| **Embedding** | A learned vector that represents a token (or a document, or an item in a recommender). | |
| **Decoder-only / causal LM** | A transformer that predicts the next token and can only look backwards (causal mask). The architecture of GPT, Llama, Qwen, DeepSeek. (`model.py`) | |
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

## Training

| Term | Meaning | Spanish docs |
|---|---|---|
| **Pre-training** | Training from scratch on next-token prediction over a large corpus. (NB 05) | preentrenamiento |
| **Perplexity** | `exp(mean loss)`. Intuition: how many tokens the model is "hesitating" between. Lower is better. | perplejidad |
| **Warmup / cosine / WSD** | Learning-rate schedules: initial ramp-up; cosine decay; or *Warmup-Stable-Decay* (constant, then a final drop). | calentamiento |
| **AdamW / weight decay** | The standard optimizer. Weight decay is not applied to norms or biases (parameter groups). | |
| **Gradient accumulation / clipping / AMP** | Simulate large batches, cap the gradient norm, and use mixed precision (bf16/fp16). | acumulación / recorte de gradiente, precisión mixta |
| **SFT (Supervised Fine-Tuning)** | Tuning the model on instruction→response pairs. The loss is computed only on the response (**loss masking**, labels `-100`). (NB 07) | enmascarado de pérdida |
| **LoRA** | Freezes W and learns a low-rank correction `W + (α/r)·B·A`. With B=0 at init, the model starts identical to the base; after training it can be *merged* at no inference cost. (`finetune.py`) | |
| **Trainable embeddings (LoRA)** | Also training the token embedding (and a tied `lm_head`) alongside the adapters. Needed when fine-tuning introduces tokens the base model never saw, such as chat role tokens. `apply_lora(..., train_embeddings=True)`; `peft` calls it `modules_to_save`. | embeddings entrenables |
| **QLoRA** | LoRA on top of a base model quantized to 4 bits (NF4). | |
| **Distillation** | A *student* model learns to imitate a *teacher's* distribution (KL between logits). | destilación |
| **RLHF / preference tuning** | Aligning the model with human preferences: chosen vs rejected response. (NB 08) | alineamiento con preferencias |
| **DPO (Direct Preference Optimization)** | Optimizes preferences directly, with no reward model: `−log σ(β·[(log π−log π_ref)(y_w) − (log π−log π_ref)(y_l)])`. Equals ln 2 at init. | |
| **Likelihood displacement / DPO + NLL (RPO)** | DPO can widen the margin by pushing down the probability of *both* the chosen and the rejected answer. Adding the NLL (SFT loss) of the chosen answer counters it (`nll_weight` in `finetune.dpo_loss`). | |
| **GRPO (Group Relative Policy Optimization)** | RL from DeepSeekMath/R1: samples a group of responses per prompt and uses the reward normalized within the group as the advantage (no *critic*). | |
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
| **Quantization (INT8 / INT4 / NF4)** | Storing weights with fewer bits. *absmax* scales by the maximum absolute value; per-block scaling reduces the error; NF4 uses levels that are optimal for normally distributed weights. (`quant.py`, NB 09) | cuantización |

## Applications

| Term | Meaning | Spanish docs |
|---|---|---|
| **RAG (Retrieval-Augmented Generation)** | Retrieving relevant chunks of your documents and passing them to the LLM as context. (`rag.py`, NB 10) | |
| **Chunking** | Splitting documents into (overlapping) pieces before indexing them. | |
| **Dense retrieval / BM25 / hybrid** | Search by embeddings (semantic), by terms (lexical), or both combined. | búsqueda densa / léxica / híbrida |
| **RRF (Reciprocal Rank Fusion)** | Fusing rankings by summing `1/(60 + rank)`. Robust, with no hyperparameters to tune. | |
| **recall@k / MRR / nDCG** | Retrieval metrics: is the right document in the top-k?, inverse rank of the first correct one, and discounted cumulative gain by position. | |
| **Faithfulness / groundedness** | Whether the answer is supported by the retrieved context (rather than made up). | fidelidad |
| **LLM-as-judge** | Using an LLM to score or compare answers (pairwise, with a rubric). Position and length biases must be controlled. | |
| **Tool calling / function calling** | The model returns a structured call (`name` + JSON arguments) that your code executes, with a timeout. (`agents.py`, NB 11) | llamada a herramientas |
| **Agent loop** | Model → tool → observation → model loop, with a step limit. | bucle de agente |
| **Structured output** | Forcing JSON outputs that satisfy a schema (constrained decoding, or validate and retry). | salida estructurada |
| **Human-in-the-loop / confirmation gate** | Asking for approval before running actions with side effects (payments, deletions, sending messages). | compuerta de confirmación |
| **Prompt injection** | Malicious text inside the data (a web page, an email, a document) that tries to give orders to the model. (NB 13) | inyección de prompts |
| **Semantic cache** | Reusing answers to *similar* questions (by embedding similarity). Risk: *false hits*. | caché semántica |
| **Sequential recommender (SASRec)** | Recommendation as "next-item prediction" with a causal transformer over the user's history. (`recsys.py`, NB 12) | recomendador secuencial |
| **hit@k / nDCG@k** | Recommendation metrics: is the real next item among the k recommended ones?, weighted by position. | |
| **MockLLM** | Deterministic test double shipped with the project: lets every pipeline run without a real model. It measures that the *pipeline* works, not the quality of an LLM. | |

## Agents, LLMOps and MCP

| Term | Meaning | Spanish docs |
|---|---|---|
| **Workflow vs agent** | In a workflow the code fixes the control flow and the LLM fills in steps; in an agent the LLM chooses the next step itself, in a loop. Start with the simplest one that works. (`agentic.py`, NB 14) | workflow frente a agente |
| **Agent budget** | Hard limits on steps, tool calls, tokens, cost and time for one agent run. When one is hit, the run stops cleanly with an explicit `stopped_reason`. (`agentic.Budget`) | presupuesto del agente |
| **Loop detection** | Stopping an agent that requests the same tool with the same arguments several times in a row (it is ignoring the observation). | detección de bucles |
| **Plan-and-execute** | Pattern that first asks the LLM for a numbered plan, then runs each step with a tool-using agent and finally synthesizes the answer. Inspectable and approvable. (`agentic.PlanAndExecute`) | |
| **Reflection / verifier** | Draft → verify → revise loop. The verifier checks the draft and returns feedback; code (tests, schema validators) is far more reliable than an LLM critic. (`agentic.reflect_and_revise`) | reflexión / verificador |
| **Short-term / long-term memory** | A token-budgeted conversation window with a running summary, vs a vector store of facts recalled by similarity (RAG over the agent's own history). | memoria a corto / largo plazo |
| **Semantic router** | Sends each request to a specialist by embedding similarity between the request and each specialist's description: fast, cheap, deterministic. (`agentic.Supervisor`) | router semántico |
| **Trajectory evaluation** | Scoring the path an agent took (which tools, in which order, how many extra calls), not only its final answer. (`agentic.evaluate_trajectory`) | evaluación de trayectorias |
| **Trace / span** | A trace is one end-to-end request; a span is one unit of work inside it (LLM call, retrieval, tool call) with timing, attributes, status and a parent. Spans form a tree. (`observability.py`, NB 15) | traza / span |
| **OTel GenAI semantic conventions** | Standard OpenTelemetry attribute names for LLM telemetry (`gen_ai.request.model`, `gen_ai.usage.input_tokens`, ...), so traces can go to any backend. | convenciones semánticas GenAI |
| **p50 / p95 / p99** | Latency percentiles: the value below which 50 / 95 / 99 % of requests fall. Reported instead of the mean, because the tail is what users notice. | |
| **SLO (Service-Level Objective)** | A target for a service metric, e.g. "p95 < 2 s, error rate < 1 %". Violations trigger alerts. (`observability.check_slos`) | objetivo de nivel de servicio |
| **Online evaluation score** | A quality score attached to a production trace: user feedback, an LLM-judge rating, a heuristic. (`Tracer.score`) | puntuación de evaluación online |
| **PII redaction** | Replacing personal data (e-mails, phone and card numbers, API keys) with placeholders before prompts and responses reach logs or traces. (`observability.redact_pii`) | redacción de PII |
| **Head sampling** | Deciding at the start of a trace (the root span) whether to keep the whole trace. *Tail sampling* decides at the end, e.g. to keep every error. | |
| **MCP (Model Context Protocol)** | Open protocol that standardizes how LLM applications get tools, data and prompts from external programs, over JSON-RPC 2.0. (`mcp.py`, NB 16) | |
| **MCP host / client / server** | The host is the user-facing app (Claude Desktop, an IDE, your agent); it runs one client per server; the server is the program that exposes capabilities. | |
| **Tool / resource / prompt (MCP)** | The three server primitives: tools the model calls, read-only resources the application loads into context, and prompt templates the user picks. | |
| **JSON-RPC request vs notification** | A request has an `id` and always gets a response (`result` or `error`); a notification has no `id` and never gets one (e.g. `notifications/initialized`). | petición / notificación |
| **Protocol error vs tool error (`isError`)** | A JSON-RPC `error` means the request itself was wrong (unknown method, bad params); a result with `isError: true` means the tool ran and failed, so the model can see it and retry. | error de protocolo / de herramienta |
| **stdio transport** | The host launches the server as a subprocess and exchanges one JSON message per line on stdin/stdout. stdout is reserved for the protocol: logs go to stderr. | transporte stdio |
| **Tool poisoning** | Attack where a malicious MCP server hides instructions in a tool's description (read by the model, rarely by the user) or changes it after approval. Mitigations: allowlists, namespacing, reviewing descriptions, confirmation for risky tools. | envenenamiento de herramientas |
