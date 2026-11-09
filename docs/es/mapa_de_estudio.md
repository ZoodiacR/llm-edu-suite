# Mapa de estudio: libros y cursos ↔ notebooks

🇬🇧 [English version](../en/study_map.md) · [← Volver al índice](README.md)

Esta página cruza el curso con material de referencia muy usado para estudiar ingeniería de LLMs: qué capítulo o clase se practica en qué notebook, y qué temas **todavía no tienen parte práctica** en el repo. Úsala para leer un capítulo y después abrir el notebook correspondiente, o al revés.

Leyenda: ✅ cubierto con código y resultados medidos · 🟡 cubierto en parte (teoría, o solo una variante) · ❌ sin parte práctica todavía.

## Libros

### Build a Large Language Model (From Scratch), Sebastian Raschka

| Capítulo | Notebook | |
|---|---|---|
| 2 Texto, tokenización, BPE, ventana deslizante | [01](../../notebooks/01_architecture/01_tokenization.ipynb), [05](../../notebooks/02_pretraining_and_inference/05_pretraining.ipynb) | ✅ |
| 3 Atención (self, causal, multi-cabeza) | [02](../../notebooks/01_architecture/02_attention_rope_norm.ipynb) | ✅ (además RoPE, GQA, MLA) |
| 4 Modelo GPT completo, generar texto | [00](../../notebooks/00_start_here/00_mini_llm_end_to_end.ipynb), [02](../../notebooks/01_architecture/02_attention_rope_norm.ipynb) | ✅ (arquitectura tipo Llama) |
| 5 Pre-entrenamiento, decodificación, guardar pesos | [05](../../notebooks/02_pretraining_and_inference/05_pretraining.ipynb), [06](../../notebooks/02_pretraining_and_inference/06_inference_serving.ipynb) | ✅ |
| 5.5 Cargar pesos preentrenados en tu propio modelo | — | ❌ |
| 6 Fine-tuning para clasificación (cabeza de clasificación) | — | ❌ |
| 7 Fine-tuning con instrucciones, evaluación con LLM juez | [07](../../notebooks/03_post_training/07_finetuning_sft_lora.ipynb), [15](../../notebooks/06_llmops/15_observability_llmops.ipynb) | ✅ (con `trl`) |
| Apéndice LoRA | [07](../../notebooks/03_post_training/07_finetuning_sft_lora.ipynb) | ✅ (y QLoRA) |

### Hands-On Large Language Models, Jay Alammar y Maarten Grootendorst

| Capítulo | Notebook | |
|---|---|---|
| 1–3 Tokens, embeddings, el interior de un LLM | [01](../../notebooks/01_architecture/01_tokenization.ipynb)–[06](../../notebooks/02_pretraining_and_inference/06_inference_serving.ipynb) | ✅ |
| 4 Clasificación de texto | [13](../../notebooks/04_applications/13_modern_llm_applications.ipynb) | 🟡 solo con LLM (zero/few-shot); falta embeddings + clasificador y un encoder ajustado |
| 5 Clustering y topic modeling | — | ❌ |
| 6 Prompt engineering | [11](../../notebooks/05_agents_and_mcp/11_agents_tools_structured_output.ipynb), [13](../../notebooks/04_applications/13_modern_llm_applications.ipynb) | ✅ |
| 7 Generación avanzada: cadenas, memoria, agentes | [11](../../notebooks/05_agents_and_mcp/11_agents_tools_structured_output.ipynb), [14](../../notebooks/05_agents_and_mcp/14_agentic_ai.ipynb) | ✅ |
| 8 Búsqueda semántica y RAG | [10](../../notebooks/04_applications/10_rag.ipynb) | 🟡 falta el **reranker cross-encoder** en práctica |
| 9 LLMs multimodales | — | ❌ |
| 10 Crear modelos de embeddings (contrastivo) | — | ❌ |
| 11 Ajustar modelos de representación (BERT) | — | ❌ |
| 12 Ajustar modelos generativos (QLoRA, DPO) | [07](../../notebooks/03_post_training/07_finetuning_sft_lora.ipynb), [08](../../notebooks/03_post_training/08_preference_rl_dpo_grpo.ipynb) | ✅ |

### AI Engineering, Chip Huyen

| Capítulo | Notebook | |
|---|---|---|
| 2 Modelos fundacionales: muestreo, salida estructurada | [06](../../notebooks/02_pretraining_and_inference/06_inference_serving.ipynb), [11](../../notebooks/05_agents_and_mcp/11_agents_tools_structured_output.ipynb) | ✅ (falta cómputo en inferencia: best-of-N, self-consistency) |
| 3–4 Metodología de evaluación, evaluar sistemas | [10](../../notebooks/04_applications/10_rag.ipynb), [15](../../notebooks/06_llmops/15_observability_llmops.ipynb), [12](../../notebooks/04_applications/12_recommender_systems.ipynb) | ✅ (juez, bootstrap, datasets) · 🟡 evaluación comparativa (Elo / Bradley-Terry) |
| 5 Prompt engineering y prompts defensivos | [13](../../notebooks/04_applications/13_modern_llm_applications.ipynb) | ✅ |
| 6 RAG y agentes | [10](../../notebooks/04_applications/10_rag.ipynb), [14](../../notebooks/05_agents_and_mcp/14_agentic_ai.ipynb) | ✅ |
| 7 Fine-tuning (memoria, PEFT, fusión de modelos) | [07](../../notebooks/03_post_training/07_finetuning_sft_lora.ipynb) | ✅ · ❌ fusión de modelos (model merging) |
| 8 Ingeniería de datasets (curación, síntesis, dedup) | [07](../../notebooks/03_post_training/07_finetuning_sft_lora.ipynb) (destilación) | 🟡 falta deduplicación, filtrado de calidad y datos sintéticos a escala |
| 9 Optimización de inferencia | [06](../../notebooks/02_pretraining_and_inference/06_inference_serving.ipynb), [09](../../notebooks/02_pretraining_and_inference/09_quantization.ipynb) | ✅ desde cero · 🟡 sin motor de serving real medido |
| 10 Arquitectura: guardrails, router, gateway, caché, feedback | [13](../../notebooks/04_applications/13_modern_llm_applications.ipynb), [14](../../notebooks/05_agents_and_mcp/14_agentic_ai.ipynb), [15](../../notebooks/06_llmops/15_observability_llmops.ipynb) | 🟡 falta gateway / routing por coste entre modelos |

### Hands-On LLM Serving and Optimization

| Capítulo | Notebook | |
|---|---|---|
| 1–5 Serving: diseño, buenas prácticas, retos | [06](../../notebooks/02_pretraining_and_inference/06_inference_serving.ipynb) | 🟡 conceptos implementados a mano |
| 6–7 Optimización (cuantización, KV-cache, batching, especulativo) | [06](../../notebooks/02_pretraining_and_inference/06_inference_serving.ipynb), [09](../../notebooks/02_pretraining_and_inference/09_quantization.ipynb) | ✅ |
| 8–9 Frameworks de serving (vLLM, SGLang, TGI) y optimización en la práctica | — | ❌ benchmark real de TTFT, TPOT y throughput bajo carga |

### Large Language Models: The Hard Parts

| Capítulo | Notebook | |
|---|---|---|
| 2–3 La brecha de evaluación, herramientas de evaluación | [10](../../notebooks/04_applications/10_rag.ipynb), [15](../../notebooks/06_llmops/15_observability_llmops.ipynb) | ✅ |
| 4 De los datos al contexto | [10](../../notebooks/04_applications/10_rag.ipynb), [13](../../notebooks/04_applications/13_modern_llm_applications.ipynb) | ✅ |
| 5 Salida estructurada | [11](../../notebooks/05_agents_and_mcp/11_agents_tools_structured_output.ipynb) | ✅ |
| 6–7 Seguridad y evaluación de seguridad | [13](../../notebooks/04_applications/13_modern_llm_applications.ipynb) | 🟡 solo prompt injection; faltan jailbreaks, clasificadores de seguridad y sobre-rechazo |
| 8 Alineamiento: caso práctico | [08](../../notebooks/03_post_training/08_preference_rl_dpo_grpo.ipynb) | ✅ |
| Apéndice: despliegue local | [instalación](instalacion.md) | ✅ (Ollama) |

### Build an AI Agent (From Scratch)

| Capítulo | Notebook | |
|---|---|---|
| 1–4 LLM, herramientas, agente ReAct básico | [11](../../notebooks/05_agents_and_mcp/11_agents_tools_structured_output.ipynb), [14](../../notebooks/05_agents_and_mcp/14_agentic_ai.ipynb) | ✅ |
| 5 Bases de conocimiento con RAG | [10](../../notebooks/04_applications/10_rag.ipynb) | ✅ |
| 6 Memoria | [14](../../notebooks/05_agents_and_mcp/14_agentic_ai.ipynb) | ✅ |
| 7 Planificación y reflexión | [14](../../notebooks/05_agents_and_mcp/14_agentic_ai.ipynb) | ✅ |
| 8 Ejecución de código | — | ❌ sandbox de ejecución de código |
| 9 Sistemas multi-agente | [14](../../notebooks/05_agents_and_mcp/14_agentic_ai.ipynb) | 🟡 solo teoría |
| 10 Evaluar agentes | [14](../../notebooks/05_agents_and_mcp/14_agentic_ai.ipynb), [11](../../notebooks/05_agents_and_mcp/11_agents_tools_structured_output.ipynb) | ✅ |

### AI Agents with MCP · Building AI Agent Platforms

| Tema | Notebook | |
|---|---|---|
| Host, cliente, servidor; tools, resources, prompts; stdio | [16](../../notebooks/05_agents_and_mcp/16_mcp_model_context_protocol.ipynb) | ✅ |
| Capacidades de cliente: sampling, roots, elicitation; varios servidores | — | ❌ |
| Streamable HTTP con autenticación | [16](../../notebooks/05_agents_and_mcp/16_mcp_model_context_protocol.ipynb) | 🟡 explicado, no ejecutado |
| Plataforma de agentes: desplegar como servicio, identidad, gobierno, coste | [15](../../notebooks/06_llmops/15_observability_llmops.ipynb) | 🟡 solo observabilidad |

### Hands-On ML (Géron) y Transactional ML with Data Streams

Son la base de ML clásico y de aprendizaje en streaming. En el curso se tocan de lado: métricas, validación y LightGBM en el [12](../../notebooks/04_applications/12_recommender_systems.ipynb). El aprendizaje online y la deriva de datos (concept drift) no tienen parte práctica (❌).

## Cursos de Stanford

**CS336, Language Modeling from Scratch (2025)**, el curso que construye un LM desde cero:

| Bloque | Notebook | |
|---|---|---|
| Tokenización | [01](../../notebooks/01_architecture/01_tokenization.ipynb) | ✅ |
| Arquitecturas e hiperparámetros, MoE | [02](../../notebooks/01_architecture/02_attention_rope_norm.ipynb)–[04](../../notebooks/01_architecture/04_ffn_moe.ipynb) | ✅ |
| GPUs, kernels (FlashAttention, Triton), profiling | [03](../../notebooks/01_architecture/03_efficient_attention.ipynb) | 🟡 teoría; sin profiling ni kernels |
| Paralelismo (datos, tensor, pipeline, FSDP) | [05](../../notebooks/02_pretraining_and_inference/05_pretraining.ipynb) | 🟡 teoría (una sola GPU) |
| Leyes de escala | [05](../../notebooks/02_pretraining_and_inference/05_pretraining.ipynb) | ✅ |
| Inferencia | [06](../../notebooks/02_pretraining_and_inference/06_inference_serving.ipynb) | ✅ |
| Evaluación | [10](../../notebooks/04_applications/10_rag.ipynb), [15](../../notebooks/06_llmops/15_observability_llmops.ipynb) | ✅ |
| Datos: filtrado, deduplicación, mezclas | — | ❌ |
| Alineamiento: SFT, RLHF, RL con recompensas verificables | [07](../../notebooks/03_post_training/07_finetuning_sft_lora.ipynb), [08](../../notebooks/03_post_training/08_preference_rl_dpo_grpo.ipynb) | ✅ |

**CME 295, Transformers & Large Language Models (2025)**, el curso panorámico:

| Bloque | Notebook | |
|---|---|---|
| Transformer, atención, tokenización, embeddings | [01](../../notebooks/01_architecture/01_tokenization.ipynb)–[02](../../notebooks/01_architecture/02_attention_rope_norm.ipynb) | ✅ |
| Variantes: encoder-only (BERT) y encoder-decoder (T5) | — | ❌ el curso solo construye decoder-only |
| Entrenamiento, ajuste, ajuste por preferencias | [05](../../notebooks/02_pretraining_and_inference/05_pretraining.ipynb), [07](../../notebooks/03_post_training/07_finetuning_sft_lora.ipynb), [08](../../notebooks/03_post_training/08_preference_rl_dpo_grpo.ipynb) | ✅ |
| Modelos de razonamiento y cómputo en inferencia | [08](../../notebooks/03_post_training/08_preference_rl_dpo_grpo.ipynb) | 🟡 GRPO sí; falta self-consistency, best-of-N y su curva coste-acierto |
| RAG, agentes, evaluación | [10](../../notebooks/04_applications/10_rag.ipynb)–[16](../../notebooks/05_agents_and_mcp/16_mcp_model_context_protocol.ipynb) | ✅ |

## Lo que falta, por prioridad

Prioridad según lo central que es el tema (cuántas fuentes lo tratan) y lo que aporta a la práctica.

| # | Tema | Fuentes | Propuesta |
|---|---|---|---|
| 1 | **Cargar pesos reales en el modelo propio** (SmolLM2 en `TransformerLM`) y comprobar que los logits coinciden con Hugging Face | Raschka 5.5, CS336 | sección nueva en el 06: une las dos mitades del curso |
| 2 | **Embeddings y rerankers**: ajustar un modelo de embeddings (pérdida contrastiva) y añadir un cross-encoder, medido sobre el RAG del 10 | Alammar caps. 8 y 10, Huyen cap. 6 | notebook nuevo |
| 3 | **Razonamiento y cómputo en inferencia**: CoT, self-consistency, best-of-N con verificador, curva coste-acierto | CME 295, CS336, Huyen cap. 2 | notebook nuevo |
| 4 | **Serving real**: TTFT, TPOT y throughput bajo concurrencia; GGUF cuantizados; vLLM en WSL2 opcional | LLM Serving caps. 8–9, Huyen cap. 9 | notebook nuevo (ya en el roadmap) |
| 5 | **Encoders (BERT/ModernBERT)**: clasificación ajustada frente a embeddings + clasificador frente a LLM zero-shot, con coste por petición | Raschka cap. 6, Alammar caps. 4 y 11, CME 295 | notebook nuevo |
| 6 | **Ingeniería de datos**: deduplicación MinHash, filtrado de calidad, datos sintéticos, descontaminación | CS336, Huyen cap. 8 | notebook nuevo |
| 7 | **Seguridad**: jailbreaks, clasificador de seguridad local, tasa de rechazo frente a sobre-rechazo | The Hard Parts caps. 6–7 | ampliar el 13 o nuevo |
| 8 | **Agentes avanzados**: multi-agente (supervisor, handoffs) y ejecución de código en sandbox | Build an AI Agent caps. 8–9 | ampliar el 14 |
| 9 | **MCP avanzado**: sampling, roots, elicitation, varios servidores, Streamable HTTP | AI Agents with MCP cap. 2 | ampliar el 16 |
| 10 | Multimodal (CLIP, modelos de visión en Ollama) | Alammar cap. 9 | notebook nuevo (opcional) |
| 11 | Sistemas: profiling, memoria, `torch.compile`, paralelismo explicado | CS336 | notebook nuevo (limitado a una GPU) |
| 12 | Fusión de modelos (task arithmetic, TIES) | Huyen cap. 7 | sección en el 07 |
| 13 | Clustering y topic modeling; aprendizaje online y deriva | Alammar cap. 5, Transactional ML | opcional |
