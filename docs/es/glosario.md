🇬🇧 [English version](../en/glossary.md)

# Glosario de términos (ES ↔ EN)

[← Volver al índice](README.md)

Los términos se mantienen en inglés cuando es así como aparecen en papers, librerías y ofertas de trabajo. Entre paréntesis, el módulo o notebook donde se implementa.

## Tokenización y arquitectura

| Término | Significado |
|---|---|
| **Token** | Unidad mínima que procesa el modelo: un fragmento de palabra, una palabra frecuente o un byte. |
| **BPE (Byte-Pair Encoding)** | Algoritmo que construye el vocabulario fusionando repetidamente el par de símbolos más frecuente. La variante *byte-level* parte de los 256 bytes, por lo que nunca produce `<UNK>`. (`tokenizer.py`, NB 01) |
| **Fertility** | Número medio de tokens por palabra. Más alto = el tokenizador "rompe" más el texto (típico en idiomas poco representados). |
| **Compression ratio** | Bytes de texto por token. Más alto = secuencias más cortas = inferencia más barata. |
| **Embedding** | Vector aprendido que representa un token (o un documento, o un ítem en recomendación). |
| **Decoder-only / causal LM** | Transformer que predice el siguiente token y solo puede mirar hacia atrás (máscara causal). Es la arquitectura de GPT, Llama, Qwen, DeepSeek. (`model.py`) |
| **Self-attention** | Cada posición calcula una media ponderada de las demás: `softmax(QKᵀ/√d)·V`. Coste O(T²) en la longitud T. |
| **MHA / GQA / MQA** | *Multi-Head*, *Grouped-Query* y *Multi-Query Attention*. GQA comparte cada cabeza K/V entre varias cabezas Q para reducir la KV-cache. (`layers.py`, NB 02) |
| **MLA (Multi-head Latent Attention)** | Técnica de DeepSeek-V2/V3: comprime K y V en un vector latente pequeño que es lo único que se guarda en caché. |
| **RoPE (Rotary Position Embedding)** | Codifica la posición rotando pares de dimensiones de Q y K; el producto QKᵀ pasa a depender solo de la distancia relativa. |
| **RMSNorm** | Normalización que solo divide por la raíz cuadrática media (sin restar la media). Más barata que LayerNorm; estándar en Llama. |
| **Pre-norm / sandwich norm / QK-norm** | Dónde se colocan las normalizaciones: antes de cada sub-bloque, antes y después, o sobre Q y K para estabilizar el entrenamiento. |
| **SwiGLU** | FFN con compuerta: `W_down(SiLU(W_gate·x) ⊙ W_up·x)`. Estándar en LLMs modernos. |
| **MoE (Mixture of Experts)** | Sustituye la FFN por N expertos y un *router* que activa solo k de ellos por token: muchos parámetros totales, pocos activos. (NB 04) |
| **Aux-loss vs aux-free balancing** | Dos formas de evitar que el router use siempre los mismos expertos: una pérdida auxiliar, o un sesgo por experto ajustado sin gradiente (DeepSeek-V3). |
| **Shared expert** | Experto que procesa siempre todos los tokens, además de los enrutados. |
| **Sliding-window / Longformer / block-sparse / linear attention** | Variantes de atención eficiente para contextos largos: ventana local, ventana + tokens globales, bloques seleccionados por contenido, y atención lineal de coste O(T). (NB 03) |

## Entrenamiento

| Término | Significado |
|---|---|
| **Pre-training** | Entrenar desde cero prediciendo el siguiente token sobre un corpus grande. (NB 05) |
| **Perplexity** | `exp(pérdida media)`. Intuición: entre cuántos tokens "duda" el modelo. Menor es mejor. |
| **Warmup / cosine / WSD** | Planes de *learning rate*: subida inicial; bajada en coseno; o *Warmup-Stable-Decay* (constante y caída final). |
| **AdamW / weight decay** | Optimizador estándar. El *weight decay* no se aplica a normas ni a biases (grupos de parámetros). |
| **Gradient accumulation / clipping / AMP** | Simular lotes grandes, limitar la norma del gradiente y usar precisión mixta (bf16/fp16). |
| **SFT (Supervised Fine-Tuning)** | Ajustar el modelo con pares instrucción→respuesta. La pérdida solo se calcula sobre la respuesta (**loss masking**, etiquetas `-100`). (NB 07) |
| **LoRA** | Congela W y aprende una corrección de rango bajo `W + (α/r)·B·A`. Con B=0 al inicio, el modelo arranca idéntico al base; tras entrenar se puede *fusionar* (merge) sin coste en inferencia. |
| **QLoRA** | LoRA sobre un modelo base cuantizado a 4 bits (NF4). |
| **Distillation** | Un modelo *student* aprende a imitar la distribución de un *teacher* (KL entre logits). |
| **RLHF / preference tuning** | Alinear el modelo con preferencias humanas: respuesta elegida vs rechazada. (NB 08) |
| **DPO (Direct Preference Optimization)** | Optimiza las preferencias directamente, sin modelo de recompensa: `−log σ(β·[(log π−log π_ref)(y_w) − (log π−log π_ref)(y_l)])`. Al inicio vale ln 2. |
| **Likelihood displacement / DPO + NLL (RPO)** | Efecto por el que DPO agranda el margen bajando la probabilidad de *ambas* respuestas, elegida y rechazada. Se contrarresta sumando la NLL (pérdida SFT) de la respuesta elegida (`nll_weight` en `finetune.dpo_loss`). |
| **GRPO (Group Relative Policy Optimization)** | RL de DeepSeekMath/R1: muestrea un grupo de respuestas por prompt y usa como ventaja la recompensa normalizada dentro del grupo (sin *critic*). |
| **Verifiable reward** | Recompensa calculable por programa (¿es correcta la respuesta?), sin juez humano ni modelo. |
| **KL penalty** | Término que impide que la política se aleje demasiado del modelo de referencia. |

## Inferencia y sistemas

| Término | Significado |
|---|---|
| **KV-cache** | Guardar K y V de los tokens ya procesados para no recalcularlos en cada paso de generación. Su tamaño suele ser el cuello de botella de memoria. (NB 02, 06) |
| **Prefill / decode** | Fase de procesar el prompt (paralela) y fase de generar token a token (secuencial, limitada por memoria). |
| **TTFT / TPOT / throughput** | *Time To First Token*, tiempo por token de salida y tokens/segundo totales. |
| **Greedy / temperature / top-k / top-p** | Estrategias de muestreo: el más probable; aplanar o afilar la distribución; quedarse con los k más probables; o con el núcleo de probabilidad acumulada p. |
| **Speculative decoding** | Un modelo pequeño (*draft*) propone varios tokens y el grande los verifica en un solo paso. Con el esquema de aceptación correcto, la distribución de salida es **idéntica** a la del modelo grande. (`inference.py`, NB 06) |
| **Continuous batching / PagedAttention** | Técnicas de vLLM para servir muchas peticiones a la vez gestionando la KV-cache como memoria paginada. |
| **Quantization (INT8 / INT4 / NF4)** | Guardar pesos con menos bits. *absmax* escala por el máximo absoluto; por bloques reduce el error; NF4 usa niveles óptimos para pesos con distribución normal. (`quant.py`, NB 09) |

## Aplicaciones

| Término | Significado |
|---|---|
| **RAG (Retrieval-Augmented Generation)** | Recuperar fragmentos relevantes de tus documentos y pasarlos como contexto al LLM. (`rag.py`, NB 10) |
| **Chunking** | Partir documentos en fragmentos (con solapamiento) antes de indexarlos. |
| **Dense retrieval / BM25 / hybrid** | Búsqueda por embeddings (semántica), por términos (léxica) o combinada. |
| **RRF (Reciprocal Rank Fusion)** | Fusionar rankings sumando `1/(60 + posición)`. Robusto y sin hiperparámetros que ajustar. |
| **recall@k / MRR / nDCG** | Métricas de recuperación: ¿está el documento correcto en el top-k?, inverso de la posición del primero correcto, y ganancia acumulada con descuento por posición. |
| **Faithfulness / groundedness** | Si la respuesta está respaldada por el contexto recuperado (y no inventada). |
| **LLM-as-judge** | Usar un LLM para puntuar o comparar respuestas (pairwise, con rúbrica). Hay que controlar sesgos de posición y longitud. |
| **Tool calling / function calling** | El modelo devuelve una llamada estructurada (`nombre` + argumentos JSON) que tu código ejecuta. (`agents.py`, NB 11) |
| **Agent loop** | Bucle modelo → herramienta → observación → modelo, con límite de pasos. |
| **Structured output** | Forzar salidas JSON que cumplan un esquema (decodificación restringida, o validar y reintentar). |
| **Human-in-the-loop / confirmation gate** | Pedir aprobación antes de ejecutar acciones con efectos (pagos, borrados, envíos). |
| **Prompt injection** | Texto malicioso dentro de los datos (una web, un email, un documento) que intenta dar órdenes al modelo. (NB 13) |
| **Semantic cache** | Reutilizar respuestas de preguntas *parecidas* (por similitud de embeddings). Riesgo: *false hits*. |
| **Sequential recommender (SASRec)** | Recomendación como "predicción del siguiente ítem" con un transformer causal sobre el historial del usuario. (`recsys.py`, NB 12) |
| **hit@k / nDCG@k** | Métricas de recomendación: ¿el siguiente ítem real está entre los k recomendados?, ponderado por posición. |
| **MockLLM** | Doble de prueba determinista incluido en el proyecto: permite ejecutar todos los pipelines sin un modelo real. No mide calidad de un LLM, sino que el *pipeline* funciona. |

## Agentes, LLMOps y MCP

| Término | Significado |
|---|---|
| **Workflow vs agent** | En un *workflow* el código fija el flujo de control y el LLM rellena los pasos; en un agente es el LLM quien elige el siguiente paso, en bucle. Empieza por lo más simple que funcione. (`agentic.py`, NB 14) |
| **Agent budget** | Límites duros de pasos, llamadas a herramientas, tokens, coste y tiempo para una ejecución de un agente. Al alcanzar uno, la ejecución se detiene limpiamente con un `stopped_reason` explícito. (`agentic.Budget`) |
| **Loop detection** | Detener un agente que pide la misma herramienta con los mismos argumentos varias veces seguidas (está ignorando la observación). |
| **Plan-and-execute** | Patrón que primero pide al LLM un plan numerado, luego ejecuta cada paso con un agente con herramientas y por último sintetiza la respuesta. Se puede inspeccionar y aprobar. (`agentic.PlanAndExecute`) |
| **Reflection / verifier** | Bucle borrador → verificar → revisar. El verificador comprueba el borrador y devuelve feedback; el código (tests, validadores de esquema) es mucho más fiable que un crítico LLM. (`agentic.reflect_and_revise`) |
| **Short-term / long-term memory** | Una ventana de conversación con presupuesto de tokens y resumen acumulado, frente a un almacén vectorial de hechos recuperados por similitud (RAG sobre el propio historial del agente). |
| **Semantic router** | Envía cada petición a un especialista según la similitud de embeddings entre la petición y la descripción de cada especialista: rápido, barato y determinista. (`agentic.Supervisor`) |
| **Trajectory evaluation** | Puntuar el camino que siguió un agente (qué herramientas, en qué orden, cuántas llamadas de más), no solo su respuesta final. (`agentic.evaluate_trajectory`) |
| **Trace / span** | Una traza es una petición de principio a fin; un span es una unidad de trabajo dentro de ella (llamada al LLM, recuperación, herramienta) con tiempos, atributos, estado y un padre. Los spans forman un árbol. (`observability.py`, NB 15) |
| **OTel GenAI semantic conventions** | Nombres de atributo estándar de OpenTelemetry para telemetría de LLMs (`gen_ai.request.model`, `gen_ai.usage.input_tokens`, ...), para que las trazas puedan ir a cualquier backend. |
| **p50 / p95 / p99** | Percentiles de latencia: el valor por debajo del cual cae el 50 / 95 / 99 % de las peticiones. Se reportan en lugar de la media, porque la cola es lo que notan los usuarios. |
| **SLO (Service-Level Objective)** | Objetivo para una métrica del servicio, p. ej. "p95 < 2 s, tasa de error < 1 %". Las violaciones disparan alertas. (`observability.check_slos`) |
| **Online evaluation score** | Puntuación de calidad asociada a una traza de producción: feedback del usuario, nota de un juez LLM, una heurística. (`Tracer.score`) |
| **PII redaction** | Sustituir datos personales (emails, teléfonos, números de tarjeta, claves de API) por marcadores antes de que los prompts y las respuestas lleguen a logs o trazas. (`observability.redact_pii`) |
| **Head sampling** | Decidir al inicio de una traza (en el span raíz) si se conserva la traza entera. El *tail sampling* decide al final, p. ej. para conservar todos los errores. |
| **MCP (Model Context Protocol)** | Protocolo abierto que estandariza cómo las aplicaciones LLM obtienen herramientas, datos y prompts de programas externos, sobre JSON-RPC 2.0. (`mcp.py`, NB 16) |
| **MCP host / client / server** | El host es la aplicación que usa la persona (Claude Desktop, un IDE, tu agente); ejecuta un cliente por cada servidor; el servidor es el programa que expone las capacidades. |
| **Tool / resource / prompt (MCP)** | Las tres primitivas de un servidor: herramientas que llama el modelo, recursos de solo lectura que la aplicación carga en el contexto y plantillas de prompt que elige el usuario. |
| **JSON-RPC request vs notification** | Una petición lleva `id` y siempre recibe respuesta (`result` o `error`); una notificación no lleva `id` y nunca la recibe (p. ej. `notifications/initialized`). |
| **Protocol error vs tool error (`isError`)** | Un `error` de JSON-RPC significa que la petición en sí estaba mal (método desconocido, parámetros incorrectos); un resultado con `isError: true` significa que la herramienta se ejecutó y falló, para que el modelo lo vea y reintente. |
| **stdio transport** | El host lanza el servidor como subproceso e intercambia un mensaje JSON por línea por stdin/stdout. stdout queda reservado para el protocolo: los logs van a stderr. |
| **Tool poisoning** | Ataque en el que un servidor MCP malicioso esconde instrucciones en la descripción de una herramienta (que lee el modelo, rara vez el usuario) o la cambia después de aprobarla. Mitigaciones: listas blancas, espacios de nombres, revisar descripciones, confirmación para herramientas arriesgadas. |
