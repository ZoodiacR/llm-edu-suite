🇬🇧 [English version](../en/glossary.md)

# Glosario de términos (ES ↔ EN)

[← Volver al índice](README.md)

Los términos se mantienen en inglés cuando es así como aparecen en papers, librerías y ofertas de trabajo. Entre paréntesis, el módulo, la librería o el notebook (NB) donde aparece cada uno.

## Tokenización y arquitectura

| Término | Significado |
|---|---|
| **Token** | Unidad mínima que procesa el modelo: un fragmento de palabra, una palabra frecuente o un byte. |
| **BPE (Byte-Pair Encoding)** | Algoritmo que construye el vocabulario fusionando repetidamente el par de símbolos más frecuente. La variante *byte-level* parte de los 256 bytes, por lo que nunca produce `<UNK>`. (`tokenizer.py`, NB 01) |
| **Fertility** | Número medio de tokens por palabra. Más alto = el tokenizador "rompe" más el texto (típico en idiomas poco representados). |
| **Compression ratio** | Bytes de texto por token. Más alto = secuencias más cortas = inferencia más barata. |
| **Embedding** | Vector aprendido que representa un token (o un documento, o un ítem en recomendación). |
| **Decoder-only / causal LM** | Transformer que predice el siguiente token y solo puede mirar hacia atrás (máscara causal). Es la arquitectura de GPT, Llama, Qwen, DeepSeek y SmolLM2. (`model.py`) |
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

## Entrenamiento y post-training

| Término | Significado |
|---|---|
| **Pre-training** | Entrenar desde cero prediciendo el siguiente token sobre un corpus grande. (NB 05) |
| **Perplexity** | `exp(pérdida media)`. Intuición: entre cuántos tokens "duda" el modelo. Menor es mejor. |
| **Warmup / cosine / WSD** | Planes de *learning rate*: subida inicial; bajada en coseno; o *Warmup-Stable-Decay* (constante y caída final). |
| **AdamW / weight decay** | Optimizador estándar. El *weight decay* no se aplica a normas ni a biases (grupos de parámetros). |
| **Gradient accumulation / clipping / AMP** | Simular lotes grandes, limitar la norma del gradiente y usar precisión mixta (bf16/fp16). |
| **Gradient checkpointing** | Recalcular las activaciones durante el *backward* en lugar de guardarlas: menos memoria a cambio de un ~30 % más de cómputo. Activado por defecto en `scripts/finetune.py`. |
| **SFT (Supervised Fine-Tuning)** | Ajustar el modelo con pares instrucción→respuesta, normalmente renderizados con la **plantilla de chat** del modelo. (NB 07) |
| **Completion-only loss / loss masking** | Calcular la pérdida de SFT solo sobre los tokens de la respuesta: los del prompt reciben la etiqueta `-100` y la entropía cruzada los ignora. TRL lo hace automáticamente con datasets *prompt-completion*. (NB 07) |
| **SFTTrainer** | El *trainer* de TRL para SFT: acepta un dataset conversacional o prompt-completion, aplica la plantilla de chat, enmascara el prompt y admite un `peft_config` para entrenar adaptadores LoRA. Se configura con `SFTConfig`. (`scripts/finetune.py`, NB 07) |
| **LoRA** | Congela W y aprende una corrección de rango bajo `W + (α/r)·B·A`. Con B=0 al inicio, el modelo arranca idéntico al base; tras entrenar se puede *fusionar* (`merge_and_unload` en `peft`) sin coste en inferencia. (NB 07) |
| **LoRA `modules_to_save`** | Opción de `peft` que entrena (y guarda) copias completas de los módulos elegidos junto a los adaptadores, típicamente el embedding y el `lm_head`. Hace falta cuando el fine-tuning introduce tokens que el modelo base nunca aprendió, como nuevos tokens de rol del chat. (NB 07) |
| **QLoRA** | LoRA sobre un modelo base cuyos pesos congelados se guardan a 4 bits (NF4); los adaptadores siguen en bf16 y cada multiplicación descuantiza el peso primero. Ahorra memoria de pesos (un 7B pasa de ~14 GB a ~3,6 GB), no de activaciones, y cada paso es algo más lento. En `transformers`, `BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", ...)` + `prepare_model_for_kbit_training`. (NB 07) |
| **Double quantization** (doble cuantización) | Cuantizar también las escalas de cada bloque (a 8 bits): el sobrecoste baja de 0,5 a unos 0,127 bits por parámetro. Una de las tres ideas de QLoRA (`bnb_4bit_use_double_quant=True`). (NB 07) |
| **Paged optimizer** (optimizador paginado) | El estado del optimizador vive en memoria unificada de CUDA y se pagina a la RAM cuando un pico de memoria lo exige, en lugar de fallar con un out-of-memory (`optim="paged_adamw_8bit"`). (NB 07) |
| **Distillation** | Un modelo *student* aprende a imitar la distribución de un *teacher* (KL entre logits). (NB 07) |
| **RLHF / preference tuning** | Alinear el modelo con preferencias humanas: respuesta elegida vs rechazada. (NB 08) |
| **DPO (Direct Preference Optimization)** | Optimiza las preferencias directamente, sin modelo de recompensa: `−log σ(β·[(log π−log π_ref)(y_w) − (log π−log π_ref)(y_l)])`. Al inicio vale ln 2. |
| **DPOTrainer** | El *trainer* de TRL para DPO y sus variantes (`loss_type`). Con un `peft_config`, el modelo de referencia es la misma red con el adaptador desactivado, así que no se guarda una segunda copia en memoria. Se configura con `DPOConfig` (`beta`, `loss_type`, `loss_weights`). (`scripts/finetune.py`, NB 08) |
| **Likelihood displacement / DPO + NLL (RPO)** | Efecto por el que DPO agranda el margen bajando la probabilidad de *ambas* respuestas, elegida y rechazada. Se contrarresta sumando la NLL (pérdida SFT) de la respuesta elegida: en TRL, `loss_type=["sigmoid", "sft"]`. |
| **GRPO (Group Relative Policy Optimization)** | RL de DeepSeekMath/R1: muestrea un grupo de respuestas por prompt y usa como ventaja la recompensa normalizada dentro del grupo (sin *critic*). Un grupo en el que todas las muestras tienen la misma recompensa no aporta gradiente. |
| **GRPOTrainer** | El *trainer* de TRL para GRPO: acepta un dataset de prompts y una o varias **funciones de recompensa** en Python `(completions, **columnas) -> list[float]`, muestrea `num_generations` respuestas por prompt y optimiza la política con penalización KL (`beta`). Se configura con `GRPOConfig`. (`scripts/finetune.py`, NB 08) |
| **Verifiable reward** | Recompensa calculable por programa (¿es correcta la respuesta?), sin juez humano ni modelo. Una recompensa *densa* da crédito parcial, así que es menos probable que los grupos empaten. |
| **KL penalty** | Término que impide que la política se aleje demasiado del modelo de referencia. |

## Inferencia y sistemas

| Término | Significado |
|---|---|
| **KV-cache** | Guardar K y V de los tokens ya procesados para no recalcularlos en cada paso de generación. Su tamaño suele ser el cuello de botella de memoria. (NB 02, 06) |
| **Prefill / decode** | Fase de procesar el prompt (paralela) y fase de generar token a token (secuencial, limitada por memoria). |
| **TTFT / TPOT / throughput** | *Time To First Token*, tiempo por token de salida y tokens/segundo totales. |
| **Greedy / temperature / top-k / top-p** | Estrategias de muestreo: el más probable; aplanar o afilar la distribución; quedarse con los k más probables; o con el núcleo de probabilidad acumulada p. `model.next_token_distribution` muestra la distribución resultante. |
| **Speculative decoding** | Un modelo pequeño (*draft*) propone varios tokens y el grande los verifica en un solo paso. Con el esquema de aceptación correcto, la distribución de salida es **idéntica** a la del modelo grande. (`inference.py`, NB 06) |
| **Continuous batching / PagedAttention** | Técnicas de vLLM para servir muchas peticiones a la vez gestionando la KV-cache como memoria paginada. |
| **Quantization (INT8 / INT4 / NF4)** | Guardar pesos con menos bits. *absmax* escala por el máximo absoluto; por bloques reduce el error; NF4 usa niveles óptimos para pesos con distribución normal. (NB 09) |
| **optimum-quanto** | Librería de Hugging Face que cuantiza en el sitio un modelo de PyTorch / `transformers`: `quantize(model, weights=qint8 / qint4 / qfloat8)` sustituye las capas lineales por versiones cuantizadas y `freeze(model)` guarda los pesos cuantizados. Funciona en CPU y GPU. (NB 09) |
| **NF4 (NormalFloat4)** | Formato de 4 bits con 16 niveles colocados en los cuantiles de una normal, con una escala absmax por bloque de 64 pesos. Como los pesos preentrenados son aproximadamente normales, pierde menos precisión que int4 con niveles equiespaciados. Es el formato de QLoRA (`bitsandbytes`). (NB 07, 09) |
| **GGUF / Q4_K_M** | El formato de fichero y el esquema de cuantización a 4 bits por bloques que usan llama.cpp y Ollama. `qwen2.5:3b` en Ollama es un GGUF Q4_K_M de unos 1,9 GB. |
| **Ollama** | Servidor local de modelos abiertos: `ollama pull` descarga un modelo y el servidor (puerto 11434) ofrece una API propia y otra compatible con OpenAI (`/v1`). Es el proveedor por defecto del curso para los notebooks 10–16 (`qwen2.5:3b`, `nomic-embed-text`), usado a través de `ChatOllama` / `OllamaEmbeddings` de `langchain-ollama`. |

## Aplicaciones: LangChain y RAG

| Término | Significado |
|---|---|
| **RAG (Retrieval-Augmented Generation)** | Recuperar fragmentos relevantes de tus documentos y pasarlos como contexto al LLM. (NB 10, `scripts/rag_demo.py`) |
| **Chunking** | Partir documentos en fragmentos (con solapamiento) antes de indexarlos, p. ej. con el `RecursiveCharacterTextSplitter` de LangChain. |
| **Runnable** | La interfaz común de LangChain: cada componente (prompt, modelo, parser, retriever, función envuelta en `RunnableLambda`) tiene `invoke`, `batch` y `stream` (y sus versiones asíncronas), así que todos se componen y se trazan igual. |
| **LCEL (LangChain Expression Language)** | Componer Runnables con `\|` (`prompt \| llm \| StrOutputParser()`). `RunnableParallel` ejecuta ramas sobre la misma entrada, `RunnablePassthrough` la deja pasar sin cambios y `.assign(...)` añade claves a un diccionario. |
| **Vector store** | Base de datos de embeddings que devuelve los vecinos más cercanos de un vector de consulta: `FAISS` e `InMemoryVectorStore` en el curso; pgvector, Qdrant o Weaviate en producción. |
| **Retriever** | Un Runnable que convierte una consulta en una lista de `Document`. `vector_store.as_retriever(search_kwargs={"k": 4})`, `BM25Retriever`, `EnsembleRetriever` y `MultiQueryRetriever` son todos retrievers. |
| **Dense retrieval / BM25 / hybrid** | Búsqueda por embeddings (semántica), por términos (léxica) o combinada. |
| **RRF (Reciprocal Rank Fusion)** | Fusionar rankings sumando `1/(60 + posición)`. Solo usa posiciones, no puntuaciones, así que no necesita calibración. |
| **EnsembleRetriever** | Retriever de LangChain (en `langchain_classic` en LangChain 1.x) que ejecuta varios retrievers y fusiona sus rankings con **RRF ponderado**. El curso combina BM25 y FAISS con pesos 0,5 / 0,5. |
| **Reranker / cross-encoder** | Segunda etapa, más cara, que vuelve a puntuar los mejores candidatos de un recuperador rápido leyendo consulta y documento juntos. Un LLM puede hacer ese papel, pero no siempre compensa: en recomendación, un LLM pequeño reordenando la lista de un modelo colaborativo la empeoró (v2 del NB 12), y la versión actual usa un ranker LambdaMART. |
| **recall@k / hit@k / MRR / nDCG** | Métricas de recuperación: ¿está el documento correcto en el top-k?, inverso de la posición del primero correcto, y ganancia acumulada con descuento por posición. `rag_demo.py` las calcula a nivel de fichero y a nivel de *evidencia* (¿el chunk contiene de verdad la respuesta?). |
| **Faithfulness / groundedness** | Si la respuesta está respaldada por el contexto recuperado (y no inventada). |
| **RAGAS** | Librería para evaluar RAG con un LLM como juez. Sus cuatro métricas principales: **faithfulness** (¿las afirmaciones de la respuesta están respaldadas por el contexto recuperado?), **answer relevancy** (¿la respuesta contesta a la pregunta? se mide generando preguntas a partir de la respuesta y comparando embeddings), **context precision** (¿los chunks relevantes salen arriba?) y **context recall** (¿el contexto recuperado cubre la respuesta de referencia?). Las dos últimas necesitan una respuesta de referencia. (NB 10, NB 13, `rag_demo.py --ragas`) |
| **LLM-as-judge** | Usar un LLM para puntuar o comparar respuestas (pairwise, con rúbrica). Hay que controlar sesgos de posición y longitud, y validar al juez antes de fiarse de él. |
| **Judge strength** (fuerza del juez) | Lo capaz que es el juez respecto al sistema que evalúa. Un juez débil no solo añade ruido: puede inventar diferencias entre sistemas. En el NB 10, el 3B juzgándose a sí mismo informaba de faithfulness 0,38 → 0,54 entre recuperación densa e híbrida; con `qwen2.5:7b` como juez, 0,87 en ambas. Regla: un juez más fuerte que el sistema, fijo mientras se comparan versiones (`config.judge_model()`, `JUDGE_MODEL`). |
| **Structured output** | Forzar salidas que cumplan un esquema. En LangChain, `llm.with_structured_output(ModeloPydantic)` usa el tool calling o el modo JSON del proveedor y devuelve un objeto validado. (NB 11) |
| **Tool calling / function calling** | El modelo devuelve una llamada estructurada (`nombre` + argumentos JSON) que tu código ejecuta. En LangChain una herramienta es una función decorada con `@tool`, y `llm.bind_tools(...)` se la muestra al modelo. (NB 11) |
| **Agent loop** | Bucle modelo → herramienta → observación → modelo, con límite de pasos. `create_agent` de `langchain.agents` lo construye sobre LangGraph. |
| **Middleware (agentes de LangChain)** | Ganchos que envuelven las llamadas del agente al modelo y a las herramientas para añadir comportamiento sin reescribir el bucle: `ModelCallLimitMiddleware` / `ToolCallLimitMiddleware` (presupuestos), `HumanInTheLoopMiddleware` (aprobación antes de ejecutar una herramienta), `ToolErrorMiddleware` (convertir excepciones de herramientas en mensajes), `SummarizationMiddleware` (comprimir historiales largos), o tu propio `AgentMiddleware`. (NB 11, 14, 16) |
| **Human-in-the-loop / confirmation gate** | Pedir aprobación antes de ejecutar acciones con efectos (pagos, borrados, envíos). |
| **Prompt injection** | Texto malicioso dentro de los datos (una web, un email, un documento, el resultado de una herramienta) que intenta dar órdenes al modelo. (NB 11, 13, 16) |
| **Semantic cache** | Reutilizar respuestas de preguntas *parecidas* (por similitud de embeddings). Riesgo: *false hits*. (NB 13) |

## Sistemas de recomendación

| Término | Significado |
|---|---|
| **Recommendation funnel** (embudo de recomendación) | La arquitectura en etapas de un recomendador industrial: **generación de candidatos** (de millones a cientos, barata, se mide por recall), **ranking** (de cientos a decenas, con muchas features, se mide por nDCG) y **re-ranking** (reglas, diversidad, exploración, explicaciones). (NB 12) |
| **Candidate generation / retrieval** (generación de candidatos) | La primera etapa del embudo: varias fuentes baratas (modelo secuencial, two-tower + ANN, co-ocurrencia, contenido, populares) cuya **unión** forma el conjunto de candidatos. Su recall es el techo del ranker: lo que no se recupera no se puede ordenar. (NB 12) |
| **Sequential recommender (SASRec)** (recomendador secuencial) | Recomendación como "predicción del siguiente ítem" con un transformer causal sobre el historial del usuario; el vocabulario es el catálogo. Se entrena mejor con la softmax completa que con negativos muestreados; a escala de millones, **softmax muestreada con corrección logQ** (restar a cada logit el log de su probabilidad de muestreo). (NB 12) |
| **Two-tower model** (modelo de dos torres) | Dos redes que llevan usuario e ítem al mismo espacio; la puntuación es su producto escalar. Como la torre de ítems no depende del usuario, sus vectores se precalculan y se sirven con un índice ANN. Con **id dropout** (ocultar el id al azar en el entrenamiento) aprende a colocar ítems nuevos desde su contenido. (NB 12) |
| **ANN / HNSW** (búsqueda aproximada de vecinos) | *Approximate nearest neighbours*: índices que devuelven casi los mismos vecinos que la búsqueda exacta en mucho menos tiempo. HNSW es el de grafo jerárquico (FAISS, ScaNN, bases vectoriales); su recall frente a la búsqueda exacta es un parámetro de calidad propio. (NB 10, 12) |
| **Learning to rank** (aprendizaje de ranking) | Entrenar un modelo para ordenar: **pointwise** (predecir cada etiqueta por separado), **pairwise** (preferir el positivo a cada negativo, RankNet) o **listwise** (optimizar la calidad de la lista entera). (NB 12) |
| **LambdaMART** | Árboles con gradient boosting cuyos gradientes son los de RankNet por pares, ponderados por cuánto cambiaría el nDCG al intercambiar los dos ítems. El ranker industrial por defecto; en LightGBM, `objective="lambdarank"`. Solo aporta sobre una primera etapa fuerte si tiene muchas consultas de un periodo posterior y features que esa etapa no ve. (NB 12) |
| **Cold start** (arranque en frío) | Recomendar ítems (o a usuarios) sin historial de interacciones. Lo colaborativo no puede: la fila del ítem nunca se entrenó. Se resuelve con contenido (texto, metadatos) para colocarlo y exploración para darle exposición. (NB 12) |
| **Exploration slot / bandit** (ranura de exploración) | Reservar posiciones de la lista para ítems nuevos o inciertos, para aprender cuánto valen (dilema exploración/explotación). Los *bandits* ($\varepsilon$-greedy, UCB, Thompson sampling, contextuales como LinUCB) lo deciden de forma adaptativa; una ranura fija es la versión mínima medible offline. (NB 12) |
| **MMR (Maximal Marginal Relevance)** (relevancia marginal máxima) | Construir la lista de forma voraz eligiendo cada vez $\lambda \cdot$ relevancia $- (1-\lambda) \cdot$ máxima similitud con lo ya elegido. Con $\lambda = 1$ es el ranking puro; más bajo, más diversidad. (NB 12) |
| **Intra-list diversity** (diversidad intra-lista) | Uno menos la similitud media entre los pares de ítems de una lista (en el NB 12, Jaccard de géneros). Offline solo puede parecer neutra o peor para la precisión; su beneficio se mide online. La **calibración** (Steck, 2018) pide además que la mezcla de géneros de la lista se parezca a la del historial. |
| **Leave-last-out / leave-last-two** (dejar fuera el último) | Partición temporal por usuario: el último evento es el objetivo de test (y el penúltimo, el de validación); todo lo anterior es historial. Una partición aleatoria filtraría el futuro. (`recsys.temporal_split`, NB 12) |
| **Full ranking vs sampled metrics** (ranking completo frente a métricas muestreadas) | Ordenar el objetivo contra **todo** el catálogo o contra unos 100 negativos aleatorios. Las muestreadas inflan los números y pueden invertir qué modelo parece mejor (Krichene y Rendle, 2020). (`recsys.target_ranks`, NB 12) |

## Agentes y LangGraph

| Término | Significado |
|---|---|
| **Workflow vs agent** | En un *workflow* el código fija el flujo de control y el LLM rellena los pasos; en un agente es el LLM quien elige el siguiente paso, en bucle. Empieza por lo más simple que funcione. (NB 14) |
| **StateGraph** | El constructor de LangGraph para un grafo cuyos nodos leen y actualizan un **estado** compartido y tipado (un `TypedDict`, o `MessagesState` para chat). `builder.compile()` lo convierte en un Runnable. (NB 14) |
| **Node** | Una función `estado -> actualización parcial del estado` en un grafo de LangGraph: una llamada al LLM, una recuperación, una herramienta, Python normal. |
| **Edge / conditional edge** | Una arista fija el siguiente nodo (`add_edge("a", "b")`, desde `START` o hacia `END`). Una arista condicional (`add_conditional_edges`) llama a una función del estado que devuelve el nombre del siguiente nodo: enrutado, bucles, reintentos. |
| **Checkpointer** | Guarda el estado del grafo tras cada paso (`InMemorySaver` en el curso; SQLite o Postgres en producción). Da al grafo memoria a corto plazo, le permite reanudar tras un fallo o una pausa y habilita el *time travel*. |
| **Thread (`thread_id`)** | La clave bajo la que un checkpointer guarda una conversación: `config={"configurable": {"thread_id": "..."}}`. Mismo hilo = misma memoria. |
| **Store** | La memoria a largo plazo de LangGraph: un almacén clave-valor organizado en espacios de nombres (p. ej. `("users", user_id)`), compartido entre hilos, con búsqueda semántica opcional si se le da un modelo de embeddings (`InMemoryStore`). |
| **Interrupt** | `interrupt(valor)` pausa un grafo dentro de un nodo y devuelve `valor` a quien lo llamó; la ejecución se reanuda con `Command(resume=...)`. Necesita un checkpointer. Es la base del human-in-the-loop en LangGraph. |
| **`recursion_limit` / agent budget** | Límites duros de una ejecución de un agente: el `recursion_limit` de LangGraph limita el número de pasos, y los middlewares de límite de llamadas limitan las llamadas al modelo y a las herramientas. Al alcanzar uno, la ejecución se detiene en lugar de dar vueltas. |
| **Plan-and-execute** | Patrón que primero pide al LLM un plan numerado, luego ejecuta cada paso con un agente con herramientas y por último sintetiza la respuesta. Se puede inspeccionar y aprobar. (NB 14) |
| **Reflection / verifier** | Bucle borrador → verificar → revisar. El verificador comprueba el borrador y devuelve feedback; el código (tests, validadores de esquema) es mucho más fiable que un crítico LLM. (NB 14) |
| **Router / supervisor** | Un nodo que envía cada petición a un especialista (un subagente o una cadena), mediante una clasificación del LLM o por similitud de embeddings. (NB 13, 14) |
| **Short-term / long-term memory** | La conversación del hilo actual (checkpointer, quizá resumida) frente a los hechos que sobreviven entre hilos (store, recuperados por similitud). |
| **Trajectory evaluation** | Puntuar el camino que siguió un agente (qué herramientas, en qué orden, cuántas llamadas de más), no solo su respuesta final. (NB 11) |

## LLMOps y LangSmith

| Término | Significado |
|---|---|
| **LangSmith** | La plataforma de LangChain para trazar, monitorizar y evaluar aplicaciones LLM. Con `LANGSMITH_TRACING=true` y una clave de API, se registra cada llamada de LangChain / LangGraph y cada función decorada con `@traceable`. Opcional en el curso. (NB 15) |
| **Run** | Una unidad de trabajo registrada por LangSmith: una llamada al LLM, un retriever, una herramienta, una cadena, un nodo de un grafo. Cada run tiene entradas, salidas, tiempos, tokens, errores, etiquetas, metadatos y un run padre. |
| **Trace** | El árbol de runs que produce una petición de nivel superior, desde el run raíz hasta cada llamada anidada. Es el árbol de spans de OpenTelemetry. |
| **Project** | El contenedor de LangSmith al que se envían las trazas (`LANGSMITH_PROJECT`, o `setup(project=...)` por notebook). Los paneles y los filtros funcionan por proyecto. |
| **Dataset** | Un conjunto versionado de ejemplos (entradas + salidas de referencia) guardado en LangSmith para la evaluación offline. |
| **Experiment** | Una ejecución puntuada de tu aplicación sobre un dataset (`evaluate(target, data=..., evaluators=[...])`). Comparar dos experimentos es comparar dos versiones de la aplicación sobre los mismos ejemplos. |
| **Evaluator** | Una función que puntúa un run, con la firma `(inputs, outputs, reference_outputs) -> puntuación`: una heurística, una comprobación por código o un LLM-as-judge. Se usa offline en experimentos y online sobre trazas de producción. |
| **Feedback** | Una puntuación o un comentario asociado a un run (`client.create_feedback(run_id, key, score=...)`): el pulgar del usuario, la nota de un evaluador, una heurística. Es la señal de la evaluación online. |
| **OTel GenAI semantic conventions** | Nombres de atributo estándar de OpenTelemetry para telemetría de LLMs (`gen_ai.request.model`, `gen_ai.usage.input_tokens`, ...), para que las trazas puedan ir a cualquier backend. LangSmith puede recibir y exportar OpenTelemetry. |
| **p50 / p95 / p99** | Percentiles de latencia: el valor por debajo del cual cae el 50 / 95 / 99 % de las peticiones. Se reportan en lugar de la media, porque la cola es lo que notan los usuarios. |
| **SLO (Service-Level Objective)** | Objetivo para una métrica del servicio, p. ej. "p95 < 2 s, tasa de error < 1 %". Las violaciones disparan alertas. (NB 15) |
| **PII redaction** | Sustituir datos personales (emails, teléfonos, números de tarjeta, claves de API) por marcadores antes de que los prompts y las respuestas lleguen a logs o trazas. (NB 15) |
| **Head sampling** | Decidir al inicio de una traza si se conserva entera (en LangSmith, `LANGSMITH_TRACING_SAMPLING_RATE=0.1` conserva el 10 % de las trazas). El *tail sampling* decide al final, p. ej. para conservar todos los errores. |
| **Bootstrap confidence interval** (intervalo de confianza bootstrap) | Remuestrear con reemplazo las puntuaciones por ejemplo (p. ej. 2.000 veces), recalcular la media cada vez y tomar los cuantiles 2,5 % y 97,5 %: un intervalo del 95 % sin suponer ninguna distribución. Con 20 preguntas sale ancho, y eso es información. (`evaluation.bootstrap_ci`, NB 10, 12) |
| **Paired bootstrap** (bootstrap pareado) | El mismo remuestreo aplicado a las diferencias por ejemplo entre dos sistemas evaluados sobre los mismos ejemplos (preguntas, usuarios). Emparejar cancela la variación por dificultad de cada ejemplo; si el intervalo de la diferencia excluye el 0, la diferencia difícilmente es ruido. (`evaluation.paired_bootstrap_diff`, NB 10, 12) |

## MCP

| Término | Significado |
|---|---|
| **MCP (Model Context Protocol)** | Protocolo abierto que estandariza cómo las aplicaciones LLM obtienen herramientas, datos y prompts de programas externos, sobre JSON-RPC 2.0. (NB 16, `scripts/mcp_server.py`) |
| **MCP host / client / server** | El host es la aplicación que usa la persona (Claude Code, Claude Desktop, un IDE, tu agente); ejecuta un cliente por cada servidor; el servidor es el programa que expone las capacidades. |
| **Tool / resource / prompt (MCP)** | Las tres primitivas de un servidor: herramientas que llama el modelo, recursos de solo lectura que la aplicación carga en el contexto (URIs fijas o plantillas de URI como `docs://{name}`) y plantillas de prompt que elige el usuario. |
| **Tool annotations** | Pistas que el servidor asocia a una herramienta (`readOnlyHint`, `destructiveHint`, `idempotentHint`, `openWorldHint`) para que el host pueda decidir, por ejemplo, pedir confirmación antes de una llamada destructiva. Son pistas, no garantías. |
| **FastMCP** | La API de alto nivel para servidores del SDK oficial de MCP para Python (`mcp.server.fastmcp.FastMCP` en 1.x): `@server.tool()`, `@server.resource(uri)` y `@server.prompt()` convierten funciones en primitivas, derivando el JSON Schema de las anotaciones de tipo y la descripción del docstring; `server.run()` sirve por stdio y `server.run(transport="streamable-http")` por HTTP. |
| **MCP adapters (`langchain-mcp-adapters`)** | Librería que convierte las herramientas de uno o varios servidores MCP en herramientas de LangChain: `MultiServerMCPClient({...}).get_tools()`, listas para `create_agent`. `tool_name_prefix` pone espacio de nombres a herramientas de servidores distintos. (NB 16) |
| **JSON-RPC request vs notification** | Una petición lleva `id` y siempre recibe respuesta (`result` o `error`); una notificación no lleva `id` y nunca la recibe (p. ej. `notifications/initialized`). |
| **Protocol error vs tool error (`isError`)** | Un `error` de JSON-RPC significa que la petición en sí estaba mal (método desconocido, parámetros incorrectos); un resultado con `isError: true` significa que la herramienta se ejecutó y falló, para que el modelo lo vea y reintente. |
| **stdio transport** | El host lanza el servidor como subproceso e intercambia un mensaje JSON por línea por stdin/stdout. stdout queda reservado para el protocolo: los logs van a stderr. |
| **Streamable HTTP transport** | El servidor escucha en un único endpoint HTTP (p. ej. `/mcp`); el cliente envía mensajes JSON-RPC por POST y el servidor responde con JSON o con un *event stream*, manteniendo una sesión con la cabecera `mcp-session-id`. Para servidores remotos o compartidos. |
| **Tool poisoning** | Ataque en el que un servidor MCP malicioso esconde instrucciones en la descripción de una herramienta (que lee el modelo, rara vez el usuario) o la cambia después de aprobarla. Mitigaciones: listas blancas, espacios de nombres, revisar descripciones, confirmación para herramientas arriesgadas. |
