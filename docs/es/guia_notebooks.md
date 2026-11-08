# Guía de notebooks

🇬🇧 [English version](../en/notebooks_guide.md) · [← Volver al índice](README.md)

Los 17 notebooks uno a uno: qué construyes, qué conceptos te llevas, qué se mide y qué resultados salen. Los notebooks están en inglés; esta guía te acompaña en español.

**Cómo usar esta guía.** Lee la ficha antes de abrir el notebook, ejecútalo celda a celda y vuelve a la ficha para contrastar tus números con los publicados. Los resultados son los de la ejecución subida al repositorio (portátil con RTX 3060 de 6 GB). En CPU todo funciona igual, pero los números pueden variar un poco: los modelos pequeños son sensibles a la aritmética de cada hardware. Donde un resultado es frágil, el notebook lo dice.

| Tópico | Notebooks | Tiempo aprox. (GPU / CPU) |
|---|---|---|
| Empieza aquí | [00](#00) | 40 s / 5 min |
| Arquitectura | [01](#01) · [02](#02) · [03](#03) · [04](#04) | 5 min / 12 min |
| Pre-entrenamiento e inferencia | [05](#05) · [06](#06) · [09](#09) | 2 min / 10 min |
| Post-training | [07](#07) · [08](#08) | 9 min / 9 min (modelos tan pequeños apenas ganan en GPU) |
| Aplicaciones | [10](#10) · [12](#12) · [13](#13) | 30 s / 1 min |
| Agentes y MCP | [11](#11) · [14](#14) · [16](#16) | 10 s |
| LLMOps | [15](#15) | 15 s |

---

<a id="00"></a>
## 00 · Un mini LLM completo en un notebook
`notebooks/00_start_here/00_mini_llm_end_to_end.ipynb` · también como script: `python scripts/mini_llm.py`

**Qué construyes.** Un GPT entero sin importar nada del proyecto, solo PyTorch: un tokenizador BPE a nivel de bytes, un transformer moderno (embeddings, RoPE, RMSNorm, atención causal, SwiGLU, cabeza de salida compartida), el bucle de entrenamiento (AdamW, warmup + coseno, clipping), la evaluación, el muestreo y el guardado.

**Conceptos clave**
- Cómo encajan todas las piezas antes de estudiarlas por separado.
- La comprobación de cordura: la pérdida inicial debe ser ≈ ln(V).
- **Pérdida irreducible**: como el corpus lo genera una gramática conocida, se puede calcular el mínimo teórico (4·ln 8 nats por frase) y medir a qué distancia está el modelo.
- Por qué el BPE deja de fusionar cuando cada palabra ya es un token.

**Resultados.** 1,86M de parámetros, vocabulario de 455 tokens. Entrena en **~22 s en GPU**. Pérdida de validación 0,778 frente a un suelo teórico de 0,774: **a un 0,6% del óptimo**. El **82%** de las frases muestreadas son gramaticales. Se aprecia el bucle típico de la decodificación greedy.

**Ejercicios.** Entrénalo con tiny Shakespeare (`text_path`), quita RoPE y observa qué pasa con la gramática, o añade una KV-cache a `generate`.

---

<a id="01"></a>
## 01 · Tokenización
`notebooks/01_architecture/01_tokenization.ipynb` · módulo `tokenizer`

**Qué construyes.** Un BPE byte-level con entrenamiento, codificación, decodificación, tokens especiales y guardado/carga.

**Conceptos clave.** Por qué se usan sub-palabras; por qué el BPE byte-level nunca produce `<UNK>`; cómo el tamaño del vocabulario intercambia longitud de secuencia por tamaño de la tabla de embeddings; la **fertilidad** (tokens por palabra) como medida de lo bien que un tokenizador cubre un dominio o idioma.

**Resultados.** La frase de ejemplo ocupa 52 bytes y se codifica en 7 tokens (7,43 bytes/token), con round-trip exacto. Fertilidad **1,00 en el dominio de entrenamiento frente a 3,90 fuera de él** (código y español): esto explica por qué los idiomas poco representados salen más caros en tokens.

---

<a id="02"></a>
## 02 · Atención, RoPE y normalización
`notebooks/01_architecture/02_attention_rope_norm.ipynb` · módulos `layers`, `model`

**Qué construyes.** RMSNorm (verificado contra `torch.nn.RMSNorm`), RoPE con escalado tipo Llama 3.1, la atención escrita a mano (verificada contra `F.scaled_dot_product_attention`), MHA → GQA → MQA → MLA, y la KV-cache.

**Conceptos clave.** RoPE solo depende de la distancia relativa (el notebook lo comprueba numéricamente). GQA comparte cabezas K/V para reducir la caché. MLA (DeepSeek) comprime K/V en un vector latente. La generación incremental con caché debe dar exactamente lo mismo que el forward completo.

**Resultados.** Entrenando la misma red con cada variante: **MHA 2048 B/token de caché frente a MLA 160 B/token, con la misma perplejidad (2,65)**. GQA-2 y MQA quedan en medio (512 y 256 B/token). En DeepSeek-V3 la reducción por capa es de unas 57 veces.

---

<a id="03"></a>
## 03 · Atención eficiente para contextos largos
`notebooks/01_architecture/03_efficient_attention.ipynb` · módulo `layers`

**Qué construyes.** Ventana deslizante (SWA), Longformer, FlexAttention (si tu PyTorch la trae), block-sparse seleccionado por contenido, atención lineal (con su forma recurrente de estado constante) e híbridos.

**Conceptos clave.** Coste O(T²) frente a O(T·w) frente a O(T); campo receptivo de SWA apilada; por qué la atención lineal tiene un estado de tamaño fijo; **elegir la métrica correcta**: la perplejidad global no sirve en esta tarea, la precisión de copia sí.

**Resultados.** Tarea de copia a ~42 tokens de distancia, 1500 pasos:

| Variante | Precisión de copia |
|---|---|
| densa (GQA) | **99,9%** |
| block-sparse | **88,7%** |
| híbrida (1 capa global de cada 4) | 48,4% |
| ventana deslizante w=16 | 0,6% |
| lineal | 0,3% |

Con solo 300 pasos incluso la densa está en 0,3%: las ***induction heads*** aparecen de golpe durante el entrenamiento (transición de fase).

---

<a id="04"></a>
## 04 · FFN y Mixture of Experts
`notebooks/01_architecture/04_ffn_moe.ipynb` · módulo `layers`

**Qué construyes.** SwiGLU con la regla de tamaño de Llama, una capa MoE con routing top-k, expertos compartidos, balanceo con pérdida auxiliar y balanceo *aux-free* (DeepSeek-V3).

**Conceptos clave.** Parámetros totales frente a activos; por qué el router colapsa sin balanceo; por qué el ajuste del sesgo aux-free debe ser pequeño; comparación justa a igual cómputo activo.

**Resultados.** Con top-2 de 8 expertos, el modelo tiene 3× más parámetros totales que activos. El balanceo aux-free lleva un router desequilibrado de **4,0× a 1,15×** con `bias_lr=0,01`, mientras que con 0,1 oscila (se ve en la gráfica). En el corpus de juguete el MoE no supera al modelo denso (ppl 3,31 frente a 3,24); el notebook explica por qué el MoE rinde a gran escala.

---

<a id="05"></a>
## 05 · Pre-entrenamiento
`notebooks/02_pretraining_and_inference/05_pretraining.ipynb` · módulos `training`, `data`

**Qué construyes.** El pipeline completo: texto → tokenizador → modelo → bucle con AdamW, warmup + coseno, clipping, checkpoints y dashboard (pérdida, norma del gradiente, LR); después generación, guardado y carga.

**Conceptos clave.** Comprobaciones de cordura (pérdida inicial ≈ ln V, test de que el modelo no ve el futuro); cómo leer el dashboard; **pérdida irreducible** (el término *E* de Chinchilla); ablación de schedule (coseno frente a WSD); sanity check de escalado.

**Resultados.** Perplejidad de validación 2,79 frente a un suelo teórico de 2,75 (**un 1,5% por encima del suelo** en pérdida). A igual número de pasos, el modelo de 1,67M aprende mucho más rápido que el de 125K (pérdida 1,054 frente a 1,259 en el paso 50). Coseno y WSD acaban prácticamente iguales.

---

<a id="06"></a>
## 06 · Inferencia y serving
`notebooks/02_pretraining_and_inference/06_inference_serving.ipynb` · módulos `inference`, `model`

**Qué construyes.** Estrategias de muestreo, KV-cache, **decodificación especulativa desde cero** (con su comprobación de que conserva la distribución), batching y medición de TTFT y throughput.

**Conceptos clave.** Temperatura, top-k y top-p como transformaciones de la distribución (gráfica y entropía); diversidad (distinct-2); por qué la decodificación especulativa es exacta; el decode está limitado por memoria; y el *overhead* de lanzar kernels en GPU con modelos pequeños.

**Resultados.** Al empezar una palabra, top-k 20 reduce la entropía de 4,25 a 2,89 nats. Greedy produce la misma continuación 16 de 16 veces. La especulativa **acepta el 98% de los tokens del draft, ~4,6 tokens por pasada del modelo grande**. Con batch 16 se obtienen ~17× los tokens/s de batch 1. En GPU, con un modelo tan pequeño, la KV-cache apenas acelera (el notebook explica por qué y cuándo cambia).

---

<a id="07"></a>
## 07 · Fine-tuning I: SFT, LoRA y destilación
`notebooks/03_post_training/07_finetuning_sft_lora.ipynb` · módulo `finetune`

**Qué construyes.** SFT con *loss masking* (pérdida solo sobre la respuesta), LoRA desde cero (idéntico al base al inicio y fusionable), ablación de rango y destilación con teacher y student.

**Conceptos clave.**
- Plantillas de chat y etiquetas `-100`.
- **Por qué LoRA puede fallar**: los tokens de rol nuevos tienen embeddings congelados. La solución es entrenar también los embeddings, equivalente a `modules_to_save` en peft.
- Rango y capacidad.
- Destilación por secuencias (el teacher escribe datos) frente a destilación por logits.

**Resultados**

| Método | Exact match |
|---|---|
| base | 0% |
| SFT completo | **100%** |
| LoRA solo adaptadores | 0% (fallo explicado) |
| LoRA + embeddings (16,5% de los parámetros) | **95,3%** |
| ablación de rango r = 1 / 4 / 16 (800 pasos) | 1,3% / 1,3% / 48,0% |
| student con 1k etiquetas humanas | 24,7% |
| student + 6k respuestas del teacher | **98,0%** |
| student + destilación por logits | 44,7%: frágil, el notebook explica por qué |

---

<a id="08"></a>
## 08 · Fine-tuning II: DPO y GRPO
`notebooks/03_post_training/08_preference_rl_dpo_grpo.ipynb` · módulo `finetune`

**Qué construyes.** DPO con su análisis (pérdida inicial ln 2, precisión de preferencia, margen, log-probs), DPO + NLL, y GRPO con recompensas verificables (exacta frente a densa).

**Conceptos clave.**
- ***Likelihood displacement***: DPO puede bajar también la probabilidad de la respuesta buena.
- El ancla NLL (`rpo_alpha` en trl).
- Ventaja relativa al grupo en GRPO.
- Diseño de recompensas y la **ley de Goodhart**: la recompensa proxy no es la métrica.
- Por qué el valor de la pérdida de policy gradient es ≈ 0 aunque su gradiente no lo sea.

**Resultados** (desde un SFT parcial al 70,7%):

| Método | Exact match | Perplejidad |
|---|---|---|
| SFT | 70,7% | 1,76 |
| + DPO | 46,7% | 3,66 |
| **+ DPO + NLL** | **80,0%** | **1,46** |
| + GRPO (recompensa exacta) | 74,0% | 1,61 |
| + GRPO (recompensa densa) | 72,7% | 1,59 |

---

<a id="09"></a>
## 09 · Cuantización
`notebooks/02_pretraining_and_inference/09_quantization.ipynb` · módulo `quant`

**Qué construyes.** Cuantización absmax por tensor, por fila y por bloques; NF4 (QLoRA); `QuantizedLinear`; cuantización del modelo completo.

**Conceptos clave.** Por qué escalas más finas reducen el error; NF4 coloca sus niveles donde están los pesos (distribución normal); **las métricas que de verdad detectan la degradación**: KL frente al modelo fp y acuerdo top-1, que es lo que usa llama.cpp. La perplejidad sola es poco sensible.

**Resultados.** A 4 bits el error relativo del peso baja del 96,7% (por tensor) al 9,6% (bloques de 32). En el modelo completo: INT8 tiene un acuerdo top-1 del 99,9%; **NF4 un 98,6% con 0,31× de memoria**; 3 bits 94,8%; y 2 bits 87,9%, con una KL 100 veces mayor, mientras que la perplejidad solo sube un 1,2%.

---

<a id="10"></a>
## 10 · RAG
`notebooks/04_applications/10_rag.ipynb` · módulos `rag`, `evaluation`

**Qué construyes.** Chunking, embeddings, BM25, búsqueda híbrida con *Reciprocal Rank Fusion*, prompts con citas, expansión de consultas y la evaluación completa.

**Conceptos clave.** Búsqueda densa frente a léxica y por qué combinarlas; métricas de recuperación (recall@k, MRR, nDCG) separadas de las de generación (token-F1, *faithfulness*); juez LLM *pairwise*.

**Resultados.** Densa recall@1 0,88; BM25 e **híbrida 1,00**. Token-F1 de las respuestas 0,70, con soporte en el contexto 1,00. En la comparación RAG frente a sin RAG, el juez prefiere RAG en 7 de 8 casos.

---

<a id="11"></a>
## 11 · Herramientas, agentes y salida estructurada
`notebooks/05_agents_and_mcp/11_agents_tools_structured_output.ipynb` · módulo `agents`

**Qué construyes.** Un registro de tools con JSON Schema, validación de argumentos antes de ejecutar, el bucle del agente, una puerta de confirmación humana y salida estructurada con reintento.

**Conceptos clave.** Los errores de las tools vuelven al modelo como observación, no como excepción; límites de pasos; decodificación restringida frente a validar y reintentar; cómo evaluar un agente.

**Resultados.** Selección de herramienta 5/5, argumentos válidos 5/5, respuesta correcta 5/5, con 2 pasos de media.

---

<a id="12"></a>
## 12 · Sistemas de recomendación
`notebooks/04_applications/12_recommender_systems.ipynb` · módulo `recsys`

**Qué construyes.** Recomendación secuencial como "predicción del siguiente ítem" (estilo SASRec), embeddings colaborativos y de contenido, y re-ranking con un LLM.

**Conceptos clave.** Evaluación *leave-one-out*; líneas base (aleatoria, popularidad, contenido); arranque en frío; un re-ranker LLM solo vale si mejora las métricas offline.

**Resultados.** hit@10: aleatorio 0,032, popularidad 0,097, contenido 0,059 y **SASRec 0,210**. El re-ranking con MockLLM **empeora** (0,187 → 0,153). Es un resultado honesto y el notebook explica qué aprender de él.

---

<a id="13"></a>
## 13 · Aplicaciones modernas con LLMs
`notebooks/04_applications/13_modern_llm_applications.ipynb` · módulo `applications`

**Qué construyes.** Extracción estructurada, clasificación zero/few-shot, resumen map-reduce, caché semántica, guardia contra *prompt injection*, procesamiento por lotes con coste y generación de datos sintéticos con filtrado.

**Conceptos clave.** Precisión por campo; matriz de confusión; ROUGE frente a *faithfulness*; el umbral de la caché intercambia aciertos por respuestas equivocadas; defensa en profundidad contra inyección; proyección de costes.

**Resultados.** Extracción 3/3 en todos los campos. Clasificación 100%. Caché semántica con 50% de aciertos y 1 falso acierto en 2 pares casi iguales. Guardia: detecta 4/4 ataques con 0/4 falsos positivos. Coste proyectado de 43 $ por millón de registros. Datos sintéticos: 82% conservados tras el filtrado.

---

<a id="14"></a>
## 14 · IA agéntica
`notebooks/05_agents_and_mcp/14_agentic_ai.ipynb` · módulo `agentic`

**Qué construyes.** Presupuestos y detección de bucles, plan-and-execute, reflexión guiada por un verificador, memoria a corto plazo (resumen) y a largo plazo (vectorial), un supervisor multi-agente y evaluación de trayectorias.

**Conceptos clave.** Workflow frente a agente; todo agente necesita un **motivo de parada**; evaluar el *camino* (herramientas correctas, en orden y sin desperdicio), no solo la respuesta; la reflexión necesita un verificador fiable; la memoria es recuperación y hay que medir su recall; el enrutado es clasificación.

**Resultados**
- Plan-and-execute resuelve **6/6 tareas multi-paso frente a 2/6** de un solo bucle (con MockLLM; el notebook explica el matiz).
- Las 4 políticas de parada se disparan correctamente.
- La memoria corta se mantiene en ~160 tokens frente a 640 del historial completo.
- Recall@1 de la memoria larga 4/6, con un embedder léxico.
- Enrutado: 67% (embeddings) frente a 42% (LLM).

---

<a id="15"></a>
## 15 · Observabilidad y LLMOps
`notebooks/06_llmops/15_observability_llmops.ipynb` · módulo `observability`

**Qué construyes.** Trazas y spans con la semántica de OpenTelemetry, un wrapper para LLMs y tools, redacción de PII, métricas por percentiles, SLOs, puntuaciones de calidad por traza y exportación (JSONL y OTLP).

**Conceptos clave.** Traza, span, árbol; por qué p95/p99 y no la media; reintentos visibles en la traza; calidad *online* (groundedness, corrección); privacidad; muestreo.

**Resultados.** 120 peticiones simuladas con latencia de cola larga y 4% de fallos. Por petición: **p50 45 ms, p95 130 ms, p99 171 ms**, 1% de errores tras reintentos, ~0,00014 $ por petición. Se cumplen los SLOs. Token-F1 medio 0,88, y drill-down a la peor traza.

---

<a id="16"></a>
## 16 · MCP: Model Context Protocol
`notebooks/05_agents_and_mcp/16_mcp_model_context_protocol.ipynb` · módulo `mcp`, script `scripts/mcp_server.py`

**Qué construyes.** Un servidor y un cliente MCP desde cero, los mensajes JSON-RPC vistos "en el cable", tools, resources y prompts, un agente que usa tools MCP, un servidor real por stdio y controles de seguridad.

**Conceptos clave.** Host, cliente y servidor; *handshake* (`initialize`); peticiones frente a notificaciones; errores de protocolo frente a `isError`; stdio (nunca escribir en stdout); allowlist, *namespacing*, confirmación de acciones destructivas, inyección en resultados de tools y *tool poisoning*.

**Resultados.** **7/7 checks de conformidad** por stdio real. La búsqueda acierta el documento en 6/6 preguntas. Ida y vuelta por stdio ~0,2 ms (despreciable frente a una llamada a un LLM). La inyección escondida en el resultado de una tool queda detectada y aislada.

---

## Y después

- **Refuerza lo aprendido:** cada notebook termina con *Takeaways* y el equivalente de producción. Repite el experimento con la herramienta real (`trl`, `peft`, vLLM, LangChain, LangGraph, el SDK de MCP).
- **Cambia los datos:** pasa tus propios documentos a `rag_demo.py` o a `scripts/mcp_server.py --docs`, y tu propio texto al notebook 00.
- Consulta el [glosario](glosario.md) para cualquier término y la [referencia de módulos](referencia_modulos.md) para la API.
