# Guía de notebooks

🇬🇧 [English version](../en/notebooks_guide.md) · [← Volver al índice](README.md)

Los 17 notebooks uno a uno: qué construyes, qué herramientas usas, qué conceptos te llevas, qué se mide y qué resultados salen. Los notebooks están en inglés; esta guía te acompaña en español.

El curso tiene dos mitades. **Los notebooks 00–06** construyen un modelo de lenguaje desde cero en PyTorch, con el paquete propio del curso, `llm_edu_suite`, para que veas cada pieza. **Los notebooks 07–16** hacen los mismos trabajos con los frameworks que usa la industria: `transformers`, `peft`, `trl`, `bitsandbytes` y `optimum-quanto` de Hugging Face sobre SmolLM2 (07–09), LangChain, LangGraph y LangSmith (10–15) y el SDK oficial de MCP (16). El modelo local de 10–16 es `qwen2.5:3b` servido con Ollama, con `nomic-embed-text` para los embeddings.

**Cómo usar esta guía.** Lee la ficha antes de abrir el notebook, ejecútalo celda a celda y vuelve a la ficha para contrastar tus números. Todos los números de esta guía salen de la ejecución subida al repositorio (portátil con una RTX 3060 de 6 GB). Los tuyos se moverán un poco: los modelos pequeños son sensibles a la aritmética de cada máquina, y todo lo que genera o juzga un LLM cambia entre ejecuciones. Donde un resultado es frágil, el notebook lo dice, y esta guía también. Cada notebook incluye además apartados de teoría ("Background: ...") antes del código que los usa; la versión larga y en español de esa teoría, con las ecuaciones, está en los [fundamentos teóricos](fundamentos.md).

**Qué necesita cada bloque**
- **00–06**: solo PyTorch y `llm_edu_suite`. Funcionan también en CPU.
- **07–09**: `transformers`, `peft`, `trl`, `bitsandbytes` (QLoRA en el 07) y `optimum-quanto`. La primera vez se descarga SmolLM2 del Hugging Face Hub. Cada notebook limita su propia VRAM (`VRAM_BUDGET_GB`) para caber en la GPU junto a Ollama: 2 GB por defecto en el 07, que entrena nueve modelos y cuyo modelo de 4 bits deja ~0,1 GB ocupados, y 1,5 GB en el 08 y el 09.
- **10–16**: **Ollama tiene que estar en marcha**, con `qwen2.5:3b` y `nomic-embed-text` descargados. La primera celda (`setup()` de `llm_edu_suite.config`) lo comprueba. El notebook 10 usa además `qwen2.5:7b` como juez de RAGAS si está descargado (`ollama pull qwen2.5:7b`, opcional; si no, juzga el 3B). En el notebook 12, Ollama solo hace falta para los embeddings de contenido y la capa de LLM (peticiones y explicaciones); SASRec, el two-tower y el ranker de LightGBM son PyTorch y LightGBM puros, y MovieLens-1M se descarga la primera vez en `data/cache/`.
- **LangSmith es opcional.** Con `LANGSMITH_TRACING=true` y `LANGSMITH_API_KEY` en `.env`, cada llamada de LangChain/LangGraph de 10–16 queda trazada. Sin clave todo funciona igual, y el notebook 15 se salta las celdas que necesitan el servidor de LangSmith e indica que las ha omitido. La ejecución publicada se hizo con LangSmith activado.

| Tópico | Notebooks | Tiempo en GPU de la ejecución final |
|---|---|---|
| Empieza aquí | [00](#00) | 32 s |
| Arquitectura | [01](#01) · [02](#02) · [03](#03) · [04](#04) | 5 s · 31 s · 180 s · 48 s (≈ 4,5 min) |
| Pre-entrenamiento e inferencia | [05](#05) · [06](#06) · [09](#09) | 35 s · 61 s · 53 s (≈ 2,5 min) |
| Post-training | [07](#07) · [08](#08) | ≈ 14 min · 645 s (≈ 25 min) |
| Aplicaciones | [10](#10) · [12](#12) · [13](#13) | ≈ 19 min · ≈ 25 min · 140 s (≈ 46 min) |
| Agentes y MCP | [11](#11) · [14](#14) · [16](#16) | 56 s · 119 s · 65 s (≈ 4 min) |
| LLMOps | [15](#15) | 129 s |

Los notebooks 07, 10 y 12 se volvieron a ejecutar en v2.1 y sus tiempos son aproximados: el 07 entrena ahora también QLoRA, el 10 usa un juez de 7B (unos 17 de sus 19 minutos son la puntuación de RAGAS) y el 12 incluye ~11 minutos de construcción de features del ranker. En total, alrededor de hora y media para los 17, en GPU y con Ollama sin compartir la GPU con otros trabajos. Si otro proceso usa la GPU, los tiempos de 10–16 crecen mucho (las llamadas al LLM hacen cola). Los tiempos en CPU no se midieron en esta ejecución: 00–06 funcionan bien en CPU, pero cuenta con que 07–09 sean mucho más lentos sin GPU.

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

**Resultados.** 1,86M de parámetros, vocabulario de 455 tokens (199 fusiones). Pérdida inicial 6,144 frente a ln V = 6,120. Los 800 pasos de entrenamiento tardan **~19 s en GPU**. Pérdida de validación 0,778 frente a un suelo teórico de 0,774: **a un 0,6% del óptimo**. El **82%** de las frases muestreadas (40/49) son gramaticales. Se aprecia el bucle típico de la decodificación greedy.

**Ejercicios.** Entrénalo con tiny Shakespeare (`text_path`), quita RoPE y observa qué pasa con la gramática, o añade una KV-cache a `generate`.

---

<a id="01"></a>
## 01 · Tokenización
`notebooks/01_architecture/01_tokenization.ipynb` · módulo `tokenizer`

**Qué construyes.** Un BPE byte-level con entrenamiento, codificación, decodificación, tokens especiales y guardado/carga.

**Conceptos clave.** Por qué se usan sub-palabras; por qué el BPE byte-level nunca produce `<UNK>`; cómo el tamaño del vocabulario intercambia longitud de secuencia por tamaño de la tabla de embeddings; la **fertilidad** (tokens por palabra) como medida de lo bien que un tokenizador cubre un dominio o idioma.

**Resultados.** 255 fusiones, vocabulario de 514 tokens con 3 especiales. La frase de ejemplo ocupa 52 bytes y se codifica en 7 tokens (7,43 bytes/token), con round-trip exacto. Fertilidad **1,00 en el dominio de entrenamiento frente a 3,90 fuera de él** (código y español): esto explica por qué los idiomas poco representados salen más caros en tokens.

---

<a id="02"></a>
## 02 · Atención, RoPE y normalización
`notebooks/01_architecture/02_attention_rope_norm.ipynb` · módulos `layers`, `model`

**Qué construyes.** RMSNorm (verificado contra `torch.nn.RMSNorm`), RoPE con escalado tipo Llama 3.1, la atención escrita a mano (verificada contra `F.scaled_dot_product_attention`), MHA → GQA → MQA → MLA, y la KV-cache.

**Conceptos clave.** RoPE solo depende de la distancia relativa (el notebook lo comprueba numéricamente). GQA comparte cabezas K/V para reducir la caché. MLA (DeepSeek) comprime K/V en un vector latente. La generación incremental con caché debe dar exactamente lo mismo que el forward completo (diferencia máxima ~1e-7 en GQA y MLA).

**Resultados.** Entrenando la misma red con cada variante: **MHA 2048 B/token de caché frente a MLA 160 B/token, con la misma perplejidad (2,65)**. GQA-2 y MQA quedan en medio (512 y 256 B/token, perplejidad 2,64). En DeepSeek-V3 la reducción por capa es de unas 57 veces.

---

<a id="03"></a>
## 03 · Atención eficiente para contextos largos
`notebooks/01_architecture/03_efficient_attention.ipynb` · módulo `layers`

**Qué construyes.** Ventana deslizante (SWA), Longformer, FlexAttention (si tu PyTorch la trae), block-sparse seleccionado por contenido, atención lineal (con su forma recurrente de estado constante) e híbridos.

**Conceptos clave.** Coste O(T²) frente a O(T·w) frente a O(T); campo receptivo de SWA apilada; por qué la atención lineal tiene un estado de tamaño fijo; **elegir la métrica correcta**: la perplejidad global no sirve en esta tarea, la precisión de copia sí.

**Resultados.** Tarea de copia (un tramo de 12 tokens repetido tras un hueco de 30, es decir, a ~42 tokens de distancia), 1500 pasos:

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

**Resultados.** Con top-2 de 8 expertos, la capa tiene 3× (2,99×) más parámetros totales que activos. El balanceo aux-free lleva un router desequilibrado de **4,0× a 1,15×** con `bias_lr=0,01`, mientras que con 0,1 oscila (solo baja a 1,68×; se ve en la gráfica). En el corpus de juguete el MoE no supera al modelo denso: densa ppl 3,24, MoE con pérdida auxiliar 3,31, MoE aux-free 3,24, con los mismos parámetros activos. El notebook explica por qué el MoE rinde a gran escala.

---

<a id="05"></a>
## 05 · Pre-entrenamiento
`notebooks/02_pretraining_and_inference/05_pretraining.ipynb` · módulos `training`, `data`

**Qué construyes.** El pipeline completo: texto → tokenizador → modelo → bucle con AdamW, warmup + coseno, clipping, checkpoints y dashboard (pérdida, norma del gradiente, LR); después generación, guardado y carga.

**Conceptos clave.** Comprobaciones de cordura (pérdida inicial ≈ ln V, test de que el modelo no ve el futuro); cómo leer el dashboard; **pérdida irreducible** (el término *E* de Chinchilla); ablación de schedule (coseno frente a WSD); sanity check de escalado.

**Resultados.** Perplejidad de validación 2,79 frente a un suelo teórico de 2,75 (**un 1,5% por encima del suelo** en pérdida). A igual número de pasos, el modelo de 1,67M aprende mucho más rápido que el de 125K (pérdida 1,054 frente a 1,259 en el paso 50). Coseno y WSD acaban prácticamente iguales (1,024 frente a 1,027).

---

<a id="06"></a>
## 06 · Inferencia y serving
`notebooks/02_pretraining_and_inference/06_inference_serving.ipynb` · módulos `inference`, `model`

**Qué construyes.** Estrategias de muestreo, KV-cache, **decodificación especulativa desde cero** (con su comprobación de que conserva la distribución), batching y medición de TTFT y throughput.

**Conceptos clave.** Temperatura, top-k y top-p como transformaciones de la distribución (gráfica y entropía); diversidad (distinct-2); por qué la decodificación especulativa es exacta; el decode está limitado por memoria; y el *overhead* de lanzar kernels en GPU con modelos pequeños.

**Resultados.** Al empezar una palabra, la entropía es de 4,25 nats con T=1,0 y baja a 2,89 con T=0,7 + top-k 20. Greedy produce la misma continuación 16 de 16 veces. La especulativa **acepta el 98% de los tokens del draft, ~4,6 tokens por pasada del modelo grande**. Con batch 16 se obtienen ~16× los tokens/s de batch 1 (2180 frente a 137). En GPU, con un modelo tan pequeño, la KV-cache apenas acelera (1,2×); el notebook explica por qué y cuándo cambia.

---

<a id="07"></a>
## 07 · Fine-tuning con Hugging Face: SFT, LoRA y destilación
`notebooks/03_post_training/07_finetuning_sft_lora.ipynb` · `transformers`, `peft`, `trl` · también como script: `python scripts/finetune.py sft`

**Qué construyes.** Ajustas `HuggingFaceTB/SmolLM2-135M-Instruct` para que convierta una petición de reserva en JSON estricto (`name`, `restaurant`, `people` como entero, `day` en minúsculas, `time` en formato 24 h). 300 ejemplos etiquetados; 40 peticiones de test con nombres y restaurantes que no aparecen en el entrenamiento. Línea base few-shot → SFT completo → LoRA → QLoRA sobre un base de 4 bits → la trampa del "token especial nuevo" → ablación de rango → destilación desde un teacher de 360M.

**Herramientas.** `AutoTokenizer.apply_chat_template`, `AutoModelForCausalLM`; `trl.SFTConfig` / `SFTTrainer` con un dataset prompt/completion (pérdida solo sobre la respuesta); `peft.LoraConfig(r=16, lora_alpha=32, target_modules="all-linear")`, `get_peft_model`, `PeftModel.from_pretrained`, `merge_and_unload()`; `tokenizer.add_tokens` + `resize_token_embeddings`, y `LoraConfig(modules_to_save=["embed_tokens"], ensure_weight_tying=True)`; para QLoRA, `BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)`, `peft.prepare_model_for_kbit_training` y `optim="paged_adamw_8bit"`.

**Conceptos clave**
- Plantilla de chat y *loss masking*: las etiquetas del prompt son `-100` y la pérdida cubre la respuesta **incluido `<|im_end|>`** (en el ejemplo, 42 de 110 tokens). El notebook comprueba que TRL construye exactamente las mismas etiquetas.
- Memoria: AdamW cuesta ~16 bytes por parámetro entrenable (2,2 GB para 135M, ~112 GB para 7B). El fine-tuning completo solo cabe en el presupuesto del notebook (2 GB) con pesos bf16 + Adafactor.
- LoRA empieza idéntico al modelo base (`B = 0`) y se fusiona sin coste.
- **QLoRA**: los pesos congelados se guardan en NF4 (16 niveles en los cuantiles de una normal), con doble cuantización de las escalas y un optimizador paginado; los adaptadores siguen en bf16 y cada multiplicación descuantiza primero. Ahorra memoria de pesos, no de activaciones, y el adaptador se fusiona en una copia bf16 del base.
- **La trampa del token nuevo**: con embeddings compartidos e inicialización en la media, un token nuevo nunca puede ser el más probable si los embeddings están congelados. Solución: entrenar el embedding.
- La pérdida de entrenamiento no dice la precisión en la tarea; las métricas por campo dicen dónde están los errores (aquí, casi siempre en `time`).

**Resultados** (40 peticiones de test; cada una vale 2,5 puntos)

| Modelo | JSON válido | Exact match | Campos | `time` | Entrenables | Segundos | VRAM pico |
|---|---|---|---|---|---|---|---|
| base, zero-shot | 5% | 0% | 2% | 0% | – | – | – |
| base, few-shot (4 ejemplos) | 100% | 17,5% | 73% | 25% | – | – | – |
| SFT completo | 100% | 35% | 85,5% | 37,5% | 134,5 M | 52 | 1,19 GB |
| LoRA r=2 | 100% | 20% | 80,5% | 30% | 0,6 M | 37 | 1,46 GB |
| LoRA r=16 | 100% | 32,5% | 86,5% | 32,5% | 4,9 M | 41 | 1,41 GB |
| QLoRA r=16 (base NF4) | 100% | 37,5% | 86,5% | 45% | 4,9 M | 47 | 1,34 GB |
| LoRA r=64 | 100% | 45% | 89% | 45% | 19,5 M | 72 | 1,25 GB |
| teacher 360M, LoRA r=16 | 100% | **87,5%** | 97,5% | 92,5% | 8,7 M | 181 | 1,27 GB |
| student 135M, etiquetas humanas + del teacher | 100% | **55%** | 91% | 55% | 4,9 M | 71 | 1,54 GB |

- LoRA r=16 entrena el 3,5% de los parámetros y queda a una petición del SFT completo. El adaptador ocupa 19,6 MB (el modelo en bf16, 269 MB). Fusionado en fp32 cambia los logits como mucho 1e-4 y mantiene el mismo token greedy en todas las posiciones.
- **QLoRA** no pierde calidad: 37,5% de exact match frente al 32,5% de LoRA, con el mismo 86,5% de campos. Son dos peticiones de 40, ruido, así que la lectura honesta es "igual", no "mejor". Los pesos en memoria bajan de 269 MB a 110 MB (210 capas lineales en 4 bits; el embedding atado de 28M parámetros se queda en bf16), pero el pico de entrenamiento apenas se mueve (1,41 → 1,34 GB): a 135M dominan las activaciones y la capa de salida de 49k tokens. Cada paso es algo más lento (47 s frente a 41 s) porque descuantiza. El ahorro decisivo llega con modelos grandes: los pesos congelados de un 7B pasan de 14 GB a 3,6 GB, los de un 13B de 26 a 6,7 GB y los de un 70B de 140 a 36,1 GB.
- **Token nuevo `<|json|>`**: con embeddings congelados no se emite nunca (0% de las respuestas; puesto ~24.000 como primer token, probabilidad ≈ 0, pérdida de entrenamiento 0,30) y el modelo degenera. Con el embedding entrenable (33,2M parámetros) abre el 100% de las respuestas (p ≈ 0,93) y el exact match vuelve a 32,5%. `trainable_token_indices` falla dentro de `SFTTrainer` con un `lm_head` compartido en las versiones usadas (trl 1.17, peft 0.21).
- **Destilación**: el teacher etiqueta 600 peticiones nuevas en 86 s, el 90% exactamente bien; el student pasa de 32,5% a 55%. Matices honestos: el student también dio 57 pasos de optimización en lugar de 38, y aprende de etiquetas que fallan un 10%.
- Las palancas más grandes fueron el tamaño del modelo y más datos (aunque los etiquete una máquina); el rango vino después.

---

<a id="08"></a>
## 08 · Preferencias y RL con TRL: DPO y GRPO
`notebooks/03_post_training/08_preference_rl_dpo_grpo.ipynb` · `trl`, `peft` · también como script: `python scripts/finetune.py dpo` / `grpo`

**Qué construyes.** Sobre la tarea de reservas del notebook 07, un modelo SFT entrenado a medias (LoRA, 30% de exact match), y encima DPO y DPO + NLL con 400 pares de preferencias (256 de ellos usan como respuesta rechazada el propio error del modelo SFT). Después, GRPO con recompensas verificables sobre sumas y restas de dos cifras, partiendo del modelo instruct, con una recompensa **escasa** (exacta) y otra **densa** (crédito parcial + bonus de formato).

**Herramientas.** `trl.DPOConfig(beta=0.1, loss_type="sigmoid")` / `DPOTrainer` con `peft_config` (el modelo de referencia es la misma red con el adaptador apagado); DPO + NLL como `loss_type=["sigmoid", "sft"]`, `loss_weights=[1.0, 1.0]` (antes `rpo_alpha`); `trl.GRPOConfig(num_generations=8, max_completion_length=32, temperature=1.0, beta=0.04)` / `GRPOTrainer(reward_funcs=[reward_exact, reward_partial])` con `reward_weights=[1, 0]` o `[0, 1]`, para registrar las dos recompensas y que solo una guíe la actualización.

**Conceptos clave**
- DPO como clasificación sobre recompensas implícitas; los mejores pares son los errores de tu propio modelo.
- ***Likelihood displacement***: DPO puede bajar también la probabilidad de la respuesta buena. Vigila `logps/chosen`, no solo la precisión de preferencia.
- El ancla NLL sobre la respuesta elegida como opción por defecto barata y robusta.
- GRPO: ventaja relativa al grupo en lugar de un crítico; los grupos donde todas las muestras reciben la misma recompensa no aportan gradiente.
- Diseño de recompensas y **ley de Goodhart**: la recompensa densa es un proxy, no la métrica.

**Resultados**

Extracción de reservas (40 peticiones de test):

| Método | JSON válido | Exact match | Campos | `time` |
|---|---|---|---|---|
| SFT (punto de partida) | 100% | 30% | 86% | 30% |
| + DPO | 95% | **7,5%** | 77,5% | 10% |
| + DPO + NLL | 100% | 32,5% | 86,5% | 32,5% |

- DPO solo **empeora** el modelo: log π(elegida) cae de −1,91 a −5,63 (rechazada: −4,81 → −12,74), aunque sus métricas de entrenamiento parezcan buenas. Con NLL la elegida se mantiene (−1,91 → −2,01) mientras la rechazada sigue bajando (−4,81 → −8,22).
- Honesto: DPO + NLL gana una sola petición (30% → 32,5%), que es ruido. Al modelo le falta una *regla* (12 h → 24 h), no una preferencia; más datos de SFT fueron la palanca más fuerte en el 07.

Aritmética (100 problemas de test, 40 pasos de GRPO):

| Modelo | Exact match | Recompensa densa media | Error absoluto mediano | Tiempo |
|---|---|---|---|---|
| instruct, sin RL | 5% | 0,090 | 46 | – |
| GRPO, recompensa escasa | **46%** | 0,553 | 2 | 200 s |
| GRPO, recompensa densa | 42% | 0,566 | 2 | 194 s |

- Sin una sola respuesta etiquetada, GRPO pasa de 5% a 42–46% en unos 3,5 minutos. Las respuestas se acortan (~27 → ~13 tokens): la política aprende que un `96 - 20 = …` corto se premia más a menudo que una explicación que se corta a medias.
- La recompensa densa da señal a más grupos, pero no produjo un modelo mejor: un poco más alta en su proxy y un poco más baja en exact match. Es la dirección que predice Goodhart, pero 4 problemas de 100 están dentro del ruido.
- La VRAM pico fue como mucho 1,40 GB en todo el notebook (0,88 GB en GRPO).

---

<a id="09"></a>
## 09 · Cuantización en la práctica
`notebooks/02_pretraining_and_inference/09_quantization.ipynb` · `optimum-quanto`

**Qué construyes.** Primero por dentro, en pocas líneas de torch: absmax por tensor, por fila y por bloques, y NF4, sobre una matriz de pesos real de SmolLM2-135M (tiene un valor atípico a 15 desviaciones típicas). Después un modelo real, SmolLM2-360M-Instruct, cuantizado a int8, float8, int4 e int2.

**Herramientas.** `optimum.quanto.quantize(model, weights=qint8 | qfloat8 | qint4 | qint2)` y `freeze(model)` (int4/int2 en grupos de 128), dejando `lm_head` en bf16. Medición contra el modelo fp32 sobre ~500 tokens de texto no visto. Tabla de producción: bitsandbytes (QLoRA), GPTQ, AWQ, GGUF (el `qwen2.5:3b` del curso es un GGUF `Q4_K_M`), FP8, torchao, cuantización de la KV-cache.

**Conceptos clave.** Por qué escalas más finas reducen el error (y cuestan ~0,25 bits extra por peso con bloques de 64); NF4 coloca sus niveles en los cuantiles de una normal; **las métricas que de verdad detectan la degradación**: KL frente al modelo original y acuerdo top-1, que es lo que usa llama.cpp. La perplejidad sola esconde diferencias. El ahorro de memoria está garantizado; la velocidad necesita un kernel fusionado.

**Resultados.** En la matriz real, el error relativo del peso a 4 bits pasa de 58,7% (por tensor) a 16,7% (por fila) y 11,4% (bloques de 64); NF4 lo baja a 9,3%. A 8 bits: 3,4% / 0,9% / 0,6%.

| SmolLM2-360M | Pesos | Perplejidad | KL vs fp32 | Acuerdo top-1 | Tokens/s |
|---|---|---|---|---|---|
| fp32 (referencia, CPU) | – | 17,16 | – | – | – |
| bf16 | 0,72 GB | 17,13 | 0,0010 | 97,2% | 24,0 |
| int8 | 0,41 GB | 17,32 | 0,0048 | 96,6% | 16,8 |
| float8 | 0,41 GB | 17,13 | 0,0094 | 95,1% | 15,2 |
| int4 (g128) | 0,26 GB | 22,16 | 0,27 | 73,3% | 10,5 |
| int2 (g128) | 0,19 GB | ~228.000 | 9,40 | 1,4% | 8,3 |

- bf16 tampoco es exacto: su 97,2% de acuerdo es el suelo de ruido.
- int8 y float8 salen casi gratis. float8 tiene **la misma perplejidad que bf16 pero el doble de KL que int8**: la perplejidad esconde lo que la KL muestra.
- int4 deja la memoria en un tercio de bf16, pero en un modelo tan pequeño cuesta una cuarta parte del acuerdo top-1 y un +29% de perplejidad. int2 con redondeo simple destruye el modelo (sopa de palabras).
- Honesto: en esta GPU **los modelos cuantizados van más lentos** que bf16. Con batch 1, un modelo de 360M está limitado por el lanzamiento de kernels, y quanto añade trabajo de desempaquetado; la aceleración real necesita un kernel fusionado para el formato.

---

<a id="10"></a>
## 10 · RAG con LangChain, evaluado con RAGAS
`notebooks/04_applications/10_rag.ipynb` · LangChain, FAISS, RAGAS · también como script: `python scripts/rag_demo.py`

**Qué construyes.** Un sistema RAG completo sobre la propia documentación del curso (`data/knowledge_base/*.md`, 5 ficheros, ~190 KB): carga, troceado con tres tamaños, embeddings, índice vectorial, BM25, fusión híbrida, una cadena LCEL con citas numeradas, multi-query (RAG-Fusion) y una evaluación que separa recuperación y generación.

**Herramientas.** `DirectoryLoader` + `TextLoader`; `RecursiveCharacterTextSplitter` (300 / 800 / 1500 caracteres → 978 / 358 / 174 chunks); `OllamaEmbeddings(nomic-embed-text)` (768 dimensiones) y `ChatOllama(qwen2.5:3b)` a través de `llm_edu_suite.config`; `FAISS` (`IndexFlatL2` exacto, embeddings en lotes de 64); `BM25Retriever` con un tokenizador que pasa a minúsculas; `EnsembleRetriever` (*Reciprocal Rank Fusion*, `c = 60`, en `langchain_classic`), verificado contra una reimplementación de seis líneas; LCEL con `ChatPromptTemplate`, `RunnableParallel`, `.assign`, `StrOutputParser`; RAGAS 0.4 con `llm_factory` sobre la API compatible con OpenAI de Ollama y las métricas `Faithfulness`, `AnswerRelevancy`, `ContextPrecision`, `ContextRecall`; el juez sale de `config.judge_model()` (`qwen2.5:7b` si está descargado, o lo que diga `JUDGE_MODEL`) y los intervalos de `evaluation.bootstrap_ci` y `paired_bootstrap_diff`.

**Conceptos clave.** Búsqueda densa frente a léxica y por qué combinarlas (la documentación técnica está llena de cadenas exactas); el tamaño de chunk como hiperparámetro; métricas a nivel de fichero frente a nivel de evidencia (¿el *chunk* recuperado contiene la evidencia?); hit@k y MRR; las cuatro métricas de RAGAS y cuáles necesitan referencia; intervalos bootstrap y la diferencia pareada para no leer ruido; **validar al juez** antes de fiarse de él, porque un juez débil no solo mete ruido, también inventa efectos.

**Resultados** (20 preguntas etiquetadas en `data/eval/rag_eval.json`, 4 por documento; chunk 800, k = 4)

| Recuperador | Fichero hit@1 | Evidencia hit@1 | Evidencia hit@4 | Evidencia MRR |
|---|---|---|---|---|
| densa | 0,95 | 0,65 | 0,85 | 0,73 |
| BM25 | 0,90 | 0,70 | **0,95** | 0,80 |
| **híbrida (RRF)** | 0,95 | **0,80** | 0,90 | **0,84** |
| híbrida + multi-query | 0,95 | 0,75 | 0,90 | 0,82 |

- El hit@4 a nivel de fichero es 1,00 para todos (saturado); las diferencias solo aparecen a nivel de evidencia. La híbrida es la mejor en la parte alta del ranking, y BM25 resulta sorprendentemente fuerte en este corpus.
- **Multi-query no ayudó** (MRR 0,82, con una llamada extra al LLM por pregunta).
- El tamaño de chunk no empuja a todos en la misma dirección (MRR de evidencia con 300 / 800 / 1500): la densa empeora, 0,77 / 0,73 / 0,67, mientras que BM25 (0,74 / 0,80 / 0,82) y la híbrida (0,75 / 0,84 / 0,89) mejoran.
- Sin recuperación, el modelo se inventa qué significa MCP ("Management and Configuration Protocol").

RAGAS, con `qwen2.5:7b` como juez de las respuestas que genera `qwen2.5:3b` (1038 s para 40 respuestas × 4 métricas, la mayor parte del tiempo del notebook). Entre corchetes, el intervalo bootstrap del 95% sobre las 20 preguntas:

| Configuración | Faithfulness | Answer relevancy | Context precision | Context recall |
|---|---|---|---|---|
| densa | 0,87 [0,74, 0,97] | 0,78 [0,64, 0,89] | 0,79 [0,62, 0,92] | 0,95 [0,85, 1,00] |
| híbrida | 0,87 [0,74, 0,96] | 0,75 [0,60, 0,86] | **0,92** [0,80, 0,99] | 0,95 [0,85, 1,00] |
| híbrida − densa (pareado) | −0,00 [−0,16, +0,17] | −0,03 [−0,16, +0,10] | **+0,13 [+0,03, +0,25]** | +0,00 [−0,15, +0,15] |

- Solo hay **una** diferencia clara: *context precision* sube de 0,79 a 0,92 y su intervalo excluye el cero. La híbrida coloca más arriba los chunks útiles, lo mismo que decía el MRR por etiquetas. En las otras tres métricas no hay diferencia: cuando el pasaje correcto está en algún lugar del top 4, el generador de 3B contesta igual de bien con cualquiera de las dos listas.
- **El juez pesó más que el sistema.** Cuando el propio 3B se juzgaba a sí mismo (la versión anterior del notebook), informaba de faithfulness 0,38 → 0,54 y answer relevancy 0,44 → 0,30 de densa a híbrida, y daba relevancia 0 a respuestas correctas. Con el juez de 7B esas "diferencias" desaparecen dentro de los intervalos. Un juez débil no solo añade ruido: puede inventar efectos.
- Las filas bajas de la tabla por pregunta separan los tres modos de fallo. "¿Dónde guarda `mini_llm.py` su checkpoint?" es un fallo de **recuperación** (la evidencia no está en el top 4; precision y recall 0). La pregunta sobre la precisión de copia del notebook 03 es un fallo de **generación**: la evidencia estaba en primera posición, las dos métricas de contexto valen 1,00 y aun así el 3B contestó "I don't know". La pregunta de DPO + NLL muestra un problema de **etiqueta**: nuestra comprobación por cadena de evidencia no encuentra el pasaje, pero el juez sí encuentra el dato en el contexto (recall 1,00). Las etiquetas de referencia también hay que revisarlas.
- Coste: un juez más fuerte es más lento (unos 17 minutos con el 7B frente a unos 7 con el 3B). La evaluación es una partida de presupuesto: un set fijo, pequeño y representativo en cada cambio, no todo.

---

<a id="11"></a>
## 11 · Herramientas, function calling y salida estructurada con LangChain
`notebooks/05_agents_and_mcp/11_agents_tools_structured_output.ipynb` · `langchain-core`, `langchain`, `langchain-ollama`, `pydantic`

**Qué construyes.** Cuatro tools para una tienda pequeña (tiempo, calculadora, consulta de pedidos, búsqueda en la documentación), el bucle de tool calling a mano, el mismo bucle con `create_agent`, el manejo de errores, salida estructurada con validación y reintentos, y una evaluación de trayectorias sobre peticiones etiquetadas.

**Herramientas.** `@tool` (nombre, docstring y type hints se convierten en el esquema) y `args_schema` de Pydantic (descripciones por campo, `pattern` con regex); `convert_to_openai_tool`; `llm.bind_tools`, `AIMessage.tool_calls`, `ToolMessage`, `tool.invoke(tool_call)`; `langchain.agents.create_agent`; `GenericFakeChatModel` como modelo con guion para pruebas deterministas; `ToolErrorMiddleware`; `with_structured_output(Schema, method="json_schema" | "function_calling" | "json_mode", include_raw=True)` con un `field_validator` de Pydantic.

**Conceptos clave.** El modelo nunca ejecuta nada: lo hace tu código. Las descripciones y los esquemas son prompts. Los esquemas de las tools se reenvían en cada llamada (~390 tokens de prompt aquí). Los errores de validación vuelven al modelo automáticamente; las excepciones de la tool, solo si lo activas. La decodificación restringida garantiza la *forma* de la salida, no su *contenido*. Reintentar *con el error*, no a ciegas. Evaluar la trayectoria, no solo el texto.

**Resultados**
- El bucle a mano resuelve "pedido A100 más un 21% de impuestos" en dos pasos de tool (`order_status` y luego `calculate`) → 145,20 $. `create_agent` lanza en paralelo, en un mismo turno, la consulta del tiempo y la del pedido.
- Con tools disponibles, `qwen2.5:3b` se negó a responder "¿quién escribió el Quijote?", algo que sabía responder solo.
- Un `order_id` inválido (`a-100`) vuelve como `ToolMessage` con `status=error`. Sin middleware, una excepción dentro de la tool rompe la ejecución; con `ToolErrorMiddleware` el modelo lee el error y reintenta con `2**10` → 1024.
- **Reintento con feedback**: el resumen pasó de 22 palabras a 13 y otra vez 13; tras 3 intentos seguía sin cumplir el límite de 12 palabras y la función devolvió `None`. El feedback ayudó, pero no bastó: limita los intentos y registra los fallos.
- **Salida estructurada, 8 emails etiquetados**: los tres métodos producen 8/8 objetos válidos, pero la precisión por campo es 85% con `json_schema` (devolvió `order_id = null` en los emails que mencionan un pedido), 98% con `function_calling` y 98% con `json_mode`.
- **Evaluación del agente, 14 peticiones**: selección de tool 79%, argumentos válidos 86%, **éxito de la tarea 64%**, latencia mediana 0,8 s, 864 tokens de media. Fallos: aritmética sin calculadora, `c300` en minúsculas que rompió el patrón (el modelo pidió el id al usuario en vez de corregirlo), una charla que llamó a `search_docs`, una traducción que rechazó y una petición de varios pasos que se saltó la consulta del pedido.

---

<a id="12"></a>
## 12 · Sistemas de recomendación: el embudo de la industria, con LLMs donde ayudan
`notebooks/04_applications/12_recommender_systems.ipynb` · PyTorch, FAISS, LightGBM, LangChain · módulos `recsys`, `model`, `evaluation`

**Qué construyes.** El embudo de un recomendador industrial, etapa por etapa, sobre datos reales: MovieLens-1M (1.000.209 valoraciones de 6.040 usuarios sobre 3.706 películas), que se descarga la primera vez en `data/cache/` (la licencia de GroupLens no permite redistribuirlo, así que no está en el repositorio). Dos líneas base (popularidad e ItemKNN), SASRec como modelo de lenguaje sobre películas, embeddings de contenido con y sin palabras clave escritas por un LLM, un two-tower con índice FAISS, la unión de cinco fuentes de candidatos, un ranker LambdaMART, una ranura de exploración para estrenos, MMR para la diversidad y una capa de LLM que convierte peticiones en filtros y escribe explicaciones ancladas en el historial. Una regla atraviesa el notebook: **una etapa solo entra en el sistema si mejora a la anterior en validación**.

**Herramientas.** `llm_edu_suite.recsys` (`download_movielens_1m`, `load_movielens_1m`, `temporal_split`, `target_ranks`, `rank_metrics`); el `TransformerLM` del notebook 02 como SASRec (2 capas, 64 dimensiones, ventana de 200 películas, softmax completa, parada temprana por nDCG de validación); `OllamaEmbeddings(nomic-embed-text)` para el texto de cada película; un two-tower en PyTorch con *id dropout*; un índice HNSW de FAISS frente a la búsqueda exacta; `lightgbm` con `objective="lambdarank"`; `evaluation.paired_bootstrap_diff` en cada comparación; LangChain con `with_structured_output(method="function_calling", include_raw=True)` y una búsqueda sin filtros como respaldo; `scripts/generate_item_keywords.py` para las palabras clave (ya generadas en `data/recsys/ml1m_llm_keywords.json`) y `data/recsys/conversational_eval.json` (20 peticiones con restricciones verificables).

**Conceptos clave**
- El embudo: generación de candidatos (se mide con el recall del conjunto), ranking (nDCG@10 de la lista) y re-ranking (se mide por lo que intercambia). El ranker solo ordena lo que trajo la recuperación.
- Protocolo honesto: *leave-last-two* en orden temporal (el último evento a test, el penúltimo a validación), **ranking completo** sobre las 3.706 películas en lugar de 100 negativos muestreados (Krichene y Rendle, 2020), un 10% de las películas (370) reservadas como estrenos fríos (747 de los 6.040 objetivos de test son fríos) y bootstrap pareado en cada comparación.
- Recomendación secuencial como predicción del siguiente token, y por qué la softmax completa gana a un negativo muestreado (a escala de millones: softmax muestreada con corrección logQ).
- *Cold start*: lo colaborativo nunca acierta una película nueva; el contenido la coloca y la exploración le da exposición.
- Learning to rank (pointwise, pairwise, listwise) y la receta que hace funcionar a un ranker de segunda etapa: muchas consultas de un periodo posterior y features que la primera etapa no ve.
- Usar el LLM para el lenguaje que rodea al recomendador, no para ordenar.

**Resultados** (test, 6.040 usuarios, ranking completo)

| Sistema | hit@10 | nDCG@10 | hit@10 conocidas | hit@10 nuevas |
|---|---|---|---|---|
| popularidad | 0,036 | 0,018 | 0,041 | 0 |
| ItemKNN | 0,125 | 0,067 | 0,143 | 0 |
| contenido (texto de metadatos, `nomic-embed-text`) | 0,033 | 0,017 | 0,031 | **0,047** |
| contenido + palabras clave del LLM | 0,023 | 0,012 | 0,023 | 0,020 |
| two-tower (con *id dropout*) | 0,194 | 0,109 | 0,221 | 0,001 |
| SASRec | 0,293 | 0,172 | 0,334 | 0 |
| **ranker LambdaMART** | **0,307** | **0,181** | **0,349** | 0,009 |
| ranker + 1 ranura de exploración | 0,300 | 0,179 | 0,336 | 0,044 |

- **SASRec** es el mejor modelo individual: más del doble que ItemKNN y ocho veces la popularidad, dentro del rango publicado para este dataset con ranking completo. La parada temprana se quedó con el paso 5.500. Nunca acierta una película nueva, porque sus filas de salida nunca se entrenaron.
- **Las palabras clave del LLM empeoraron el contenido** (hit@10 0,033 → 0,023) y la validación las rechazó. Muchas son genéricas ("nostalgia", "coming-of-age") y diluyen la señal del título y los géneros; algunas son falsas (a *The Ref* la describe como una película de deportes de instituto). En el two-tower empatan en validación (nDCG@10 0,1122 frente a 0,1110) y se queda la versión simple.
- **Candidatos.** La unión de cinco fuentes contiene el objetivo para el 68,4% de los usuarios (el top 100 de SASRec solo, el 60,8%) y es la única vía por la que entran estrenos (recall en fríos 0,149). Para el 32% restante ningún ranker puede acertar. El índice HNSW devuelve el 98,7% del top 100 exacto a ~0,01 ms por usuario, sin ganancia de velocidad con 3.706 ítems: aquí se ve el precio de la aproximación sin su beneficio.
- **El ranker.** Un primer intento, entrenado con un único positivo por usuario, no superó a SASRec y no pasó la puerta. Con la receta de RecTools y de Kaggle OTTO (los modelos de recuperación se ajustan sobre el historial menos los 8 últimos eventos: 48.320 eventos → 35.762 consultas y 1,82 M filas, 50 negativos muestreados, y features que SASRec no ve: valoraciones explícitas, demografía, co-visitas, días desde el evento anterior, tendencias) pasa la puerta en validación (nDCG +0,0059 [+0,0022, +0,0096]) y en test sube el hit@10 1,4 puntos [+0,7, +2,1] y el nDCG +0,0092 [+0,0055, +0,0128]. Las features de SASRec aportan la mayor parte de la ganancia; después vienen `days_since_prev`, `is_cold`, la ocupación, las co-visitas, el año y la valoración media.
- La feature de palabras clave del LLM en el ranker da +0,0012 [−0,0012, +0,0035]: el intervalo incluye el cero, y no compensa una hora de LLM por cada refresco del catálogo.
- **Exploración.** La ranura la llena el two-tower, el mejor ordenando estrenos en validación. Una ranura sube el hit@10 en películas nuevas de 0,009 a 0,044 por 0,7 puntos globales (0,307 → 0,300); dos, 0,290 / 0,074; tres, 0,275 / 0,094. Dónde parar es una decisión de producto, y un bandit la tomaría de forma adaptativa.
- **Diversidad (MMR con Jaccard de géneros).** λ = 0,8 sube la diversidad intra-lista de 0,659 a 0,719 por 0,4 puntos de hit@10 (0,307 → 0,303); λ = 0,6 da 0,775 / 0,288 y λ = 0,2, 0,852 / 0,222. Offline la diversidad solo puede parecer neutra o peor; su beneficio se mide online.

Capa de LLM (`qwen2.5:3b`):

| Medida | Resultado |
|---|---|
| cumplimiento de restricciones@5, 20 peticiones verificables | búsqueda por embeddings 0,60 [0,46, 0,74] · **LLM → filtros 0,87 [0,80, 0,93]** |
| fallos de salida estructurada | 1 de 221 (el respaldo mantuvo el sistema en pie) |
| hit@10 dentro del conjunto filtrado (200 usuarios, petición construida con el género y la década de su siguiente película real) | popularidad 0,285 · similitud con la petición 0,325 · **SASRec 0,555** |
| explicaciones ancladas (anclas = películas del historial más cercanas en el espacio de SASRec), 30 usuarios | citan un ancla el 97% · mencionan una película fuera del historial el 7% (2/30), frente al 13% (4/30) sin anclas |

- El LLM gana justo donde los embeddings son débiles: rangos de fechas y negaciones ("una película de crímenes de los 90 como Pulp Fiction" pasa de 0,0 a 0,8). Los fallos que quedan (pedir Animation sin Children's, poner el título de referencia en el campo equivocado) se arreglan con mejores descripciones de campo.
- La división del trabajo es clara: el LLM entiende *qué* se pide y el historial decide *cuál* de las películas que encajan.
- Las explicaciones sin ancla inventan atributos aunque no citen títulos (a *The Usual Suspects* le atribuye elementos bélicos); con ancla son verdaderas, aunque no siempre convincentes, porque los embeddings de SASRec codifican co-visionado, no contenido. Con 30 usuarios, 7% frente a 13% es una pista, no una prueba.
- **La conclusión honesta:** los LLMs pequeños no superan a los modelos colaborativos ordenando, pero son excelentes en el lenguaje que rodea al recomendador.

---

<a id="13"></a>
## 13 · Patrones de aplicación con LangChain
`notebooks/04_applications/13_modern_llm_applications.ipynb` · `langchain-core`, `pydantic`

**Qué construyes.** Seis patrones de producción, cada uno con su métrica: extracción estructurada, clasificación zero-shot y few-shot, resumen map-reduce, caché (exacta y semántica), defensa contra *prompt injection* y procesamiento por lotes con cuenta de tokens y coste. Todo a temperatura 0 y con conjuntos pequeños etiquetados a mano (de 8 a 20 elementos).

**Herramientas.** Pydantic + `with_structured_output` (`json_schema` y `function_calling`); `FewShotChatMessagePromptTemplate` y una etiqueta `Literal`; map-reduce en LCEL con `RecursiveCharacterTextSplitter` y `.batch`; `set_llm_cache(InMemoryCache())` y una caché semántica escrita sobre `BaseCache` con `nomic-embed-text`; un clasificador LLM con veredicto estructurado; `.batch(max_concurrency=n)` y `get_usage_metadata_callback()`.

**Conceptos clave.** Con `json_schema`, `ChatOllama` pasa el esquema solo al decodificador, así que el modelo nunca *lee* las descripciones de los campos; los ejemplos few-shot son un cambio de prompt que hay que evaluar; las entradas largas se truncan en silencio (mira `usage_metadata["input_tokens"]`); una caché sirve las respuestas equivocadas igual de rápido que las buenas; la inyección se gestiona por capas, no se resuelve; proyectar coste y duración antes de escalar.

**Resultados**
- **Extracción (8 registros)**: todas las variantes devuelven 8/8 objetos válidos, pero todos los campos están bien en el 62% con `json_schema`, el 88% con `json_schema` + las reglas en el mensaje de sistema, y el 88% con `function_calling`.
- **Clasificación (20 tickets)**: zero-shot 100%, few-shot 100% en la ejecución publicada (95% en una ejecución anterior, con un ticket de envíos clasificado como técnico). Los ejemplos no aportaron nada.
- **Resumen**: el documento tiene ~4135 tokens, pero la llamada única ("stuff") solo procesó 2050 (truncado sin ningún error): 9 de 13 temas (perdió el principio: tokenización, atención, MoE) y 779 palabras en lugar de un párrafo de 150; 6 de 13 en otra ejecución. Map-reduce (6+1 llamadas): 11 de 13 temas, 210 palabras, compresión 12×, 30,4 s frente a 15,7 s.
- **Caché exacta**: 0,54 s la primera llamada, 1–2 ms las siguientes. También cachea la respuesta *equivocada* del modelo de 3B sobre la KV-cache. **Trampa encontrada al medir:** el `llm_string` de `ChatOllama` (la parte del modelo en la clave de caché) no incluye ni el nombre del modelo ni la temperatura, así que una llamada a T=0,5 recibió la respuesta de T=0 (3 ms, un acierto).
- **Caché semántica**: las paráfrasis tienen coseno 0,73–0,95 y los casi-iguales 0,56–0,93 (se solapan). Para no tener falsos aciertos el umbral debe ser 0,94, y entonces solo acierta el 20% de las paráfrasis (3 aciertos en 25 peticiones).
- **Prompt injection (10 documentos)**: éxito del ataque 90% con el prompt ingenuo, **100% al delimitar** (con este modelo no ayudó nada) y 40% delimitando + un recordatorio después del documento ("sándwich"). El guardián LLM detecta 10/10 ataques con 0/10 falsos positivos en su propio conjunto y bloquea 8 de los 10 documentos inyectados. Combinados: ≈ 0,2 × 0,4 = 8% de éxito esperado, no cero.
- **Lotes (40 registros)**: 8,5 s con concurrencia 1 y 8,3 s con 4 (~210 ms por registro; un único Ollama local apenas gana). ~131 tokens de entrada y 9 de salida por registro → ~25 $ por millón de registros a 0,15 $/M de entrada y 0,60 $/M de salida; en local, un millón de registros tardaría 57 horas.

---

<a id="14"></a>
## 14 · IA agéntica con LangGraph
`notebooks/05_agents_and_mcp/14_agentic_ai.ipynb` · `langgraph`, `langchain`

**Qué construyes.** Un `StateGraph` de triaje de tickets, un agente ReAct, límites con motivo de parada explícito, una puerta de aprobación para reembolsos, memoria a corto y largo plazo, plan-and-execute como grafo explícito, reflexión guiada por tests unitarios, un enrutador con subgrafos especialistas y una evaluación de ReAct frente a plan-and-execute.

**Herramientas.** `StateGraph`, `START`/`END`, `add_conditional_edges`, reducers (`Annotated[..., operator.add]`, `add_messages`), `stream(stream_mode="updates")`, `draw_mermaid()`; `create_agent`; `recursion_limit`, `ModelCallLimitMiddleware`, `ToolCallLimitMiddleware` y un `AgentMiddleware` propio (`WallClockLimit`); `interrupt` + `Command(resume=...)` y `HumanInTheLoopMiddleware`; `InMemorySaver` + `thread_id`; `InMemoryStore` con búsqueda semántica, `ToolRuntime` y `@dynamic_prompt`; `Command(goto=...)` y grafos compilados como subgrafos.

**Conceptos clave.** Workflow frente a agente (usa el diseño más simple que funcione); todo agente necesita límites y un **motivo de parada**; la aprobación vive en tu código, no en el prompt; memoria = checkpointer (por hilo) + store (entre hilos), y la memoria larga es recuperación, así que se mide su recall; la reflexión necesita un verificador fiable; el enrutado es clasificación; evaluar el *camino* (precisión y recall de tools), no solo la respuesta.

**Resultados**
- **Límites**: las cuatro paradas se disparan con su motivo (`recursion_limit`, `model_call_limit`, `tool_call_limit`, `timeout`) frente a un modelo falso que repite sin fin.
- **Humano en el bucle**: con la primitiva, el reembolso de 20 $ se aprueba y el de 450 $ se rechaza; con `HumanInTheLoopMiddleware` y una política de "hasta 40 $", A100 (20) se aprueba y B200 (45,5) se rechaza. Solo se emitió el reembolso aprobado.
- **Memoria**: el mismo hilo recuerda el pedido (8 mensajes guardados); un hilo nuevo no. La memoria larga tiene recall@1 de **5/6**; un dato guardado en un hilo se usa en otro.
- **Reflexión con tests (8 funciones)**: 4/8 pasan al primer intento y **siguen siendo 4/8** con hasta 3 rondas de feedback. Cuando el modelo de 3B no sabe arreglar el fallo, ver el assert que falla no le ayuda.
- **Enrutador**: 67% en 12 peticiones (0,23 s por llamada). Los saludos y agradecimientos fueron a `support`, y "Where is my order B200?" también.
- **ReAct frente a plan-and-execute (10 tareas de varios pasos)**:

| Diseño | Éxito | Precisión de tools | Recall de tools | Llamadas al modelo | Tiempo por tarea |
|---|---|---|---|---|---|
| **ReAct** | **80%** | 90% | 70% | 2,1 | 1,8 s |
| plan-and-execute | 60% | 87% | 90% | 6,6 | 3,0 s |

Honesto: con este modelo y estas tareas cortas **planificar no compensó**. ReAct lo hizo mejor con un tercio de las llamadas (qwen2.5:3b lanza tools en paralelo). El plan obliga a hacer cada subtarea (más recall), pero añade pasos redundantes y cada traspaso pierde información (un estado inventado, `"shased"`, y una cuenta mal hecha en el ejemplo). Los fallos de ReAct fueron cuentas hechas "de cabeza" y una cifra de la política inventada en vez de consultada. Con 10 tareas, un acierto son 10 puntos.

---

<a id="15"></a>
## 15 · LLMOps con LangSmith: trazas, monitorización y evaluación
`notebooks/06_llmops/15_observability_llmops.ipynb` · `langsmith`, LangGraph · **LangSmith es opcional**: sin clave funcionan la aplicación y los evaluadores locales, y se saltan las celdas que necesitan el servidor

**Qué construyes.** Un pequeño asistente de soporte (un grafo RAG de LangGraph sobre `data/sample_docs/` más una tool de pedidos), tráfico sintético y, después, el trabajo de un ingeniero de guardia con el SDK de LangSmith: árboles de trazas, hilos, percentiles, SLOs, puntuaciones online, redacción de PII, muestreo, exportación y un experimento offline que compara dos versiones.

**Herramientas.** Trazado automático de LangChain/LangGraph (`LANGSMITH_TRACING=true`); `@traceable` (`run_type="llm"`, `"tool"`, `"retriever"`), `tracing_context`, `get_current_run_tree`, `langsmith_extra={"run_id": ...}`; `client.runs.query(...)` (asíncrono, con filtros como `has(tags, "traffic")`); metadato `thread_id`; `client.create_feedback`; `langsmith.anonymizer.create_anonymizer` y `Client(anonymizer=...)`; `tracing_sampling_rate`; `langsmith.evaluate` con datasets y experimentos; `InMemoryVectorStore`.

**Conceptos clave.** Proyecto, traza, run, tags/metadata, hilo, feedback, dataset → experimento; por qué p50/p95/p99 y no la media; los SLOs como contrato; heurísticas baratas en cada traza para decidir cuáles debe abrir una persona; redactar antes de subir (el modelo sigue viendo el texto original); el muestreo conserva o descarta trazas enteras; contrastar al juez LLM con evaluadores deterministas.

**Resultados**
- Un `graph.invoke` produce un árbol de 7 runs (grafo, nodos, retriever, prompt y `ChatOllama` con 115+12 tokens).
- **Tráfico**: 18 peticiones (2 a pedidos inexistentes, a propósito), 18 trazas y 112 runs. Por petición: **p50 0,22 s, p95 2,80 s, p99 4,24 s**, media 0,93 s, 11,1% de errores (la tool: 33%), 104 tokens por petición (1605 de entrada y 270 de salida en total), ~0,00002 $ por petición a precios ilustrativos (LangSmith deja `total_cost` vacío para un modelo local). Los prompts son seis veces más largos que las respuestas: el contexto es la palanca del coste.
- **SLOs**: p95 < 4 s **OK**, tasa de errores < 5% **INCUMPLIDO** (a propósito), tokens < 600 **OK**. El SLO de latencia depende de lo ocupada que esté la GPU (el notebook vio p95 entre ~3 y ~10 s según la ejecución).
- **Groundedness online**: media 0,81 sobre 16 respuestas. Las tres más bajas son respuestas de la tool de pedidos: A101 y A103 son **fallos reales** (el modelo tenía el estado y dijo que no tenía información de seguimiento); A100 (0,43) es una falsa alarma.
- **PII**: email, teléfono, tarjeta y clave de API quedan sustituidos en los 8 runs de la traza; los nombres como "Ana" pasan.
- **Muestreo** a 0,25: se conservan 13 de 40 trazas (varía entre ejecuciones).
- **Experimento offline (12 ejemplos)**:

| Versión | exact_match | contains_fact | correctness (juez LLM) |
|---|---|---|---|
| closed_book | 0,00 | 0,17 | 0,17 |
| **rag** | 0,58 | **1,00** | **0,83** |

El juez coincide con `contains_fact` en 20 de 24 respuestas: una vez detecta una afirmación falsa que la búsqueda de palabras clave no vio, y dos veces se equivoca (rechaza una respuesta correcta por añadir la zona horaria y acepta otra que no da ninguna cifra). `exact_match` mide la redacción, no la corrección.

---

<a id="16"></a>
## 16 · MCP con el SDK oficial
`notebooks/05_agents_and_mcp/16_mcp_model_context_protocol.ipynb` · `mcp` (SDK 1.x), `langchain-mcp-adapters` · script `scripts/mcp_server.py`

**Qué construyes.** Un servidor MCP con FastMCP (tools, resources, plantillas y prompts), una sesión cliente en memoria, los mensajes JSON-RPC por stdio vistos "en el cable", los dos tipos de error, el servidor de la base de conocimiento (`scripts/mcp_server.py`) por stdio, un agente de LangChain que usa sus tools, controles de seguridad y una evaluación del servidor.

**Herramientas.** `mcp.server.fastmcp.FastMCP` con `@server.tool(annotations=ToolAnnotations(...))`, `@server.resource` (URIs fijas y con plantilla) y `@server.prompt`; `create_connected_server_and_client_session` (en memoria, para tests); `ClientSession`, `stdio_client`, `StdioServerParameters`, `McpError`; `MultiServerMCPClient(tool_name_prefix=True)`, `client.session(...)` + `load_mcp_tools(session)`; `create_agent`; `HumanInTheLoopMiddleware` con `InMemorySaver` y `Command`; interceptores de llamadas a tools que analizan y envuelven los resultados.

**Conceptos clave.** Host, cliente y servidor; tres primitivas con tres responsables distintos (el modelo elige tools, la aplicación carga resources, el usuario elige prompts); JSON-RPC 2.0, *handshake* (`initialize` + `notifications/initialized`), peticiones frente a notificaciones; errores de tool (`isError`, vuelven al modelo) frente a errores de protocolo (`error` de JSON-RPC, fallos del cliente); stdio: nunca escribir en stdout; las anotaciones son *pistas*; allowlist, *namespacing*, confirmación de todo lo que no sea de solo lectura, inyección en resultados y *tool poisoning*.

**Resultados**
- **14/14 checks de conformidad** (handshake, listados, llamadas válidas e inválidas, resources, plantilla, prompt, ping).
- Detalles de implementación que el notebook saca a la luz: FastMCP 1.x devuelve una tool desconocida como resultado con `isError` (la especificación lo describe como error de protocolo), usa `-32602` para un método desconocido (JSON-RPC define `-32601`) y el código `0` para un resource que no existe.
- `search_docs` (BM25) pone el documento correcto el primero en **6/6** preguntas (top-3: 6/6); es un conjunto fácil.
- **Agente**: el dato clave está en 6/6 respuestas y cita el fichero correcto en **5/6** (la respuesta sobre envío exprés no lleva cita); mediana de 0,8 s por pregunta.
- **Latencia de un `tools/call`** (p50): 1,4 ms en memoria, 2,7 ms por stdio, 3,6 ms con el adaptador de LangChain y una sesión, y **~2,3 s con la tool por defecto, que abre una sesión por llamada**, porque arranca un subproceso y repite el handshake cada vez. Mantén la sesión abierta.
- **Seguridad**: la descripción envenenada (`<IMPORTANT>`) se detecta y esa tool no llega al modelo; el resultado inyectado llega dentro de un sobre `<untrusted_tool_result>` con un aviso; el revisor rechaza `docs_create_ticket` y la tool destructiva no se ejecuta.

---

## Y después

- **Refuerza lo aprendido:** cada notebook termina con *Takeaways*. Los scripts empaquetan los mismos pipelines: `scripts/finetune.py` (SFT, DPO, GRPO), `scripts/rag_demo.py` (RAG y su evaluación) y `scripts/mcp_server.py`.
- **Cambia los datos:** pasa tus propios documentos a `scripts/rag_demo.py --docs` o a `scripts/mcp_server.py --docs`, y tu propio texto al notebook 00.
- **Conecta el servidor a un host real:** `claude mcp add llm-edu-docs -- python /RUTA/ABSOLUTA/scripts/mcp_server.py`, o arráncalo con `--http`.
- Consulta el [glosario](glosario.md) para cualquier término, los [fundamentos teóricos](fundamentos.md) para la teoría con sus ecuaciones y la [referencia de módulos](referencia_modulos.md) para la API.
