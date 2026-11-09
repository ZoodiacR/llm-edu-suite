# Fundamentos teóricos

🇬🇧 [English version](../en/foundations.md) · [← Volver al índice](README.md)

> La teoría detrás de cada técnica y cada patrón del curso: qué problema resuelve, cuál es la idea (con sus ecuaciones), por qué funciona, con qué compite, por qué es lo que usa hoy la industria y qué sustituyó, y cómo falla. Está pensada para leerla antes de cada notebook o en paralelo. Las definiciones cortas están en el [glosario](glosario.md); aquí no se repiten, se enlazan.

**Cómo leer esta página.** Sigue el mismo orden que el [mapa por tópicos](README.md#mapa-por-tópicos). Cada apartado termina con el notebook que lo implementa y una línea **En el curso** con lo que se mide allí; las cifras que aparecen son las que publica el propio notebook, medidas con modelos pequeños (SmolLM2, `qwen2.5:3b`) y sets de evaluación pequeños, así que valen como señales grandes, no como benchmarks. Cuando un punto está en discusión en la literatura, se dice.

## Índice

1. [El modelo por dentro](#1-el-modelo-por-dentro)
   - [1.1 Tokenización](#11-tokenización) · [1.2 El bloque transformer](#12-el-bloque-transformer-y-el-flujo-residual) · [1.3 Atención, RoPE y normalización](#13-atención-rope-y-normalización) · [1.4 Atención eficiente](#14-atención-eficiente-para-contextos-largos) · [1.5 FFN y Mixture of Experts](#15-ffn-y-mixture-of-experts) · [1.6 Encoders y encoder-decoders](#16-encoders-y-encoder-decoders)
2. [Pre-training e inferencia](#2-pre-training-e-inferencia)
   - [2.1 El objetivo](#21-el-objetivo-predecir-el-siguiente-token) · [2.2 Optimizador y schedule](#22-optimizador-y-plan-de-learning-rate) · [2.3 Leyes de escalado](#23-leyes-de-escalado) · [2.4 Decodificación](#24-estrategias-de-decodificación) · [2.5 KV-cache](#25-prefill-decode-y-kv-cache) · [2.6 Batching y serving](#26-batching-y-serving) · [2.7 Speculative decoding](#27-speculative-decoding) · [2.8 Cuantización](#28-cuantización)
3. [Post-training](#3-post-training)
   - [3.1 SFT](#31-supervised-fine-tuning) · [3.2 LoRA](#32-lora-y-fine-tuning-eficiente) · [3.3 Destilación](#33-destilación) · [3.4 RLHF, DPO y GRPO](#34-de-rlhf-a-dpo-y-grpo)
4. [Aplicaciones](#4-aplicaciones)
   - [4.1 RAG](#41-retrieval-augmented-generation) · [4.2 Métricas de recuperación](#42-métricas-de-recuperación) · [4.3 RAGAS y LLM-as-judge](#43-ragas-y-llm-as-judge) · [4.4 Patrones de aplicación](#44-patrones-de-aplicación) · [4.5 Recomendación](#45-sistemas-de-recomendación)
5. [Agentes y MCP](#5-agentes-y-mcp)
   - [5.1 Tool calling](#51-tool-calling) · [5.2 Decodificación restringida](#52-decodificación-restringida-y-salida-estructurada) · [5.3 ReAct, plan-and-execute y reflexión](#53-react-plan-and-execute-y-reflexión) · [5.4 Grafos de estado](#54-langgraph-agentes-como-máquinas-de-estado) · [5.5 Memoria](#55-memoria) · [5.6 Límites y human-in-the-loop](#56-límites-y-human-in-the-loop) · [5.7 MCP](#57-model-context-protocol)
6. [LLMOps](#6-llmops)
   - [6.1 Trazas y spans](#61-trazas-spans-y-opentelemetry-genai) · [6.2 Evaluación offline y online](#62-evaluación-offline-y-online) · [6.3 Percentiles y SLOs](#63-percentiles-y-slos) · [6.4 PII y muestreo](#64-pii-y-muestreo-de-trazas) · [6.5 Estadística de sets pequeños](#65-estadística-de-sets-de-evaluación-pequeños)

---

## 1. El modelo por dentro

Términos de esta parte: [glosario, Tokenización y arquitectura](glosario.md#tokenización-y-arquitectura).

Un modelo de lenguaje causal es una distribución de probabilidad sobre secuencias de tokens, factorizada con la regla de la cadena:

$$p(x_1, \dots, x_T) = \prod_{t=1}^{T} p(x_t \mid x_{<t})$$

La red nunca modela frases enteras: dado un prefijo, devuelve una distribución sobre el siguiente token. Todo lo que hay en esta parte (tokenizador, atención, FFN) existe para calcular bien ese factor.

```
texto ─► tokenizador ─► ids ─► embedding ─► L bloques ─► norm ─► lm_head ─► logits (T × V) ─► softmax
```

> **Notebook de entrada:** [00 · Un mini LLM completo](../../notebooks/00_start_here/00_mini_llm_end_to_end.ipynb) · **En el curso:** el modelo completo en un solo notebook; la pérdida inicial debe estar cerca de $\ln V$ y, con el corpus sintético, la pérdida irreducible se calcula exacta ($4 \ln 8 = 8.32$ nats por frase).

### 1.1 Tokenización

**Problema.** El transformer no lee texto: su primera capa es una tabla de embeddings $E \in \mathbb{R}^{V \times C}$ y la última produce $V$ logits. Hay que fijar, antes de entrenar, una función `encode` de texto a enteros y su inversa. Esa decisión fija la longitud de las secuencias (y por tanto el coste de la atención, que es $O(n^2)$), cuántos parámetros hay en el embedding y en la cabeza de salida, y qué idiomas o dominios salen caros.

**Idea central.** Dos fuerzas tiran en sentidos opuestos. Las secuencias cortas abaratan entrenamiento e inferencia; un vocabulario pequeño y bien entrenado da embeddings bien estimados. Las palabras completas caen en la cola de Zipf (muchas palabras vistas pocas veces) y producen `<UNK>`; los bytes generan secuencias 4-5 veces más largas. Las sub-palabras son el compromiso. **BPE** (Sennrich et al., 2016) parte de símbolos base y repite, $M$ veces, la fusión del par adyacente más frecuente:

$$(a^*, b^*) = \arg\max_{(a,b)} \; c(a,b), \qquad c(a,b) = \left|\lbrace i : s_i = a,\ s_{i+1} = b \rbrace\right|$$

Cada fusión acorta el corpus exactamente en $c(a^*, b^*)$ símbolos: BPE es un compresor voraz, no óptimo, pero barato y determinista. Para codificar texto nuevo se reaplican las reglas por orden de aprendizaje.

**Por qué byte-level BPE es el estándar.** GPT-2 cambió el alfabeto base a los 256 valores de byte de UTF-8: cualquier cadena se puede codificar y la decodificación es exacta, sin `<UNK>`. Las fusiones aprenden solas qué secuencias de bytes (acentos, CJK, emoji) merecen un token. El precio es pequeño: un carácter raro cuesta 2-4 tokens y, en streaming, hay que acumular secuencias UTF-8 incompletas. tiktoken, Llama 3 y Qwen usan byte-level BPE; SentencePiece (BPE o Unigram sobre texto crudo, con *byte fallback*) sigue presente en T5, Llama 1-2 y Gemma.

| Método | Construcción del vocabulario | Segmentación | Lo usan |
|---|---|---|---|
| BPE | de abajo arriba, fusiona el par más frecuente | reaplica fusiones por rango | GPT, Llama, Mistral, Qwen |
| WordPiece | de abajo arriba, fusiona el par que más sube la verosimilitud | coincidencia más larga desde la izquierda | BERT |
| Unigram LM | de arriba abajo, poda piezas de un conjunto grande | segmentación más probable (Viterbi) | T5, ALBERT |

**Detalles que importan.** Un *pre-tokenizador* (regex) separa letras, dígitos y espacios antes de BPE: acelera el entrenamiento (se cuenta cada palabra única una vez) y evita tokens accidentales como `"dog."`. Los tokens especiales (BOS, EOS, PAD, marcadores de rol del chat template) reciben ids que ninguna fusión puede producir, y un tokenizador de producción no debe convertir en token especial la cadena `<|eos|>` escrita por un usuario. Vocabulario más grande comprime más pero cuesta $V \cdot C$ parámetros por matriz; por eso la tendencia (Llama 3 con 128k, Gemma con 256k) va ligada a cubrir más idiomas y código.

**Modos de fallo.** Injusticia entre idiomas (un tokenizador entrenado en inglés cobra más tokens por el mismo contenido en otros idiomas; Petrov et al., 2023); ceguera a nivel de carácter (contar letras de `"strawberry"`); números segmentados de forma inconsistente; tokens infra-entrenados que provocan conductas raras (`SolidGoldMagikarp`); un prompt que termina en espacio. La relación entre compresión y calidad final está en debate: la única prueba definitiva es entrenar modelos con cada tokenizador. Los modelos sin tokenizador (ByT5, MEGABYTE, Byte Latent Transformer) son investigación activa y todavía no han desplazado a BPE.

**Papers clave.** Sennrich, Haddow y Birch (2016); Radford et al. (2019, GPT-2); Kudo (2018); Kudo y Richardson (2018, SentencePiece); Rust et al. (2021); Petrov et al. (2023).

> **Notebook:** [01 · Tokenización](../../notebooks/01_architecture/01_tokenization.ipynb) · **En el curso:** BPE en Python puro con round-trip, compresión y fertility; la compresión en el dominio sube rápido con el vocabulario y se satura, y el texto fuera de dominio apenas mejora.

### 1.2 El bloque transformer y el flujo residual

**Problema.** Antes del Transformer, los modelos de lenguaje eran n-gramas y luego RNN/LSTM, que leen token a token: el entrenamiento es secuencial y el gradiente conecta mal tokens lejanos.

**Idea central.** El Transformer (Vaswani et al., 2017) sustituye la recurrencia por atención, y la variante *decoder-only* con máscara causal permite calcular las $T$ predicciones de una secuencia en una sola pasada (teacher forcing). Un modelo moderno apila $L$ bloques pre-norm idénticos:

$$h = x + \mathrm{Attn}\big(\mathrm{Norm}(x)\big), \qquad y = h + \mathrm{FFN}\big(\mathrm{Norm}(h)\big)$$

La **atención** es el único sitio donde las posiciones intercambian información (entre tokens); la **FFN** transforma cada posición por separado (dentro del token). El **flujo residual** $x$ es la suma acumulada donde cada sub-capa lee y escribe.

**Lo que usa la industria hoy y lo que sustituyó.** La receta consolidada por Llama (Touvron et al., 2023) cambió varias piezas de GPT-2:

| Componente | GPT-2 (2019) | Receta actual | Motivo |
|---|---|---|---|
| Posición | embeddings absolutos aprendidos | RoPE | posiciones relativas, sin tabla de longitud máxima |
| Normalización | LayerNorm post/pre | RMSNorm pre-norm | más barata, igual de estable |
| FFN | MLP con GELU | SwiGLU | mejor perplejidad al mismo número de parámetros |
| Biases | en las capas lineales | casi eliminados | más simple, sin beneficio medible a escala |
| Cabeza de salida | — | a menudo atada al embedding | ahorra $V \cdot C$ parámetros en modelos pequeños |

El decoder-only ganó a encoder-decoder y encoder-only para modelos generales porque un único objetivo (continuar texto) cubre cualquier tarea expresable como texto y mejora de forma predecible con la escala.

**Hardware.** En entrenamiento y *prefill* se procesan $T$ posiciones a la vez: multiplicaciones de matrices grandes, el límite es el cómputo. En *decode* se procesa un token por paso: hay que leer todos los pesos y toda la KV-cache para producir un token, el límite es el ancho de banda de memoria. Buena parte de la evolución de la arquitectura (GQA, MLA) ataca este segundo régimen.

> **Notebook:** [00 · Un mini LLM completo](../../notebooks/00_start_here/00_mini_llm_end_to_end.ipynb) y [02 · Atención, RoPE y normalización](../../notebooks/01_architecture/02_attention_rope_norm.ipynb) · **En el curso:** el bloque completo se escribe a mano y se comprueba que el modelo aprende la gramática del corpus solo con predicción del siguiente token.

### 1.3 Atención, RoPE y normalización

#### Atención escalada y multi-cabeza

**Idea.** Cada token proyecta su estado en una consulta $q = W_Q x$, una clave $k = W_K x$ y un valor $v = W_V x$:

$$\mathrm{Attn}(Q,K,V) = \mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{D_h}} + M\right)V$$

Es un diccionario "blando": en vez de devolver el valor de la clave idéntica, devuelve una mezcla de todos los valores ponderada por la coincidencia, y el softmax hace que la mezcla sea diferenciable. La máscara causal $M$ pone $-\infty$ por encima de la diagonal, de modo que el futuro recibe peso exactamente cero.

**Intuición de $\sqrt{D_h}$.** Si las componentes de $q$ y $k$ tienen media 0 y varianza 1, $q^\top k$ tiene varianza $D_h$. Con $D_h = 128$ los logits tendrían desviación típica ~11, el softmax se saturaría en una sola clave y el gradiente hacia las demás sería casi nulo. Dividir por $\sqrt{D_h}$ devuelve la varianza a 1 al inicializar.

**Varias cabezas.** Un softmax solo produce una ponderación por consulta; con $H$ cabezas de tamaño $D_h = C/H$ una capa sigue varias relaciones a la vez (token previo, sujeto, paréntesis que cierra). El coste es $O(T^2 C)$ por capa. **FlashAttention** (Dao et al., 2022-2024) calcula *exactamente* lo mismo sin materializar la matriz $T \times T$, con tiles y un softmax "online": la memoria baja a $O(T)$ y la velocidad sube porque se evita tráfico a memoria lenta, pero los FLOPs siguen siendo cuadráticos.

#### Normalización: de LayerNorm post-norm a RMSNorm pre-norm

Sin normalización, el flujo residual crece capa a capa y la escala que ve cada sub-capa cambia durante el entrenamiento. **RMSNorm** (Zhang y Sennrich, 2019) solo reescala:

$$\mathrm{RMSNorm}(x) = \gamma \odot \frac{x}{\sqrt{\tfrac{1}{C}\sum_i x_i^2 + \epsilon}}$$

La hipótesis de sus autores (lo importante es reescalar, no re-centrar) se ha confirmado en la práctica a escala; ahorra una reducción en una operación limitada por memoria. Más importante que *qué* norma es *dónde*: con **post-norm** (Transformer original, BERT) los gradientes cerca de la salida son grandes al inicio y hacía falta un warmup cuidadoso (Xiong et al., 2020); **pre-norm** deja un camino identidad sin normalizar desde la entrada hasta la salida y entrena estable con menos ajuste. Hay variantes en estudio: normas antes y después de cada sub-capa (Gemma 2) y **QK-norm**, que normaliza $q$ y $k$ para que los logits de atención no crezcan (OLMo 2, Gemma 3). Se critica que, con pre-norm, las capas tardías aportan relativamente menos; qué variante es mejor sigue abierto.

#### RoPE: posición relativa por rotación

**Problema.** Sin máscara, la atención es equivariante a permutaciones: "perro muerde a hombre" y "hombre muerde a perro" serían iguales.

**Idea.** RoPE (Su et al., 2021) divide cada cabeza en $D_h/2$ pares y rota la consulta de la posición $m$ un ángulo $m\theta_i$ y la clave de la posición $n$ un ángulo $n\theta_i$, con $\theta_i = \text{base}^{-2i/D_h}$. Como las rotaciones componen sumando ángulos:

$$\big(R(m\theta)q\big)^\top \big(R(n\theta)k\big) = q^\top R\big((n-m)\theta\big)\,k$$

La puntuación depende solo de la distancia. Las rotaciones no cambian normas; los primeros pares giran rápido (resolución local) y los últimos casi no se mueven (canales casi sin posición).

| Enfoque | Relativo | Parámetros extra | Inconveniente |
|---|---|---|---|
| Sinusoidal absoluto (2017) | indirectamente | ninguno | mezcla posición y contenido, extrapola mal |
| Absoluto aprendido (GPT-2, BERT) | no | $T_{\max} \times C$ | no puede pasar de $T_{\max}$ |
| Sesgo relativo (T5) | sí | pocos | toca la matriz $T \times T$, complica los kernels fusionados |
| ALiBi | sí | ninguno | sesgo de recencia fuerte |
| RoPE | sí | ninguno | necesita una regla de escalado para alargar contexto |

RoPE se impuso porque es relativa, no añade parámetros y actúa sobre $q$ y $k$ *antes* de la atención, así que FlashAttention funciona sin cambios y una clave se rota una vez y se guarda en caché para siempre. **Extensión de contexto:** más allá de la longitud de entrenamiento, los pares lentos alcanzan ángulos nunca vistos y la perplejidad se dispara. Position Interpolation, el escalado "NTK-aware", YaRN y el escalado de Llama 3.1 modifican la tabla de ángulos: estiran los pares de baja frecuencia y dejan los de alta frecuencia (que ya dieron muchas vueltas en entrenamiento). Que la codificación posicional sea siempre necesaria en decoders causales está en discusión (NoPE, Kazemnejad et al., 2023).

#### KV-cache y la familia MHA → GQA → MLA

En generación, las claves y valores de los tokens pasados no cambian, así que se guardan. Por secuencia:

$$\text{bytes} = 2 \times L \times H_{kv} \times D_h \times T \times b$$

Cada paso de decode lee toda la caché, así que su tamaño fija el batch máximo (memoria) y la latencia por token con contexto largo (ancho de banda).

| Variante | Caché por token y capa | Calidad (según lo publicado) | Complejidad |
|---|---|---|---|
| MHA | $2 H D_h$ | referencia | simple |
| MQA (Shazeer, 2019) | $2 D_h$ | menor y menos estable | simple |
| GQA (Ainslie et al., 2023) | $2 H_{kv} D_h$ | cercana a MHA | simple, compatible con todos los kernels |
| MLA (DeepSeek-V2, 2024) | $d_c + d^R$ (512 + 64) | igual o mejor que MHA en los ablations de DeepSeek | proyecciones extra, absorción de pesos, RoPE desacoplado |

GQA es el estándar de los modelos densos abiertos porque reduce la caché 4-8 veces con un cambio mínimo; además un checkpoint MHA se puede convertir promediando cabezas y "uptraining". MLA guarda un único vector latente $c_t = W^{DKV} x_t$ por token y reconstruye claves y valores; la **absorción** pliega $W^{UK}$ en la consulta para atender directamente sobre los latentes. RoPE rompería esa absorción, de ahí el **RoPE desacoplado** (una parte pequeña de la clave, compartida entre cabezas, que lleva la posición). Si la ventaja de calidad de MLA sobre GQA se sostiene fuera del setup de DeepSeek no está zanjado.

**Modos de fallo.** Un `start_pos` desplazado en uno (la calidad cae sin ningún error), rotar dos veces las claves en caché, olvidar la máscara causal en un prefill de varios tokens, cachear en MLA las claves expandidas en vez del latente. Y los *attention sinks* (Xiao et al., 2023): los modelos concentran mucha atención en los primeros tokens, así que expulsarlos de la caché rompe el modelo.

**Papers clave.** Vaswani et al. (2017); Ba et al. (2016); Zhang y Sennrich (2019); Xiong et al. (2020); Su et al. (2021); Shazeer (2019); Ainslie et al. (2023); DeepSeek-AI (2024, DeepSeek-V2); Dao et al. (2022).

**Dos convenciones de RoPE.** Los pares que se rotan pueden ser dimensiones contiguas $(2i, 2i+1)$ (el Llama original de Meta y este curso) o separadas por media cabeza $(i, i + d_h/2)$ (*rotate_half*, de GPT-NeoX y Hugging Face). Como el producto escalar no cambia si se permutan igual las dimensiones de los dos vectores, $q^\top k = (Pq)^\top (Pk)$, basta con permutar las filas de $W_q$ y $W_k$ dentro de cada cabeza para pasar de una a otra. Olvidarlo no da ningún error: el modelo funciona, pero peor. El notebook 06 lo mide al cargar SmolLM2 en nuestro código.

> **Notebook:** [02 · Atención, RoPE y normalización](../../notebooks/01_architecture/02_attention_rope_norm.ipynb) · **En el curso:** se verifica numéricamente que RoPE depende solo de $n-m$, que la atención a mano coincide con el kernel fusionado y que decode incremental con caché iguala a la pasada completa; con la forma de Llama-2-70B, MHA necesitaría 2.6 MB por token (10 GB por secuencia con 4k de contexto) y GQA con 8 cabezas KV lo divide entre 8.

### 1.4 Atención eficiente para contextos largos

**Problema.** Por token y capa, proyecciones y FFN cuestan del orden de $24C^2$ FLOPs, y la mezcla de atención unos $4TC$. La atención domina cuando $T$ supera en un orden de magnitud a $C$ (decenas de miles de tokens para $C = 4096$). En inferencia hay un segundo muro: cada token generado lee una KV-cache que crece con $T$. FlashAttention resuelve la memoria $O(T^2)$, no el cómputo $O(T^2)$ ni el tamaño de la caché.

**Familias de soluciones.**

| Familia | Idea | Ejemplos | Lo que se pierde |
|---|---|---|---|
| Patrón fijo disperso | ventana local, posiciones con paso, tokens globales | Sparse Transformer, Longformer, BigBird, Mistral 7B | acceso directo fuera del patrón |
| Disperso por contenido | un puntuador barato elige qué bloques o tokens atender | Reformer, Routing Transformer, NSA, MoBA, DSA | algo de recall si el selector falla; kernels complejos |
| Bajo rango / kernel | aproximar el softmax con matrices delgadas | Linformer, Performer, Linear Transformers | la nitidez del softmax |
| Recurrente / state-space | resumir el pasado en un estado de tamaño fijo | RetNet, Mamba, Mamba-2, Gated DeltaNet | recuerdo exacto de tokens arbitrarios |
| Híbridos | mayoría de capas baratas y algunas densas | Jamba, Griffin, Gemma 2/3, Qwen3-Next | las capas densas siguen pagando $O(T^2)$ |

**Sliding window.** La consulta $i$ atiende a $j$ si $i - w < j \le i$: coste $O(TwC)$ y caché $O(w)$ en forma de buffer circular. Tras $L$ capas el campo receptivo es $L(w-1)$, pero la información lejana llega *retransmitida* y difuminada; el campo receptivo dice qué es alcanzable, no qué se conserva.

**Atención lineal.** Si la similitud factoriza, $\mathrm{sim}(q,k) = \phi(q)^\top\phi(k)$, la asociatividad del producto de matrices evita la matriz $T \times T$ y el modelo se vuelve una RNN con estado fijo:

$$o_t = \frac{\phi(q_t)^\top S_t}{\phi(q_t)^\top z_t}, \qquad S_t = S_{t-1} + \phi(k_t)\,v_t^\top$$

$S_t$ es una memoria asociativa lineal: su capacidad está acotada (unas $D_h$ asociaciones antes de interferir) y la versión aditiva nunca olvida. Las capas recurrentes modernas atacan exactamente eso: el *gating* añade olvido ($S_t = \alpha_t S_{t-1} + k_t v_t^\top$) y la *regla delta* sobrescribe el valor guardado bajo una clave ($S_t = S_{t-1} + \beta_t k_t (v_t - S_{t-1}^\top k_t)^\top$), base de Gated DeltaNet en Qwen3-Next.

**Por qué la atención densa es difícil de sustituir.** Puede recuperar cualquier token pasado exactamente, en un paso, por contenido. Copiar un nombre del prompt, seguir una variable o responder desde un documento recuperado dependen de eso; el hueco de calidad entre atención y alternativas se explica en gran parte por el *associative recall* (Arora et al., 2023) y los modelos de estado fijo no pueden copiar cadenas largas como la atención (Jelassi et al., 2024). El mecanismo que lo implementa son las **induction heads** (Olsson et al., 2022): una cabeza de token previo en una capa temprana más una cabeza de inducción en una capa posterior que busca "quién fue precedido por A" y copia lo que siguió.

**Lo que sobrevivió en producción.** Entre 2019 y 2021 se propusieron docenas de "efficient transformers"; pocos llegaron a los grandes decoders, porque FlashAttention hizo la atención densa suficientemente rápida y muchas aproximaciones perdían justo en recuperación en contexto. Hoy se usan ventanas locales, dispersión aprendida por contenido y recurrencias lineales con compuertas, casi siempre **mezcladas con algunas capas de atención completa**. La proporción del híbrido es el gran parámetro de diseño, y cuál es la mejor es una pregunta empírica abierta.

**Modos de fallo.** Recuerdo más allá de la ventana que se degrada en silencio (por eso existen needle-in-a-haystack y RULER); expulsar los primeros tokens de la caché sin conservar los *attention sinks*; un selector disperso que no elige el bloque relevante.

> **Notebook:** [03 · Atención eficiente](../../notebooks/01_architecture/03_efficient_attention.ipynb) · **En el curso:** una tarea de copia a ~42 tokens de distancia muestra la transición de fase de las induction heads (a 300 pasos el modelo denso no copia casi nada, a 1500 casi todo), que SWA con $w=16$ y la atención lineal no copian, y que un híbrido con una capa global de cuatro recupera parte de la capacidad.

### 1.5 FFN y Mixture of Experts

**La FFN.** Con la FFN original de $d_{ff} = 4C$, la atención tiene unos $4C^2$ parámetros por capa y la FFN $8C^2$: unos dos tercios de los parámetros no-embedding viven en la FFN. Una lectura útil es la de **memoria clave-valor** (Geva et al., 2021):

$$\mathrm{FFN}(x) = \sum_{i=1}^{d_{ff}} \sigma(k_i^\top x)\, v_i$$

Cada fila de $W_1$ es un patrón que se compara con el estado y cada columna de $W_2$ un vector que se escribe en el flujo residual. "El conocimiento vive en la FFN" es una buena aproximación de trabajo (ROME localiza y edita hechos en MLPs intermedios), no una ley: otros estudios encuentran hechos repartidos.

**SwiGLU.** El estándar actual añade una compuerta:

$$\mathrm{FFN}(x) = W_{down}\big(\mathrm{SiLU}(W_{gate}x) \odot W_{up}x\big)$$

Cada unidad oculta es el producto de dos rasgos lineales distintos, una interacción multiplicativa que una capa lineal más no-linealidad puntual no expresa directamente, y la rama lineal no se satura. La evidencia es empírica y consistente (Shazeer, 2020, que atribuye el éxito literalmente a la "benevolencia divina"). Con tres matrices, $d_{ff} = \tfrac{8}{3}C$ iguala el presupuesto de parámetros; Llama 3 8B lo agranda a propósito ($d_{ff} = 14336 = 3.5C$).

**MoE: el problema que resuelve.** En un modelo denso, duplicar parámetros duplica FLOPs por token. Una capa MoE sustituye la FFN por $E$ expertos y un router que elige $k \ll E$:

$$g = \mathrm{softmax}(W_r x), \qquad \mathcal{T} = \mathrm{TopK}(g, k), \qquad y = \sum_{e \in \mathcal{T}} \frac{g_e}{\sum_{j \in \mathcal{T}} g_j}\, \mathrm{FFN}_e(x)$$

El top-k no es diferenciable: el único gradiente que recibe el router llega a través de los valores de compuerta de los expertos elegidos. La economía es de tres presupuestos distintos:

| Cantidad | FFN densa | MoE con $E$ expertos, top-$k$ |
|---|---|---|
| Parámetros en memoria | $P$ | unos $E \cdot P_e$ |
| FLOPs por token | proporcional a $P$ | proporcional a $k \cdot P_e$ (parámetros activos) |
| Comunicación | nada extra | all-to-all cuando los expertos están en GPUs distintas |

MoE compra **capacidad** a cómputo fijo y la paga en **memoria** y **comunicación** (Mixtral 8x7B necesita memoria para ~47B parámetros aunque cada token use ~13B). Por eso brilla en entrenamientos grandes y serving distribuido, y mucho menos en una GPU de consumo.

**Balanceo de carga, la dificultad central.** El router tiende al "rico se hace más rico": un experto elegido algo más al principio recibe más gradiente, mejora y se elige aún más. Con capacidad fija por experto, los tokens que desbordan se descartan. Dos soluciones:

- **Pérdida auxiliar** (Shazeer 2017, simplificada en Switch): $\mathcal{L}_{aux} = \alpha E \sum_e f_e P_e$, con $f_e$ la fracción de tokens enrutados (sin gradiente) y $P_e$ la probabilidad media del router (con gradiente). Compite con la pérdida de lenguaje: $\alpha$ grande daña la calidad, pequeño no equilibra.
- **Sin pérdida auxiliar** (Wang et al., 2024; DeepSeek-V3): un sesgo por experto que solo interviene en la *selección*, $\mathcal{T} = \mathrm{TopK}(s_e + b_e, k)$, ajustado sin gradiente con $b_e \leftarrow b_e - \gamma\,\mathrm{sign}(\text{load}_e - \overline{\text{load}})$. Funciona como un precio de mercado o el término integral de un controlador. $\gamma$ debe ser pequeño (0.001 en DeepSeek-V3) o la carga oscila; DeepSeek-V3 mantiene además una pérdida de balanceo muy pequeña como red de seguridad.

**Expertos finos y compartidos** (DeepSeekMoE): partir cada experto en $m$ más pequeños y elegir $mk$ multiplica las combinaciones posibles; uno o varios expertos siempre activos absorben el conocimiento común. Los expertos no se especializan por temas humanos ("experto en biología"): en Mixtral el enrutamiento correlaciona más con sintaxis y tipo de token.

**Modos de fallo.** Colapso del router y expertos muertos; inestabilidad por logits del router que crecen (z-loss, router en fp32); descarte de tokens (MegaBlocks evita la capacidad fija); sobreajuste en fine-tuning con pocos datos (ST-MoE); throughput de serving que depende de la mezcla de tráfico.

**Papers clave.** Shazeer et al. (2017); Lepikhin et al. (2020, GShard); Fedus et al. (2021, Switch); Zoph et al. (2022, ST-MoE); Jiang et al. (2024, Mixtral); Dai et al. (2024, DeepSeekMoE); Wang et al. (2024); DeepSeek-AI (2024, V3); Shazeer (2020); Geva et al. (2021).

> **Notebook:** [04 · FFN y MoE](../../notebooks/01_architecture/04_ffn_moe.ipynb) · **En el curso:** se fuerza un router muy desequilibrado y el sesgo aux-free equilibra la carga sin gradiente, mientras que un `bias_lr` grande (0.1) oscila; la comparación densa frente a MoE a igual cómputo activo muestra una ventaja pequeña o nula en un corpus de juguete, que es lo esperable.

### 1.6 Encoders y encoder-decoders

**Problema.** Todo lo anterior describe un decoder-only, el diseño de GPT, Llama y Qwen. Pero el transformer original (Vaswani et al., 2017) tenía dos mitades, y cada una dio lugar a una familia propia. Para entender un texto que ya está completo (clasificarlo, extraer entidades, convertirlo en un embedding) no hace falta generar nada, y leer solo hacia la izquierda desperdicia la mitad del contexto. Para transformar una secuencia en otra distinta (traducir, transcribir audio) interesa leer la entrada entera una vez y escribir la salida aparte.

**La diferencia es la máscara.** Las tres familias apilan el mismo bloque (atención, FFN, residuales, normalización). Lo que cambia es a qué posiciones puede atender cada token:

| Familia | Atención | Objetivo de entrenamiento | Miembros conocidos | Uso hoy |
|---|---|---|---|---|
| encoder-only | bidireccional: cada token ve a todos | rellenar tokens ocultos | BERT, RoBERTa, DeBERTa, ModernBERT | clasificación, entidades, embeddings, rerankers, filtros de seguridad |
| decoder-only | causal: cada token ve solo el pasado | predecir el siguiente token | GPT, Llama, Qwen, SmolLM2 | generación, chat, agentes |
| encoder-decoder | encoder bidireccional + decoder causal con cross-attention | reconstruir tramos corruptos o mapear entrada a salida | T5, BART, Flan-T5, Whisper | traducción, reconocimiento de voz, reescritura estructurada |

En código, un encoder es la misma capa de atención llamada sin máscara causal (`is_causal=False`); en este repositorio, `ModelConfig(causal=False)`.

**Masked language modeling (BERT).** Un modelo bidireccional no puede entrenarse prediciendo el siguiente token: lo estaría viendo. BERT (Devlin et al., 2019) oculta un subconjunto aleatorio $\mathcal{M}$ del 15 % de las posiciones y pide recuperarlas a partir del resto:

$$\mathcal{L}_{\text{MLM}} = -\sum_{i \in \mathcal{M}} \log p_\theta\big(x_i \mid x_{\setminus \mathcal{M}}\big)$$

De las posiciones elegidas, el 80 % se sustituye por `[MASK]`, el 10 % por un token aleatorio y el 10 % se deja igual. La mezcla existe porque `[MASK]` no aparece nunca al usar el modelo después: si solo viera `[MASK]` en las posiciones que se puntúan, no aprendería representaciones útiles de los tokens reales. RoBERTa (Liu et al., 2019) mostró que BERT estaba infraentrenado: más datos, máscaras dinámicas y sin la tarea de "siguiente frase".

**El coste de MLM y ELECTRA.** Un LM causal aprende de todas las posiciones de cada secuencia; MLM, solo del 15 %. ELECTRA (Clark et al., 2020) lo resuelve con *replaced token detection*: un generador pequeño corrompe algunos tokens y el encoder (el discriminador $D$) clasifica **todas** las posiciones como originales o sustituidas,

$$\mathcal{L}_{\text{disc}} = -\sum_{t=1}^{T} \Big[\mathbb{1}(\tilde x_t = x_t)\log D(\tilde x, t) + \mathbb{1}(\tilde x_t \ne x_t)\log\big(1 - D(\tilde x, t)\big)\Big]$$

con lo que cada posición da señal y, a igual cómputo, el encoder resultante es mejor. DeBERTa-v3 usa este objetivo.

**Span corruption (T5).** T5 (Raffel et al., 2020) sustituye tramos contiguos de la entrada por centinelas (`<extra_id_0>`, `<extra_id_1>`, ...) y el decoder tiene que escribir los tramos que faltan, cada uno precedido de su centinela:

$$\mathcal{L}_{\text{span}} = -\sum_{j=1}^{|y|} \log p_\theta\big(y_j \mid y_{<j}, \tilde x\big)$$

donde $\tilde x$ es la entrada corrupta e $y$ la secuencia de centinelas y tramos. La salida es corta (solo lo que falta), así que el preentrenamiento es barato, y el formato texto a texto permite tratar cualquier tarea igual.

**Cabeza de clasificación.** Para una tarea con un conjunto fijo de etiquetas, la frase se resume en un vector: el estado del token especial inicial `[CLS]` o la media de los estados de todos los tokens (*mean pooling*, con la máscara de atención para no contar el padding). Encima va una capa lineal que se ajusta junto con todo el encoder:

$$p(y \mid x) = \mathrm{softmax}\big(W\, h_{\text{[CLS]}}(x) + b\big)$$

Es la receta de `AutoModelForSequenceClassification`. Una sola pasada, sin generación; la salida es una distribución sobre las etiquetas y nunca una respuesta fuera de ellas. Si en lugar de ajustar el encoder se congela y solo se entrena la capa lineal, se tiene un **linear probe**: mide cuánta información útil para la tarea hay ya en las representaciones.

**ModernBERT.** BERT-base (2018) fue el encoder por defecto durante seis años. ModernBERT (Warner et al., 2024) lo reconstruye con lo que aprendieron los decoders: RoPE en lugar de posiciones aprendidas, FFN GeGLU, pre-normalización, capas de atención global alternadas con capas de ventana deslizante de 128 tokens, *unpadding* (no se gasta cómputo en el padding), 8.192 tokens de contexto y preentrenamiento sobre 2 billones de tokens con código. La versión base tiene 150M de parámetros.

**Cross-attention.** En un encoder-decoder, el encoder lee la entrada $X$ una vez, de forma bidireccional, y produce los estados $H_X$. El decoder genera la salida $Y$ de forma causal y, en cada capa, añade entre la self-attention y la FFN una capa de cross-attention cuyas consultas vienen de la salida que se escribe y cuyas claves y valores vienen del encoder:

$$\mathrm{CrossAttn}(Y, X) = \mathrm{softmax}\!\Big(\frac{(Y W_Q)(H_X W_K)^\top}{\sqrt{d_h}}\Big) H_X W_V$$

No lleva máscara causal: cada token de salida puede leer toda la entrada. Las claves y valores de $H_X$ se calculan una sola vez por entrada y se reutilizan en todos los pasos de decodificación. Flan-T5 (Chung et al., 2022) es T5 con instruction tuning.

**Por qué ganó el decoder-only, y dónde sobreviven los otros.** El decoder-only tiene un único objetivo que convierte cualquier texto en señal de entrenamiento (todas las posiciones puntúan), escala de forma predecible y aprende tareas nuevas desde el prompt (*in-context learning*) sin cabeza específica. Para un asistente general que tiene que generar, ganó. Pero:

- cuando la tarea es un **conjunto fijo de etiquetas** y hay algo de datos etiquetados, un encoder pequeño ajustado suele ser igual o más preciso que un LLM, con una fracción del coste y la latencia;
- los **modelos de embeddings y los rerankers** (cross-encoders) son encoders, y también muchos filtros de seguridad (Prompt Guard de Meta es un mDeBERTa ajustado);
- cuando entrada y salida son secuencias distintas, como en **traducción y voz** (Whisper, NLLB), los encoder-decoders siguen siendo competitivos. Si son mejores a igual cómputo está en discusión; T5Gemma (Google, 2025) vuelve sobre la pregunta.

**Modos de fallo.** Un encoder no genera: no hay forma natural de escribir con él de izquierda a derecha. Las representaciones de un encoder preentrenado solo con MLM no están hechas para similitud: sin un entrenamiento contrastivo, su media no es un buen embedding de frase. Con pocas etiquetas por clase, un encoder ajustado se queda corto. Los modelos ajustados son sobreconfiados con texto fuera de distribución (se calibran con *temperature scaling* en validación), y las etiquetas derivan cuando aparecen peticiones nuevas que se fuerzan en clases viejas (una clase "otros" o un umbral de confianza). En los mapas de atención, los *attention sinks* acaparan probabilidad y pueden tapar el alineamiento real. Y si casi-duplicados caen en train y en test, la accuracy se infla.

**Papers clave.** Vaswani et al. (2017); Devlin et al. (2019, BERT); Liu et al. (2019, RoBERTa); Clark et al. (2020, ELECTRA); Raffel et al. (2020, T5); He et al. (2021, DeBERTa); Chung et al. (2022, Flan); Warner et al. (2024, ModernBERT); Casanueva et al. (2020, Banking77).

> **Notebook:** [17 · Encoders y encoder-decoders](../../notebooks/01_architecture/17_encoders_and_encoder_decoders.ipynb) (se lee después del 04) · **En el curso:** con el mismo tamaño y 800 pasos, el decoder recupera el primer token de una palabra oculta el 2.9 % de las veces y el encoder que ve los dos lados el 100 % (el mismo encoder sin el lado derecho cae al 2.5 %); en Banking77 (77 intenciones), ModernBERT-base ajustado llega al 93.0 % en test (69.6 % con 8 ejemplos por intención, 89.3 % con 32), `nomic-embed-text` + regresión logística al 93.3 % y ModernBERT congelado al 89.2 %, mientras que `qwen2.5:3b` se queda en 48.9 % zero-shot y 61.9 % few-shot con decodificación restringida, y en 15.2 % con function calling (181 de 231 salidas inválidas); latencia por mensaje de 18 ms (embeddings + regresión), 40 ms (encoder) y ~250 ms (LLM); Flan-T5-small reparte sus 77M en 19M de encoder, 25M de decoder con cross-attention y 16M de embedding compartido, y su cross-attention de la última capa la dominan dos attention sinks.

---

## 2. Pre-training e inferencia

Términos de esta parte: [glosario, Entrenamiento y post-training](glosario.md#entrenamiento-y-post-training) e [Inferencia y sistemas](glosario.md#inferencia-y-sistemas).

### 2.1 El objetivo: predecir el siguiente token

**Idea.** El pre-training minimiza la entropía cruzada media del token verdadero:

$$\mathcal{L}(\theta) = -\frac{1}{T}\sum_{t=1}^{T} \log p_\theta(x_t \mid x_{<t}), \qquad \mathrm{PPL} = e^{\mathcal{L}}$$

Es auto-supervisado: los objetivos son la entrada desplazada una posición, así que cualquier texto es dato de entrenamiento. Con la máscara causal, una secuencia de longitud $T$ da $T$ ejemplos en una pasada. Como la pérdida depende del tokenizador, las comparaciones entre modelos con vocabularios distintos se hacen en *bits por byte*.

**Por qué hay un suelo.** La entropía cruzada esperada se descompone:

$$H(p, q_\theta) = H(p) + D_{\mathrm{KL}}(p \,\Vert\, q_\theta) \;\ge\; H(p)$$

El entrenamiento solo reduce el término KL; la entropía del texto es irreducible. Con texto real se desconoce; con un corpus sintético se calcula.

**Lo que sustituyó.** BERT (enmascarado, solo un 15 % de posiciones con pérdida y sin forma natural de generar), T5 (span corruption, dos pilas) o UL2 (mezcla de denoisers) perdieron frente al LM causal como opción por defecto: cada token es un objetivo, entrenamiento e inferencia coinciden y escala sin trucos. GPT-3 mostró el **in-context learning**, y eso lo consolidó. Para código se añade *fill-in-the-middle*.

**Los datos deciden casi todo.** El pipeline típico (extracción, idioma, filtros heurísticos, clasificador de calidad, deduplicación exacta y MinHash, PII, descontaminación de benchmarks, pesos por dominio y una fase final de *annealing* con datos de alta calidad) es donde los datasets abiertos más han mejorado (RefinedWeb, FineWeb-Edu). Repetir datos unas pocas épocas es casi tan bueno como datos nuevos (Muennighoff et al., 2023).

### 2.2 Optimizador y plan de learning rate

**AdamW.** Casi todos los LLMs se entrenan con:

$$m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t, \qquad v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2$$

$$\theta_t = \theta_{t-1} - \eta_t\Big(\frac{\hat m_t}{\sqrt{\hat v_t} + \epsilon} + \lambda\,\theta_{t-1}\Big)$$

La división por $\sqrt{\hat v_t}$ da a cada parámetro su propio paso, independiente de la escala del gradiente (que varía órdenes de magnitud entre embeddings, atención y FFN). El *weight decay* desacoplado es la "W": la L2 dentro de Adam se dividiría por $\sqrt{\hat v}$ y regularizaría menos justo los parámetros con gradiente grande. El decay se aplica solo a matrices, nunca a ganancias de normas ni biases. $\beta_2 = 0.95$ (desde GPT-3) promedia unos 20 pasos en vez de 1000 y reacciona antes a cambios de escala, lo que reduce picos de pérdida. El *clipping* de la norma global a 1.0 protege contra lotes malos. Coste de memoria: unos **16 bytes por parámetro** con precisión mixta (contabilidad de ZeRO), antes de activaciones. Alternativas como Adafactor, Lion, Sophia o Muon existen; si superan a un AdamW bien ajustado a escala está en debate.

**Schedule.** El *warmup* existe porque al principio las estimaciones de $v_t$ se basan en pocos pasos y un paso grande sobre una red aleatoria puede llevarla a una región mala. La caída final deja que los parámetros se asienten en el mínimo en vez de rebotar alrededor. El coseno:

$$\eta_t = \eta_{min} + \tfrac{1}{2}(\eta_{max} - \eta_{min})\Big(1 + \cos\big(\pi\,\tfrac{t - t_{w}}{T - t_{w}}\big)\Big)$$

obliga a fijar $T$ de antemano. **WSD** (warmup-stable-decay; MiniCPM, Hägele et al., 2024) mantiene el LR constante y solo cae al final: se puede continuar desde cualquier checkpoint estable o ramificar varias caídas cortas para evaluar presupuestos distintos. Su ventaja es operativa.

**Estabilidad.** Picos de pérdida ligados a logits de atención grandes (QK-norm, z-loss, LR más bajo, en el peor caso reiniciar desde un checkpoint saltando lotes, como hizo PaLM); divergencia por LR pico demasiado alto ($\mu$P ayuda a transferir el LR entre anchos); bugs de datos (shuffle roto, tokenizador distinto, validación filtrada en entrenamiento).

### 2.3 Leyes de escalado

**Idea.** Entrenar $N$ parámetros sobre $D$ tokens cuesta $C \approx 6ND$ FLOPs (2 por parámetro y token hacia delante, 4 hacia atrás). Kaplan et al. (2020) encontraron que la pérdida cae como una ley de potencias en $N$, $D$ y $C$ durante muchos órdenes de magnitud, lo que permite planificar entrenamientos grandes desde pequeños. Chinchilla (Hoffmann et al., 2022) ajustó:

$$L(N, D) = E + \frac{A}{N^{\alpha}} + \frac{B}{D^{\beta}}$$

y, minimizando con $C$ fijo, concluyó que parámetros y datos deben crecer en igual proporción, unos **20 tokens por parámetro** (su modelo de 70B y 1.4T tokens superó a Gopher de 280B con el mismo cómputo). El consejo previo de Kaplan (crecer el modelo más rápido que los datos) se debía en buena parte a no ajustar el schedule a la longitud del entrenamiento; y el propio ajuste de Chinchilla se ha revisado (Besiroglu et al., 2024), así que los exponentes son aproximados.

**Óptimo de entrenamiento no es óptimo de despliegue.** Un modelo servido a millones de usuarios se paga sobre todo en inferencia, que escala con $N$. Por eso se **sobre-entrenan** modelos pequeños muy por encima de 20 tokens por parámetro (Llama 3 8B, unos 15T tokens, ~1.900 por parámetro): la pérdida sigue bajando, más despacio por FLOP.

> **Notebook:** [05 · Pre-training](../../notebooks/02_pretraining_and_inference/05_pretraining.ipynb) · **En el curso:** pérdida inicial $\approx \ln V$ y test de fuga de la máscara causal como sanity checks; el corpus sintético tiene un suelo exacto ($4 \ln 10 = 9.21$ nats por frase) que los modelos alcanzan en ~100 pasos, el modelo grande llega antes y coseno y WSD acaban prácticamente iguales.

### 2.4 Estrategias de decodificación

**Problema.** El modelo da logits; hay que convertirlos en texto. Los métodos que buscan la moda (greedy, beam search) sirven cuando hay una respuesta correcta (traducción, extracción), pero en texto abierto producen bucles insulsos: maximizar verosimilitud genera texto *más* predecible que el humano, y la repetición se auto-refuerza (Holtzman et al., 2020). El muestreo puro tiene el problema contrario: la cola larga de tokens improbables suma probabilidad suficiente para descarrilar.

**Idea.** Temperatura y truncamiento:

$$p_i(T) = \frac{e^{z_i / T}}{\sum_j e^{z_j / T}}, \qquad V_p = \text{menor } S \text{ con } \sum_{i \in S} p_i \ge p$$

La temperatura no cambia el orden de los tokens, solo cuánta masa se llevan los primeros. Top-k recorta a un tamaño fijo; top-p (nucleus) recorta a un conjunto que se adapta a la confianza del modelo, y por eso desplazó a top-k en casi todas las APIs; min-p usa un umbral relativo al token más probable y es popular en runtimes abiertos (cuánto mejora a top-p está en discusión).

| Estrategia | Se adapta a la confianza | Uso típico |
|---|---|---|
| greedy / beam | no | extracción, traducción, evaluaciones deterministas |
| temperatura | no | siempre combinada con un truncamiento |
| top-k | no | defecto de la época de GPT-2 |
| top-p | sí | chat (a menudo $T \approx 0.7$, $p \approx 0.9$) |
| min-p | sí | runtimes abiertos, escritura creativa |

**Matices.** Las penalizaciones de repetición también castigan repeticiones correctas (un nombre de variable). El orden en que cada motor aplica temperatura, top-k y top-p cambia la distribución. Y $T = 0$ no es perfectamente determinista en un servidor: el resultado de una matmul puede depender del batch. La salida estructurada (apartado 5.2) es también una estrategia de decodificación.

### 2.5 Prefill, decode y KV-cache

**Idea.** La generación tiene dos fases con física distinta. **Prefill** procesa el prompt de una vez: cada peso leído se usa $P$ veces, está limitado por cómputo y fija el *time to first token* (TTFT). **Decode** produce un token por pasada: cada peso se lee para un solo vector (o $B$ con batch $B$), está limitado por ancho de banda y fija el *time per output token* (TPOT). La intensidad aritmética de un paso de decode es:

$$\text{intensidad aritmética} \approx \frac{2NB \text{ FLOPs}}{2N \text{ bytes}} = B \ \text{FLOPs por byte}$$

Una GPU de centro de datos hace del orden de cien o más FLOPs por byte leído (el "ridge point" del roofline). Con $B = 1$ la GPU está casi siempre esperando a la memoria. Esa única cuenta explica casi todo lo demás: el techo de velocidad a batch 1 es aproximadamente ancho de banda / tamaño del modelo en bytes; el batching sale casi gratis hasta acercarse al ridge point; la cuantización acelera decode porque reduce bytes, no FLOPs; y el speculative decoding aprovecha el cómputo ocioso.

La **KV-cache** convierte $O(N^2)$ trabajo de proyecciones en $O(N)$ y es una optimización *exacta*: con greedy, con y sin caché deben salir los mismos tokens (salvo empates rotos por redondeo). Su tamaño, no el de los pesos, suele decidir cuántas secuencias caben en una GPU; de ahí GQA, MLA, ventanas y caché en fp8.

### 2.6 Batching y serving

| Técnica | Problema que resuelve | Referencia |
|---|---|---|
| Batching estático | amortiza la lectura de pesos | — |
| Continuous batching | en el estático, las secuencias terminadas dejan huecos hasta que acaba la más larga; aquí el scheduler entra y saca secuencias en cada paso | Orca (Yu et al., 2022) |
| PagedAttention | buffers contiguos por secuencia desperdician memoria; bloques fijos con tabla, como páginas de memoria virtual | vLLM (Kwon et al., 2023) |
| Prefix caching | muchos prompts comparten system prompt o few-shot; reusar sus bloques KV | SGLang (Zheng et al., 2024) |
| Chunked prefill | el prefill de un prompt largo bloquea a todos los que están en decode | Sarathi-Serve (Agrawal et al., 2024) |
| Desagregación prefill/decode | las dos fases quieren hardware distinto | DistServe (Zhong et al., 2024) |
| CUDA graphs | el coste de lanzar kernels domina en pasos pequeños | — |

**Compromiso latencia-throughput.** El tiempo de una petición es aproximadamente $\text{TTFT} + N_{out} \times \text{TPOT}$. Subir el batch aumenta el throughput agregado casi gratis mientras decode esté limitado por memoria; cuando el paso pasa a estar limitado por cómputo, cada secuencia extra ralentiza a todas. Un sistema de serving elige el mayor batch que cumple el objetivo de latencia en p99 ("goodput"). Los benchmarks que solo dan throughput pico o solo latencia de una petición esconden esa elección.

### 2.7 Speculative decoding

**Problema.** En decode, verificar $k+1$ posiciones cuesta casi lo mismo que generar una.

**Idea.** Un modelo borrador barato propone $k$ tokens y el modelo objetivo los verifica en una pasada (Leviathan et al., 2023; Chen et al., 2023):

```
borrador:  propone  d1 d2 d3 d4          (k = 4 pasos baratos)
objetivo:  una pasada puntúa 1..5        acepta d1, acepta d2, rechaza d3 -> remuestrea
salida:    d1 d2 r3                       (3 tokens por pasada del objetivo)
```

Cada token propuesto $x \sim q$ se acepta con probabilidad $\min(1, p(x)/q(x))$; en el primer rechazo se muestrea del residuo $p'(x) \propto \max(0, p(x) - q(x))$. La salida tiene **exactamente** la distribución del objetivo:

$$P(x) = q(x)\min\Big(1, \frac{p(x)}{q(x)}\Big) + (1-\beta)\,p'(x) = \min(p, q) + \max(0, p - q) = p(x)$$

con $\beta = \sum_y \min(p(y), q(y)) = 1 - \mathrm{TV}(p,q)$. Un borrador malo hace el método lento, nunca incorrecto. Con tasa de aceptación $\alpha$ independiente por token:

$$\mathbb{E}[\text{tokens por ronda}] = \frac{1 - \alpha^{k+1}}{1 - \alpha}$$

La ganancia real es menor porque los $k$ pasos del borrador no son gratis.

**Cuándo no ayuda.** Con batch grande desaparece el cómputo ocioso; a temperatura alta las distribuciones discrepan más; el borrador debe compartir tokenizador. Las variantes modernas (Medusa, EAGLE, prompt lookup, la cabeza MTP de DeepSeek-V3) eliminan el modelo borrador separado pero conservan la misma regla de aceptación, que es la que garantiza la corrección.

> **Notebook:** [06 · Inferencia y serving](../../notebooks/02_pretraining_and_inference/06_inference_serving.ipynb) · **En el curso:** entropía y truncamiento de cada estrategia en el punto donde el modelo duda (greedy repite una única continuación 16 veces), igualdad exacta con y sin KV-cache, speculative decoding con $k = 4$ (techo de 5 tokens por ronda) y un test estadístico de que la distribución no cambia; en un modelo de juguete el coste de lanzar kernels domina, así que la mejora por caché parece modesta.

### 2.8 Cuantización

**Problema.** Un modelo de 7B en bf16 ocupa 14 GB solo en pesos. La cuantización ataca tres problemas distintos que conviene separar: **memoria** (bytes por peso; el ahorro está garantizado), **ancho de banda** (bytes leídos por token en decode; solo si hay un kernel que opere sobre el formato empaquetado) y **cómputo** (requiere pesos *y* activaciones en baja precisión y hardware que lo soporte).

**Idea.** Con escala simétrica por grupo:

$$s = \frac{\max|w|}{2^{b-1}-1},\qquad q = \mathrm{round}(w/s),\qquad \hat w = s\,q, \qquad \mathbb{E}[(w-\hat w)^2] = \frac{s^2}{12}$$

Cada bit extra divide $s$ por dos y la varianza del error por cuatro. Lo que decide si un esquema funciona es $s$, fijado por el **valor más grande del grupo**: un outlier engorda el paso para todos. Por eso las escalas son por bloques pequeños (32-128 pesos), lo que cuesta $16/g$ bits extra por peso, y por eso NF4 coloca sus 16 niveles en los cuantiles de una normal. Los enteros reparten niveles uniformemente; los floats (fp8 E4M3/E5M2), logarítmicamente, lo que absorbe parte del rango dinámico.

**Por qué los LLMs lo toleran y dónde no.** Están sobre-parametrizados y entrenados con ruido; los errores de redondeo en un producto escalar largo se cancelan en parte. Pero las activaciones son más difíciles que los pesos (LLM.int8() encontró dimensiones con valores enormes a partir de unos pocos miles de millones de parámetros), los modelos pequeños sufren más, y embeddings, cabeza de salida y primeros y últimos bloques son más sensibles.

| Método | Datos | Coste | Punto óptimo |
|---|---|---|---|
| RTN (quanto, bitsandbytes) | no | segundos | 8 bits; 4 bits con grupos pequeños en modelos grandes |
| LLM.int8() | no | al cargar | 8 bits con outliers en fp16 |
| NF4 + doble cuantización | no | al cargar | base 4 bits para QLoRA |
| GPTQ | calibración pequeña | minutos-horas | 4 bits (y 3) solo pesos |
| AWQ | calibración pequeña | minutos | 4 bits solo pesos |
| SmoothQuant / FP8 | escalas de activación | minutos | W8A8 en hardware int8/fp8 |
| QAT, BitNet | entrenamiento completo | un entrenamiento | 2 bits o menos |

**GPTQ** minimiza el error de la *salida* de cada capa, $\min_{\hat W} \Vert WX - \hat W X\Vert_F^2$, cuantizando columna a columna y compensando el error en las columnas pendientes con la inversa del Hessiano (la idea de Optimal Brain Surgeon). **AWQ** detecta los canales importantes por la magnitud de sus *activaciones* y los escala antes de cuantizar con la identidad $Wx = (W\,\mathrm{diag}(s))(\mathrm{diag}(s)^{-1}x)$. **SmoothQuant** usa la misma identidad al revés para W8A8, y QuaRot/SpinQuant rotan con matrices ortogonales para repartir los outliers.

**Lo que hace la industria.** FP8 en pesos y activaciones para serving en GPUs recientes (casi sin pérdida, con tensor cores nativos); 4 bits solo pesos para local y coste (AWQ y GPTQ en servidor, GGUF k-quants en llama.cpp y Ollama); FP8 también en entrenamiento (DeepSeek-V3). Con presupuesto de memoria fijo, un modelo más grande a ~4 bits suele ganar a uno más pequeño en más precisión (Dettmers y Zettlemoyer, 2023).

**Medir fidelidad.** La perplejidad solo mira el token verdadero y puede no moverse aunque el modelo se comporte distinto. Mejor comparar contra el original con $D_{\mathrm{KL}}(p_{ref} \,\Vert\, p_q)$ por posición (no negativa, no se cancela) y el **acuerdo top-1** (¿greedy diría lo mismo?). Un modelo cuantizado puede igualar la precisión del original y cambiar muchas respuestas individuales (Dutta et al., 2024).

**Modos de fallo.** Outliers con escala por tensor; RTN a 2-3 bits (colapso); habilidades concretas que se rompen primero (matemáticas, código, contexto largo, idiomas no ingleses); calibración de otro dominio; formato sin kernel fusionado, que ahorra memoria pero es *más lento*.

> **Notebook:** [09 · Cuantización](../../notebooks/02_pretraining_and_inference/09_quantization.ipynb) · **En el curso:** con SmolLM2-360M y fp32 como referencia, bf16 ya da KL 0.001 y 97.2 % de acuerdo top-1 (el suelo de ruido); int8 y float8 bajan los pesos de 0.72 a 0.41 GB con ~96 % de acuerdo; int4 llega a 0.26 GB pero cae al 73 % de acuerdo y +29 % de perplejidad; int2 destruye el modelo. Los modelos cuantizados fueron *más lentos* que bf16 (~24 → 17 y 10.5 tok/s) por falta de kernel fusionado.

---

## 3. Post-training

Términos de esta parte: [glosario, Entrenamiento y post-training](glosario.md#entrenamiento-y-post-training).

### 3.1 Supervised fine-tuning

**Problema.** Un modelo instruct ya conversa, pero no conoce *tu* tarea, tu formato ni tus convenciones.

**Idea.** SFT es exactamente el objetivo del pre-training sobre un dataset curado de pares (prompt, respuesta ideal), y solo sobre los tokens de la respuesta:

$$\mathcal{L}_{\text{SFT}}(\theta) = -\,\mathbb{E}_{(x,y)\sim\mathcal{D}} \sum_{t=1}^{|y|} \log \pi_\theta(y_t \mid x, y_{<t})$$

El **enmascarado de la pérdida** (labels a `-100` en el prompt) convierte la verosimilitud conjunta $\log \pi(x,y)$ en la condicional $\log \pi(y \mid x)$, que es lo que se pide en inferencia; la pérdida incluye el token de fin de turno para que el modelo aprenda a parar. El entrenamiento debe usar exactamente el **chat template** del modelo. Mantener pérdida en el prompt puede ayudar con respuestas muy cortas y pocos datos (Shi et al., 2024); el enmascarado sigue siendo el defecto.

**Intuición y límites.** SFT es *behaviour cloning*: imita las demostraciones, errores incluidos, y no recibe señal de que una respuesta incorrecta sea peor que otra. Además sufre **exposure bias**: entrena condicionado en tokens correctos y en inferencia condiciona en los suyos, así que un error temprano lo lleva a estados que nunca vio. Enseña de forma fiable **forma** (formato, estilo, conjunto de etiquetas, aplicar una regla que el modelo ya medio conoce) y es mala vía para añadir **hechos nuevos**: los ejemplos con hechos desconocidos se aprenden despacio y aumentan las alucinaciones (Gekhman et al., 2024). La "hipótesis de alineamiento superficial" de LIMA (pocos miles de ejemplos bastan para el formato) es útil pero discutida.

| Enfoque | Cambia | Bueno para | Débil en |
|---|---|---|---|
| Prompting / few-shot | nada | iterar rápido | formatos largos y estrictos; cada llamada paga los ejemplos |
| RAG | el contexto | hechos que cambian, citas, control de acceso | enseñar un comportamiento |
| SFT (completo o LoRA) | los pesos | formatos, estilo, habilidades estrechas | añadir conocimiento |
| Destilación | pesos de un modelo menor | bajar el coste de serving | hereda los errores del profesor |

En la práctica se combinan: un modelo ajustado a tu formato, alimentado con contexto recuperado. La regla es medir antes un **baseline few-shot**: el fine-tuning tiene que ganar a un buen prompt.

### 3.2 LoRA y fine-tuning eficiente

**Problema.** El fine-tuning completo cuesta unos 16 bytes por parámetro entrenable (112 GB para 7B) y una copia completa del modelo por tarea.

**Idea.** Congelar $W$ y aprender una actualización de bajo rango (Hu et al., 2021):

$$h = W x + \frac{\alpha}{r}\, B (A x), \qquad A\in\mathbb{R}^{r\times k},\ B\in\mathbb{R}^{d\times r},\ B=0 \text{ al inicio}$$

$B = 0$ hace que el modelo arranque exactamente como el base; $\alpha/r$ desacopla la magnitud de la actualización del rango. Una matriz $4096 \times 4096$ con $r = 16$ pasa de 16.8 M parámetros entrenables a 131 k (factor 128), y gradientes y estados de Adam bajan igual. Tras entrenar, `merge_and_unload()` pliega $BA$ en $W$: **cero latencia extra**. Ahorra memoria de optimizador y gradientes, no de activaciones.

**Por qué funciona.** El fine-tuning tiene **dimensión intrínseca** baja (Aghajanyan et al., 2021): basta optimizar unos cientos o miles de parámetros proyectados en el espacio completo. Hu et al. vieron que $\Delta W$ amplifica unas pocas direcciones ya presentes en $W$. Es un hallazgo sobre tareas *estrechas*: con un dominio lejano (otro idioma, pre-training continuado) la actualización necesaria tiene más rango y LoRA se queda atrás. Biderman et al. (2024) lo resumen: "LoRA aprende menos y olvida menos".

| | Fine-tuning completo | LoRA | QLoRA |
|---|---|---|---|
| Parámetros entrenables | todos | $r(d+k)$ por matriz adaptada | igual que LoRA |
| Base congelada en memoria | — | 16 bits | 4 bits (NF4) |
| Calidad en tareas estrechas | referencia | normalmente cercana | cercana, algo más de ruido |
| Artefacto por tarea | modelo completo | adaptador pequeño | adaptador pequeño |

LoRA desplazó a los adapters (latencia extra) y al prefix/prompt tuning (secuencia efectiva más larga) porque no añade latencia, iguala al fine-tuning completo en muchas tareas estrechas y lo soporta todo el ecosistema (`peft`, TRL, vLLM con varios adaptadores sobre un mismo base). Variantes: rsLoRA ($\alpha/\sqrt{r}$), DoRA.

**QLoRA: la misma LoRA sobre un base de 4 bits.** LoRA congela los pesos base, pero siguen ocupando memoria en bf16: 2 bytes por parámetro, unos 14 GB para un 7B antes de guardar una sola activación. QLoRA (Dettmers et al., 2023) guarda esos pesos congelados en 4 bits y entrena adaptadores LoRA normales en bf16 encima:

$$h = \operatorname{dequant}(W_0^{\text{NF4}})\,x + \frac{\alpha}{r}\, B A x$$

El peso de 4 bits se descuantiza a bf16 bloque a bloque solo para la multiplicación y se descarta; el gradiente atraviesa la capa congelada hasta $A$ y $B$, que siguen en bf16, así que el optimizador nunca toca un número de 4 bits. Las tres ideas del artículo: **NF4** (16 niveles en los cuantiles de una normal, con una escala absmax por bloque de 64 pesos; los pesos preentrenados son aproximadamente normales, ver el apartado [2.8](#28-cuantización)), **doble cuantización** (las escalas de cada bloque se cuantizan a su vez a 8 bits, y el sobrecoste baja de 0.5 a unos 0.127 bits por parámetro) y **optimizadores paginados** (el estado del optimizador vive en memoria unificada de CUDA y se pagina a la RAM cuando un pico de memoria lo exige, en lugar de morir con un out-of-memory). El precio: cada paso es más lento porque descuantiza en cada forward y backward, y el adaptador es para un base *cuantizado*; para fusionarlo se recarga el base en bf16 y se aplica allí. La regla práctica: si el modelo cabe en bf16 con margen para entrenar, LoRA (más rápida, fusión directa); si no cabe, QLoRA. En `transformers` es `BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)`, más `prepare_model_for_kbit_training` y `optim="paged_adamw_8bit"`.

**Reglas prácticas y modos de fallo.** Adaptar todas las capas lineales a $r = 8$-16 suele ganar a un rango grande solo en `q_proj` y `v_proj`; LoRA necesita un LR ~10 veces mayor que el fine-tuning completo (hallazgo empírico); al comparar rangos, comprueba si $\alpha$ escala con $r$. La trampa del **token nuevo**: con embeddings atados y congelados, una fila inicializada en la media de los embeddings nunca puede ganar el argmax, porque $h \cdot \bar e$ es la media de los $h \cdot e_i$ y una media no supera al máximo; hay que entrenar esa fila (`modules_to_save` o `trainable_token_indices`).

> **Notebook:** [07 · SFT, LoRA y destilación](../../notebooks/03_post_training/07_finetuning_sft_lora.ipynb) · **En el curso:** extracción de reservas a JSON con SmolLM2-135M; few-shot llega al 17.5 % de exact match, SFT completo al 35.0 % y LoRA (3.5 % de los parámetros, adaptador de 20 MB) al 32.5 %; el rango sube el exact match de 20 % ($r=2$) a 45 % ($r=64$); con embeddings congelados el token nuevo nunca se emite (queda en torno al puesto 24.000) y entrenando la fila sale primero con p ≈ 0.93; QLoRA r=16 sobre el base en NF4 queda en 37.5 % de exact match y 86.5 % de campos (LoRA r=16: 32.5 % y 86.5 %, dos peticiones de diferencia, ruido), los pesos en memoria bajan de 269 MB a 110 MB pero el pico de entrenamiento apenas se mueve (1.41 → 1.34 GB) porque a 135M dominan las activaciones; para un 7B los pesos congelados pasarían de 14 GB a unos 3.6 GB.

### 3.3 Destilación

**Problema.** Los datos etiquetados son caros; los prompts sin etiquetar, baratos. Y un modelo grande que funciona cuesta mucho de servir.

**Idea.** Un profesor fuerte transfiere lo que sabe a un estudiante pequeño. Hinton, Vinyals y Dean (2015) observaron que la distribución de salida del profesor lleva más información que su respuesta principal ("conocimiento oscuro": qué errores son plausibles) y la exponen suavizando con temperatura:

$$\mathcal{L} = (1-\lambda)\,\mathrm{CE}(y, p_s) + \lambda\,\tau^2\,D_{\mathrm{KL}}\big(p_t^{\tau} \,\Vert\, p_s^{\tau}\big), \qquad p_i^\tau = \frac{e^{z_i/\tau}}{\sum_j e^{z_j/\tau}}$$

El factor $\tau^2$ compensa que el gradiente del KL suavizado escale como $1/\tau^2$. La dirección del KL importa: el **directo** es *mode-covering* (el estudiante reparte masa por todo lo que el profesor podría decir) y el **inverso** es *mode-seeking*; MiniLLM defiende el inverso para generación y GKD entrena sobre muestras del propio estudiante (on-policy), lo que ataca el exposure bias.

| | Por secuencia (etiquetas sintéticas) | Por logits | On-policy (GKD, MiniLLM) |
|---|---|---|---|
| Necesita del profesor | solo texto | logits en cada posición | logits sobre muestras del estudiante |
| Tokenizador | cualquiera | debe coincidir | debe coincidir |
| Coste | una generación por prompt | pasada del profesor por paso | muestreo + pasada del profesor por paso |
| Riesgo principal | copia los errores | memoria para dos modelos | lento, más piezas |

**Por qué domina la versión por secuencia.** Kim y Rush (2016) mostraron que entrenar sobre las salidas decodificadas del profesor funciona sorprendentemente bien: se pierde el conocimiento oscuro, pero funciona con tokenizadores distintos, con un profesor detrás de una API y es SFT normal. Así se hicieron Alpaca, Vicuna y muchas destilaciones de razonamiento (DeepSeek-R1-Distill).

**Modos de fallo.** El estudiante está acotado por el profesor y hereda sus errores sistemáticos; una brecha de capacidad grande puede hacer de un profesor muy fuerte un peor profesor (Mirzadeh et al., 2020); los términos de uso de algunas APIs prohíben entrenar modelos competidores con sus salidas.

> **Notebook:** [07 · SFT, LoRA y destilación](../../notebooks/03_post_training/07_finetuning_sft_lora.ipynb) · **En el curso:** un profesor de 360M entrenado con las mismas 300 etiquetas llega al 87.5 %, etiqueta 600 peticiones nuevas con un 90 % de acierto, y el estudiante de 135M pasa del 32.5 % al 55 % de exact match (con la salvedad de que también dio más pasos de optimización).

### 3.4 De RLHF a DPO y GRPO

**Problema.** SFT no dice *cuánto* de mala es una respuesta y solo muestra buen comportamiento en estados tomados de los datos. Además, a menudo es más fácil **comparar** dos respuestas que **escribir** la ideal.

**RLHF clásico** (Christiano et al., 2017; Stiennon et al., 2020; InstructGPT, 2022): SFT, después un modelo de recompensa ajustado con comparaciones humanas mediante **Bradley-Terry**,

$$P(y_w \succ y_l \mid x) = \sigma\big(r(x, y_w) - r(x, y_l)\big)$$

y por último RL con PPO sobre el objetivo regularizado:

$$\max_{\pi}\; \mathbb{E}_{x,\,y\sim\pi}\big[r_\phi(x,y)\big] \;-\; \beta\,D_{\mathrm{KL}}\big(\pi(\cdot|x)\,\Vert\,\pi_{\text{ref}}(\cdot|x)\big)$$

El KL importa tanto como la recompensa: el modelo de recompensa solo es fiable cerca de sus datos, y sin correa la política encuentra salidas que puntúan alto por motivos equivocados (**reward hacking**; Gao et al., 2023, midieron cómo la calidad real sube y luego baja). PPO necesita cuatro modelos a la vez (política, referencia, recompensa y un crítico de tamaño similar), muestrear en el bucle y muchos hiperparámetros sensibles: funciona, pero es caro y frágil.

**DPO: el modelo de recompensa está escondido en la política.** Rafailov et al. (2023) observaron que el objetivo anterior tiene óptimo cerrado, $\pi^*(y|x) \propto \pi_{\text{ref}}(y|x)\exp(r(x,y)/\beta)$, que se puede invertir:

$$r(x,y) = \beta \log\frac{\pi^*(y\mid x)}{\pi_{\text{ref}}(y\mid x)} + \beta \log Z(x)$$

Al sustituir en Bradley-Terry, el $\log Z(x)$ intratable se cancela en la diferencia (ambas respuestas comparten prompt) y queda una pérdida de clasificación sobre pares:

$$\mathcal{L}_{\text{DPO}} = -\,\mathbb{E}\,\log\sigma\Big(\beta\log\frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta\log\frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)}\Big)$$

Sin muestreo, sin crítico, sin modelo de recompensa: solo política y referencia (con LoRA, la referencia es la misma red con el adaptador apagado). El gradiente pesa más los pares mal ordenados y **solo restringe la diferencia** de log-probabilidades. Eso causa el fallo más instructivo: la **likelihood displacement**: cuando elegida y rechazada se parecen mucho, la forma más barata de ensanchar el margen puede ser bajar *las dos* y mandar la masa a respuestas que no están en ningún conjunto (Razin et al., 2024). El arreglo estándar es re-anclar la elegida con un término SFT: $\mathcal{L}_{\text{DPO}} + \lambda\,\mathrm{NLL}(y_w)$ (DPO+NLL, RPO). Otras limitaciones: es offline (no explora; el DPO iterativo cierra parte de la brecha), sobreajusta preferencias deterministas (IPO), explota la longitud. Si DPO iguala a un PPO bien ajustado está **en debate** (Xu et al., 2024 encontraron PPO por delante en código); su popularidad viene de la simplicidad.

**RL con recompensas verificables y GRPO.** En matemáticas o código, el modelo de recompensa se sustituye por un programa (tests, comparación numérica, un esquema): RLVR. Por el gradiente de política

$$\nabla_\theta\, \mathbb{E}_{y\sim\pi_\theta}[r] = \mathbb{E}\big[(r - b)\,\nabla_\theta \log\pi_\theta(y\mid x)\big]$$

cualquier baseline $b$ reduce la varianza sin sesgar. PPO aprende $b$ con un crítico; **GRPO** (Shao et al., 2024) lo estima con un grupo de $G$ muestras del mismo prompt:

$$A_i = \frac{r_i - \operatorname{mean}(r_1,\dots,r_G)}{\operatorname{std}(r_1,\dots,r_G)}$$

con el ratio recortado de PPO y un KL a la referencia. Consecuencias: no hay crítico; un grupo donde todas las respuestas puntúan igual no aporta gradiente (por eso la fracción de grupos de varianza cero es el diagnóstico clave, y DAPO los filtra); y la normalización introduce sesgos (Dr. GRPO propone quitar la división por longitud y por desviación típica). DeepSeek-R1-Zero mostró que GRPO con recompensas simples de corrección y formato produce cadenas de razonamiento largas sin trazas supervisadas. Cuánto *crea* RLVR nuevas capacidades frente a *afilar* las que el modelo base ya puede muestrear es una pregunta abierta (Yue et al., 2025).

**Diseño de recompensa.** Es la mayor parte del trabajo. Una recompensa densa crea varianza en más grupos, pero cada término es un comportamiento por el que pagas (ley de Goodhart); los verificadores tienen falsos positivos y, con suficientes muestras, la política los encuentra.

| | RLHF con PPO | DPO | GRPO (RLVR) |
|---|---|---|---|
| Señal | modelo de recompensa aprendido | comparaciones, directamente | un programa que comprueba |
| Modelos en memoria | política, referencia, recompensa, crítico | política, referencia | política, referencia (ninguna si $\beta=0$) |
| Muestreo en entrenamiento | sí | no | sí, $G$ por prompt |
| Fallo típico | reward hacking, inestabilidad | sobreajuste, likelihood displacement | verificador laxo, grupos de varianza cero |
| Uso hoy | alineamiento de chat en grandes labs | alineamiento de modelos abiertos, estilo, seguridad | razonamiento: matemáticas, código, herramientas |

**Papers clave.** Christiano et al. (2017); Schulman et al. (2017, PPO); Ouyang et al. (2022); Rafailov et al. (2023); Azar et al. (2023, IPO); Shao et al. (2024, DeepSeekMath); DeepSeek-AI (2025, R1); Razin et al. (2024); Gao et al. (2023).

> **Notebook:** [08 · DPO y GRPO](../../notebooks/03_post_training/08_preference_rl_dpo_grpo.ipynb) · **En el curso:** DPO puro gana en sus propias métricas de entrenamiento pero hunde el exact match de 30 % a 7.5 % (la log-prob de la respuesta correcta cae de −1.91 a −5.63); DPO+NLL la mantiene (−1.91 → −2.01) y no rompe nada, aunque solo sube a 32.5 %; GRPO lleva la aritmética de 5 % a 46 % (recompensa exacta) y 42 % (densa) en 40 pasos, sin etiquetas, y la recompensa densa baja la fracción de grupos de varianza cero de 60-80 % a ~5-25 %.

---

## 4. Aplicaciones

Términos de esta parte: [glosario, Aplicaciones: LangChain y RAG](glosario.md#aplicaciones-langchain-y-rag).

### 4.1 Retrieval-augmented generation

**Problema.** El conocimiento **paramétrico** de un modelo está congelado, no se puede citar ni borrar para un usuario y es poco fiable en hechos raros: el modelo responde con fluidez lo sepa o no. Un conocimiento **no paramétrico** (una colección de documentos que se busca en tiempo de consulta) resuelve eso.

**Idea.** Recuperar y luego leer. Aproximadamente:

$$P(\text{respuesta correcta}) \approx P(\text{evidencia recuperada}) \times P(\text{respuesta correcta} \mid \text{evidencia en contexto})$$

Los dos factores fallan por motivos distintos y se arreglan con herramientas distintas (chunking, índices y consultas para el primero; prompt, modelo y tamaño de contexto para el segundo), así que **se evalúan por separado**.

| Enfoque | Actualizar conocimiento | Citas | Control de acceso |
|---|---|---|---|
| Confiar en el modelo | re-entrenar | no | no |
| Fine-tuning sobre los documentos | re-entrenar; malo inyectando hechos | no | no |
| RAG | re-indexar | sí | filtrar al recuperar |
| Contexto largo (pegarlo todo) | editar el prompt | posible | por petición |

El término viene de Lewis et al. (2020), que entrenaban recuperador y generador juntos; lo que la industria llama RAG hoy es más simple: un modelo de embeddings comercial, un índice y un LLM instruct que lee los pasajes, sin entrenamiento conjunto. Ganó porque cada pieza se cambia y se actualiza por separado. Recuperar es mejor que el fine-tuning no supervisado para inyectar hechos (Ovadia et al., 2024); el contexto largo es alternativa real para corpus pequeños, pero cada consulta paga todo el contexto y el modelo usa peor lo que está en medio (*Lost in the Middle*, Liu et al., 2024).

**Recuperación dispersa y densa.** **BM25** puntúa por términos de la consulta, ponderados por rareza, con saturación de frecuencia ($k_1$) y normalización por longitud ($b$):

$$\text{BM25}(q, d) = \sum_{t \in q} \text{IDF}(t)\cdot\frac{f(t,d)\,(k_1+1)}{f(t,d) + k_1\,(1 - b + b\,|d|/\text{avgdl})}$$

Exacto con identificadores, sin entrenamiento, y víctima del *vocabulary mismatch*. La **recuperación densa** (DPR, Karpukhin et al., 2020) usa un bi-encoder entrenado con pérdida contrastiva InfoNCE,

$$\mathcal{L} = -\log\frac{\exp(\mathrm{sim}(q, d^+)/\tau)}{\exp(\mathrm{sim}(q, d^+)/\tau) + \sum_j \exp(\mathrm{sim}(q, d^-_j)/\tau)}$$

que acerca paráfrasis pero difumina cadenas exactas, nombres y números, y cae fuera de su dominio (en BEIR, BM25 fue un baseline duro). Solo el bi-encoder permite pre-calcular los documentos; el **cross-encoder** lee consulta y documento juntos y se usa para reordenar los 20-100 primeros; ColBERT (interacción tardía) está en medio.

**Híbrido con RRF, el defecto de la industria.** Como fallan en consultas distintas, se ejecutan ambos y se fusionan rankings (no puntuaciones, que están en escalas incomparables):

$$\text{RRF}(d) = \sum_{r} w_r \cdot \frac{1}{c + \text{rank}_r(d)}, \qquad c = 60$$

RRF (Cormack et al., 2009) ignora magnitudes, no necesita calibración y premia el acuerdo entre recuperadores. Una combinación convexa ajustada puede superarlo (Bruch et al., 2023), así que "RRF siempre gana" no está zanjado; es el defecto seguro sin datos etiquetados.

**Chunking.** La tensión de fondo: la recuperación quiere unidades pequeñas y de un tema; la generación quiere contexto suficiente. Fijo, recursivo por separadores, por estructura (encabezados), semántico, *small-to-big* (recuperar el trozo pequeño, mostrar la sección padre) o *contextual chunks* (prefijar una descripción escrita por un LLM). El tamaño es un hiperparámetro que se mide. Para escalar, índices aproximados (HNSW, IVF, product quantization), cuyo recall es otro parámetro de calidad independiente del modelo de embeddings.

**Transformaciones de consulta.** Multi-query/RAG-Fusion, HyDE (embeber una respuesta hipotética), query2doc, step-back, descomposición: todas gastan llamadas al LLM antes de recuperar para atacar el vocabulary mismatch, y todas arriesgan latencia y *deriva* del significado.

**Modos de fallo.**

| Síntoma | Causa probable | Arreglo |
|---|---|---|
| la respuesta está en los docs pero no se recupera | chunks mal dimensionados, vocabulario distinto | medir tamaños, híbrido, multi-query/HyDE |
| se recupera pero el modelo lo ignora | demasiado contexto, modelo pequeño | menos y mejores chunks, reranker, exigir citas |
| hacen falta hechos de varios sitios | recuperación de un solo paso | recuperación iterativa o agéntica |
| falla en nombres, códigos, versiones | solo denso | añadir BM25 |

### 4.2 Métricas de recuperación

Para cada consulta, $\text{rank}_q$ es la posición del primer resultado relevante y $\text{rel}_q(i) \in \lbrace 0,1 \rbrace$:

$$\text{hit@}k = \frac{1}{|Q|}\sum_q \mathbb{1}[\text{rank}_q \le k], \qquad \text{MRR} = \frac{1}{|Q|}\sum_q \frac{1}{\text{rank}_q}$$

$$\text{nDCG@}k = \frac{\text{DCG@}k}{\text{IDCG@}k}, \qquad \text{DCG@}k = \sum_{i=1}^{k} \frac{\text{rel}_q(i)}{\log_2(i+1)}$$

hit@k coincide con recall@k cuando hay un único elemento relevante. MRR premia poner la evidencia *primero*, y eso importa en RAG porque el modelo lee mejor los primeros chunks y un $k$ menor abarata el prompt. nDCG generaliza a varios relevantes con relevancia graduada (estándar en BEIR). La evaluación a nivel de **fichero** se satura pronto; a nivel de **evidencia** (¿contiene el chunk la cadena que responde?) es más estricta y más cercana a lo que necesita el LLM. Etiquetar con cadenas literales es barato y transparente pero no ve paráfrasis; la alternativa es un juez LLM (siguiente apartado).

> **Notebook:** [10 · RAG con LangChain y RAGAS](../../notebooks/04_applications/10_rag.ipynb) · **En el curso:** sobre la documentación del curso, a nivel de fichero todos los recuperadores aciertan (hit@4 = 1.00); a nivel de evidencia el híbrido es el mejor arriba del ranking (hit@1 0.80, MRR 0.84 frente a 0.73 del denso), BM25 llega a hit@4 0.95, multi-query no ayuda (MRR 0.82 con una llamada extra) y el denso empeora con chunks grandes (MRR 0.77 → 0.67).

### 4.3 RAGAS y LLM-as-judge

**Problema.** Buena recuperación es necesaria, no suficiente: el modelo puede ignorar el contexto, añadir hechos o contestar otra pregunta. Y la calidad de texto libre no tiene métrica determinista barata.

**Idea de RAGAS** (Es et al., 2023): descomponer "¿es buena esta respuesta?" en comprobaciones que un juez hace bien una a una.

$$\text{faithfulness} = \frac{\left|\lbrace s \in S(a) : s \text{ se infiere de } c_{1..K} \rbrace\right|}{|S(a)|}, \qquad \text{context recall} = \frac{\left|\lbrace s \in S(g) : s \text{ atribuible a } c_{1..K} \rbrace\right|}{|S(g)|}$$

*Faithfulness* mide fundamentación, no corrección (repetir fielmente un pasaje equivocado puntúa 1). *Answer relevancy* genera preguntas a partir de la respuesta y mide su similitud con la original (y pone 0 si la respuesta es evasiva). *Context precision* es una precisión media sobre el ranking, como MRR pero con un juez decidiendo relevancia. Juntas localizan el fallo: recall bajo apunta a recuperación; recall alto y faithfulness baja, al generador. Faithfulness y relevancy no necesitan referencia (sirven en tráfico real); precision y recall sí.

**Por qué funciona el juez.** Calificar suele ser más fácil que generar. Zheng et al. (2023) encontraron que jueces fuertes coinciden con humanos tanto como los humanos entre sí, y documentaron sesgos:

| Sesgo | Qué ocurre | Mitigación |
|---|---|---|
| posición | en comparaciones gana la primera (o segunda) opción | invertir el orden y promediar |
| verbosidad | lo largo parece mejor | rúbrica que excluya la longitud |
| auto-preferencia | favorece a su propia familia | juez de otra familia o un panel |
| capacidad limitada | acepta respuestas que no sabría resolver | dar una referencia; rúbrica estrecha |
| sensibilidad al prompt | pequeños cambios mueven las notas | congelar y versionar el prompt del juez |
| deriva de escala | escalas 1-10 usadas de forma inconsistente | veredictos binarios; G-Eval pondera por probabilidades |

**La regla de trabajo:** un juez LLM es un instrumento de medida que hay que **validar**. Etiquetar a mano unas decenas de casos y medir acuerdo con kappa de Cohen, $\kappa = (p_o - p_e)/(1 - p_e)$, que corrige el acuerdo por azar; congelar juez y prompt mientras se comparan versiones; leer diferencias más que valores absolutos; poner el razonamiento *antes* del veredicto en la salida estructurada. Un panel de jueces pequeños de familias distintas puede igualar a uno grande (Verga et al., 2024); ARES corrige jueces pequeños con un set humano mediante prediction-powered inference.

**La fuerza del juez.** Un juez no puede verificar lo que no entiende, y un juez débil no solo añade ruido: puede *inventar* efectos, diferencias entre sistemas que no existen, porque sus errores no se reparten por igual entre configuraciones (una respuesta más corta, otro formato o un contexto distinto basta para que falle más en una que en otra). Tres consecuencias prácticas. El juez debe ser claramente más fuerte que el sistema evaluado y, mejor, no ser el mismo modelo que genera (que además trae auto-preferencia). Toda comparación lleva un intervalo (bootstrap pareado, apartado [6.5](#65-estadística-de-sets-de-evaluación-pequeños)): con 20 preguntas, una diferencia de 0.1 en una métrica de RAGAS suele caber en el ruido. Y cambiar de juez cambia el instrumento de medida: los números de dos jueces distintos no se comparan entre sí, solo las diferencias medidas con el mismo juez. Un juez más fuerte cuesta tiempo, así que se reserva para un set fijo, pequeño y representativo.

> **Notebook:** [10 · RAG con LangChain y RAGAS](../../notebooks/04_applications/10_rag.ipynb) y [15 · Observabilidad y evaluación](../../notebooks/06_llmops/15_observability_llmops.ipynb) · **En el curso:** con `qwen2.5:7b` como juez (y `qwen2.5:3b` generando), la única diferencia clara entre recuperación densa e híbrida es context precision, 0.79 → 0.92 (+0.13, IC 95 % [+0.03, +0.25]); faithfulness (0.87 → 0.87), answer relevancy (0.78 → 0.75) y context recall (0.95 → 0.95) no cambian dentro de sus intervalos, y la evaluación tarda 1038 s. Cuando el propio 3B se juzgaba a sí mismo, informaba de faithfulness 0.38 → 0.54 y relevancy 0.44 → 0.30: efectos que el juez de 7B no confirma. En el notebook 15 (todavía con el juez de 3B) el juez coincide con un check determinista en 20 de 24 respuestas, y los desacuerdos muestran qué se le escapa a cada uno.

### 4.4 Patrones de aplicación

**Qué sustituyeron.** Antes de los LLM instruct, cada tarea tenía su pipeline: NER más regex para extraer, un clasificador entrenado con miles de tickets, un resumidor extractivo, una lista negra de palabras. El in-context learning (GPT-3) y el instruction tuning (FLAN, InstructGPT) permitieron que un solo modelo hiciera todo eso a partir de una descripción. El patrón actual es: escribir la tarea como prompt, restringir la salida a un esquema, medir en un set pequeño etiquetado y entrenar un modelo dedicado solo cuando el volumen o la latencia lo justifiquen.

**Salida estructurada.** Un modelo Pydantic con buenas descripciones de campo, más decodificación restringida (apartado 5.2), elimina el código de parseo y cualquier etiqueta inválida. Lo que queda son errores **semánticos** que parecen válidos: un campo mal en un JSON perfecto. Un detalle que cambia resultados: según el método, el modelo puede no *leer* las descripciones de campo (si el esquema solo va al decodificador), así que las reglas conviene repetirlas en el mensaje de sistema.

**Few-shot.** Los ejemplos no actualizan pesos; son condicionamiento. Min et al. (2022) vieron que aleatorizar sus etiquetas daña menos de lo esperado: formato, espacio de etiquetas y distribución de entradas pesan mucho. Zhao et al. (2021) documentaron sesgos de etiqueta mayoritaria, de recencia y de token común. Así que elección, equilibrio y orden de los ejemplos cambian las predicciones: los ejemplos son un cambio de prompt que se evalúa como cualquier otro. Alternativas para clasificación: embeddings más regresión logística, o un encoder pequeño ajustado (SetFit), cuando el volumen y la estabilidad de etiquetas lo justifican.

**Map-reduce para entradas largas.** Muchos servidores **truncan en silencio** un prompt que excede la ventana. Y aunque quepa, la calidad baja antes del límite (*Lost in the Middle*).

| Estrategia | Llamadas | Paralelo | Riesgo principal |
|---|---|---|---|
| stuff (todo en un prompt) | 1 | — | truncado silencioso, lost-in-the-middle |
| map-reduce | $n+1$ (o un árbol) | sí | cada trozo se resume sin contexto; el reduce descarta temas |
| refine (incremental) | $n$ | no | lo posterior sobrescribe lo anterior |

No hay ganador claro en textos largos (BooookScore, Chang et al., 2024); map-reduce gana en tiempo y robustez cuando los trozos son independientes. Evaluar un resumen sin referencia es medir **cobertura** (¿menciona lo que debe?) y **fidelidad** (¿dice solo lo que dice la fuente?).

**Caching.** Tres cosas distintas reciben ese nombre:

| Capa | Qué se reutiliza | Clave | Riesgo |
|---|---|---|---|
| caché de respuesta exacta | la respuesta | prompt + ajustes del modelo | respuestas viejas; claves que omiten ajustes |
| caché semántica | la respuesta | embedding del prompt bajo un umbral | contestar otra pregunta (falso acierto) |
| prefix cache | la KV-cache de un prefijo | el prefijo de tokens | ninguno: la salida se calcula igual |

Con tasa de acierto $h$, $\mathbb{E}[t] = t_{\text{lookup}} + (1-h)\,t_{\text{model}}$. La caché exacta paga con cualquier $h$; la semántica cuesta un embedding en *cada* petición y su umbral es un clasificador binario con su compromiso precisión-recall. Los embeddings acercan textos del mismo tema, así que dos preguntas que difieren en una palabra (laborables/fines de semana, plan pro/básico) están cerca por construcción: negaciones, números y nombres son el punto débil.

**Prompt injection.** No hay un canal separado para los datos: SQL injection se resolvió con consultas parametrizadas que mandan código y datos por separado; un LLM recibe una única secuencia de tokens y está entrenado para seguir instrucciones aparezcan donde aparezcan. La inyección *indirecta* (en contenido recuperado; Greshake et al., 2023) es la peligrosa para RAG y agentes; es LLM01 en el OWASP Top 10 para LLMs.

| Capa | Ejemplos | Límite |
|---|---|---|
| diseño de prompt | delimitadores, "datos, no instrucciones", sandwich, spotlighting | convención que el modelo puede ignorar |
| entrenamiento | instruction hierarchy, StruQ | requiere controlar el entrenamiento |
| detección | clasificadores LLM, Prompt Guard, Llama Guard | atacantes adaptativos, falsos positivos |
| diseño de sistema | mínimo privilegio, confirmación humana, dual-LLM, CaMeL | restringe lo que la aplicación puede hacer |

El consenso: las tres primeras capas reducen la probabilidad de éxito; solo la última acota las consecuencias. Las capas se multiplican, y por eso ninguna es opcional.

**Economía del batch.** $\text{coste} \approx N\,(t_{\text{in}}\,p_{\text{in}} + t_{\text{out}}\,p_{\text{out}})$; los tokens de salida cuestan varias veces más porque decode es secuencial. La ley de Little, $L = \lambda W$, explica cuándo sube el throughput al aumentar la concurrencia: solo si el servidor procesa más peticiones sin que cada una tarde proporcionalmente más (una API alojada, o un motor con continuous batching). Un job serio necesita reintentos con backoff exponencial, `return_exceptions=True`, contabilidad idempotente y, si la latencia no importa, las APIs batch asíncronas con descuento.

> **Notebook:** [13 · Patrones de aplicación](../../notebooks/04_applications/13_modern_llm_applications.ipynb) · **En el curso:** extracción con `json_schema` sin reglas en el prompt acierta 5 de 8 registros completos y con las reglas 7 de 8; few-shot no aporta nada en clasificación (20/20 en ambos); el "stuff" procesó solo ~2.050 de ~4.100 tokens sin error y cubrió 9 de 13 temas, map-reduce 11 de 13; la caché semántica necesita umbral 0.94 para cero falsos aciertos y entonces solo acierta 2 de 10 paráfrasis; la inyección triunfa 9/10 con prompt ingenuo, 10/10 solo delimitando y 4/10 con sandwich, y un clasificador LLM detecta 10/10 con 0/10 falsos positivos en su set.

### 4.5 Sistemas de recomendación

**Problema.** Para un usuario $u$ en un contexto $c$, estimar qué ítem $i$ es lo "correcto" a continuación (verlo, escucharlo, comprarlo, disfrutarlo) y enseñar los pocos mejores. Tres hechos condicionan cualquier sistema real. **Escala**: catálogos de millones de ítems y respuestas en decenas de milisegundos, así que no se puede pasar un modelo caro por cada ítem. **Feedback implícito y sesgado**: casi siempre se observa lo que la gente pulsó o vio, no lo que no le gustó, y solo entre lo que el sistema ya le enseñó (sesgo de exposición); lo popular recibe más exposición y, por tanto, más datos (sesgo de popularidad). **El mundo cambia**: llegan ítems nuevos cada día sin historial (*cold start*) y los gustos derivan.

**El embudo.** La respuesta a la escala, popularizada por el artículo de YouTube (Covington, Adams y Sargin, 2016), es partir el problema en etapas con objetivos distintos:

| Etapa | Entrada → salida | Qué se le pide | Métrica propia | Modelos típicos |
|---|---|---|---|---|
| generación de candidatos | catálogo → cientos | no perder lo bueno; barata por ítem | recall@K del conjunto de candidatos | two-tower + ANN, modelo secuencial, co-ocurrencia ítem-ítem, contenido, populares |
| ranking | cientos → decenas | ordenar con precisión; puede usar muchas features | nDCG@10, hit@10 de la lista final | árboles boosted (LambdaMART), modelos profundos (Wide & Deep, DCN, DLRM) |
| re-ranking | decenas → la página | reglas, diversidad, exploración, explicaciones | lo que intercambia: precisión frente a cobertura o diversidad | MMR, calibración, bandits, reglas de negocio |

Cada fuente de candidatos ve al usuario de otra forma, así que los sistemas toman la **unión** de varias. La consecuencia más útil para depurar: el ranker solo puede ordenar lo que la recuperación trajo, y el recall del conjunto de candidatos es un techo que ningún ranker supera. Mejorar recuperación y mejorar ranking son proyectos distintos, y se miden por separado.

**Recomendación secuencial como modelado de lenguaje.** Si el historial de un usuario es una frase cuyas palabras son ids de ítems, predecir lo siguiente es predecir el siguiente token. SASRec (Kang y McAuley, 2018) es exactamente eso: un transformer decoder-only con atención causal cuyo vocabulario es el catálogo; en el curso, el `TransformerLM` del notebook 02 con `vocab_size = n_items + 1`. Funciona porque el gusto es secuencial (sagas, rachas de un género, estados de ánimo) y la atención pondera el historial como una recencia aprendida y dependiente del contexto. La industria va en esa dirección: HSTU de Meta (Zhai et al., 2024) escala como un LLM, y Netflix describe un modelo fundacional autoregresivo sobre la secuencia de interacciones de cada miembro (2025).

El detalle de entrenamiento que más pesa es la pérdida. El SASRec original usaba una pérdida binaria con un negativo muestreado por positivo. La **softmax completa** sobre el catálogo, la pérdida de un modelo de lenguaje,

$$\mathcal{L} = -\log \frac{\exp(s_{u,i^+})}{\sum_{j \in \mathcal{I}} \exp(s_{u,j})},$$

es mucho más fuerte con la misma arquitectura: Klenitskiy y Vasilev (2023) pasan de hit@10 0.250 a 0.315 en MovieLens-1M. Con unos miles de ítems es barata; con millones, el denominador no cabe y se usa **softmax muestreada**: el denominador se estima con un subconjunto de negativos, a menudo los positivos de los otros usuarios del batch (*in-batch negatives*). Pero esos negativos no salen uniformes: un ítem popular aparece como negativo mucho más a menudo, y el modelo aprende a infravalorarlo. La **corrección logQ** (Yi et al., 2019) resta a cada logit el logaritmo de la probabilidad con que ese ítem se muestrea, $s_{u,j} - \log Q(j)$, y devuelve el estimador hacia la softmax completa.

**Two-tower y búsqueda aproximada.** Dos redes llevan usuario e ítem al mismo espacio y la puntuación es el producto escalar, $s(u, i) = \langle f(u), g(i) \rangle$. Como la torre de ítems no depende del usuario, todos los vectores de ítem se precalculan y se meten en un índice de vecinos aproximados (ANN): servir es calcular un vector de usuario y buscar los más cercanos, milisegundos sobre millones de ítems. Los índices habituales son grafos (**HNSW**), particiones (IVF) y compresión (product quantization), con FAISS o ScaNN; su recall frente a la búsqueda exacta es un parámetro de calidad propio, independiente de lo bueno que sea el modelo. La limitación de diseño: un producto escalar no puede modelar interacciones finas entre features del usuario y del ítem (eso lo hace el ranker). Para *cold start*, la torre de ítems puede usar contenido (texto, género, año); si además se oculta al azar el id durante el entrenamiento (*id dropout*, como DropoutNet, Volkovs et al., 2017), la torre aprende a colocar un ítem solo a partir de su contenido.

**Learning to rank.** El ranker puntúa cada par (usuario, candidato) a partir de features y ordena. Hay tres formas de entrenarlo:

| Enfoque | Pérdida | Ejemplo |
|---|---|---|
| pointwise | predecir cada etiqueta por separado (log-loss sobre "¿fue el siguiente?") | regresión logística, GBDT binario |
| pairwise | preferir el positivo a cada negativo: $\log(1 + e^{-(s_+ - s_-)})$ | RankNet |
| listwise | optimizar la calidad de la lista ordenada entera | LambdaMART, softmax / XE-NDCG |

**LambdaMART** (Burges, 2010) son árboles con gradient boosting cuyos gradientes son los de RankNet para cada par, multiplicados por cuánto cambiaría el nDCG si se intercambiaran los dos ítems:

$$\lambda_{ij} = \frac{-\sigma}{1 + e^{\sigma (s_i - s_j)}}\,\bigl|\Delta \text{nDCG}_{ij}\bigr|,$$

así que los árboles se concentran en la parte alta de la lista. Lleva más de una década como ranker industrial por defecto (buscadores, comercio electrónico, competiciones de recomendación en Kaggle) porque los árboles aceptan features heterogéneas (puntuaciones, conteos, categorías, ratios) sin escalar, entrenan en minutos y se inspeccionan con facilidad; en LightGBM es `objective="lambdarank"`. Dos condiciones deciden si un ranker de segunda etapa aporta algo sobre una primera etapa fuerte. **Muchos grupos de etiquetas de un periodo posterior** al que vieron los modelos de recuperación: si se entrena con un único positivo por usuario, la mejor feature es la propia puntuación de la primera etapa y los árboles reaprenden "ordena por SASRec" más ruido. La receta de RecTools y de las mejores soluciones de Kaggle OTTO es ajustar la recuperación sobre el historial menos los últimos $R$ eventos y convertir cada uno de esos eventos en una consulta del ranker. Y **features que la primera etapa no ve**: valoraciones explícitas, demografía, co-visitas, tiempo desde el evento anterior, tendencias recientes. Sin eso, el ranker es solo coste.

**Cold start y exploración.** Nada colaborativo recomienda un ítem que nadie ha visto: su fila de embedding nunca se entrenó. Lo resuelven dos cosas distintas. El **contenido** coloca el ítem nuevo (texto y metadatos embebidos, torres con *id dropout*). La **exploración** le da exposición para aprender cuánto vale, que es el dilema exploración/explotación: un sistema que solo explota no aprende nunca de lo nuevo. La herramienta de producción son los **bandits**: $\varepsilon$-greedy (explorar al azar una fracción del tiempo), UCB (puntuar por media más un bonus de incertidumbre, $\hat\mu_i + c\sqrt{\ln t / n_i}$) o Thompson sampling (muestrear de la posterior de cada ítem), y sus versiones contextuales (LinUCB; BaRT en la home de Spotify, McInerney et al., 2018). La versión mínima que se puede medir offline es una **ranura de exploración** fija: los primeros puestos para el ranking y uno reservado para el mejor ítem nuevo según un modelo que entienda contenido. Offline solo se ve su coste en ítems conocidos y su alcance en los nuevos; el beneficio de aprender se ve online.

**Diversidad, MMR y calibración.** Una lista de diez casi iguales es precisa y aburrida. **MMR** (Carbonell y Goldstein, 1998) construye la lista de forma voraz, eligiendo cada vez

$$\arg\max_{i \notin S} \; \lambda\,\text{rel}(i) - (1 - \lambda) \max_{j \in S} \text{sim}(i, j),$$

con $\lambda = 1$ el ranking puro y $\lambda$ más bajo más diversidad. La **diversidad intra-lista** se mide como uno menos la similitud media entre pares de la lista, $\text{ILD} = 1 - \frac{2}{k(k-1)} \sum_{i<j} \text{sim}(i, j)$. La **calibración** (Steck, 2018) pide otra cosa: que la mezcla de géneros de la lista $q$ se parezca a la del historial $p$, penalizando $\mathrm{KL}(p \,\|\, q)$, para que quien ve un 70 % de dramas y un 30 % de comedias no reciba diez dramas. Una advertencia: hit@10 premia acertar *el* siguiente ítem, así que cualquier cambio de la lista solo puede parecer neutro o peor offline; los beneficios (menos fatiga, descubrimiento, retención) se ven en tests A/B de semanas, y por eso esta perilla se ajusta online.

**Protocolo de evaluación offline.** Cuatro decisiones separan una evaluación honesta de una que engaña:

| Decisión | Opción honesta | Qué falla con la otra |
|---|---|---|
| partición | temporal: *leave-last-out* por usuario (el último evento es test, el anterior validación) o un corte global en el tiempo | una partición aleatoria filtra el futuro al entrenamiento |
| candidatos de la métrica | **ranking completo** sobre todo el catálogo | las métricas muestreadas (el objetivo contra 100 negativos aleatorios) inflan los números y pueden invertir qué modelo parece mejor (Krichene y Rendle, 2020) |
| agregación | por segmentos: ítems conocidos frente a nuevos, usuarios con poco o mucho historial | una media global esconde el *cold start* por completo |
| comparación | **bootstrap pareado** sobre usuarios, intervalo del 95 % de la diferencia | uno o dos puntos de hit@10 con unos miles de usuarios pueden ser ruido |

Con un único ítem relevante, hit@k es la fracción de usuarios cuyo objetivo está en el top $k$ y nDCG@k vale $1/\log_2(p+1)$ para un acierto en la posición $p$ (1 en la primera, 0.5 en la tercera, 0.29 en la décima). Una regla de trabajo útil: **una etapa solo entra en el sistema si mejora a la anterior en validación**, con un intervalo que excluya el cero; el test se mira una vez, al final. Y lo offline predice lo online solo de forma imperfecta: la decisión final es un test A/B.

**Dónde ayudan los LLMs y dónde no.** La evidencia publicada y la de este curso apuntan en la misma dirección:

| Uso del LLM | Veredicto | Por qué |
|---|---|---|
| re-ordenar las recomendaciones de un modelo colaborativo fuerte | no compensa con modelos pequeños | el LLM no ve la señal de co-visionado que codifica el modelo, solo descripciones cortas; cuesta segundos por usuario frente a milisegundos |
| enriquecer ítems offline (palabras clave, descripciones; KAR, Xi et al., 2023; LLMRec, Wei et al., 2024) | depende; hay que medirlo | funciona como feature aparte en modelos entrenados y con LLMs grandes; mezclado en el texto del ítem puede diluir la señal |
| entender peticiones en lenguaje natural (salida estructurada → filtros) | ayuda claramente | fechas, negaciones y restricciones duras son justo lo que un embedding no sabe imponer |
| explicar ("porque viste X e Y") | ayuda si está anclado | con las películas del historial que motivaron la recomendación es verificable; sin ancla inventa atributos |

La división de trabajo resultante: el LLM entiende *qué* se pide y lo cuenta; el historial del usuario y los modelos colaborativos deciden *cuál*. La línea de investigación que intenta fundir ambas cosas es la recuperación generativa con *semantic IDs* (TIGER, Rajput et al., 2023), en la que el modelo genera el identificador del ítem token a token.

> **Notebook:** [12 · Sistemas de recomendación](../../notebooks/04_applications/12_recommender_systems.ipynb) · **En el curso:** en MovieLens-1M con ranking completo, SASRec entrenado con softmax completa llega a hit@10 0.293 (ItemKNN 0.125, popularidad 0.036) y nunca acierta una película nueva; la unión de cinco fuentes recupera el objetivo para el 68.4 % de los usuarios; LambdaMART con la receta de RecTools/OTTO sube a 0.307 (+1.4 puntos, IC 95 % [+0.7, +2.1]) tras pasar la puerta de validación, que el primer intento no pasó; una ranura de exploración lleva el hit@10 en películas nuevas de 0.009 a 0.044 por 0.7 puntos globales; MMR con $\lambda = 0.8$ sube la diversidad de 0.659 a 0.719 por 0.4 puntos; las palabras clave del LLM empeoran el contenido y no aportan como feature, pero el LLM que traduce peticiones a filtros cumple las restricciones en el 87 % del top 5 frente al 60 % de la búsqueda por embeddings.

---

## 5. Agentes y MCP

Términos de esta parte: [glosario, Agentes y LangGraph](glosario.md#agentes-y-langgraph) y [MCP](glosario.md#mcp).

### 5.1 Tool calling

**Problema.** Un modelo preentrenado tiene cuatro límites que ningún prompt quita: conocimiento con fecha de corte, ningún acceso a tus datos privados, poca fiabilidad en trabajo simbólico exacto (aritmética, fechas, contar) y ninguna capacidad de *actuar*. Las herramientas atacan los cuatro, siempre que el modelo pueda expresar una petición que un programa ejecute. La dificultad real es el puente: el modelo emite tokens, el programa necesita un nombre de función y argumentos tipados.

**Idea.** Cada herramienta se describe con nombre, descripción y un **JSON Schema** de argumentos. El modelo responde con texto o con una llamada estructurada `{name, args, id}`; tu código valida, ejecuta y devuelve un mensaje de herramienta con el mismo `id`. El modelo nunca ejecuta nada. El esquema tiene doble función: es *documentación* para el modelo (lo único que sabe de la herramienta) y *validador* para tu código. Por eso una descripción vaga produce llamadas malas igual que un prompt vago produce respuestas malas.

**Lo que sustituyó.** MRKL y ReAct (2022) pedían al modelo líneas `Action:` que se parseaban con cadenas y se rompían con una palabra de más. Toolformer y Gorilla entrenaron el uso de APIs. Desde 2023 la industria usa tool calling **nativo**: el formato forma parte del entrenamiento del modelo (Qwen2.5 escribe las llamadas como JSON dentro de etiquetas `<tool_call>`) y la capa de serving lo convierte en datos estructurados. El entrenamiento enseña cuatro habilidades que fallan por separado: **decidir** si hace falta una herramienta, **elegir** cuál, **rellenar** los argumentos (el *slot filling* clásico) y **usar** el resultado.

**Un agente es un bucle con estructura de coste.** El modelo es una función sin estado de una lista de mensajes a un mensaje nuevo, así que cada llamada reenvía toda la conversación y los esquemas. Con prompt fijo de $P$ tokens y $g$ tokens añadidos por paso, $k$ pasos cuestan aproximadamente

$$\text{tokens de entrada} \approx kP + g\,\frac{k(k-1)}{2}$$

lineal en el prompt fijo y cuadrático en el historial. El número de pasos, no la longitud de la respuesta, domina coste y latencia; cada herramienta añadida cuesta tokens en cada petición la use o no.

**Errores como observaciones.** Los modelos encuentran mal sus propios errores sin ayuda (Huang et al., 2024), pero corrigen razonablemente cuando un verificador externo se los señala (CRITIC). Un validador de esquema es justo ese verificador: el mensaje de error es parte del diseño. Los errores de validación se devuelven al modelo; los de ejecución, solo si se decide, porque pueden filtrar rutas, SQL o secretos.

**Evaluación.** Se evalúa la **trayectoria**, no solo el texto final: precisión de selección de herramienta (incluido "ninguna" para charla), validez de argumentos, éxito de la tarea, latencia y tokens. Es la descomposición del Berkeley Function-Calling Leaderboard (AST y ejecución, más una categoría de "no llamar"); τ-bench añade *pass^k*, la probabilidad de acertar en las $k$ repeticiones, que expone la inconsistencia.

| Fallo | Mitigación |
|---|---|
| decisión equivocada (responde de memoria o llama para charlar) | descripciones claras, system prompt, evaluación |
| herramienta equivocada | menos herramientas por prompt, nombres distintos, enrutado |
| argumentos malos | validación de esquema, error devuelto al modelo, normalizar dentro de la herramienta |
| bucles | límites de pasos y gasto |
| inyección vía resultado | tratar la salida como datos, confirmación para acciones |
| agencia excesiva | mínimo privilegio, solo lectura por defecto |

> **Notebook:** [11 · Tools y salida estructurada](../../notebooks/05_agents_and_mcp/11_agents_tools_structured_output.ipynb) · **En el curso:** el modelo calculó `(120 - 45.5) / 2` "de cabeza" sin llamar a la calculadora (acertó, pero la trayectoria es incorrecta), usó `search_docs` para una charla y se saltó un paso en una petición multi-paso; los esquemas cuestan unos 390 tokens de prompt por llamada antes de cualquier conversación, y con 14 casos cada uno vale ~7 puntos.

### 5.2 Decodificación restringida y salida estructurada

**Tres niveles de garantía.**

| Nivel | Mecanismo | Qué se garantiza |
|---|---|---|
| prompting | "responde en JSON con las claves..." | nada; normalmente cumple |
| JSON mode | el decodificador solo permite tokens que mantienen un JSON válido | JSON sintácticamente válido, cualquier clave |
| decodificación restringida por esquema | solo tokens que mantienen la salida válida *para tu esquema* | claves, tipos y valores de enum correctos |

**Idea.** El esquema se compila en una gramática (expresión regular o gramática libre de contexto para el anidamiento) y, en cada paso, se aplica una máscara a los logits:

$$p(y_t \mid y_{<t}) \;\propto\; \exp\big(z_{t,y_t} + m_t(y_t)\big), \qquad m_t(v) = \begin{cases} 0 & \text{si } y_{<t}\,v \text{ es prefijo válido} \\ -\infty & \text{si no} \end{cases}$$

Todo lo muestreado es, por construcción, prefijo de un documento válido. Lo difícil es la eficiencia: la gramática trabaja con caracteres, el modelo con tokens que abarcan varios símbolos, y el vocabulario tiene ~150.000 entradas. Outlines (Willard y Louf, 2023) precalcula, por estado de un autómata finito, los tokens permitidos; XGrammar (Dong et al., 2024) lo extiende a gramáticas libres de contexto; llama.cpp (debajo de Ollama) lo hace con GBNF. Algunos motores son incluso *más rápidos* con gramática porque emiten tokens forzados sin ejecutar el modelo.

**Lo que no garantiza.** El *contenido* (un campo perfectamente tipado puede estar mal o ser `null`); una distribución sin sesgo (la máscara se aplica token a token y empuja al modelo por caminos que consideraba improbables; Park et al., 2024); la misma calidad de razonamiento (Tam et al., 2024 vieron caídas con formatos estrictos; otros lo atribuyen al prompt; **en debate**). La práctica en la que coinciden ambos lados: poner un campo `reasoning` de texto libre *antes* de los campos de respuesta, porque las claves se generan en orden. Las reglas de negocio ("como máximo 12 palabras") quedan fuera de cualquier decodificador: validación en tu código, reintento *con el error* (un reintento ciego a temperatura 0 repite lo mismo), número máximo de intentos y un plan B.

> **Notebook:** [11 · Tools y salida estructurada](../../notebooks/05_agents_and_mcp/11_agents_tools_structured_output.ipynb) · **En el curso:** los tres métodos de `with_structured_output` producen un `Ticket` válido en los 8 correos, pero `json_schema` devuelve `order_id = null` en todos los que mencionan pedido mientras `function_calling` y `json_mode` lo extraen: la forma está garantizada, el contenido no; el reintento con el error baja el resumen de 22 a 13 palabras sin llegar al límite de 12.

### 5.3 ReAct, plan-and-execute y reflexión

**Qué hace "agéntico" a un sistema.** La definición útil es de control de flujo: un sistema es agéntico en la medida en que **un modelo decide qué pasa después**. Es un espectro: llamada única, workflow fijo, router, bucle de herramientas, agente autónomo con sus propios sub-objetivos. Cada escalón compra flexibilidad y paga en latencia, coste, varianza y dificultad para testear. La recomendación de la industria (Anthropic, *Building effective agents*, 2024) es usar el diseño más simple que funcione y añadir autonomía solo donde una evaluación muestre que ayuda.

**ReAct.** Antes había dos líneas separadas: chain-of-thought (razona con lo que hay en los pesos; un error temprano se propaga) y agentes solo-acción (llaman herramientas sin un sitio donde descomponer el objetivo). ReAct (Yao et al., 2022) amplía el espacio de acciones con *pensamientos*, acciones que solo cambian el propio contexto. El bucle se retroalimenta en los dos sentidos: razonar mejora la elección de la acción y cada observación ancla el siguiente pensamiento en datos reales. El tool calling nativo conserva el bucle y cambia la superficie: la línea `Action` es un campo `tool_calls` estructurado y el pensamiento es texto visible, tokens de razonamiento ocultos o nada.

**Plan-and-execute.** ReAct decide paso a paso con todo el historial: el contexto crece y el modelo puede perder el objetivo. Separar planificador y ejecutor (Plan-and-Solve; ReWOO, que escribe el plan completo con huecos para los resultados; LLMCompiler, que lo trata como grafo de dependencias y paraleliza):

| | ReAct | Plan-and-execute |
|---|---|---|
| Llamadas al modelo | una por paso, menos con llamadas paralelas | planificador + al menos una por paso + redactor |
| Se adapta a sorpresas | en cada paso | solo con un paso de re-planificación |
| Contexto por llamada | crece con todo el historial | cada ejecutor ve su paso |
| Inspección | la traza, después | el plan se puede ver, editar o aprobar antes |
| Fallo típico | se salta un paso, pierde el objetivo | pasos redundantes, información perdida en los traspasos |

Ninguno es mejor en general.

**Reflexión y el hueco generación-verificación.** Self-Refine (el mismo modelo genera, critica y revisa), Reflexion (convierte una señal de fallo en una lección escrita que se antepone al siguiente intento: RL donde los "pesos" son palabras) y CRITIC (verifica con herramientas). El patrón que se sostiene en toda esta literatura: comprobar suele ser mucho más fácil que producir, así que un verificador barato y fiable más unos pocos reintentos mejora a un generador que acierta a veces. Cuando el verificador es el mismo modelo juzgándose sin información externa, la ganancia a menudo desaparece o se invierte (Huang et al., 2024). Coste esperado: $1 + \sum_{r \ge 2} P(\text{se llega a la ronda } r)$ generaciones por tarea; la reflexión sale barata cuando casi todo pasa a la primera, que es justo cuando menos ayuda.

**Errores compuestos.** Si cada paso acierta con probabilidad $p$, una tarea de $k$ pasos acierta con $p^k$; $0.95^{10} \approx 0.60$. Las ejecuciones largas necesitan pasos muy fiables o puntos de control donde se detecten errores.

**Multi-agente.** Prompts más pequeños, menos herramientas por decisión, trabajo en paralelo; pero más llamadas y traspasos donde se pierde contexto. Los practicantes discrepan públicamente (Anthropic describió en 2025 un sistema de investigación multi-agente mejor que uno solo a mucho más coste en tokens; Cognition defendió ese mismo año agentes de un solo hilo). La discrepancia es sobre todo de tipo de tarea. La forma menos controvertida es un **router delante de especialistas**, y un router es un **clasificador**: set etiquetado, accuracy, matriz de confusión; un router que falla 1 de cada 10 limita el sistema a ~90 % aunque los especialistas sean perfectos.

> **Notebook:** [14 · IA agéntica con LangGraph](../../notebooks/05_agents_and_mcp/14_agentic_ai.ipynb) · **En el curso:** con `qwen2.5:3b`, el agente ReAct supera a plan-and-execute con aproximadamente un tercio de las llamadas (plan-and-execute tiene más recall de herramientas y menos precisión); la reflexión con tests unitarios no reparó nada en una ejecución (4/8 antes y después) y alguna tarea en otras; el router acierta entre 67 y 75 % según la ejecución.

### 5.4 LangGraph: agentes como máquinas de estado

**Problema.** En 2023 los agentes más visibles (AutoGPT, BabyAGI) eran bucles abiertos: ejecuciones que no terminaban, acciones repetidas, objetivos a la deriva, costes impredecibles, imposible testear una parte o reanudar tras un fallo.

**Idea.** Devolver el control de flujo al código y dar al modelo control solo en puntos concretos. Una aplicación es un **grafo** sobre un estado tipado: los nodos leen el estado y devuelven actualizaciones; las aristas (fijas o condicionales) deciden el siguiente nodo; un **reducer** por clave combina actualizaciones (por ejemplo, `add_messages` añade en vez de sobrescribir). El runtime sigue el modelo **Pregel / bulk synchronous parallel** (Malewicz et al., 2010): en cada *super-step* se ejecutan todos los nodos activados, sus actualizaciones se fusionan con los reducers (eso hace seguras las ramas paralelas) y las aristas deciden el siguiente super-step.

**Propiedades que da.** Transiciones explícitas que se testean con modelos falsos; bucles visibles donde poner límites; **checkpoints** (el estado solo cambia entre nodos, así que se guarda tras cada paso: reanudar, pausar para un humano el tiempo que haga falta, "viajar en el tiempo"); y autonomía mixta (una rama guionizada junto a un bucle ReAct libre). `create_agent` es un grafo de dos nodos (`model`, `tools`) con una arista condicional que codifica toda la regla de parada de ReAct: si el último mensaje tiene llamadas, ir a `tools`; si no, terminar.

> **Notebook:** [14 · IA agéntica con LangGraph](../../notebooks/05_agents_and_mcp/14_agentic_ai.ipynb) · **En el curso:** un grafo de triaje con funciones puras para ver la maquinaria, el agente ReAct, plan-and-execute, reflexión y un router con sub-grafos, todos testeados con modelos falsos en `tests/test_agents_langgraph.py`.

### 5.5 Memoria

La memoria de un agente se describe con categorías de la ciencia cognitiva (CoALA, Sumers et al., 2023):

| Tipo | Qué guarda | En LangGraph |
|---|---|---|
| de trabajo | la ventana de contexto actual | los `messages` del estado |
| episódica | lo que pasó en interacciones anteriores | checkpoints de otros hilos o resúmenes |
| semántica | hechos sobre el usuario y el mundo | el store, con búsqueda por embedding |
| procedimental | cómo hacer las cosas | el prompt y el propio código |

La memoria a corto plazo es el hilo de conversación guardado por un checkpointer con `thread_id`; crece con cada turno y cada llamada la reenvía, de ahí la summarization o el recorte. La de largo plazo es un store por namespaces con búsqueda semántica. Generative Agents puntúa recuerdos por recencia, importancia y relevancia; MemGPT pagina memoria como un sistema operativo. La restricción de fondo es que el contexto es limitado y caro y se usa de forma desigual: **la memoria a largo plazo es un problema de recuperación**, con todos sus modos de fallo, y se mide como tal (recall). La mitad difícil es *escribir*: qué merece guardarse, deduplicación, actualizaciones, privacidad; un hecho erróneo o inyectado se recupera en todas las conversaciones posteriores.

### 5.6 Límites y human-in-the-loop

**Por qué los agentes no paran.** En ReAct la única salida es que el modelo decida no llamar a una herramienta, y nada en su entrenamiento lo garantiza: puede leer un resultado como incompleto, reintentar con los mismos argumentos o seguir buscando porque una búsqueda más siempre parece útil. La terminación se impone desde fuera, como un servidor impone timeouts. Capas, de más gruesa a más fina: límite de recursión (super-steps; un backstop que lanza excepción), límites de llamadas al modelo y a herramientas (el presupuesto en las unidades que pagas, que terminan con un mensaje), límite de tiempo o de coste por middleware. Principio general de fiabilidad: la primera línea degrada con elegancia; la última no se puede saltar. Cada ejecución termina con una **razón de parada** que se cuenta y se alerta.

**Puertas de aprobación en código.** Las herramientas de solo lectura corren libres; las que mueven dinero, envían correo o borran datos esperan a una persona. Un system prompt que dice "pregunta antes de reembolsar" es una instrucción que el modelo puede ignorar o de la que un documento inyectado puede disuadirle; un `interrupt` en el grafo es código por el que el camino a la herramienta pasa físicamente. Es el principio de **mínima agencia**: el modelo propone, un control determinista decide. Pausar exige **persistencia**: una persona puede responder en segundos o en días desde otra máquina, así que el estado se escribe en almacenamiento (la misma idea que Temporal o Step Functions). Detalle crucial: al reanudar, LangGraph vuelve a ejecutar el nodo interrumpido desde su primera línea, así que todo efecto secundario antes del `interrupt` se ejecuta dos veces; va en un nodo posterior o debe ser idempotente.

> **Notebook:** [14 · IA agéntica con LangGraph](../../notebooks/05_agents_and_mcp/14_agentic_ai.ipynb) · **En el curso:** un modelo falso que pide la misma herramienta para siempre solo se detiene por el límite de recursión (con excepción) sin middleware, y con middleware termina con mensaje y razón; el límite de recursión por defecto de la versión instalada es 10.007, por eso cada ejecución pasa el suyo; la aprobación de reembolsos se decide con una política (hasta 40 dólares) fuera del modelo.

### 5.7 Model Context Protocol

**Problema.** Varias aplicaciones de IA (asistente, plugin del IDE, bot de soporte, agente interno) y muchos sistemas que alcanzar (wiki, tickets, CRM, bases de datos). Si cada aplicación escribe su conector para cada sistema, el trabajo crece como $N \times M$, cada conector se mantiene aparte y se comporta distinto.

**Idea.** La respuesta clásica a un problema $N \times M$ es un **estándar en medio**: $N + M$ piezas en lugar de $N \times M$, como ODBC/JDBC para bases de datos o el Language Server Protocol (2016) para editores e idiomas, del que MCP toma la estructura explícitamente: JSON-RPC 2.0, sesión con estado que empieza con `initialize` y negociación de capacidades, subproceso stdio como transporte local.

```
HOST (Claude Desktop, un IDE, tu agente LangGraph)   <- frontera de confianza: aquí vive la política
  modelo <-> lógica del host (aprobaciones, contexto)
     cliente A ──stdio──► servidor A (ficheros)
     cliente B ──stdio──► servidor B (git)
     cliente C ──Streamable HTTP──► servidor C (SaaS remoto)
```

Un **cliente por servidor** es una decisión de aislamiento: un servidor no ve a los demás ni la conversación completa. Las tres primitivas tienen dueños distintos, y esa asignación es el diseño: **tools** las controla el modelo (ahí se concentra el riesgo y ahí van las aprobaciones), **resources** la aplicación (decide qué entra al contexto), **prompts** el usuario (plantillas explícitas, como comandos de barra). En el lado cliente, *sampling* permite a un servidor usar el modelo del host sin tener su propia clave, *elicitation* pedir datos al usuario y *roots* preguntar en qué carpetas puede trabajar. La versión se negocia por fecha y las capacidades no declaradas no se usan, lo que permite añadir funciones sin romper implementaciones antiguas.

**Por qué no basta con OpenAPI y function calling.** Pregunta justa y todavía debatida (los plugins de ChatGPT de 2023 hacían justo eso y se retiraron en 2024). MCP añade un transporte local sin puertos ni autenticación, sesión con capacidades negociadas, comunicación en ambos sentidos (notificaciones, progreso, peticiones del servidor al cliente) y primitivas pensadas para modelos. Los contraargumentos también son reales: otro protocolo que asegurar y operar, OAuth para servidores remotos, y para una sola aplicación llamando a una sola API documentada, una llamada directa es más simple. Úsalo cuando la misma capacidad deba llegar a varios hosts o quieras el ecosistema existente. La adopción es lo que da valor a un protocolo: Anthropic lo publicó en noviembre de 2024, OpenAI, Google y Microsoft lo soportaron en 2025 y en diciembre de 2025 pasó a la Agentic AI Foundation de la Linux Foundation.

**Dos canales de error.** Un *error de herramienta* es un resultado normal con `isError: true` y va al **modelo** como observación para que se corrija; un *error de protocolo* es un objeto `error` de JSON-RPC con código numérico, es un bug del **cliente** y nunca llega al modelo. Las implementaciones difieren en detalles, así que el servidor real se prueba con **tests de contrato**.

**Coste.** El transporte es barato (milisegundos frente a segundos de inferencia); arrancar un proceso y el handshake son lo caro, así que las sesiones se reutilizan. El coste real es el **contexto**: cada herramienta conectada envía nombre, descripción y esquema en cada llamada, y demasiadas herramientas empeoran la selección; los hosts responden con allowlists, subconjuntos por tarea o búsqueda de herramientas.

**Seguridad: los servidores son código no confiable y texto no confiable.**

| Fallo | Descripción | Control |
|---|---|---|
| tool poisoning | instrucciones ocultas en la *descripción* de una herramienta | revisar servidores, fijar versiones, escanear descripciones |
| rug pull | el servidor cambia una descripción tras ser aprobado | re-confirmar cambios, fijar versiones |
| shadowing | un servidor redefine o instruye sobre herramientas de otro | namespaces por servidor |
| inyección indirecta | instrucciones en los datos devueltos | tratar resultados como datos, confirmación para escribir |
| confused deputy | reenviar un token emitido para otra cosa | aceptar solo tokens propios (RFC 8707); prohibido reenviarlos |
| la "trifecta letal" | datos privados + contenido no confiable + canal de salida (Willison, 2025) | quitar al menos una de las tres |

Las anotaciones (`readOnlyHint`, `destructiveHint`) son *pistas*: un servidor malicioso puede mentir. Las defensas tienen que estar fuera del modelo: mínimo privilegio, confirmación en código para todo lo que no sea de solo lectura, y diseñar para que una inyección con éxito no pueda hacer mucho daño.

> **Notebook:** [16 · MCP con el SDK oficial](../../notebooks/05_agents_and_mcp/16_mcp_model_context_protocol.ipynb) · **En el curso:** los 14 checks de conformidad pasan; `search_docs` acierta 6/6 en top-1 en un set fácil; el agente responde con el dato clave en 6 de 6 y cita el fichero correcto en 5 de 6; un `tools/call` cuesta ~1.4 ms en memoria y ~2.7 ms por stdio, frente a 2-3 s por llamada con el adaptador que abre una sesión nueva cada vez.

---

## 6. LLMOps

Términos de esta parte: [glosario, LLMOps y LangSmith](glosario.md#llmops-y-langsmith).

### 6.1 Trazas, spans y OpenTelemetry GenAI

**Problema.** La monitorización clásica supone que una petición triunfa o falla y que el éxito es correcto. Los LLM rompen eso: los fallos son **semánticos** (una alucinación devuelve HTTP 200), el comportamiento es no determinista y depende de la entrada, una petición son varios pasos (recuperación, varias llamadas al modelo, herramientas), el coste es proporcional a tokens que cambian con cada release, y la calidad es una distribución, no un bit. LLMOps une dos tradiciones: el **tracing distribuido** de SRE (qué pasó, dónde se fue el tiempo) y la **evaluación** de ML (cómo de buenas son las salidas).

**Idea.** El modelo traza/span viene de Dapper (Sigelman et al., 2010): una **traza** es el registro de una petición, un árbol de **spans** temporizados con padre. El mecanismo clave es la **propagación de contexto**: el span padre pasa el id de traza y el suyo propio, de modo que el hijo se cuelga del sitio correcto incluso entre hilos, procesos o servicios. Dapper llevó a Zipkin, OpenTracing y OpenCensus, que se fusionaron en 2019 en **OpenTelemetry**. Un *run* de LangSmith es un span con campos específicos (mensajes, tokens, `run_type`); Langfuse y Arize Phoenix tienen modelos muy parecidos. Las subidas van en lotes desde un hilo de fondo: el tracing casi no añade latencia, y un script que termina enseguida debe hacer *flush*.

**Convenciones semánticas GenAI de OTel.** Una traza solo es portable si todos nombran igual las cosas: `gen_ai.operation.name`, `gen_ai.provider.name` (antes `gen_ai.system`), `gen_ai.request.model`, `gen_ai.usage.input_tokens`... Estaban marcadas como *en desarrollo* y los nombres han cambiado entre versiones, así que se comprueba la versión antes de construir dashboards. Su valor es estratégico: instrumentar con ellas permite enviar las trazas a LangSmith, Langfuse, Phoenix, Datadog o Grafana sin tocar código.

**Señales.** Trazas (estructura y tiempos), métricas (agregados baratos para alertar), logs (eventos) y, propio de LLMs, **feedback**: una puntuación pegada a una traza después (usuarios, heurísticas, jueces, revisores). El feedback convierte trazas en señal de calidad y es el puente entre monitorización y evaluación.

### 6.2 Evaluación offline y online

| | Offline | Online |
|---|---|---|
| Cuándo | antes de publicar un cambio | continuamente, en producción |
| Datos | dataset fijo y etiquetado | peticiones reales, sin referencia |
| Evaluadores | con referencia (exact match, hechos clave, juez con referencia) | sin referencia (fundamentación, formato, juez, feedback de usuario) |
| Pregunta | ¿es B mejor que A en los casos que importan? | ¿está sana la producción hoy y qué trazas debe leer una persona? |
| Riesgo | el dataset deja de representar el tráfico | las señales baratas son ruidosas; las caras solo en muestra |

Se alimentan entre sí: las trazas malas, revisadas y etiquetadas, entran en el dataset, que así sigue lo que los usuarios preguntan de verdad. Ese bucle es el flujo de trabajo central de LLMOps.

Online, sin referencias, se elige en una **pirámide** de coste frente a fidelidad: checks deterministas en el 100 % del tráfico (formato, longitud, solapamiento léxico), modelos pequeños (NLI, similitud de embeddings) en la mayoría, juez LLM en una muestra, señales implícitas y pulgares de usuario (gratis pero escasas y sesgadas hacia los descontentos), y revisión humana de las peores. Las heurísticas baratas sirven para **ordenar qué trazas abrir**, no para calificar. Offline, el exact match mide redacción, no corrección, en texto libre; los checks de hechos clave son deterministas pero ciegos a redacciones imprevistas; el juez LLM cubre el resto si se valida (apartado 4.3).

> **Notebook:** [15 · Observabilidad y evaluación](../../notebooks/06_llmops/15_observability_llmops.ipynb) · **En el curso:** la versión con RAG sube la corrección juzgada de 0.17 (closed book) a 0.83 y el hecho clave de 0.17 a 1.00, mientras el exact match (0.00 frente a 0.58) solo mide cuánto se copia la redacción; la heurística de fundamentación (media 0.81) encontró en tráfico dos bugs reales del prompt de la herramienta de pedidos que el set offline no podía ver.

### 6.3 Percentiles y SLOs

**Por qué percentiles.** La latencia nunca es normal: está acotada por abajo y tiene cola larga. Para una llamada al LLM,

$$T \;\approx\; T_{\text{cola}} + T_{\text{prefill}} + n_{\text{out}} \cdot T_{\text{por token}}$$

y como la longitud de salida varía mucho y las colas llegan en ráfagas, la cola es larga por construcción. La media la arrastran unos pocos casos lentos y no dice nada de ellos; p50 es la experiencia típica, p95 "1 de cada 20 es más lenta", p99 la cola que dispara timeouts. La cola pesa más de lo que parece (Dean y Barroso, 2013, *The Tail at Scale*): si una acción necesita 100 llamadas y cada una es lenta el 1 % de las veces, la acción es lenta con probabilidad $1 - 0.99^{100} \approx 63\%$, y los agentes tienen justo esa estructura. Dos consecuencias: los percentiles necesitan muchas muestras (el p95 de 18 peticiones lo deciden una o dos), y **no se promedian** (la media de p95 horarios no es el p95 diario), por eso se guardan histogramas o sketches fusionables (HDR, t-digest, DDSketch).

**SLI, SLO, SLA y error budget** (libro de SRE de Google, 2016). Un SLI es una medida (fracción de eventos buenos), un SLO el objetivo sobre una ventana, un SLA un contrato con consecuencias. Un objetivo del 99 % *permite* explícitamente un 1 % de eventos malos: mientras queda presupuesto, se publican cambios arriesgados; agotado, primero la fiabilidad. Se alerta por **burn rate** en una ventana corta y otra larga a la vez, para que un pico breve no despierte a nadie y uno sostenido sí. Los LLM añaden SLIs propios: TTFT, tokens o coste por petición, fracción que pasa un check de fundamentación o formato, tasa de rechazos y fracción de ejecuciones que terminan por límite de pasos.

> **Notebook:** [15 · Observabilidad y evaluación](../../notebooks/06_llmops/15_observability_llmops.ipynb) · **En el curso:** 18 peticiones con 11.1 % de error (dos pedidos inexistentes, la herramienta falla 2 de 6) y 1605 tokens de entrada frente a 270 de salida, así que el contexto recuperado, no la respuesta, es la palanca de coste; p50/p95 de 0.22 s / 2.8 s con la GPU libre y mucho más con la GPU compartida, por lo que el SLO de latencia unas ejecuciones pasa y otras no: con 18 peticiones el p95 es casi la segunda más lenta.

### 6.4 PII y muestreo de trazas

**PII.** Las trazas son una copia de datos de usuario con otro propósito (depurar y evaluar) y otro público (ingenieros, proveedores), y el RGPD las trata como cualquier copia: minimización, limitación de finalidad y de conservación.

| Técnica | Qué hace | Qué conserva |
|---|---|---|
| eliminar (`hide_inputs` / `hide_outputs`) | el contenido no se envía | tiempos, tokens, errores |
| redactar / enmascarar | sustituye valores detectados por una etiqueta (`<email>`) | el texto alrededor, la traza sigue legible |
| seudonimizar | sustituye por un token consistente o un hash con clave | agrupar por usuario sin saber quién es; legalmente sigue siendo dato personal |

Detectar es lo difícil: las regex son rápidas y precisas para estructuras fijas (correos, tarjetas con Luhn, teléfonos, claves con prefijo) y ciegas a todo lo demás; los modelos NER (Presidio combina ambos) encuentran nombres y direcciones con latencia y errores. El recall nunca es 100 %, otra razón para limitar quién lee trazas y cuánto se guardan. Y decidir *dónde* se redacta: en el cliente de tracing protege el almacén de trazas, no al proveedor del modelo, que sigue recibiendo el texto original.

**Muestreo.** Con mucho tráfico no hacen falta todas las trazas.

| | Head-based | Tail-based |
|---|---|---|
| Se decide | al empezar la traza | al terminar |
| Basado en | id de traza o un sorteo | cualquier cosa: errores, latencia, feedback |
| Coste | despreciable | hay que guardar todos los spans hasta que acaba (un collector) |
| ¿Conserva todos los errores? | no | sí, si la política lo dice |

La decisión es por traza completa (una traza a medias deja árboles con huecos). El muestreo uniforme mantiene las métricas agregadas sin sesgo (los conteos se recuperan dividiendo por la tasa); lo que sufre son los eventos raros. El compromiso habitual es una fracción pequeña de todo más todos los errores y todo el feedback negativo, que exige tail-based o una vía de log aparte.

> **Notebook:** [15 · Observabilidad y evaluación](../../notebooks/06_llmops/15_observability_llmops.ipynb) · **En el curso:** el anonymizer sustituye correo, teléfono, tarjeta y clave de API en la raíz y en todos los spans hijos, y deja pasar un nombre como "Ana"; con tasa 0.25 sobre 40 trazas se conservaron 7, 12 y 14 en distintas ejecuciones (desviación típica teórica de ~2.7).

### 6.5 Estadística de sets de evaluación pequeños

Todo el curso evalúa con sets de 8 a 40 elementos, y eso tiene consecuencias que conviene saber calcular.

**Error típico de una proporción.** Con $n$ ejemplos y tasa medida $\hat p$:

$$\mathrm{SE} \approx \sqrt{\frac{\hat p(1-\hat p)}{n}} \;\le\; \frac{0.5}{\sqrt{n}}$$

Con $n = 12$ el intervalo del 95 % abarca aproximadamente ±0.28; con $n = 100$, ±0.10. En un set de 40, cada elemento vale 2.5 puntos; en uno de 20, 5. Un set de una docena de ejemplos puede mostrar un efecto grande (una capacidad que una versión tiene y la otra no), no una diferencia de pocos puntos.

**Diseño pareado.** La mayor parte de la variación en una puntuación viene de lo difícil que es cada ejemplo; comparar dos versiones sobre los **mismos** ejemplos la cancela. Para puntuaciones binarias, el test de McNemar solo mira los **pares discordantes** (uno acierta y el otro no); los que ambos aciertan o fallan no dicen nada sobre cuál es mejor. Para puntuaciones continuas, el **bootstrap pareado**: remuestrear con reemplazo las diferencias por pregunta (por ejemplo 2.000 veces) y tomar los cuantiles 2.5 y 97.5; si el intervalo excluye el 0, la diferencia difícilmente es ruido.

**Otras fuentes de varianza.** A temperatura 0 las salidas pueden variar entre ejecuciones (kernels de GPU no deterministas bit a bit, batching); un juez LLM cambia veredictos en casos límite; *pass^k* expone inconsistencia que una sola ejecución oculta. Medidas útiles: repetir ejecuciones e informar del rango, mirar siempre la tabla por ejemplo (los que cambian de veredicto entre versiones son donde está la explicación) y poner coste y latencia en la misma tabla que la calidad.

**Qué va en el dataset.** Importa tanto como su tamaño: intenciones comunes, casos límite conocidos, entradas adversarias y cada fallo de producción ya corregido (un set de regresión). Versionado, para que cada experimento esté ligado a los ejemplos exactos, y fuera de cualquier ajuste de prompts, o las puntuaciones medirán memorización del test.

> **Notebook:** [15 · Observabilidad y evaluación](../../notebooks/06_llmops/15_observability_llmops.ipynb) y [10 · RAG con LangChain y RAGAS](../../notebooks/04_applications/10_rag.ipynb) · **En el curso:** intervalos bootstrap y diferencia pareada híbrido-denso en RAGAS (notebook 10), bootstrap pareado sobre 6.040 usuarios en cada comparación entre etapas del recomendador (notebook 12), el cálculo de ±0.28 para $n = 12$ en el notebook 15, y en todos los notebooks de aplicaciones el recordatorio de cuánto vale un elemento del set antes de leer una diferencia.

---

[← Volver al índice](README.md) · [Glosario](glosario.md) · [Guía de notebooks](guia_notebooks.md)
