# Fundamentos teóricos

🇬🇧 [English version](../en/foundations.md) · [← Volver al índice](README.md)

> La teoría detrás de cada técnica y cada patrón del curso: qué problema resuelve, cuál es la idea (con sus ecuaciones), por qué funciona, con qué compite, por qué es lo que usa hoy la industria y qué sustituyó, y cómo falla. Está pensada para leerla antes de cada notebook o en paralelo. Las definiciones cortas están en el [glosario](glosario.md); aquí no se repiten, se enlazan.

**Cómo leer esta página.** Sigue el mismo orden que el [mapa por tópicos](README.md#mapa-por-tópicos). Cada apartado termina con el notebook que lo implementa y una línea **En el curso** con lo que se mide allí; las cifras que aparecen son las que publica el propio notebook, medidas con modelos pequeños (SmolLM2, `qwen2.5:3b`) y sets de evaluación pequeños, así que valen como señales grandes, no como benchmarks. Cuando un punto está en discusión en la literatura, se dice.

## Índice

1. [El modelo por dentro](#1-el-modelo-por-dentro)
   - [1.0 La visión de conjunto](#10-la-visión-de-conjunto-del-texto-al-siguiente-token) · [1.1 Tokenización](#11-tokenización) · [1.2 El bloque transformer](#12-el-bloque-transformer-y-el-flujo-residual) · [1.3 Atención, RoPE y normalización](#13-atención-rope-y-normalización) · [1.4 Atención eficiente](#14-atención-eficiente-para-contextos-largos) · [1.5 FFN y Mixture of Experts](#15-ffn-y-mixture-of-experts) · [1.6 Encoders y encoder-decoders](#16-encoders-y-encoder-decoders)
2. [Pre-training e inferencia](#2-pre-training-e-inferencia)
   - [2.0 Datos de pre-entrenamiento](#20-datos-de-pre-entrenamiento) · [2.1 El objetivo](#21-el-objetivo-predecir-el-siguiente-token) · [2.2 Optimizador y schedule](#22-optimizador-y-plan-de-learning-rate) · [2.3 Leyes de escalado](#23-leyes-de-escalado) · [2.3b Sistemas](#23b-sistemas-memoria-precisión-kernels-y-paralelismo) · [2.4 Decodificación](#24-estrategias-de-decodificación) · [2.5 KV-cache](#25-prefill-decode-y-kv-cache) · [2.6 Batching y serving](#26-batching-y-serving) · [2.7 Speculative decoding](#27-speculative-decoding) · [2.8 Cuantización](#28-cuantización) · [2.8b Serving bajo carga](#28b-serving-bajo-carga)
3. [Post-training](#3-post-training)
   - [3.1 SFT](#31-supervised-fine-tuning) · [3.2 LoRA](#32-lora-y-fine-tuning-eficiente) · [3.3 Destilación](#33-destilación) · [3.3b Fusión de modelos](#33b-fusión-de-modelos) · [3.4 RLHF, DPO y GRPO](#34-de-rlhf-a-dpo-y-grpo) · [3.5 Razonamiento y cómputo en inferencia](#35-razonamiento-y-cómputo-en-inferencia)
4. [Aplicaciones](#4-aplicaciones)
   - [4.0 Embeddings y rerankers](#40-embeddings-y-rerankers) · [4.0b Clustering y modelado de temas](#40b-clustering-y-modelado-de-temas) · [4.0c Modelos multimodales](#40c-modelos-multimodales-clip-y-modelos-de-visión-y-lenguaje) · [4.1 RAG](#41-retrieval-augmented-generation) · [4.2 Métricas de recuperación](#42-métricas-de-recuperación) · [4.3 RAGAS y LLM-as-judge](#43-ragas-y-llm-as-judge) · [4.4 Patrones de aplicación](#44-patrones-de-aplicación) · [4.4b Prompts como programas (DSPy)](#44b-prompts-como-programas-dspy-y-optimización-de-prompts) · [4.5 Recomendación](#45-sistemas-de-recomendación) · [4.5b Drift y aprendizaje online](#45b-drift-aprendizaje-online-y-actualización-continua)
5. [Agentes y MCP](#5-agentes-y-mcp)
   - [5.1 Tool calling](#51-tool-calling) · [5.2 Decodificación restringida](#52-decodificación-restringida-y-salida-estructurada) · [5.3 ReAct, plan-and-execute y reflexión](#53-react-plan-and-execute-y-reflexión) · [5.4 Grafos de estado](#54-langgraph-agentes-como-máquinas-de-estado) · [5.5 Memoria](#55-memoria) · [5.6 Límites y human-in-the-loop](#56-límites-y-human-in-the-loop) · [5.7 Equipos de agentes](#57-equipos-de-agentes-supervisor-y-handoffs) · [5.8 Ejecución de código y sandboxes](#58-ejecución-de-código-y-sandboxes) · [5.9 MCP](#59-model-context-protocol)
6. [LLMOps](#6-llmops)
   - [6.1 Trazas y spans](#61-trazas-spans-y-opentelemetry-genai) · [6.2 Evaluación offline y online](#62-evaluación-offline-y-online) · [6.3 Percentiles y SLOs](#63-percentiles-y-slos) · [6.4 PII y muestreo](#64-pii-y-muestreo-de-trazas) · [6.5 Estadística de sets pequeños](#65-estadística-de-sets-de-evaluación-pequeños) · [6.5b Frameworks de evaluación y de trazas](#65b-frameworks-de-evaluación-y-de-trazas) · [6.6 Seguridad y guardrails](#66-seguridad-rechazos-jailbreaks-y-modelos-guardián)

---

## 1. El modelo por dentro

Términos de esta parte: [glosario, Tokenización y arquitectura](glosario.md#tokenización-y-arquitectura).

Un modelo de lenguaje causal es una distribución de probabilidad sobre secuencias de tokens, factorizada con la regla de la cadena:

$$p(x_1, \dots, x_T) = \prod_{t=1}^{T} p(x_t \mid x_{<t})$$

La red nunca modela frases enteras: dado un prefijo, devuelve una distribución sobre el siguiente token. Todo lo que hay en esta parte (tokenizador, atención, FFN) existe para calcular bien ese factor.

```
texto ─► tokenizador ─► ids ─► embedding ─► L bloques ─► norm ─► lm_head ─► logits (T × V) ─► softmax
```

> **Notebooks de entrada:** primero [30 · Los transformers, en grande](../../notebooks/00_start_here/30_transformers_big_picture.ipynb) ([apartado 1.0](#10-la-visión-de-conjunto-del-texto-al-siguiente-token)) y después [00 · Un mini LLM completo](../../notebooks/00_start_here/00_mini_llm_end_to_end.ipynb) · **En el curso (00):** el modelo completo en un solo notebook; la pérdida inicial debe estar cerca de $\ln V$ y, con el corpus sintético, la pérdida irreducible se calcula exacta ($4 \ln 8 = 8.32$ nats por frase).

### 1.0 La visión de conjunto: del texto al siguiente token

**Problema.** Todos los modelos del curso, desde SmolLM2-135M hasta los 7B que sirve Ollama y los modelos frontera detrás de una API, hacen una sola cosa: leen una secuencia de ids de tokens y devuelven una probabilidad para cada posible siguiente token. Luego se elige uno, se añade y el bucle se repite. Los apartados siguientes desmontan cada pieza (tokenizador, atención, FFN, decodificación), pero estudiadas de una en una no se ve cómo encajan, y varias cosas que importan en la práctica (cuánto cuesta un token, por qué la respuesta solo aparece en las últimas capas, dónde van los parámetros y la memoria) solo se ven al seguir una frase a través de un modelo real.

**El pipeline, con tamaños reales.** Para SmolLM2-135M (30 bloques, tamaño oculto $C = 576$, 9 cabezas de query que comparten 3 cabezas de key/value, vocabulario $V = 49.152$):

```
"The capital of France is" ─► 5 ids ─► 5 vectores de 576 ─► 30 × (atención + FFN) ─► último vector ─► 49.152 logits ─► softmax ─► " the", " Paris", ...
```

Tres ideas lo sostienen todo:

- **Los tokens son ids.** El modelo nunca ve letras, solo índices en el vocabulario fijo de su tokenizador ([apartado 1.1](#11-tokenización)). El espacio inicial forma parte del token (`' capital'`), y cuántos tokens necesita un texto depende del idioma, de los dígitos y de lo raras que sean sus palabras. El coste, la latencia y la longitud de contexto se cuentan en tokens, así que el mismo contenido puede salir más barato en un idioma que en otro.
- **El significado es una dirección en el espacio.** La primera capa es una tabla $E \in \mathbb{R}^{V \times C}$, una fila por id, y consultar una fila *es* el embedding. Entrenar para predecir el siguiente token acerca las filas de los tokens que se usan en contextos parecidos, así que los vecinos más cercanos por similitud coseno tienen sentido (lunes → martes) sin que nadie haya definido una palabra. Una similitud solo significa algo frente a una referencia: los embeddings de tokens aprendidos son **anisótropos**, se apiñan en un cono estrecho alrededor de una dirección común (Gao et al., 2019), de modo que hasta dos tokens al azar están lejos de ser ortogonales.
- **La atención es una búsqueda aprendida.** En cada bloque, cada token construye una query, la compara con las keys de los tokens anteriores y hace una media ponderada de sus values ([apartado 1.3](#13-atención-rope-y-normalización)). La máscara causal solo deja mirar hacia atrás, y eso es lo que permite generar de izquierda a derecha. Cada cabeza tiene sus propios pesos, así que **las cabezas se especializan**: una puede resolver a qué se refiere un pronombre mientras la mayoría hace otras cosas, y la media sobre cabezas lo esconde.

**El flujo residual y el logit lens.** Cada bloque *suma* su salida a un vector acumulado por token, $x_\ell = x_{\ell-1} + \mathrm{Attn}_\ell(\cdot) + \mathrm{FFN}_\ell(\cdot)$ ([apartado 1.2](#12-el-bloque-transformer-y-el-flujo-residual)), y solo el último se convierte en logits, $z = W_U\,\mathrm{Norm}(x_L)$. Como todas las capas escriben en el mismo espacio, se puede aplicar la norma y la cabeza finales a un vector intermedio y leer qué "diría" el modelo a esa profundidad:

$$z_\ell = W_U\,\mathrm{Norm}(x_\ell), \qquad \ell = 1, \dots, L$$

Eso es el **logit lens** (nostalgebraist, 2020). Seguir capa a capa la posición del token correcto en el ranking muestra cómo la respuesta se va montando en profundidad y se decide en las últimas capas. Es una lectura aproximada: la cabeza solo se entrenó con la última capa, y las intermedias viven en una base algo distinta, así que en un modelo pequeño las capas del medio parecen equivocadas cuando solo están "en otro idioma". El **tuned lens** (Belrose et al., 2023) entrena una pequeña transformación afín por capa para leerlas con más justicia. Un lens o un mapa de atención *sugieren* dónde aparece un comportamiento; anular una cabeza o parchear una activación lo *confirma*, y así se localizaron y editaron hechos dentro de modelos GPT (ROME; Meng et al., 2022).

**Attention sinks.** Un softmax tiene que poner su peso en algún sitio. Cuando una cabeza no tiene nada útil a lo que atender, aparca el peso en el primer token, sea cual sea, y ese token acaba recibiendo una parte enorme de toda la atención: un **attention sink**. Xiao et al. (2024) lo describieron en varias familias de modelos abiertos y lo convirtieron en StreamingLLM: una KV-cache con ventana deslizante que conserva los primeros tokens se mantiene estable en secuencias muy largas, y una que los descarta se desmorona ([apartado 1.4](#14-atención-eficiente-para-contextos-largos)). Algunos modelos recientes añaden tokens sink explícitos o atención con compuertas (*gated attention*), para que el aparcamiento esté diseñado en vez de ser un accidente.

**La salida es una distribución, y después una elección.** La cabeza devuelve $V$ logits para la última posición, y el softmax con temperatura $T$ los convierte en probabilidades. La entropía $H = -\sum_i p_i \ln p_i$ dice cómo de repartidas están, desde 0 (un token seguro) hasta $\ln V$ (10.8 nats para una distribución uniforme sobre este vocabulario). La temperatura divide los logits: por debajo de 1 afila la distribución, por encima de 1 la aplana, y nunca cambia el orden ([apartado 2.4](#24-estrategias-de-decodificación)). El token más probable de un modelo base es la *continuación* más probable, no la respuesta a una pregunta: después de "The capital of France is", ' the' ("...is the capital of...") puede ganar a ' Paris'. Saber un dato y decirlo a continuación son cosas distintas. La decodificación greedy de un modelo base cae en bucles sosos (Holtzman et al., 2020), y por eso los sistemas reales muestrean con truncado, penalizan la repetición y ajustan el modelo con instrucciones para que la continuación más probable sea una respuesta directa (parte 3). Generar es el bucle *predecir, elegir, añadir*, y la **KV-cache** guarda las keys y los values de los tokens anteriores para que cada paso calcule solo el más nuevo ([apartado 2.5](#25-prefill-decode-y-kv-cache)).

**Dónde viven los parámetros.** Con tamaño oculto $C$, tamaño de FFN $F$, $L$ capas, $H$ cabezas de query de tamaño $D_h = C/H$, $H_{kv}$ cabezas de key/value y vocabulario $V$, las cuentas principales son

$$\underbrace{L\,\big(2C^2 + 2\,C\,D_h H_{kv}\big)}_{\text{atención: } Q, O \text{ y } K, V} \;+\; \underbrace{3\,L\,C\,F}_{\text{FFN SwiGLU}} \;+\; \underbrace{V C\ (\times 2 \text{ si la cabeza no está atada})}_{\text{embeddings y cabeza}}$$

Las normas no suman casi nada. Las capas feed-forward son la mayor parte del modelo, y en ellas se guarda buena parte de su conocimiento factual (Geva et al., 2021), por eso Mixture of Experts divide justo esas capas ([apartado 1.5](#15-ffn-y-mixture-of-experts)) y LoRA las suele incluir además de la atención. La atención con queries agrupadas (GQA; Ainslie et al., 2023) reduce el término de atención compartiendo keys y values entre cabezas de query. El término del vocabulario solo depende de $V$ y $C$, así que pesa mucho en los modelos pequeños y poco en los grandes. En inferencia, la memoria son estos parámetros por los bytes de cada uno **más la KV-cache**, que crece con la longitud de contexto × las capas × las cabezas de key/value.

**Usos en producción.** Contar tokens con el tokenizador del modelo desplegado y sobre el tráfico real, no palabras: el texto que no está en inglés, el código y los números pueden costar entre 1.5 y 3 veces más, y los tokenizadores tratan los idiomas de forma muy desigual (Petrov et al., 2023), así que conviene elegir uno adecuado a tus idiomas. Tratar los ajustes de decodificación como ajustes de producto (temperatura 0 o baja para extracción, clasificación y llamadas a tools; moderada, con top-p o min-p, para redactar; nunca greedy sobre un modelo base para texto abierto) y registrarlos con cada petición. Dimensionar el hardware con parámetros × bytes más la KV-cache: un modelo de 7B ocupa ~15 GB en bf16 y ~4.5 GB a 4 bits ([apartado 2.8](#28-cuantización)). Usar el logit lens, los mapas de atención y las sondas para localizar dónde aparece un comportamiento, y confirmarlo después con una intervención.

**Modos de fallo.** Suponer que una palabra es un token; leer una sola cabeza de atención, o la media de todas, como "lo que piensa el modelo" (los pesos de atención no son una explicación por sí solos; Jain y Wallace, 2019); tomar el token top-1 como el conocimiento del modelo; decodificar en greedy un modelo base; un tokenizador que no coincide exactamente con el modelo.

**Papers clave.** Vaswani et al. (2017), *Attention Is All You Need*; Elhage et al. (2021), *A Mathematical Framework for Transformer Circuits* (la visión del flujo residual); nostalgebraist (2020), *interpreting GPT: the logit lens*; Belrose et al. (2023), *Eliciting Latent Predictions from Transformers with the Tuned Lens*; Xiao et al. (2024), *Efficient Streaming Language Models with Attention Sinks*; Geva et al. (2021), *Transformer Feed-Forward Layers Are Key-Value Memories*; Meng et al. (2022), *Locating and Editing Factual Associations in GPT* (ROME); Gao et al. (2019), *Representation Degeneration Problem in Training Natural Language Generation Models*; Jain y Wallace (2019), *Attention is not Explanation*; Petrov et al. (2023), *Language Model Tokenizers Introduce Unfairness Between Languages*; Ainslie et al. (2023), *GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints*; Allal et al. (2025), *SmolLM2*.

> **Notebook:** [30 · Los transformers, en grande](../../notebooks/00_start_here/30_transformers_big_picture.ipynb) (se lee el primero, antes del 00) · **En el curso:** con SmolLM2-135M, "The capital of France is" son 5 tokens, el español necesita 1.73 veces los tokens del inglés para las mismas frases y `123456789` son 9 tokens; en la tabla de embeddings, Monday → Tuesday tiene un coseno de 0.92 y seven → eight 0.89, frente a 0.445 de media entre pares al azar; en media sobre las cabezas, "it" pone entre 0.00 y 0.09 de su atención en "animal" en todas las capas, mientras que la cabeza 8 de la capa 13 pone 0.73 en "animal" y 0.02 en "street", y el 61 % de toda la atención cae en el primer token; con el logit lens, el token correcto está en la posición 546 a 20.984 de 49.152 en las capas 10 y 20, entra en el top 10 en las capas 25–30 y solo pasa a top-1 en las capas 28–30, y Paris acaba en la posición 2, detrás de ' the'; con $T = 1$, ' the' se lleva el 26 % y ' Paris' el 9 %, con una entropía de 0.93 / 4.41 / 7.80 nats para $T$ = 0.5 / 1 / 1.5, y la decodificación greedy da vueltas en círculo (*"The capital of France is the capital of the country. The capital of France"*); las capas feed-forward tienen el 59.2 % de los parámetros, el embedding compartido con la cabeza el 21.0 % y la atención el 19.7 %, frente al 75 % / 14 % / 11 % (feed-forward / embeddings / atención) de Qwen2.5-7B con la misma fórmula y su configuración publicada.

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

Términos de esta parte: [glosario, Datos de pre-entrenamiento](glosario.md#datos-de-pre-entrenamiento), [Entrenamiento y post-training](glosario.md#entrenamiento-y-post-training) e [Inferencia y sistemas](glosario.md#inferencia-y-sistemas).

### 2.0 Datos de pre-entrenamiento

**Problema.** Un corpus de pre-entrenamiento empieza como crawls de la web (Common Crawl: miles de millones de páginas por snapshot) y casi todo lo que contiene no sirve para aprender: menús, plantillas, spam, listados, páginas copiadas de otras páginas. Los modelos abiertos que alcanzaron a los cerrados en 2023–2024 lo hicieron sobre todo con mejores datos, no con arquitecturas nuevas (FineWeb, DCLM, el pipeline de Llama 3). El trabajo de datos es la mayor parte del trabajo.

**El pipeline.** El crawl se reduce entre uno y dos órdenes de magnitud:

```
crawl (WARC) -> extracción de texto -> idioma -> filtros de URL y seguridad -> filtros de calidad
             -> deduplicación (exacta, casi-duplicados) -> descontaminación -> mezcla -> tokenización
```

FineWeb (Penedo et al., 2024) documenta cada paso con ablaciones: un filtro solo se quedaba si un modelo pequeño entrenado con los datos filtrados puntuaba mejor en benchmarks. Ese es el método de referencia: los filtros se eligen por su efecto en un modelo entrenado, no a ojo.

**Por qué hacen daño los duplicados.** Las crawls son muy redundantes (sitios espejo, plantillas, artículos sindicados, boilerplate); Lee et al. (2022) encontraron en C4 secuencias repetidas decenas de miles de veces. Los duplicados **gastan cómputo** (el modelo paga otra vez por texto que ya ha visto), **causan memorización** (una secuencia vista muchas veces se reproduce literalmente, un riesgo de privacidad y copyright; Carlini et al., 2021, 2023) y **filtran evaluaciones** (un benchmark copiado en la web acaba en el entrenamiento e infla las notas). La deduplicación exacta (hash del texto normalizado) es lineal y atrapa los espejos, pero un solo carácter cambiado la derrota; para lo demás hace falta detectar casi-duplicados.

**Jaccard y MinHash.** Un documento $A$ se representa por su conjunto de **shingles** (por ejemplo, 5-gramas de palabras), y el solapamiento entre dos documentos es su similitud de Jaccard:

$$J(A, B) = \frac{|A \cap B|}{|A \cup B|}$$

Comparar todos los pares es $O(N^2)$, imposible con miles de millones de documentos. **MinHash** (Broder, 1997) aplica una función hash aleatoria $h$ a cada shingle y se queda con el mínimo. Para una permutación aleatoria,

$$\Pr\big[\min h(A) = \min h(B)\big] = J(A, B)$$

porque el mínimo sobre $A \cup B$ puede ser con igual probabilidad cualquiera de sus elementos, y los dos mínimos coinciden exactamente cuando ese elemento está en $A \cap B$. Con $k$ funciones hash independientes, la fracción de mínimos que coinciden en las dos **firmas** de longitud $k$ estima $J$ con error típico

$$\mathrm{SE}(\hat J) = \sqrt{\frac{J(1-J)}{k}}$$

En la práctica, las permutaciones se sustituyen por funciones afines $h_i(x) = (a_i x + b_i) \bmod p$ con un primo grande $p$ (un primo de Mersenne hace barata la operación módulo).

**LSH: encontrar los pares candidatos.** La firma ya es pequeña, pero compararlas todas sigue siendo cuadrático. **Locality-sensitive hashing** parte cada firma en $b$ bandas de $r$ filas y mete los documentos en buckets por cada banda; dos documentos son candidatos si coinciden *exactamente* en alguna banda:

$$\Pr[\text{candidato}] = 1 - \big(1 - J^r\big)^b$$

Es una curva en S con un umbral suave cerca de

$$J^* \approx \left(\frac{1}{b}\right)^{1/r}$$

Los pares por encima se encuentran casi siempre y los de debajo casi nunca. Con $b = 16$ y $r = 8$ (128 funciones hash), $J^* \approx 0.71$. Más filas por banda afilan la curva y suben el umbral; más bandas lo bajan a cambio de más candidatos. Los candidatos se verifican (Jaccard estimado por encima del umbral) y los pares aceptados se agrupan con union-find, de modo que sobrevive un documento por cluster. La elección de $b$ y $r$ no es un detalle técnico: el umbral define qué es un "duplicado". FineWeb usa 5-gramas y un umbral cerca de 0.75, que quita copias con otra plantilla y reescrituras ligeras pero conserva documentos que solo se citan entre sí. `datatrove`, `text-dedup` y las implementaciones en Spark hacen exactamente esto a escala de miles de millones de documentos.

**Filtros heurísticos.** Quitan lo que obviamente no es prosa. Las reglas de **Gopher** (Rae et al., 2021) acotan el número de palabras y la longitud media de palabra, limitan símbolos, líneas con viñetas y puntos suspensivos, exigen stop words comunes del inglés y penalizan líneas y n-gramas repetidos. Las de **C4** (Raffel et al., 2020) conservan las líneas que terminan en puntuación y descartan páginas con llaves de código o "lorem ipsum". Son baratas, interpretables y conviene registrar qué regla quita cada documento para poder auditarlas.

**Filtros basados en modelo.** Puntúan cada documento con un clasificador pequeño entrenado para reconocer buen texto. **FineWeb-Edu** entrenó un regresor sobre un encoder BERT con 450.000 documentos que Llama-3-70B calificó por valor educativo (0 a 5); quedarse con los de nota 3 o más dio grandes ganancias en benchmarks de conocimiento con el mismo presupuesto de tokens. **DCLM** (Li et al., 2024) hizo lo mismo con un clasificador fastText. El compromiso: todo filtro quita también algo de texto bueno, y un clasificador codifica el gusto de quien etiquetó sus datos (favorece la prosa de libro de texto frente a la conversación o el código). Reglas y clasificador miden cosas distintas: las reglas quitan lo que no es texto; el clasificador juzga cuánto vale el texto.

**Descontaminación.** Si las preguntas del test están en los datos de entrenamiento, el benchmark mide memoria, no capacidad. La comprobación estándar (GPT-3, Llama) indexa n-gramas largos (8 a 13 tokens) de cada benchmark y marca los documentos de entrenamiento que los comparten. Es barata y atrapa las copias literales; la contaminación parafraseada necesita similitud de embeddings o, sencillamente, no se detecta. Y no hace falta que nadie copie el test a propósito: la web se repite, y el mismo texto existe en los dos sitios.

**Datos sintéticos y model collapse.** El texto escrito por LLMs es ya una parte grande de los datos de post-training y una parte creciente del pre-training (los "libros de texto" de los modelos Phi, Cosmopedia). Es controlable y barato, pero las muestras de un modelo son menos diversas que el texto humano, y entrenar repetidamente con salidas de modelos estrecha la distribución: las colas desaparecen primero y la variedad cae de generación en generación (***model collapse***, Shumailov et al., 2024). Los datos sintéticos necesitan la misma deduplicación y los mismos filtros que los datos web, más medidas explícitas de diversidad (léxica, como los bigramas distintos, y de embeddings), y se evalúan entrenando con ellos: un clasificador de calidad que premia lo que *parece* educativo premia justo el texto escrito para parecerlo.

**Modos de fallo.** Un umbral que quita en silencio un dominio entero (código, otros idiomas o dialectos que el clasificador ve como "baja calidad"). Una deduplicación demasiado agresiva que quita repetición legítima (licencias que tienen que estar, frases comunes). Contaminación de benchmarks descubierta después de publicar. Datos sintéticos que inflan las mismas métricas que los juzgan. Un cambio en un sitio que inunda el corpus de boilerplate sin que nadie lo vea: por eso se monitoriza la proporción de cada fuente y de cada filtro en el tiempo, la tasa de duplicados por fuente, la distribución de notas del clasificador y una muestra leída por una persona en cada versión.

**Papers clave.** Broder (1997), *On the resemblance and containment of documents*; Rae et al. (2021, Gopher); Raffel et al. (2020, T5 / C4); Carlini et al. (2021, 2023); Lee et al. (2022), *Deduplicating Training Data Makes Language Models Better*; Penedo et al. (2024), *The FineWeb Datasets*; Li et al. (2024, DataComp-LM / DCLM); Shumailov et al. (2024), *AI models collapse when trained on recursively generated data*.

> **Notebook:** [18 · Ingeniería de datos](../../notebooks/02_pretraining_and_inference/18_data_engineering.ipynb) (se lee antes del 05) · **En el curso:** sobre 12.000 documentos de FineWeb y un crawl simulado de 13.500 con duplicados plantados, el hash exacto quita las 1.100 copias plantadas y nada más (13.500 → 12.400); MinHash + LSH desde cero ($k = 128$, 16 × 8, umbral 0.7) acepta 1.633 pares (→ 10.767) con recall del 100 % en copias con otra plantilla y con el 1 % de palabras cambiadas, 72.2 % con el 3 % y 0 % con el 10 %, sobre la curva teórica y sin falsos positivos; las reglas de Gopher y C4 quitan el 12.0 % (1.173 documentos por la regla de puntuación final de C4); el clasificador de FineWeb-Edu da 1.20 de media y 2.0 % con nota ≥ 3 a FineWeb tras las reglas, frente a 3.04 y 47.6 % a FineWeb-Edu; la descontaminación con 13-gramas encuentra las 50 de 50 fugas plantadas y 29 solapes naturales (noticias sindicadas); dos `TransformerLM` de ~16M entrenados con los mismos 12.3M de tokens quedan empatados en held-out (5.547 en bruto frente a 5.537 limpio, ruido), pero en los 43 documentos repetidos 10 veces el modelo en bruto baja a 4.350 frente a 5.281 (memorización); los 60 pasajes sintéticos de `qwen2.5:3b` sacan 4.15 en el clasificador frente a 3.14 del texto humano, con 0.60 bigramas distintos frente a 0.72.

### 2.1 El objetivo: predecir el siguiente token

**Idea.** El pre-training minimiza la entropía cruzada media del token verdadero:

$$\mathcal{L}(\theta) = -\frac{1}{T}\sum_{t=1}^{T} \log p_\theta(x_t \mid x_{<t}), \qquad \mathrm{PPL} = e^{\mathcal{L}}$$

Es auto-supervisado: los objetivos son la entrada desplazada una posición, así que cualquier texto es dato de entrenamiento. Con la máscara causal, una secuencia de longitud $T$ da $T$ ejemplos en una pasada. Como la pérdida depende del tokenizador, las comparaciones entre modelos con vocabularios distintos se hacen en *bits por byte*.

**Por qué hay un suelo.** La entropía cruzada esperada se descompone:

$$H(p, q_\theta) = H(p) + D_{\mathrm{KL}}(p \,\Vert\, q_\theta) \;\ge\; H(p)$$

El entrenamiento solo reduce el término KL; la entropía del texto es irreducible. Con texto real se desconoce; con un corpus sintético se calcula.

**Lo que sustituyó.** BERT (enmascarado, solo un 15 % de posiciones con pérdida y sin forma natural de generar), T5 (span corruption, dos pilas) o UL2 (mezcla de denoisers) perdieron frente al LM causal como opción por defecto: cada token es un objetivo, entrenamiento e inferencia coinciden y escala sin trucos. GPT-3 mostró el **in-context learning**, y eso lo consolidó. Para código se añade *fill-in-the-middle*.

**Los datos deciden casi todo.** El pipeline típico (extracción, idioma, filtros heurísticos, clasificador de calidad, deduplicación exacta y MinHash, PII, descontaminación de benchmarks, pesos por dominio y una fase final de *annealing* con datos de alta calidad) es donde los datasets abiertos más han mejorado (RefinedWeb, FineWeb-Edu); el [apartado 2.0](#20-datos-de-pre-entrenamiento) lo desarrolla. Repetir datos unas pocas épocas es casi tan bueno como datos nuevos (Muennighoff et al., 2023).

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

### 2.3b Sistemas: memoria, precisión, kernels y paralelismo

**Problema.** Las leyes de escalado dicen cuánto cómputo hace falta; no dicen si el entrenamiento cabe en la GPU, a qué velocidad va ni cómo se reparte entre muchas. Esa es la parte de sistemas de la ingeniería de LLMs, y casi todas sus decisiones consisten en cambiar un recurso (memoria, cómputo, comunicación) por otro.

**Contabilidad de cómputo.** Un forward cuesta unos $2N$ FLOPs por token (una multiplicación y una suma por peso) y el backward el doble, así que entrenar cuesta

$$C \approx 6\,N\,D \quad \text{FLOPs para } D \text{ tokens}$$

más un término de atención de unos $12\,L\,d\,T$ FLOPs por token ($L$ capas de ancho $d$, contexto $T$), que solo pesa cuando $T$ es grande frente a $d$ (Kaplan et al., 2020; Chowdhery et al., 2022). La **model FLOPs utilization** mide qué fracción del pico de la GPU es trabajo útil:

$$\mathrm{MFU} = \frac{\text{tokens/s} \times (6N + 12LdT)}{\text{pico de FLOP/s}}$$

Solo cuenta los FLOPs del modelo, no los que se recalculan con checkpointing (eso sería la *hardware FLOPs utilization*, HFU). Los entrenamientos grandes bien ajustados llegan al 40–60 %.

**Contabilidad de memoria.** Con Adam y precisión mixta, cada parámetro cuesta (Rajbhandari et al., 2020):

$$\underbrace{2}_{\text{pesos bf16}} + \underbrace{2}_{\text{gradientes bf16}} + \underbrace{4 + 4 + 4}_{\text{pesos maestros fp32},\ m,\ v} = 16 \ \text{bytes} \quad\Rightarrow\quad 16\Psi \ \text{bytes para } \Psi \text{ parámetros}$$

Un 7B necesita unos 112 GB antes de la primera activación; en fp32 puro también son 16 bytes (4 + 4 + 8). Las **activaciones**, las salidas de cada capa que se guardan para el backward, crecen con batch × secuencia × ancho × capas, más un término en $T^2$ si la atención guarda la matriz de scores (Korthikanti et al., 2022, dan unos $sbh(34 + 5as/h)$ bytes por capa en 16 bits, con secuencia $s$, batch $b$, ancho $h$ y $a$ cabezas). En secuencias largas y batches grandes son ellas, no los parámetros, las que deciden qué cabe. Una parte fácil de olvidar son los logits: batch × $T$ × $V$ números guardados para la pérdida.

**Intensidad aritmética y roofline.** Que una operación esté limitada por cómputo o por memoria depende de su **intensidad aritmética** $I$, los FLOPs por byte movido. El modelo *roofline* acota el rendimiento:

$$\text{FLOP/s alcanzables} = \min\big(\text{pico de FLOP/s},\ I \times \text{ancho de banda de memoria}\big)$$

Una matmul $n \times n$ hace $O(n^3)$ trabajo sobre $O(n^2)$ datos y va cerca del pico; las operaciones elemento a elemento, las normalizaciones y el softmax hacen $O(1)$ trabajo por byte y están limitadas por memoria. Es la misma cuenta que explica el decode en el [apartado 2.5](#25-prefill-decode-y-kv-cache).

**Trucos que cambian un recurso por otro.**

- **Precisión mixta** (Micikevicius et al., 2018): calcular en bf16 o fp16 en los tensor cores (varias veces el throughput de fp32) y guardar los pesos maestros en fp32, para que las actualizaciones pequeñas no se pierdan por redondeo. bf16 tiene los 8 bits de exponente de fp32, así que no necesita *loss scaling*; fp16, con 5 bits de exponente, sí, o los gradientes pequeños se quedan en cero. fp8 (de Hopper en adelante) añade escalas por tensor o por bloque.
- **Activation checkpointing** (Chen et al., 2016): guardar solo la entrada de cada bloque y recalcular su interior durante el backward. Cuesta un forward extra (~30 % más de cómputo) y deja en memoria un tensor por bloque más el interior de un solo bloque a la vez; con un checkpoint cada $\sqrt{L}$ capas, la memoria de activaciones baja a $O(\sqrt{L})$. La recomputación selectiva (Korthikanti et al., 2022) recalcula solo las partes grandes y baratas, como la atención.
- **FlashAttention** (Dao et al., 2022; Dao, 2023): la atención estándar escribe en HBM la matriz $S = QK^\top$ de $T \times T$, la lee para el softmax y la vuelve a leer para multiplicar por $V$; su coste lo dominan esos bytes, no los FLOPs. FlashAttention es **IO-aware**: parte $Q$, $K$ y $V$ en bloques que caben en la SRAM del chip y mantiene para cada fila un máximo $m$ y una suma $\ell$ que se corrigen con cada bloque nuevo $j$ (*softmax online*):

  $$m' = \max(m, \tilde m_j), \qquad \ell' = e^{m - m'}\,\ell + e^{\tilde m_j - m'}\,\tilde\ell_j, \qquad O' = \frac{e^{m - m'}\,\ell\,O + e^{\tilde m_j - m'}\,\tilde P_j V_j}{\ell'}$$

  El resultado es exacto, la memoria es lineal en $T$ y el backward recalcula $S$ por bloques en lugar de guardarla. FlashAttention-2 reparte mejor el trabajo entre warps y FlashAttention-3 aprovecha la asincronía de Hopper; el kernel *memory-efficient* (Rabe y Staats, 2021; xformers) y la atención fusionada de cuDNN aplican la misma idea. Ninguno reduce el cómputo $O(T^2)$: eso es el [apartado 1.4](#14-atención-eficiente-para-contextos-largos).
- **Fusión de kernels y CUDA graphs.** Cada operación de PyTorch lanza un kernel con un coste fijo de unos microsegundos de CPU, y cada operación elemento a elemento lee y escribe su tensor entero en HBM. `torch.compile` (TorchDynamo captura el grafo; su backend por defecto, `inductor`, genera kernels de Triton) fusiona cadenas de operaciones en un solo kernel que lee una vez y escribe una vez. Un **CUDA graph** graba una secuencia de kernels con sus direcciones de memoria y la reproduce con un solo lanzamiento: no ahorra bytes ni FLOPs, ahorra lanzamientos, y exige formas estáticas. Es lo que usan los servidores de inferencia para el paso de decode.

**Paralelismo.**

| Estrategia | Qué se reparte | Comunicación por paso | Para qué |
|---|---|---|---|
| de datos (DDP) | el batch; cada GPU tiene el modelo entero | all-reduce de los gradientes | modelos que caben en una GPU |
| ZeRO-1 / 2 / 3, FSDP | estados del optimizador (1), + gradientes (2), + pesos (3) | reduce-scatter + all-gather | modelos grandes sin cambiar su código |
| de tensor (Megatron) | cada matmul entre GPUs | all-reduce dentro de cada capa | capas muy anchas, dentro de un nodo (enlaces rápidos) |
| de pipeline | las capas, por etapas | activaciones entre etapas | modelos muy profundos, entre nodos |
| de secuencia / contexto | la secuencia | bloques de atención | contextos muy largos |
| de expertos | los expertos de una capa MoE | all-to-all del enrutado | modelos MoE ([apartado 1.5](#15-ffn-y-mixture-of-experts)) |

El paralelismo de datos descansa en un hecho exacto: la media de los gradientes de $k$ trozos de un batch es el gradiente del batch entero, así que $k$ GPUs con batch $b$ entrenan como una con batch $kb$. El **all-reduce en anillo** hace un reduce-scatter y un all-gather de $k-1$ pasos cada uno, y en cada paso cada GPU envía un trozo de $P/k$ bytes:

$$\text{bytes enviados por GPU} = 2\,\frac{k-1}{k}\,P$$

casi independiente de $k$; por eso DDP escala mientras la red siga el ritmo, y solapa la comunicación con el backward agrupando los gradientes en *buckets*. **ZeRO** (Rajbhandari et al., 2020) quita la redundancia de DDP repartiendo los estados entre las $N_d$ GPUs de datos:

$$\text{memoria por GPU:}\quad \text{DDP } 16\Psi, \qquad \text{ZeRO-1 } 4\Psi + \frac{12\Psi}{N_d}, \qquad \text{ZeRO-2 } 2\Psi + \frac{14\Psi}{N_d}, \qquad \text{ZeRO-3 } \frac{16\Psi}{N_d}$$

ZeRO-1 y ZeRO-2 comunican lo mismo que DDP (un reduce-scatter más un all-gather); ZeRO-3 añade el all-gather de los pesos antes del forward y del backward de cada capa, alrededor de 1.5 veces el volumen de DDP. FSDP (Zhao et al., 2023) es la implementación de PyTorch (*full shard* es ZeRO-3; en FSDP2, `fully_shard`). Los modelos más grandes combinan estrategias ("paralelismo 3D/4D", Narayanan et al., 2021): tensor dentro del nodo, pipeline y FSDP entre nodos. El pipeline paga una burbuja de tiempo ocioso de $(p-1)/(m+p-1)$ con $p$ etapas y $m$ micro-batches (GPipe; 1F1B la reduce en memoria).

**Lo que sustituyó.** Antes de ZeRO, un modelo que no cabía en una GPU necesitaba paralelismo de modelo escrito a mano (Megatron) o pipelines; antes de FlashAttention, el contexto largo estaba limitado por la memoria $O(T^2)$, y por eso proliferaron las atenciones aproximadas. Hoy la pila por defecto es bf16, atención fusionada, FSDP o ZeRO-3 y, a gran escala, tensor parallelism dentro del nodo (`torchtitan`, Megatron-LM / NeMo, DeepSpeed).

**Modos de fallo.** Falta de memoria por las activaciones, no por los parámetros, que aparece al crecer la secuencia o el batch. *Fallbacks* silenciosos: una build sin FlashAttention, o Windows paginando memoria de GPU en la RAM del sistema; los dos se ven como un frenazo repentino, no como un error. fp16 sin loss scaling: gradientes que se quedan en cero. Escalado limitado por la comunicación: añadir GPUs no acelera porque la red no sigue a los all-gathers de ZeRO-3. No determinismo entre ejecuciones por los kernels de GPU, que importa al comparar diferencias pequeñas.

**Papers clave.** Chen et al. (2016), *Training Deep Nets with Sublinear Memory Cost*; Micikevicius et al. (2018), *Mixed Precision Training*; Shoeybi et al. (2019), *Megatron-LM*; Rajbhandari et al. (2020), *ZeRO*; Kaplan et al. (2020), *Scaling Laws for Neural Language Models*; Rabe y Staats (2021), *Self-attention Does Not Need O(n²) Memory*; Narayanan et al. (2021), *Efficient Large-Scale Language Model Training on GPU Clusters*; Dao et al. (2022), *FlashAttention*, y Dao (2023), *FlashAttention-2*; Korthikanti et al. (2022), *Reducing Activation Recomputation in Large Transformer Models*; Chowdhery et al. (2022, PaLM, MFU); Zhao et al. (2023), *PyTorch FSDP*.

> **Notebook:** [19 · Sistemas: memoria, precisión, kernels y paralelismo](../../notebooks/02_pretraining_and_inference/19_systems_memory_and_parallelism.ipynb) (se lee después del 05) · **En el curso:** en una RTX 3060 Laptop de 6 GB, con un `TransformerLM` de 29.4M de parámetros, los pesos fp32 ocupan 0.111 GB (0.110 predichos), los gradientes 0.118 y Adam 0.219, pero las activaciones pasan de 0.43 a 3.39 GB de batch 2 a 16; bf16 hace el paso 2.2× más rápido y baja el pico de 2.18 a 1.66 GB; el checkpointing recorta el pico un 57 % (fp32) y un 51 % (bf16) por ~22 % más de tiempo; la atención ingenua ocupa 4.1 GB a 8.192 tokens y se queda sin memoria a 16.384, mientras los kernels fusionados (memory-efficient y cuDNN; FlashAttention-2 no está en las wheels de Windows) necesitan 0.22 GB a 16.384 y cuDNN tarda 51.9 ms; el backend *math* sin fusionar desborda a la memoria compartida de Windows (8.1 GB en una tarjeta de 6 GB, 978 ms); el profiler pone el 47 % del tiempo de GPU en matmuls, con 1.136 kernels por paso; sin Triton no hay `inductor`, pero `backend="cudagraphs"` da 4.1× (8.68 → 2.11 ms) con la misma salida; la MFU es del 32 % (7.38 TFLOP/s frente a un pico medido de 23.0, 36.608 tokens/s); un DDP real con dos procesos (`gloo`, `FileStore`, `mp.spawn`) acaba con los mismos pesos que un proceso con el batch global (7.45e-9) y hace all-reduce de 0.53 MB por paso; y la calculadora da para un 7B 112 GB por GPU con DDP frente a 14 GB con FSDP en 8 GPUs (1.8 GB con 64), y para un 70B 140 GB con 8 y 17.5 GB con 64.

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

**Compromiso latencia-throughput.** El tiempo de una petición es aproximadamente $\text{TTFT} + N_{out} \times \text{TPOT}$. Subir el batch aumenta el throughput agregado casi gratis mientras decode esté limitado por memoria; cuando el paso pasa a estar limitado por cómputo, cada secuencia extra ralentiza a todas. Un sistema de serving elige el mayor batch que cumple el objetivo de latencia en p99 ("goodput"). Los benchmarks que solo dan throughput pico o solo latencia de una petición esconden esa elección. La sección [2.8b](#28b-serving-bajo-carga) mide estos compromisos en un servidor real y bajo carga.

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

### 2.8b Serving bajo carga

**Problema.** Las secciones 2.5–2.8 miran una generación cada vez. Una aplicación manda muchas peticiones a la vez a un servidor HTTP, y entonces lo que importa es cuánto espera cada usuario, qué hace el hardware con ese tráfico y cuánto cuesta. Un servidor puede parecer rápido con una petición y hundirse con ocho.

**Las métricas.** Una respuesta en streaming tiene dos fases, y cada una tiene su métrica:

$$\text{e2e} \approx \text{TTFT} + \text{TPOT} \times (N_{out} - 1), \qquad \text{throughput} = \frac{\sum \text{tokens de salida}}{\text{tiempo de reloj}}$$

El **TTFT** (time to first token) incluye el tiempo en cola, el prefill y la red: es el "¿está haciendo algo?" del usuario. El **TPOT** (time per output token, o latencia entre tokens) es el hueco entre tokens durante el decode, y $1/\text{TPOT}$ es la velocidad de lectura de un usuario (con 20–30 tokens por segundo ya se lee más despacio de lo que llega el texto). El **throughput** suma todos los usuarios y es de lo que depende la factura. Las colas de latencia son largas (sección 6.3), así que los objetivos son **SLOs sobre percentiles** (por ejemplo TTFT p95 < 1 s y TPOT p95 < 50 ms), y la capacidad útil es el **goodput**: las peticiones por segundo que cumplen el SLO, no los tokens por segundo en bruto (Zhong et al., 2024).

**Por qué las dos fases se comportan distinto.** El prefill pasa el prompt entero por el modelo en una sola pasada de matmuls grandes, así que está limitado por cómputo; el decode lee todos los pesos para cada token nuevo, así que está limitado por memoria (sección 2.5). Todas las optimizaciones de serving aprovechan esa asimetría: el batching reparte una misma lectura de pesos entre muchas secuencias, el prefix caching se salta el prefill ya hecho, la cuantización reduce los bytes leídos por token y el speculative decoding (2.7) convierte varios pasos de decode en una sola verificación. El **serving desagregado** (DistServe; Mooncake) va más allá y ejecuta el prefill y el decode en GPUs distintas, porque un prefill largo dentro del mismo batch frena el decode de todos y las dos fases piden paralelismos distintos.

**Batching.** El **batching estático** espera a juntar un batch y lo ejecuta hasta que termina la secuencia más larga, con los huecos de las ya terminadas ociosos. El **batching continuo** (Orca, Yu et al., 2022) planifica a nivel de iteración: las secuencias entran y salen en cada paso de decode, así que el batch siempre va lleno. Lo que limita entonces el batch es la memoria de la KV-cache, y **PagedAttention** (vLLM, Kwon et al., 2023) la guarda en bloques de tamaño fijo con una tabla de bloques, como las páginas de la memoria virtual, de modo que casi no se pierde nada por fragmentación y los bloques se pueden compartir entre secuencias. llama.cpp, y Ollama por encima, sirve un número fijo de **slots** (`OLLAMA_NUM_PARALLEL`): cada slot es una secuencia del batch con su parte de la KV-cache, y las peticiones que no caben esperan en una cola. El batching es un intercambio: sube el throughput total y, con él, el TPOT de cada usuario.

**Colas y ley de Little.** En cualquier sistema estable, sea cual sea la distribución de llegadas (Little, 1961):

$$L = \lambda W$$

peticiones en el sistema = tasa de llegada × tiempo en el sistema. Convierte el dimensionado en una cuenta: con 2 peticiones por segundo y 3 s por petición hay 6 en vuelo de media, así que hay que poner más slots o réplicas que eso. Y explica el codo de toda curva de latencia: en cuanto las peticiones llegan más deprisa de lo que el servidor las termina, la cola crece sin límite y el TTFT con ella. En el modelo de colas más sencillo (M/M/1), $W = 1/(\mu - \lambda)$, que se dispara cuando la utilización $\lambda/\mu$ se acerca a 1. Por eso la capacidad se planifica con curvas de latencia frente a carga y se mantiene bien por debajo de la saturación, y por eso el autoescalado mira la longitud de la cola o el uso de la KV-cache, no la CPU.

**Prefix caching.** Muchas peticiones comparten un prefijo largo: el system prompt, las definiciones de herramientas, los ejemplos few-shot, el documento del que se habla. Su KV-cache se puede guardar y reutilizar en lugar de recalcularla. El automatic prefix caching de vLLM identifica los bloques KV por un hash; **RadixAttention** de SGLang (Zheng et al., 2024) guarda los prefijos en un árbol radix que comparten todas las peticiones, con desalojo LRU; las APIs de los proveedores lo venden como **prompt caching**, con los tokens de entrada cacheados a una fracción del precio; llama.cpp reutiliza la caché de un slot cuando el siguiente prompt empieza con los mismos tokens. La coincidencia es token a token desde el principio, así que un solo token cambiado (una marca de tiempo arriba del todo) invalida todo lo que viene detrás: **lo compartido primero, el mensaje del usuario al final**.

**Formatos de pesos: GGUF y k-quants.** GGUF es el formato de un solo fichero de llama.cpp: pesos, tokenizador y metadatos juntos. Sus **k-quants** (Q4_K_M, Q5_K_M, Q6_K...) cuantizan los pesos en superbloques de 256, divididos en subbloques pequeños cuyas escalas están a su vez cuantizadas: las escalas por grupo de la sección 2.8. El sufijo `_S`/`_M`/`_L` indica cuántos tensores sensibles se dejan en más precisión, así que Q4_K_M sale a algo menos de 5 bits por peso de media. Q8_0 son bloques de 32 valores int8 con una escala fp16; FP16 va sin cuantizar. Como el decode está limitado por memoria, menos bytes significa decode más rápido además de menos memoria, siempre que el runtime tenga un kernel para ese formato (al contrario que quanto en el notebook 09). Lo que cuesta en calidad hay que medirlo en tu propia tarea.

**Gateways, cascadas y routers.** Las aplicaciones casi nunca llaman a un modelo directamente. Un **gateway** (LiteLLM, Portkey, OpenRouter, los AI gateways de la nube) ofrece a todos los clientes una única interfaz, normalmente compatible con OpenAI, y resuelve en un solo sitio lo que es transversal: enrutado, reintentos con backoff, **fallbacks** a otro modelo ante errores o **timeouts**, **circuit breakers** que dejan de mandar tráfico durante un rato a un backend que falla, límites de tasa, presupuestos, caché y logs. Encima va el enrutado consciente del coste:

- una **cascada** (FrugalGPT, Chen, Zaharia y Zou, 2023) prueba los modelos por orden de coste y se para cuando un evaluador acepta la respuesta. Su coste esperado es $c_{small} + c_{signal} + p_{esc} \cdot c_{big}$: compensa cuando la tasa de escalado $p_{esc}$ es baja **y** la señal es barata;
- un **router** (RouteLLM, Ong et al., 2024) predice antes de llamar a nadie si hace falta el modelo fuerte, con un clasificador entrenado con datos de preferencias, así que cada petición cuesta una sola llamada.

La señal de incertidumbre puede ser el acuerdo entre muestras (como en la self-consistency, 3.5), las log-probabilidades de los tokens, un verificador o un clasificador entrenado. El enrutado compensa sobre todo cuando la diferencia de precio es grande: un modelo pequeño o local delante de una API frontera.

**Lo que hace la industria.** llama.cpp u Ollama para local, edge y un solo usuario; vLLM, SGLang o TGI (y TensorRT-LLM para el máximo throughput en NVIDIA) para servir a muchos usuarios en GPUs de centro de datos, con batching continuo, PagedAttention, prefix caching entre usuarios, pesos FP8 o AWQ y speculative decoding; todos detrás de una API compatible con OpenAI y, a menudo, de un gateway. Las pruebas de carga usan longitudes de prompt y de salida y patrones de llegada realistas (`vllm bench`, `genai-perf`, Locust, k6).

**Modos de fallo.** Tráfico en paralelo contra un servidor de un solo slot (la concurrencia se convierte en cola); un prompt montado de forma que anula la caché; tamaños de contexto que recargan el modelo sin avisar; capacidad planificada con el throughput pico en lugar del codo de latencia; modelos cuantizados de más que pasan una prueba rápida y fallan en los casos límite; timeouts más cortos que una carga del modelo en frío; fallbacks que esconden un modelo principal caído; routers cuya señal cuesta más de lo que ahorra.

**Papers clave.** Little (1961), *A Proof for the Queuing Formula: L = λW*; Yu et al. (2022), *Orca*; Kwon et al. (2023), *Efficient Memory Management for Large Language Model Serving with PagedAttention*; Chen, Zaharia y Zou (2023), *FrugalGPT*; Zheng et al. (2024), *SGLang: Efficient Execution of Structured Language Model Programs*; Zhong et al. (2024), *DistServe*; Ong et al. (2024), *RouteLLM*.

> **Notebook:** [23 · Serving bajo carga](../../notebooks/02_pretraining_and_inference/23_serving_and_load.ipynb) (se lee después del 06) · **En el curso:** con `qwen2.5:3b` en Ollama y una RTX 3060 Laptop de 6 GB, el prefill va a ~3.900 tokens/s y el decode a ~110 (35×), y el TTFT crece de 101 a 645 ms entre 338 y 2.322 tokens de prompt; con un system prompt compartido, el 99.1 % del prompt sale de la caché después de la primera petición y el TTFT baja de 654 a 39 ms; con un slot, 8 peticiones en vuelo llevan el TTFT p50 a 13.4 s mientras el throughput no se mueve (84 → 67 tok/s); cuatro slots llegan a 171 tok/s (2×) con el TTFT por debajo de 100 ms hasta 4 en vuelo, a cambio de un TPOT de 23 ms en vez de 12.5; la ley de Little se cumple hasta el decimal (1.33 peticiones/s × 3.00 s = 4.0); `qwen2.5:1.5b-instruct` ocupa 1.17 GB a 130 tok/s en Q4_K_M, 1.83 GB a 105 tok/s en Q8_0 y 3.28 GB a 64 tok/s en FP16, con un 89.5 %, un 91.0 % y un 91.0 % en 200 preguntas de ARC-Easy; y en 80 problemas de GSM8K una cascada 3B → 7B que escala cuando dos respuestas del 3B no coinciden (el 16 % de los problemas) llega al 85.0 %, frente al 82.5 % de usar siempre el 3B y al 78.8 % de usar siempre el 7B, pero con el 88 % del tiempo de GPU del 7B, porque la propia comprobación de acuerdo duplica el coste del 3B.

---

## 3. Post-training

Términos de esta parte: [glosario, Entrenamiento y post-training](glosario.md#entrenamiento-y-post-training) y [Razonamiento y cómputo en inferencia](glosario.md#razonamiento-y-cómputo-en-inferencia).

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

### 3.3b Fusión de modelos

**Problema.** Con el tiempo acabas con varios fine-tunings del mismo base: uno por tarea, por cliente o por experimento. Quieres un solo modelo, y puede que no tengas los datos ni la GPU para entrenarlos juntos.

**Idea.** Tratar un fine-tuning como un vector. Ilharco et al. (2023, *Editing Models with Task Arithmetic*) llaman a la diferencia $\tau_t = \theta_t - \theta_0$ un **task vector** y mostraron que se puede operar con ella (**task arithmetic**):

$$\theta_{\text{fusionado}} = \theta_0 + \sum_t \lambda_t\, \tau_t$$

Sumar dos task vectors da un modelo que hace las dos tareas, a menudo casi tan bien como cada especialista; restar uno ($\lambda < 0$) quita un comportamiento, y el artículo lo usó para hacer un modelo menos tóxico. Con LoRA, el task vector de cada matriz adaptada es simplemente $\Delta W = \frac{\alpha}{r} BA$. Fusionar es solo aritmética: ni datos ni pasos de gradiente.

**Por qué funciona.** Los pesos no suelen ser sumables: si promedias dos redes entrenadas desde inicializaciones aleatorias distintas sale un modelo roto, porque sus neuronas están permutadas unas respecto a otras. Con los fine-tunings del *mismo* modelo preentrenado no pasa. Se quedan en la misma cuenca del paisaje de pérdida, unidos por un camino recto a lo largo del cual la pérdida sigue baja (**linear mode connectivity**, Frankle et al., 2020). Wortsman et al. (2022) lo aprovecharon en los **model soups**: promediar fine-tunings de una misma tarea con hiperparámetros distintos superó a la mejor ejecución individual en precisión y robustez, sin coste extra en inferencia. Para tareas *distintas*, la suma funciona cuando los task vectors interfieren poco, y en un espacio con millones de coordenadas por matriz dos actualizaciones sin relación son casi **ortogonales** (similitud coseno cercana a 0): sumar una apenas mueve el modelo en la dirección de la otra.

**Cuando interfieren: TIES y DARE.** Si los task vectors se solapan, la suma simple falla de dos maneras: las coordenadas pequeñas y ruidosas se acumulan, y las coordenadas en las que las tareas tiran en sentidos opuestos se anulan o se pasan de largo.

- **TIES-Merging** (Yadav et al., 2023): *recortar* cada task vector a su fracción de mayor magnitud (la *density*); *elegir* un signo por coordenada, el de mayor magnitud total; y hacer una *fusión disjunta*, promediando solo los valores que coinciden con ese signo.
- **DARE** (Yu et al., 2024, *Language Models are Super Mario*): *descartar* al azar una fracción $p$ de las coordenadas de cada task vector y *reescalar* el resto por $1/(1-p)$, para que la actualización esperada no cambie. Las actualizaciones del fine-tuning son tan redundantes que se puede descartar el 90 % o más. Se suele combinar con la elección de signo de TIES (`dare_ties`).
- **SLERP** interpola dos modelos sobre la esfera en lugar de en línea recta, lo que conserva la norma de los pesos; solo sirve para dos modelos. **LoraHub** (Huang et al., 2023) aprende los pesos de mezcla $\lambda_t$ de muchos adaptadores LoRA a partir de unos pocos ejemplos de una tarea nueva.

La herramienta de referencia para modelos completos es **MergeKit** (Goddard et al., 2024), que está detrás de muchos de los modelos abiertos fusionados de los leaderboards públicos; para adaptadores, `add_weighted_adapter` de `peft`.

**La trampa de los factores de LoRA.** Un adaptador guarda $B$ y $A$, no $\Delta W$, y

$$\big(B_1 + B_2\big)\big(A_1 + A_2\big) = B_1 A_1 + B_2 A_2 + \underbrace{B_1 A_2 + B_2 A_1}_{\text{términos cruzados}}$$

así que sumar los factores añade dos términos que ninguna tarea pidió. En `peft`, `combination_type="linear"` hace justo eso (su documentación lo llama una aproximación muy burda); `cat` concatena los factores en un adaptador de rango $r_1 + r_2$ cuyo $\Delta W$ es la suma ponderada exacta, es decir, task arithmetic; `svd` forma la suma exacta y la comprime a un rango elegido (es exacta si `svd_rank` es al menos el rango de la suma); `ties_svd` y `dare_ties_svd` aplican TIES o DARE sobre el $\Delta W$ completo y después una SVD.

| | Fusionar | Entrenar multitarea |
|---|---|---|
| Necesita | los pesos o adaptadores ya ajustados, del mismo base | todos los datasets a la vez |
| Coste | segundos de aritmética | un entrenamiento con todos los datos |
| Qué obtienes | los comportamientos uno al lado del otro, cada uno algo por debajo de su especialista | normalmente lo mejor en cada tarea, y las habilidades se pueden combinar |
| Cuándo | los adaptadores vienen de otros equipos o del Hub, ya no están los datos, no hay presupuesto para reentrenar | tienes todos los datos |

**Modos de fallo.** Fusionar fine-tunings de bases distintos (o un adaptador en un base con el que no se entrenó); fusionar factores LoRA con `linear`; usar TIES o DARE donde no hay interferencia, con lo que recortar solo tira señal; esperar que la fusión *combine* habilidades, cuando solo las pone una al lado de la otra; y evaluar la fusión en una sola tarea, cuando puede romper otra sin avisar. Hay que pasar la evaluación de cada tarea sobre el modelo fusionado.

> **Notebook:** [07 · SFT, LoRA y destilación](../../notebooks/03_post_training/07_finetuning_sft_lora.ipynb) (sección 8) · **En el curso:** un segundo adaptador LoRA aprende a pasar horas a formato 24 h y se fusiona con el de reservas. Los dos task vectors son casi ortogonales (coseno medio +0.012 en 210 matrices, 49 % de conflictos de signo donde las dos actualizaciones son grandes). Task arithmetic (`cat`) mantiene las reservas en 32.5 % de exact match, lo mismo que el adaptador de reservas solo, y convierte horas al 90 % (el especialista, 97.5 %); `linear` se queda en 27.5 % / 87.5 %, `ties_svd` en 27.5 % / 80 % y `dare_ties_svd` en 30 % / 87.5 %. Ninguna fusión lleva la habilidad de conversión al campo `time` de las reservas. Un único adaptador multitarea entrenado con los dos datasets (76 pasos) llega al 55 % de exact match y al 97.5 %, mejor que cualquier fusión en las dos tareas.

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

### 3.5 Razonamiento y cómputo en inferencia

**Problema.** Hay problemas que piden varios pasos encadenados ("primero los huevos que sobran, luego cuánto se sacan por ellos"), y un transformer hace una cantidad fija de cómputo por token: $L$ capas, sea cual sea la pregunta. Si los pasos no caben en un solo forward, el modelo no puede acertar soltando la respuesta de golpe, por mucho que sepa.

**Cadena de pensamiento: los tokens como cómputo en serie.** Escribir los pasos intermedios como tokens convierte la salida en un borrador: cada token nuevo atiende a los resultados escritos antes, así que **más tokens compran más cómputo en serie**. Wei et al. (2022) mostraron que unos pocos ejemplos resueltos en el prompt desbloquean el razonamiento en varios pasos de los modelos grandes (CoT few-shot); Kojima et al. (2022), que "Let's think step by step" basta para buena parte de ello (CoT zero-shot). También hay una base teórica: con pasos intermedios, un transformer resuelve problemas que se puede demostrar que no resuelve en una sola pasada (Feng et al., 2023; Merrill y Sabharwal, 2024). La cadena escrita no es necesariamente el cómputo que produjo la respuesta: puede ser **infiel** (Turpin et al., 2023), así que leerla no equivale a una explicación.

**Gastar más cómputo en inferencia.** Una vez que el modelo sabe razonar, hay tres formas de gastar más cómputo en un mismo problema:

- **Muestreo en paralelo y self-consistency** (Wang et al., 2022; ICLR 2023). Se muestrean $k$ caminos de razonamiento a temperatura $> 0$ y se devuelve la respuesta final más frecuente: los caminos correctos tienden a coincidir y los erróneos se dispersan. Su acierto es **maj@k**. El techo es **pass@k**, la probabilidad de que al menos una de las $k$ muestras acierte, que es lo que alcanzaría un verificador perfecto. Con $n$ muestras de las que $c$ son correctas, el estimador sin sesgo es (Chen et al., 2021)

  $$\text{pass@}k = 1 - \binom{n-c}{k}\Big/\binom{n}{k}$$

  La distancia entre pass@k y maj@k es lo máximo que puede añadir cualquier verificador sobre el voto.
- **Best-of-N con un verificador.** Un segundo modelo puntúa cada candidata y se devuelve la mejor. Un **modelo de recompensa de resultado** (ORM) puntúa la respuesta final; Cobbe et al. (2021), que crearon GSM8K para esto, entrenaron uno y vieron que muestrear muchas soluciones y dejar que el verificador eligiera superaba al fine-tuning solo. Un **modelo de recompensa de proceso** (PRM) puntúa cada paso; Lightman et al. (2023, *Let's Verify Step by Step*) encontraron que supervisar el proceso es claramente mejor que supervisar solo el resultado en MATH. La trampa está en la **capacidad de discriminar** del verificador: si no separa lo correcto de lo incorrecto mejor que el voto, añade coste y no acierto, y con suficientes candidatas la búsqueda acaba encontrando las respuestas que lo engañan (*reward hacking* en inferencia, el mismo efecto Goodhart de la [sección 3.4](#34-de-rlhf-a-dpo-y-grpo)). Los verificadores programáticos (tests unitarios, una calculadora, un solver, un esquema) no tienen ninguno de estos problemas, allí donde existen.
- **Refinamiento secuencial.** Revisar una misma respuesta varias veces (self-refine, la reflexión de la [sección 5.3](#53-react-plan-and-execute-y-reflexión)) u obligar al modelo a seguir pensando (*budget forcing*, s1; Muennighoff et al., 2025).

Snell et al. (2024) mostraron que, con un presupuesto fijo, gastar el cómputo en inferencia de forma adaptativa (más muestras o revisiones en los prompts difíciles) puede ganar a un modelo varias veces mayor en problemas de la dificultad adecuada: **el cómputo en inferencia es un eje de escalado propio**, junto a los parámetros y los datos ([sección 2.3](#23-leyes-de-escalado)). No los sustituye: en los problemas más difíciles, un modelo base mejor sigue ganando.

**Modelos de razonamiento.** En lugar de pedirlo en el prompt, el razonamiento se puede **entrenar**. DeepSeek-AI (2025, *DeepSeek-R1*) aplicó RL a gran escala con recompensas verificables (GRPO, [sección 3.4](#34-de-rlhf-a-dpo-y-grpo)) y obtuvo cadenas de pensamiento largas, con autocomprobación y vuelta atrás; o1 de OpenAI hizo algo parecido con una receta no publicada. El pensamiento se emite como tokens (a menudo entre etiquetas `<think>`) y se paga como cualquier otra salida. La **destilación** lo transfiere barato: los modelos R1-Distill son modelos pequeños de Qwen y Llama ajustados (SFT normal, [sección 3.3](#33-destilación)) con las trazas de razonamiento de R1. Si los modelos pequeños destilados razonan de forma robusta fuera de su distribución de entrenamiento es una pregunta abierta. El fallo conocido es el **overthinking**: razonamientos largos y en bucle en problemas fáciles, o que agotan el presupuesto antes de responder.

**El coste.** El razonamiento se paga en tokens de salida, los lentos y caros ([sección 2.5](#25-prefill-decode-y-kv-cache)): diez veces los tokens son, más o menos, diez veces la latencia y el precio, y los modelos de razonamiento tienen colas largas. Vienen con dos trampas prácticas: respuestas cortadas por `max_tokens` y errores al extraer la respuesta, que convierten en silencio una solución correcta en un fallo.

**Comparar estrategias: Bradley-Terry y jueces por pares.** Muchas evaluaciones comparan dos sistemas sobre la misma entrada en lugar de puntuar cada uno por separado. El modelo de **Bradley-Terry** (Bradley y Terry, 1952), el mismo de los modelos de recompensa de RLHF de la [sección 3.4](#34-de-rlhf-a-dpo-y-grpo), convierte esos resultados en puntuaciones:

$$P(i \succ j) = \frac{e^{\beta_i}}{e^{\beta_i} + e^{\beta_j}} = \sigma(\beta_i - \beta_j)$$

con las $\beta$ ajustadas por máxima verosimilitud (Chatbot Arena lo hace con millones de votos humanos; Elo es una aproximación online de la misma idea) y los intervalos sacados con bootstrap sobre las entradas. Cuando el juez es un LLM hay que medir sus sesgos (Zheng et al., 2023, *Judging LLM-as-a-Judge*): el **sesgo de posición** (preferir la primera o la segunda respuesta sin mirar el contenido), el de verbosidad y la autopreferencia. La defensa estándar es **juzgar cada par en los dos órdenes**, informar de cuántas veces el veredicto sobrevive al cambio y quedarse solo con los coherentes; y validar el juez contra la verdad conocida antes de fiarse de él ([sección 4.3](#43-ragas-y-llm-as-judge)).

| Estrategia | Coste extra | Ayuda cuando | Fallo típico |
|---|---|---|---|
| cadena de pensamiento | muchos más tokens de salida que la respuesta directa | problemas de varios pasos | razonamiento infiel o demasiado largo |
| self-consistency (maj@k) | $k$ veces la cadena de pensamiento | las respuestas se pueden comparar exactamente | se satura enseguida; no sirve para texto abierto |
| best-of-N con verificador | $k$ veces + la verificación | el verificador discrimina (tests, solvers, PRMs entrenados) | un verificador débil o hackeable elige respuestas incorrectas bien escritas |
| modelo de razonamiento | pensamiento largo, cobrado como salida | problemas difíciles con respuesta comprobable | overthinking, truncado, colas de coste |

**Papers clave.** Wei et al. (2022), *Chain-of-Thought Prompting Elicits Reasoning in Large Language Models*; Kojima et al. (2022), *Large Language Models are Zero-Shot Reasoners*; Wang et al. (2022), *Self-Consistency Improves Chain of Thought Reasoning*; Cobbe et al. (2021), *Training Verifiers to Solve Math Word Problems* (GSM8K); Chen et al. (2021), *Evaluating Large Language Models Trained on Code* (pass@k); Lightman et al. (2023), *Let's Verify Step by Step*; Turpin et al. (2023), *Language Models Don't Always Say What They Think*; Snell et al. (2024), *Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters*; DeepSeek-AI (2025), *DeepSeek-R1*; Muennighoff et al. (2025), *s1: Simple test-time scaling*; Zheng et al. (2023), *Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena*; Bradley y Terry (1952).

> **Notebook:** [20 · Razonamiento y cómputo en inferencia](../../notebooks/03_post_training/20_reasoning_test_time_compute.ipynb) (se lee después del 08) · **En el curso:** sobre 80 problemas de GSM8K con `qwen2.5:3b`, la respuesta directa acierta el 12.5 % y la cadena de pensamiento greedy el 85.0 % (246 tokens de salida frente a 3); la self-consistency con 5 muestras llega al 90.0 % (+5.0 puntos, intervalo pareado [+0.0, +11.2]) por unos 1.274 tokens por problema, mientras que pass@k se queda en el 91.2 %, así que votar ya saca casi todo lo que sacaría un verificador perfecto; best-of-5 con el juez de 7B como verificador saca un 85.0 % en los 40 primeros problemas, frente al 90.0 % del voto, porque da de media un 7.30 a las soluciones correctas y un 6.15 a las incorrectas; DeepSeek-R1-Distill 1.5B llega al 63.7 % con 870 tokens por problema, y 16 de sus 80 respuestas tocan el límite de 2.048 tokens; Bradley-Terry ordena self-consistency (+3.09) > cadena de pensamiento (+1.14) > R1-Distill (−0.58) > directa (−3.65) con intervalos que no se solapan; y el juez de 7B por pares, preguntado en los dos órdenes, mantiene el veredicto solo en el 43 % de 30 pares (el 52 % de sus veredictos van a la posición A, así que es más ruido que una preferencia fija).

---

## 4. Aplicaciones

Términos de esta parte: [glosario, Embeddings y rerankers](glosario.md#embeddings-y-rerankers), [Clustering y modelado de temas](glosario.md#clustering-y-modelado-de-temas), [Modelos multimodales](glosario.md#modelos-multimodales), [Aplicaciones: LangChain y RAG](glosario.md#aplicaciones-langchain-y-rag), [Sistemas de recomendación](glosario.md#sistemas-de-recomendación) y [Drift y aprendizaje online](glosario.md#drift-y-aprendizaje-online).

### 4.0 Embeddings y rerankers

**Problema.** El RAG, la búsqueda semántica, la recomendación y la memoria de los agentes empiezan todos por encontrar, entre miles o millones, los pocos documentos relevantes para una consulta. Leer cada documento con un modelo grande es impensable, así que recuperar es un equilibrio entre lo atentamente que lee un modelo y cuánto corpus se puede permitir leer.

**Tres formas de puntuar una consulta frente a un documento.**

| Arquitectura | Cómo puntúa | Coste en la consulta | Calidad |
|---|---|---|---|
| **bi-encoder** (modelo de embeddings) | consulta y documento codificados *por separado*, un vector cada uno; puntuación = similitud coseno | una pasada para la consulta y una búsqueda vectorial sobre vectores de documentos ya calculados | buena |
| **cross-encoder** (reranker) | consulta y documento entran *juntos* en un mismo transformer; una cabeza devuelve la relevancia | una pasada **por candidato** | la mejor |
| **interacción tardía** (ColBERT) | un vector por token; puntuación = suma, sobre los tokens de la consulta, del token del documento que mejor encaja (MaxSim) | búsqueda vectorial sobre vectores de tokens | intermedia |

El bi-encoder es rápido porque lo caro, codificar el corpus, se hace una vez y offline. El precio es que consulta y documento nunca se ven: todo lo que el modelo sabe de un documento tiene que caber en un vector antes de conocer la pregunta. Un cross-encoder (Nogueira y Cho, 2019) deja que cada token de la consulta atienda a cada token del documento, la atención completa del [apartado 1.3](#13-atención-rope-y-normalización), y por eso es más preciso y demasiado lento para todo un corpus. ColBERT (Khattab y Zaharia, 2020) guarda los vectores de los tokens y deja la interacción para un MaxSim barato. La búsqueda en producción usa el embudo de **recuperar y después reordenar** (*retrieve-then-rerank*), como en recomendación ([apartado 4.5](#45-sistemas-de-recomendación)): una primera etapa barata (BM25, embeddings o ambos) devuelve 50–100 candidatos y un cross-encoder los reordena. Un reranker solo puede subir lo que la primera etapa encontró.

**Cómo se entrenan los modelos de embeddings: aprendizaje contrastivo.** Un modelo de embeddings es un encoder ([apartado 1.6](#16-encoders-y-encoder-decoders)) con un paso de pooling (la media de los vectores de los tokens, o el vector `[CLS]`), como en Sentence-BERT (Reimers y Gurevych, 2019). Se entrena para que una consulta caiga cerca de sus documentos relevantes y lejos del resto, con **InfoNCE** (van den Oord et al., 2018) y **negativos del batch** (*in-batch negatives*): en un batch de $B$ pares (consulta, documento positivo), todos los demás documentos del batch son negativos para cada consulta.

$$\mathcal{L} = -\frac{1}{B}\sum_{i=1}^{B} \log \frac{\exp(\mathrm{sim}(q_i, d_i)/\tau)}{\sum_{j=1}^{B} \exp(\mathrm{sim}(q_i, d_j)/\tau)}$$

Es una entropía cruzada sobre el batch en la que la "clase" de la consulta $i$ es su propio documento. Tres detalles deciden si funciona bien:

- **Temperatura.** Las similitudes coseno viven en $[-1, 1]$; dividir por un $\tau$ pequeño (a menudo 0.01–0.05) hace la softmax lo bastante afilada como para aprender. `sentence-transformers` llama a esta pérdida `MultipleNegativesRankingLoss` y la escribe con un `scale` de $1/\tau = 20$.
- **Tamaño del batch.** Más negativos en el batch hacen la tarea más difícil y los embeddings mejores; por eso los modelos de embeddings se entrenan con batches de miles (CLIP usó 32.768) y existen trucos como GradCache.
- **Negativos difíciles** (*hard negatives*). Los negativos aleatorios son fáciles. Los documentos que BM25 o un modelo anterior ponen arriba pero que *no* son relevantes son los casos que de verdad confunden al modelo; minarlos es el ingrediente principal de DPR (Karpukhin et al., 2020) y de todas las recetas modernas. El riesgo es minar positivos sin etiquetar (falsos negativos), que se suelen filtrar con la puntuación de un reranker.

Los modelos modernos (E5, Wang et al., 2022; BGE, GTE, nomic-embed, Nussbaum et al., 2024) se entrenan en dos fases: un pre-entrenamiento contrastivo débilmente supervisado sobre cientos de millones de pares de textos que vienen emparejados de forma natural (título–cuerpo, pregunta–respuesta, consulta–clic) y un ajuste sobre conjuntos etiquetados más pequeños con negativos difíciles. Muchos son **asimétricos** y esperan un **prefijo de tarea**, como `search_query:` / `search_document:` (nomic) o `query:` / `passage:` (E5), porque una pregunta corta y un pasaje largo son textos de naturaleza distinta. Olvidar el prefijo falla en silencio: nada se rompe, solo baja la calidad.

**Vectores más baratos: Matryoshka y cuantización.** 100 millones de documentos con 768 dimensiones en float32 ocupan unos 300 GB de RAM. Dos ideas independientes lo reducen:

- **Matryoshka Representation Learning** (Kusupati et al., 2022) suma la pérdida contrastiva sobre varias longitudes de prefijo (768, 512, 256, 128, 64), de modo que las *primeras* $m$ dimensiones de cada vector son por sí solas un buen embedding y los vectores se pueden truncar a posteriori. Así están entrenados `text-embedding-3` de OpenAI (`dimensions=`) y nomic-embed v1.5. Un modelo entrenado sin ella no tiene motivo para poner lo importante al principio, y el truncado le hace mucho más daño.
- **La cuantización** guarda cada dimensión en int8 (4 veces más pequeño) o en un solo bit de signo (**binaria**, 32 veces más pequeño, comparada con la distancia de Hamming, que la CPU calcula en una instrucción). La receta estándar es recuperar con los códigos baratos y **reordenar** (*rescoring*) los mejores candidatos con vectores más precisos, lo que recupera casi toda la calidad perdida. Se combina con el truncado Matryoshka, y es lo que implementan las bases de datos vectoriales (cuantización binaria o por producto más rescoring).

**Cómo se mide la calidad de la recuperación.** El benchmark BEIR (Thakur et al., 2021), sobre el que se construye la parte de recuperación de MTEB (Muennighoff et al., 2023), informa de **nDCG@10**: un documento relevante en la posición $r$ suma $1/\log_2(r+1)$, y el total se divide por la mejor puntuación posible para esa consulta ([apartado 4.2](#42-métricas-de-recuperación)). Premia poner arriba los documentos relevantes. El **recall@100**, la fracción de documentos relevantes que entran en el top 100, es el material con el que trabaja un reranker. Los rankings públicos son un punto de partida, no una respuesta: MTEB promedia decenas de tareas, las puntuaciones en conjuntos públicos se pueden inflar entrenando con ellos, y el número que decide es el que se mide sobre *tus* consultas y documentos, con intervalos pareados ([apartado 6.5](#65-estadística-de-sets-de-evaluación-pequeños)).

| Palanca | Qué compra | Qué cuesta | Fallo típico |
|---|---|---|---|
| un modelo de embeddings moderno | coincidencia semántica, recall alto | una pasada del encoder por consulta; recodificar el corpus si cambia el modelo | caídas fuera de dominio; prefijos olvidados |
| ajuste contrastivo con negativos difíciles | mejoras grandes en el dominio con unos cientos de etiquetas | etiquetar; comprobar el olvido | falsos negativos entre los minados; sobreajuste |
| truncado Matryoshka | vectores más pequeños, como un dial | algo de calidad a tamaño completo | truncar un modelo que no se entrenó para ello |
| binario / int8 + rescoring | 32 / 4 veces menos memoria | un segundo paso de puntuación, pequeño | saltarse el rescoring |
| reranker cross-encoder | una lectura atenta de los mejores candidatos | una pasada por candidato | el dominio equivocado; una lista demasiado larga |

**Papers clave.** Reimers y Gurevych (2019), *Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks*; van den Oord, Li y Vinyals (2018), *Representation Learning with Contrastive Predictive Coding* (InfoNCE); Karpukhin et al. (2020), *Dense Passage Retrieval for Open-Domain Question Answering* (DPR); Khattab y Zaharia (2020), *ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT*; Nogueira y Cho (2019), *Passage Re-ranking with BERT*; Thakur et al. (2021), *BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models*; Muennighoff et al. (2023), *MTEB: Massive Text Embedding Benchmark*; Wang et al. (2022), *Text Embeddings by Weakly-Supervised Contrastive Pre-training* (E5); Kusupati et al. (2022), *Matryoshka Representation Learning*; Nussbaum et al. (2024), *Nomic Embed: Training a Reproducible Long Context Text Embedder*.

> **Notebook:** [21 · Embeddings y rerankers](../../notebooks/04_applications/21_embeddings_and_rerankers.ipynb) (se lee antes del 10) · **En el curso:** en SciFact (5.183 resúmenes, 300 afirmaciones de test), BM25 llega a un nDCG@10 de 0.652 y el clásico all-MiniLM-L6-v2 a 0.648, mientras que bge-small (0.706) y nomic-embed-text (0.708) lo superan con claridad, y olvidar los prefijos de nomic cuesta solo 0.006, dentro del ruido; InfoNCE a mano coincide con `MultipleNegativesRankingLoss` (0.18594); ajustar MiniLM con 919 pares lo sube a 0.723 con negativos del batch y a 0.762 con un negativo difícil de BM25 por par, en 22 s, sin olvido medible en NFCorpus (a ±0.005); con 64 dimensiones, el modelo Matryoshka conserva 0.646 frente a 0.589 sin esa pérdida; los códigos binarios con rescoring int8 sacan 0.760 frente a 0.762 de float32, con 48 bytes por documento en lugar de 1.536; y en esta tarea la recuperación híbrida (0.735) y los dos rerankers generales (ms-marco-MiniLM 0.692; bge-reranker-base 0.731 en el mejor caso, con 0.3–1.7 s por consulta) pierden frente al bi-encoder ajustado solo.

### 4.0b Clustering y modelado de temas

**Problema.** Un sistema con LLMs en producción genera montañas de texto sin etiquetar: mensajes de usuarios, tickets, ejecuciones fallidas de agentes, búsquedas, comentarios. Antes de construir un set de evaluación, diseñar un router o encontrar en qué falla el sistema, hay que saber qué hay ahí, y nadie lo ha etiquetado. El clustering agrupa textos parecidos; el modelado de temas dice además de qué va cada grupo.

**El pipeline moderno.** Desde BERTopic (Grootendorst, 2022), la receta estándar tiene cuatro pasos, y cada uno se puede cambiar:

| Paso | Elección habitual | Por qué |
|---|---|---|
| embeddings | un modelo de embeddings de frases ([sección 4.0](#40-embeddings-y-rerankers)) | los significados parecidos quedan cerca |
| reducir | UMAP a ~5 dimensiones | las distancias y las densidades vuelven a ser fiables |
| agrupar | HDBSCAN | encuentra el número de clusters y deja los casos atípicos como ruido |
| describir | palabras clave con c-TF-IDF y, después, un LLM que pone nombre al cluster | una persona puede saber qué es cada cluster |

**Tres familias de clustering.**

- **Por centroides: k-means** (MacQueen, 1967; inicialización k-means++, Arthur y Vassilvitskii, 2007) minimiza la distancia de cada punto al centroide de su cluster, $\sum_i \lVert x_i - \mu_{c(i)} \rVert^2$, alternando asignación y actualización. Es rápido, pero necesita $k$ por adelantado, supone clusters redondos y de tamaño parecido y asigna todos los puntos, también los atípicos.
- **Jerárquico (aglomerativo)**: empieza con cada punto solo y va fusionando el par más cercano (con enlace simple, completo, medio o de Ward). El resultado es un árbol que se puede cortar a cualquier altura, a cambio de una memoria $O(n^2)$ que lo limita a decenas de miles de puntos.
- **Por densidad: DBSCAN y HDBSCAN** (Campello, Moulavi y Sander, 2013). Los clusters son regiones densas separadas por otras poco densas. HDBSCAN construye una jerarquía sobre todos los niveles de densidad y se queda con los clusters más estables, así que **encuentra por sí solo el número de clusters**, acepta clusters de cualquier forma y tamaño y marca los puntos de las zonas poco densas como **ruido** en lugar de meterlos a la fuerza en un cluster. Su mando principal, `min_cluster_size`, es el grupo más pequeño que estás dispuesto a llamar tema.

**Por qué reducir antes: la maldición de la dimensionalidad.** Con 384–1.024 dimensiones, las distancias se concentran: el vecino más cercano y el más lejano de un punto acaban a distancias parecidas, y las estimaciones de densidad (de las que depende HDBSCAN) dejan de significar gran cosa. **UMAP** (McInnes, Healy y Melville, 2018) construye un grafo con los vecinos más cercanos de cada punto y busca una disposición en pocas dimensiones que conserve ese grafo: mantiene los vecindarios locales y descarta casi todo el ruido de las demás direcciones. Para agrupar, lo habitual es unas 5 dimensiones con `min_dist=0`; las 2 dimensiones son para dibujar, porque pierden demasiada estructura. PCA es lineal y conserva la varianza global, que no es lo que necesita el clustering; t-SNE es solo para gráficas.

**Describir los clusters: c-TF-IDF y nombres puestos por un LLM.** **c-TF-IDF** (TF-IDF por clases) concatena los documentos de cada cluster en uno solo y pondera cada palabra $w$ del cluster $c$ con

$$W_{w,c} = \mathrm{tf}_{w,c} \cdot \log\left(1 + \frac{A}{f_w}\right)$$

donde $\mathrm{tf}_{w,c}$ es la frecuencia de la palabra en el cluster, $A$ el número medio de palabras por cluster y $f_w$ la frecuencia total de la palabra en todos los clusters: frecuente aquí, rara en el resto. No necesita ningún modelo y no cuesta nada. Después, un **LLM** convierte las palabras clave principales y unos cuantos documentos de ejemplo en un nombre legible, el paso que hizo práctico el modelado de temas. Antes de los embeddings, los temas salían de **LDA** (Latent Dirichlet Allocation, Blei, Ng y Jordan, 2003): un modelo probabilístico en el que cada documento es una mezcla de temas y cada tema una distribución sobre palabras, ajustado sobre recuentos de bolsa de palabras. Ignora el orden y el significado más allá de la coaparición, funciona mal con textos cortos como los mensajes de un chat y te deja a ti la tarea de interpretar listas de palabras; se sigue usando con documentos largos y cuando importa tener un modelo generativo interpretable.

**Evaluar los clusters.**

- **Con etiquetas** (benchmarks o una muestra etiquetada a mano): el **adjusted Rand index** (ARI; Hubert y Arabie, 1985) cuenta los pares de puntos que están juntos o separados a la vez en el clustering y en las etiquetas, corregido por azar, de modo que 0 es aleatorio y 1 perfecto. La **información mutua normalizada** (NMI) mide cuánto te dice el cluster sobre la etiqueta, entre 0 y 1. La **pureza** es la proporción de puntos de un cluster que llevan su etiqueta mayoritaria; se lee fácil cluster a cluster, pero premia tener muchos clusters diminutos.
- **Sin etiquetas** (la situación real): el **silhouette score** de un punto es $s = (b - a) / \max(a, b)$, con $a$ su distancia media a su propio cluster y $b$ al cluster vecino más cercano, promediado sobre todos los puntos. Sirve para descartar valores malos de $k$, pero el texto real rara vez tiene un único número correcto de temas, porque los temas se anidan. En la práctica, una persona lee una muestra de cada cluster.

**Usos en producción.** Entender qué preguntan los usuarios, con recuentos y ejemplos, es el primer análisis sobre los logs de cualquier producto con LLMs. Los clusters dan además sets de evaluación que cubren las intenciones raras (muestreando de cada uno), una forma de encontrar fallos (agrupar las conversaciones con valoraciones negativas o llamadas a tools fallidas y arreglar el cluster más grande), intenciones y rutas diseñadas a partir de datos y la **detección de temas emergentes**: marcar los mensajes que quedan lejos de todo lo que el sistema sabe manejar (poca similitud con todas las intenciones conocidas), agrupar lo marcado, ponerle nombre y revisar cada semana los clusters más grandes. BERTopic empaqueta el pipeline, con temas a lo largo del tiempo y fusión jerárquica; cuML ejecuta UMAP y HDBSCAN en GPU para millones de documentos, o se agrupa una muestra y el resto se asigna al centroide más cercano.

**Modos de fallo.** Fiarse de los nombres del LLM sin leer los ejemplos; un cluster gigante que esconde varios temas; un modelo de embeddings que no entiende el dominio (códigos, jerga, otros idiomas); elegir $k$ por una métrica en lugar de por la decisión a la que sirve; volver a agrupar desde cero cada vez, de modo que los ids de los clusters no se conservan; datos personales en los mensajes de ejemplo que se enseñan en los dashboards.

**Papers clave.** MacQueen (1967), *Some Methods for Classification and Analysis of Multivariate Observations* (k-means); Arthur y Vassilvitskii (2007), *k-means++: The Advantages of Careful Seeding*; Campello, Moulavi y Sander (2013), *Density-Based Clustering Based on Hierarchical Density Estimates* (HDBSCAN); McInnes, Healy y Melville (2018), *UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction*; Blei, Ng y Jordan (2003), *Latent Dirichlet Allocation*; Grootendorst (2022), *BERTopic: Neural topic modeling with a class-based TF-IDF procedure*; Hubert y Arabie (1985), *Comparing Partitions* (ARI).

> **Notebook:** [25 · Clustering y modelado de temas](../../notebooks/04_applications/25_clustering_topic_modeling.ipynb) (se lee después del 21) · **En el curso:** con los 3.080 mensajes de test de Banking77 (77 intenciones, usadas solo para medir), k-means con la $k = 77$ verdadera llega a un ARI de 0.685 sobre una proyección UMAP de 5-d de los embeddings de bge-small, frente a 0.581 con los vectores tal cual y 0.576 con all-MiniLM-L6-v2 sobre UMAP; el silhouette score elige $k = 60$ y es plano entre 50 y 100 (0.721–0.742); HDBSCAN con `min_cluster_size=20` encuentra por su cuenta 73 clusters, con un ARI de 0.683 y un 5 % de ruido; el cluster mediano es en un 91 % de una sola intención, pero el más grande (122 mensajes) solo en un 32 %; los nombres de `qwen2.5:3b` apuntan a la intención mayoritaria en el 58 % de los clusters (42 de 73), y muchos de los fallos son correctos pero más amplios; y con siete intenciones apartadas como desconocidas, un marcado de novedades en el percentil 5 de la similitud de las intenciones conocidas llega a una precisión del 53 % y un recall del 56 %, y agrupar los mensajes marcados hace aparecer las 7 intenciones desconocidas como clusters propios.

### 4.0c Modelos multimodales: CLIP y modelos de visión y lenguaje

**Problema.** Muchas entradas no son texto: fotos de productos, facturas escaneadas, capturas de pantalla, gráficos. Dos familias de modelos llevan a las imágenes lo que el curso construyó para el texto, y hacen trabajos distintos: los **encoders contrastivos de imagen y texto** (CLIP, SigLIP) dan un vector por imagen y por texto en un espacio compartido, para clasificar, buscar y filtrar; los **modelos de visión y lenguaje** (VLMs) generan texto condicionado a imágenes, para describir, responder preguntas y leer documentos.

**Imágenes como tokens: el Vision Transformer.** ViT (Dosovitskiy et al., 2021) corta la imagen en parches de tamaño fijo (16×16 o 32×32 píxeles), aplana cada uno y lo proyecta linealmente a un vector (el **patch embedding**), suma embeddings de posición para la disposición en 2-D y pasa la secuencia por los mismos bloques que el texto ([sección 1.2](#12-el-bloque-transformer-y-el-flujo-residual)). Una imagen de 224×224 con parches de 32×32 son $7 \times 7 = 49$ tokens, más un token de clase cuyo vector final representa la imagen entera. Sin los sesgos inductivos de una CNN, ViT necesita muchísimos datos para entrenar, que es justo lo que aportan los pares imagen-texto a escala web.

**CLIP: un espacio compartido aprendido de los pies de foto.** CLIP (Radford et al., 2021) entrena a la vez un encoder de imagen $f$ y uno de texto $g$ con 400 millones de pares (imagen, pie de foto), usando la pérdida InfoNCE de la [sección 4.0](#40-embeddings-y-rerankers) en los dos sentidos: en un batch de $N$ pares, la imagen $i$ tiene que quedar más cerca del pie $i$ que de los otros $N-1$, y el pie $i$ más cerca de la imagen $i$ que de las demás.

$$\mathcal{L} = \tfrac{1}{2}\big(\mathrm{CE}_{\text{imagen}\to\text{texto}} + \mathrm{CE}_{\text{texto}\to\text{imagen}}\big), \qquad s_{ij} = \frac{f(x_i)^\top g(t_j)}{\tau}$$

Nadie etiqueta clases y, aun así, el espacio permite la **clasificación zero-shot**: se escribe una frase por clase, se calculan sus embeddings y cada imagen se queda con la clase de la frase más parecida. La redacción forma parte del modelo, igual que el prompt en un LLM. El nombre de la clase a secas es ambiguo ("boxer", ¿el perro o el boxeador?) y no se parece a los pies de foto que vio CLIP; una plantilla como "a photo of a {}, a type of pet." arregla las dos cosas, y el **prompt ensembling**, que promedia los embeddings de texto de varias plantillas por clase (los autores de CLIP usaron 80), suaviza las manías de cada redacción concreta. Con unas pocas imágenes etiquetadas por clase, el siguiente paso es un **linear probe** (una regresión logística sobre los embeddings de imagen congelados, la versión en imagen de embeddings + clasificador de la [sección 1.6](#16-encoders-y-encoder-decoders)); que supere al zero-shot depende de lo bien que los pies de foto de la web describan ya tus categorías.

**SigLIP.** La softmax de la pérdida de CLIP normaliza sobre todo el batch, lo que exige batches enormes y un all-gather caro entre dispositivos. SigLIP (Zhai et al., 2023) trata cada par (imagen, texto) del batch como un problema binario independiente, casan o no casan:

$$\mathcal{L} = -\frac{1}{N}\sum_{i=1}^{N}\sum_{j=1}^{N} \log \sigma\big(z_{ij}\,(t\, \mathbf{x}_i^\top \mathbf{y}_j + b)\big), \qquad z_{ij} = \begin{cases}+1 & i = j\\ -1 & i \neq j\end{cases}$$

con una temperatura $t$ y un sesgo $b$ aprendidos. Entrena mejor, también con batches más pequeños, y hay encoders SigLIP dentro de muchos VLMs actuales.

**VLMs: un encoder de imagen conectado a un LLM.** Dominan dos diseños:

| Diseño | Cómo llega la imagen al LLM | Ejemplos | Coste |
|---|---|---|---|
| **tokens visuales proyectados** | un encoder ViT/SigLIP produce vectores de parches; un **proyector** pequeño (una capa lineal o un MLP) los lleva al espacio de embeddings del LLM; entran en el prompt como tokens normales | LLaVA (Liu et al., 2023), Qwen-VL, InternVL | los tokens de imagen ocupan contexto y prefill |
| **cross-attention** | dentro de las capas del LLM se añaden bloques de cross-attention que leen las características de la imagen, como el decoder de un encoder-decoder lee al encoder | Flamingo (Alayrac et al., 2022), Llama 3.2 Vision | las imágenes no ocupan contexto; más parámetros nuevos |

El primero es el más sencillo y el más común. Suele entrenarse en dos pasos: primero se alinea el proyector con pies de foto, con el encoder y el LLM congelados, y después se hace instruction tuning con preguntas sobre imágenes.

**Resolución dinámica y lo que cuesta una imagen.** Los primeros VLMs, como LLaVA, redimensionaban todas las imágenes a un único tamaño fijo y pequeño, y el texto pequeño se volvía ilegible. Qwen2-VL y Qwen2.5-VL usan **resolución dinámica**: el número de tokens visuales crece con el de píxeles (un token por cada zona de 28×28 después de fusionar parches vecinos), y otros modelos parten las imágenes grandes en recortes. Se conserva el detalle, y la factura crece con él: **una imagen se paga en tokens**, es decir, en tiempo de prefill, en longitud de contexto y, con un proveedor, en dinero ([sección 2.8b](#28b-serving-bajo-carga)). En cuántos tokens se convierte una imagen depende de la pila de serving además del modelo, así que hay que medirlo.

**Ataques tipográficos: inyección de prompt a través de los píxeles.** Goh et al. (2021) vieron que CLIP lee tan bien el texto de las imágenes que una manzana con la etiqueta "iPod" se clasifica como un iPod. Un VLM es un LLM que lee sus tokens visuales, así que el texto que es capaz de leer en una imagen pasa a formar parte de su entrada y puede acabar obedeciéndolo, como cualquier texto no fiable ([sección 6.6](#66-seguridad-rechazos-jailbreaks-y-modelos-guardián)). Una captura de pantalla, un documento escaneado o una página web que ve un agente navegador pueden llevar instrucciones. Las defensas son las de la inyección de prompt: tratar el contenido de las imágenes como datos, no dar secretos ni herramientas peligrosas a un agente que lee imágenes y probar con imágenes adversarias.

| Uso | Familia | Por qué |
|---|---|---|
| clasificar en un conjunto fijo de etiquetas, buscar imágenes, deduplicar, filtrar | embeddings de CLIP / SigLIP (+ un linear probe si hay etiquetas) | milisegundos por imagen, un índice vectorial, fuertes en categorías comunes |
| pies de foto, preguntas sobre imágenes, respuestas abiertas, seguir instrucciones | VLM | genera texto; no hay un conjunto fijo de etiquetas |
| leer documentos (facturas, formularios, capturas) | VLM con salida estructurada, u OCR + modelo de maquetación + LLM | campos que un programa puede comprobar, con validación por reglas encima |

**Modos de fallo.** Confundir clases parecidas; objetos o campos alucinados; errores de OCR con texto pequeño o degradado; detalle perdido al reducir la imagen; texto de la imagen que se adueña de la respuesta; un conjunto de evaluación más limpio que los datos de producción.

**Papers clave.** Dosovitskiy et al. (2021), *An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale*; Radford et al. (2021), *Learning Transferable Visual Models From Natural Language Supervision* (CLIP); Zhai et al. (2023), *Sigmoid Loss for Language Image Pre-Training* (SigLIP); Alayrac et al. (2022), *Flamingo: a Visual Language Model for Few-Shot Learning*; Liu et al. (2023), *Visual Instruction Tuning* (LLaVA); Bai et al. (2025), *Qwen2.5-VL Technical Report*; Goh et al. (2021), *Multimodal Neurons in Artificial Neural Networks*.

> **Notebook:** [24 · Modelos multimodales](../../notebooks/04_applications/24_multimodal.ipynb) (se lee después del 21) · **En el curso:** con 600 imágenes de test de Oxford-IIIT Pet (37 razas), CLIP ViT-B/32 en zero-shot llega al 80.0 % con el nombre de la raza a secas, al 85.2 % con una plantilla y al 86.3 % con un ensemble de cinco, a 5.9 ms por imagen; un linear probe sobre sus embeddings congelados necesita las ~100 imágenes etiquetadas por raza para superarlo (89.0 %; 79.5 % con 16 y 34.3 % con una); `qwen2.5vl:3b` en Ollama, eligiendo entre los 37 nombres, acierta el 72.0 % de 150 imágenes, 11.3 puntos por debajo de CLIP con las mismas imágenes, a 1.45 s y unos 1.071 tokens de prompt por imagen, y esa cifra se queda en ~1.080 tanto si la imagen se limita a 224 como a 448 o 896 px (Ollama la redimensiona a un presupuesto fijo), mientras que limitarla a 224 px baja el acierto al 55 %; con un esquema JSON extrae los cinco campos de 40 facturas sintéticas, limpias y degradadas, en unos 2.3 s por documento, señal de que las facturas son demasiado fáciles; y con una raza equivocada escrita en una esquina de la imagen, CLIP sigue la etiqueta el 8 % de las veces y el VLM el 72 %.

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

que acerca paráfrasis pero difumina cadenas exactas, nombres y números, y cae fuera de su dominio (en BEIR, BM25 fue un baseline duro). Solo el bi-encoder permite pre-calcular los documentos; el **cross-encoder** lee consulta y documento juntos y se usa para reordenar los 20-100 primeros; ColBERT (interacción tardía) está en medio; el [apartado 4.0](#40-embeddings-y-rerankers) explica los tres, y cómo se entrenan, en detalle.

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

### 4.4b Prompts como programas: DSPy y optimización de prompts

**Problema.** Todos los prompts de los apartados anteriores se escriben y se ajustan a mano. Es lento, el resultado queda atado a un modelo (un prompt ajustado para un modelo suele funcionar peor con el siguiente), la redacción se mezcla con el código que parsea la respuesta, y nadie puede decir si existe un prompt mejor. Con varios pasos de LM en un pipeline, ajustar cada uno a mano contra una métrica de extremo a extremo no escala.

**Idea: separar qué hace un paso de cómo se le pide.** DSPy (Khattab et al., 2023) trata el prompt como un *parámetro* de un programa, igual que los pesos son parámetros de una red:

- una **signature** declara qué hace un paso como entradas y salidas con tipo (`message: str -> intent: Literal[...]`), con un docstring opcional como descripción de la tarea y nada sobre la redacción;
- un **módulo** decide cómo pedirlo: `Predict` pide las salidas directamente, `ChainOfThought` añade antes un campo `reasoning`, `ReAct` ejecuta un bucle con tools. Los módulos se componen como capas de PyTorch, así que recuperar → reordenar → responder es un solo programa con varios predictores;
- un **adapter** escribe el prompt concreto para un modelo y parsea la respuesta. El `ChatAdapter` por defecto pone en el mensaje de sistema los nombres de los campos, sus tipos y los valores permitidos, pide campos delimitados y comprueba los tipos de lo que vuelve. Cambiar el modelo o el adapter cambia el prompt sin tocar el programa;
- cada predictor guarda una **instrucción** y una lista de **demos** (ejemplos de entrada y salida que van en el prompt como turnos anteriores). Eso es lo que cambian los optimizadores.

**Optimizadores.** Dados un programa, una métrica y unos cientos de ejemplos etiquetados, un optimizador busca la instrucción y las demos que maximizan la métrica. De más barato a más caro:

| Optimizador | Qué cambia | Cómo |
|---|---|---|
| `LabeledFewShot` | demos | $k$ ejemplos etiquetados al azar, los mismos para todas las entradas |
| `KNNFewShot` | demos, **por entrada** | los $k$ ejemplos de entrenamiento más cercanos a la entrada en un espacio de embeddings (Liu et al., 2021) |
| `BootstrapFewShot(WithRandomSearch)` | demos | ejecuta el programa sobre datos de entrenamiento y se queda con las trazas que pasan la métrica, así que las demos pueden incluir pasos intermedios como el razonamiento; búsqueda aleatoria entre varios conjuntos de demos, puntuados en un set de validación |
| `MIPROv2` | instrucción y demos | un *modelo de prompts* propone instrucciones a partir de un resumen de los datos y del programa; una optimización bayesiana (TPE; Bergstra et al., 2011) elige la combinación instrucción × demos, puntuada en minibatches (Opsahl-Ong et al., 2024) |
| `GEPA` | instrucción | un *modelo de reflexión* lee las ejecuciones fallidas con el feedback en texto de la métrica y reescribe la instrucción, manteniendo un frente de Pareto de candidatas que ganan en ejemplos distintos (Agrawal et al., 2025) |

Descienden de la ingeniería automática de prompts, en la que un LLM propone prompts y una puntuación los selecciona (APE, Zhou et al., 2023; OPRO, Yang et al., 2024). Lo que produce un optimizador son datos, instrucciones y demos, guardados en JSON: un prompt optimizado es un artefacto versionado, que se revisa como el código y se regenera (no se edita a mano) cuando cambian el modelo, los datos o la métrica.

**La métrica es la especificación.** Un optimizador maximiza exactamente lo que mide la métrica, sobre exactamente los datos que recibe. Los sets de validación para buscar prompts son pequeños, así que la selección se sobreajusta: con 77 ejemplos una respuesta vale 1.3 puntos ([apartado 6.5](#65-estadística-de-sets-de-evaluación-pequeños)), y la mejor de muchas candidatas puntuadas en un set así está sesgada al alza: parte de por qué ganó es ruido que no se repetirá con datos nuevos. Una mejor puntuación en validación es una hipótesis hasta que un test apartado la confirma. Los optimizadores reflexivos añaden un riesgo propio: una instrucción escrita a partir de un puñado de fallos tiende a memorizarlos.

**Por qué los optimizadores no ayudaron en una búsqueda entre 77 clases.** Un optimizador busca *dentro* del espacio que le das, aquí una instrucción y unas pocas demos fijas. En una tarea como Banking77, lo difícil es saber cuál de 77 etiquetas que suenan parecido usa el banco para *este* mensaje. Lo que falta es información por entrada (qué mensajes etiquetados del entrenamiento se parecen a este), y ninguna instrucción puede aportarla, mientras que unas pocas demos fijas cubren unas pocas intenciones y arrastran al modelo hacia esas etiquetas. Recuperar los ejemplos etiquetados más cercanos a cada entrada sí la aporta. Con miles de ejemplos etiquetados y un buen modelo de embeddings, la tarea es sobre todo una búsqueda, así que el voto de los $k$ vecinos más cercanos, sin ningún LLM, es la referencia a batir ([apartado 4.0](#40-embeddings-y-rerankers)). La chain of thought tampoco ayuda: razonar sirve cuando la respuesta hay que *derivarla*, no *recordarla*. Los optimizadores compensan en programas de varios pasos, con una descripción de la tarea estable y margen real de mejora en cómo se formula cada paso.

**Cascadas y routers.** Cuando un componente barato suele estar seguro y suele acertar, que responda él y que solo el resto vaya al LLM: el patrón de cascada (FrugalGPT, Chen, Zaharia y Zou, 2023; [apartado 2.8b](#28b-serving-bajo-carga)). Aquí el componente barato es el voto de los $k$ vecinos, y su acuerdo (cuántos de los $k$ vecinos comparten la etiqueta mayoritaria) es la señal de confianza. El umbral de enrutado es un hiperparámetro, se elige en validación y nunca en test, y el resultado se juzga a la vez por precisión, latencia y llamadas al LLM por petición.

**Usos en producción.** Los programas declarativos compensan cuando el pipeline tiene varios pasos de LM, el modelo puede cambiar y hay una métrica con unos cientos de ejemplos etiquetados: reoptimizar para un modelo nuevo sale más barato que reescribir los prompts. Antes de optimizar la redacción, hay que arreglar el flujo de información (recuperar ejemplos o documentos para cada petición) y medir una referencia sin LLM. Las variantes se comparan emparejadas, sobre los mismos ejemplos. La optimización se paga una vez (cientos de llamadas) y la inferencia, siempre.

**Modos de fallo.** Un set de validación demasiado pequeño para separar a las candidatas, o que se solapa con los ejemplos de entrenamiento (un mensaje de entrenamiento que se recupera a sí mismo como demo infla la puntuación); una referencia débil; chain of thought por defecto; fiarse de la instrucción larga que escribió un optimizador reflexivo; demos fijas al azar en una tarea con muchas etiquetas; cambios silenciosos en el formato de salida al cambiar de modelo (hay que contar las respuestas que no se pueden parsear).

**Papers clave.** Khattab et al. (2023), *DSPy: Compiling Declarative Language Model Calls into Self-Improving Pipelines*; Opsahl-Ong et al. (2024), *Optimizing Instructions and Demonstrations for Multi-Stage Language Model Programs* (MIPROv2); Agrawal et al. (2025), *GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning*; Liu et al. (2021), *What Makes Good In-Context Examples for GPT-3?*; Zhou et al. (2023), *Large Language Models Are Human-Level Prompt Engineers* (APE); Yang et al. (2024), *Large Language Models as Optimizers* (OPRO); Bergstra et al. (2011), *Algorithms for Hyper-Parameter Optimization* (TPE); Chen, Zaharia y Zou (2023), *FrugalGPT: How to Use Large Language Models While Reducing Cost and Improving Performance*.

> **Notebook:** [28 · Optimización de prompts con DSPy](../../notebooks/04_applications/28_dspy_prompt_optimization.ipynb) (se lee después del 13) · **En el curso:** con la muestra de 231 mensajes de Banking77 del notebook 17 y `qwen2.5:3b`, el `Predict` zero-shot de DSPy llega a 0.558 [0.489, 0.623], frente al 0.489 del prompt escrito a mano del 17, sin ninguna respuesta que no se pueda parsear, y la chain of thought a 0.541 con más del doble de latencia; 16 demos al azar lo bajan a 0.433, mientras que los 4, 8 y 16 mensajes de entrenamiento más cercanos como demos lo suben a 0.818, 0.848 y 0.870; el voto mayoritario de los 8 vecinos más cercanos, sin LLM, llega a 0.913 con 0.02 s por mensaje; BootstrapFewShotWithRandomSearch (0.558), MIPROv2 (0.554) y GEPA (0.554) se quedan dentro del ruido del punto de partida tras 5–9 minutos y 650–870 llamadas al LM cada uno, y la instrucción de GEPA sacó un 59.7 % en validación; GEPA sobre el programa kNN se quedó con la instrucción original (0.870); un router que responde con el voto cuando al menos 4 de los 8 vecinos coinciden (umbral elegido en validación) llega a 0.922 con 0.03 s por mensaje, y el 97 % de los mensajes no llega nunca al LLM, frente a 0.926 con 1.62 s de `qwen2.5:7b` con 16 demos kNN; y el programa optimizado son 2.4 KB de JSON que dan la misma predicción en 30 de 30 mensajes después de recargarlo.

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

### 4.5b Drift, aprendizaje online y actualización continua

**Problema.** Un modelo se entrena con el pasado y se usa en el futuro. Cuando el mundo se mueve (usuarios nuevos, productos nuevos, preguntas nuevas, gustos que cambian), la calidad se degrada, casi siempre en silencio: nada da error, los resultados solo empeoran un poco cada semana. De ahí salen dos preguntas: cómo mantener el modelo al día y cómo darse cuenta de que hace falta actualizarlo, a menudo antes de que lleguen las etiquetas.

**Qué puede cambiar.** Si escribimos los datos como entradas $x$ y resultados $y$, con $P(x, y) = P(x)\,P(y \mid x)$:

| Tipo | Qué cambia | Ejemplo |
|---|---|---|
| **covariate shift** (cambio de covariables) | $P(x)$, con $P(y \mid x)$ fijo | usuarios de un país nuevo; un tipo nuevo de pregunta a un chatbot |
| **label (prior) shift** (cambio en las etiquetas) | $P(y)$ | un taquillazo se lleva buena parte de todas las visualizaciones; una oleada de fraude |
| **concept drift** (cambio de concepto) | $P(y \mid x)$ | el mismo historial lleva ahora a elecciones distintas, porque cambiaron los gustos o el catálogo |

El drift puede ser **repentino** (un lanzamiento, una caída, un cambio de política), **gradual** o **recurrente** (días laborables, temporadas). Parte lo provoca el propio modelo: un recomendador decide qué ven los usuarios y, por tanto, en qué hacen clic, así que el siguiente conjunto de entrenamiento refleja las decisiones del modelo anterior. Ese **bucle de retroalimentación** estrecha la exposición e infla el acierto aparente del modelo, salvo que lo rompan la exploración ([sección 4.5](#45-sistemas-de-recomendación)) o las propensiones registradas.

**Mantener los modelos frescos.** Las opciones cambian frescura por coste y riesgo:

- **reentrenar periódicamente** con todos los datos acumulados (estable, lento en reaccionar) o con una **ventana deslizante** que olvida lo antiguo (reacciona antes, pierde patrones raros de largo plazo);
- **warm start**: seguir entrenando el último modelo con los datos nuevos, mucho más barato que empezar de cero y lo normal con modelos grandes;
- **aprendizaje online (incremental)**: actualizar con cada evento o mini-batch, con SGD o con recuentos que decaen exponencialmente ($c \leftarrow \gamma^{\Delta t} c + 1$) para que pese más lo reciente. Es la opción más fresca, pero los datos malos entran en producción al instante, y las redes neuronales pueden sufrir **olvido catastrófico** de lo que sabían. Es el terreno del **aprendizaje continuo** (Gama et al., 2014; Lu et al., 2018).

Sea cual sea la estrategia, un modelo nuevo es un despliegue: se evalúa offline con los datos más recientes y se despliega poco a poco (un **canary** con una parte pequeña del tráfico y después un test A/B), con el modelo anterior listo para volver atrás.

**Evaluar sobre un flujo: la evaluación precuencial.** Una partición aleatoria en entrenamiento y test cuela el futuro en el entrenamiento. Sobre un flujo, el protocolo honesto es el **precuencial** ("test-then-train", Dawid, 1984; Gama et al., 2014): cada evento se predice primero con el modelo tal como está y solo después se usa para aprender. Todas las predicciones son fuera de muestra, y dibujar la puntuación a lo largo del tiempo enseña cómo envejece el modelo y cuánto frena la degradación cada estrategia de actualización.

**Detectar el drift sin etiquetas.** Las etiquetas suelen llegar tarde (una compra, una baja) o no llegar nunca, así que la monitorización vigila también las **entradas** y las **salidas**:

- **PSI** (population stability index) entre una distribución de referencia $q$ y la actual $p$ sobre unos intervalos $b$:

$$\mathrm{PSI} = \sum_b (p_b - q_b)\,\ln\frac{p_b}{q_b}$$

  una divergencia KL simetrizada; la regla práctica de la industria es < 0.1 estable, 0.1–0.25 moderado, > 0.25 un cambio importante. El test de **Kolmogorov-Smirnov** (la mayor diferencia entre dos funciones de distribución empíricas) y el de chi cuadrado hacen lo mismo variable a variable, con un p-valor; las divergencias **KL** y de **Jensen-Shannon** miden directamente la distancia entre distribuciones.
- **Con embeddings** (texto, imágenes), los tests variable a variable no funcionan: hay cientos de dimensiones y ninguna significa nada por separado. Tres tests comparan las dos nubes de vectores. El **desplazamiento del centroide** es la distancia entre sus medias: barato, pero ciego a los cambios que no mueven la media. La **maximum mean discrepancy** (MMD; Gretton et al., 2012) compara las muestras a través de un kernel,

$$\mathrm{MMD}^2 = \mathbb{E}[k(x, x')] + \mathbb{E}[k(y, y')] - 2\,\mathbb{E}[k(x, y)]$$

  con un p-valor sacado de un test de permutaciones (barajando de qué ventana viene cada punto). El **test de dos muestras con clasificador** (Lopez-Paz y Oquab, 2017) entrena un clasificador para distinguir los datos de referencia de los actuales: un AUC con validación cruzada claramente por encima de 0.5 indica que son distintos, y los aciertos del clasificador enseñan *dónde*. Rabanser, Günnemann y Lipton (2019) compararon estos métodos sobre muchos cambios distintos; lo que mejor funcionó fue reducir dimensiones antes y usar tests de dos muestras.

**Qué monitorizar.** Tres niveles, del más barato al que más significa: **entradas** (drift de variables y de embeddings, la proporción de ítems, usuarios o intenciones nunca vistos), **salidas** (distribución de las predicciones, confianza, tasas de rechazo y de fallback) y **resultados** (calidad online con etiquetas que llegan tarde: clics, compras, valoraciones, tareas completadas, feedback). Solo el último mide lo que importa; el drift en las entradas es una alarma para investigar, no una prueba de daño, y puede haber concept drift sin ningún drift en las entradas. Marcar las novedades elemento a elemento (como en la [sección 4.0b](#40b-clustering-y-modelado-de-temas)) detecta un tema nuevo mucho antes de que mueva la distribución de una ventana entera. Evidently, NannyML, WhyLabs y Arize implementan estas comprobaciones; en las aplicaciones con LLMs van junto a las trazas ([sección 6.1](#61-trazas-spans-y-opentelemetry-genai)).

**En aplicaciones con LLMs.** Los pesos de un modelo fijado no cambian, pero todo lo que lo rodea sí: lo que preguntan los usuarios, los documentos del índice del RAG, las APIs de las tools y el modelo del proveedor que hay detrás de un nombre de API. El reentrenamiento se sustituye por volver a pasar el set de evaluación con cada cambio, renovar el índice y el set de evaluación con tráfico de producción y seguir los clusters de intenciones nuevas.

**Modos de fallo.** Vigilar solo las distribuciones de entrada y que se escape el concept drift; alarmas de las que nadie se encarga; reentrenar automáticamente con datos corruptos o envenenados; bucles de retroalimentación; ventanas demasiado pequeñas para detectar nada; cambios silenciosos aguas arriba (un campo renombrado, un tokenizador nuevo, una versión nueva del modelo detrás del mismo nombre de API).

**Papers clave.** Gama, Žliobaitė, Bifet, Pechenizkiy y Bouchachia (2014), *A Survey on Concept Drift Adaptation*; Lu et al. (2018), *Learning under Concept Drift: A Review*; Gretton et al. (2012), *A Kernel Two-Sample Test*; Lopez-Paz y Oquab (2017), *Revisiting Classifier Two-Sample Tests*; Rabanser, Günnemann y Lipton (2019), *Failing Loudly: An Empirical Study of Methods for Detecting Dataset Shift*; Harper y Konstan (2015), *The MovieLens Datasets: History and Context*.

> **Notebook:** [26 · Aprendizaje online y drift](../../notebooks/04_applications/26_online_learning_drift.ipynb) (se lee después del 12) · **En el curso:** con MovieLens-1M reproducido en 22 ventanas semanales tras un corte de entrenamiento el 1 de agosto de 2000, un recomendador de co-visitación congelado cae de un hit@10 de 0.184 a 0.160, mientras que el reentrenado cada semana se mantiene en torno a 0.20 (0.207 frente a 0.175 en el conjunto de semanas, +18 %); la popularidad apenas envejece, y sus cuatro estrategias de actualización (congelada, semanal, ventana de 4 semanas, online con decaimiento) acaban a menos de 0.004 unas de otras; el PSI de la distribución de películas se queda en 0.01–0.03 todas las semanas, así que la degradación es un concept drift que la monitorización de entradas no ve, y un disparador de reentrenamiento con PSI > 0.25 no salta nunca; con Banking77 y embeddings de bge-small, los tres detectores mantienen las falsas alarmas en el 5 %, pero con 400 mensajes por ventana un cambio en el que el 20 % de los mensajes viene de intenciones nuevas solo se detecta el 25–30 % de las veces, mientras que con un 40 % MMD y el test con clasificador lo detectan siempre.

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

**Multi-agente.** Prompts más pequeños, menos herramientas por decisión, trabajo en paralelo; pero más llamadas y traspasos donde se pierde contexto. Los practicantes discrepan públicamente (Anthropic describió en 2025 un sistema de investigación multi-agente mejor que uno solo a mucho más coste en tokens; Cognition defendió ese mismo año agentes de un solo hilo). La discrepancia es sobre todo de tipo de tarea. La forma menos controvertida es un **router delante de especialistas**, y un router es un **clasificador**: set etiquetado, accuracy, matriz de confusión; un router que falla 1 de cada 10 limita el sistema a ~90 % aunque los especialistas sean perfectos. Los equipos que reparten una misma petición entre varios agentes están en [5.7](#57-equipos-de-agentes-supervisor-y-handoffs).

> **Notebook:** [14 · IA agéntica con LangGraph](../../notebooks/05_agents_and_mcp/14_agentic_ai.ipynb) · **En el curso:** con `qwen2.5:3b`, el agente ReAct supera a plan-and-execute con aproximadamente un tercio de las llamadas (plan-and-execute tiene más recall de herramientas y menos precisión); la reflexión con tests unitarios no reparó nada en una ejecución (4/8 antes y después) y alguna tarea en otras; el router acierta entre 67 y 75 % según la ejecución.

### 5.4 LangGraph: agentes como máquinas de estado

**Problema.** En 2023 los agentes más visibles (AutoGPT, BabyAGI) eran bucles abiertos: ejecuciones que no terminaban, acciones repetidas, objetivos a la deriva, costes impredecibles, imposible testear una parte o reanudar tras un fallo.

**Idea.** Devolver el control de flujo al código y dar al modelo control solo en puntos concretos. Una aplicación es un **grafo** sobre un estado tipado: los nodos leen el estado y devuelven actualizaciones; las aristas (fijas o condicionales) deciden el siguiente nodo; un **reducer** por clave combina actualizaciones (por ejemplo, `add_messages` añade en vez de sobrescribir). El runtime sigue el modelo **Pregel / bulk synchronous parallel** (Malewicz et al., 2010): en cada *super-step* se ejecutan todos los nodos activados, sus actualizaciones se fusionan con los reducers (eso hace seguras las ramas paralelas) y las aristas deciden el siguiente super-step.

**Propiedades que da.** Transiciones explícitas que se testean con modelos falsos; bucles visibles donde poner límites; **checkpoints** (el estado solo cambia entre nodos, así que se guarda tras cada paso: reanudar, pausar para un humano el tiempo que haga falta, "viajar en el tiempo"); y autonomía mixta (una rama guionizada junto a un bucle ReAct libre). `create_agent` es un grafo de dos nodos (`model`, `tools`) con una arista condicional que codifica toda la regla de parada de ReAct: si el último mensaje tiene llamadas, ir a `tools`; si no, terminar.

> **Notebook:** [14 · IA agéntica con LangGraph](../../notebooks/05_agents_and_mcp/14_agentic_ai.ipynb) · **En el curso:** un grafo de triaje con funciones puras para ver la maquinaria, el agente ReAct, plan-and-execute, reflexión y un router con sub-grafos, todos testeados con modelos falsos en `tests/test_agents_langgraph.py`, que además lanza código de ataque real contra el sandbox de la sección 11.

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

### 5.7 Equipos de agentes: supervisor y handoffs

**Problema.** Un router (5.3) manda cada petición a *un* especialista. Muchas peticiones necesitan dos o más ("¿cuánto sale mi pedido con envío exprés?" necesita facturación y la política de envíos), y un único agente con todas las herramientas y todas las instrucciones acaba con un prompt largo y más elecciones de herramienta equivocadas.

**Dos diseños.** En un diseño con **supervisor** (también llamado *agents as tools*, u *orchestrator-workers* en *Building effective agents* de Anthropic, 2024) un agente supervisor llama a agentes especialistas como si fueran herramientas: escribe una petición, recibe la respuesta final del trabajador, puede volver a llamar y redacta él la respuesta. El control siempre vuelve a él. En un diseño con **handoffs** (el estilo *swarm*: Swarm y el Agents SDK de OpenAI, `langgraph-swarm`) el agente activo *traspasa la conversación* a otro agente, que toma el relevo y puede volver a traspasarla; responde el que termina. En LangGraph, un trabajador como herramienta es una función que ejecuta un agente y devuelve su último mensaje, y un handoff es una herramienta que devuelve `Command(goto="billing", graph=Command.PARENT)`: la herramienta se ejecuta dentro del grafo del propio agente, así que el salto tiene que darse un nivel más arriba, en el grafo padre que contiene a todos los agentes.

| | supervisor | handoffs |
|---|---|---|
| Quién tiene el control | el supervisor, siempre | el agente activo en cada momento |
| Qué ve un especialista | solo la petición que escribió el supervisor | la conversación que le pasan |
| Quién escribe la respuesta | el supervisor | el último agente |
| Fallo típico | la petición o la respuesta pierde un detalle | un agente traspasa antes de tiempo, o el siguiente no entiende que le toca |
| Coste | llamadas del supervisor + las de cada trabajador | un agente cada vez, más los traspasos |

**Dónde discrepa la industria.** El sistema de investigación multi-agente de Anthropic (2025) usa un agente principal que planifica y lanza subagentes en paralelo, cada uno con su propia ventana de contexto, y mostró una mejora grande sobre un agente solo en preguntas de investigación amplias, a cambio de unas 15 veces los tokens de un chat; buena parte de la mejora venía de poder gastar más tokens en paralelo. *Don't Build Multi-Agents* de Cognition (2025) defiende lo contrario para trabajo muy acoplado, como programar: cada acción lleva decisiones implícitas, los subagentes que no ven el contexto completo de los demás toman decisiones que chocan, y un único agente con un contexto continuo (comprimido cuando crece) es más fiable. Las dos posturas encajan en cuanto se dice de qué tarea se habla: divide el trabajo **paralelo y poco acoplado**, y deja en un solo agente el trabajo **secuencial y muy acoplado**.

**Ingeniería de contexto en la frontera.** Cada traspaso es una compresión con pérdidas del contexto: un trabajador solo sabe lo que le cuentan, y el supervisor solo sabe lo que el trabajador le devuelve. Lo que cruza la frontera forma parte del diseño: una petición autocontenida (identificadores, cifras y restricciones escritos explícitamente), las respuestas anteriores que necesita el siguiente agente, y nada de la maquinaria del traspaso que pueda leer como "trabajo terminado". Un equipo añade llamadas al modelo y puntos de fallo, así que tiene que ganarse su sitio en el mismo conjunto de evaluación que el agente solo.

> **Notebook:** [14 · IA agéntica con LangGraph](../../notebooks/05_agents_and_mcp/14_agentic_ai.ipynb), sección 10 · **En el curso:** con las diez tareas de la evaluación de ReAct y `qwen2.5:3b`, el supervisor resolvió el 80 % (5.3 llamadas al modelo por tarea) frente al 70 % del agente solo (2.2 llamadas), un empate en calidad a más del doble de coste, y los handoffs el 30 % (3.4 llamadas). La primera versión con handoffs fallaba porque el especialista leía "Transferred to the billing specialist" como el final del trabajo y respondía en blanco; ocultar los mensajes del traspaso y añadir una nota de "te toca" ayudó, pero los agentes seguían traspasando antes de hacer su parte.

### 5.8 Ejecución de código y sandboxes

**Por qué dejar que el modelo escriba código.** Los modelos de lenguaje calculan mal y programan bien. Program-Aided Language models (PAL, Gao et al., 2022) y Program of Thoughts (Chen et al., 2022) hacen que el modelo escriba un programa corto para el razonamiento y dejan la aritmética al intérprete, y los dos superan a la cadena de pensamiento en problemas matemáticos con enunciado. CodeAct (Wang et al., 2024) lo convierte en un diseño de agente: las *acciones* del agente son fragmentos de Python en lugar de llamadas a herramientas en JSON, así que una sola acción puede hacer bucles, ramificar, combinar varias herramientas y reutilizar variables, y en sus experimentos las acciones en código resolvieron más tareas en menos turnos. Los intérpretes de código (el análisis de datos de ChatGPT, la herramienta de ejecución de código de Claude, los agentes de análisis de datos) son esta idea hecha producto.

**Feedback de ejecución.** Ejecutar el código le da al modelo una señal que no puede sacar de su propio texto: un traceback, una salida vacía, un número impreso. Un bucle que le devuelve esa señal y le deja reintentar arregla errores y llamadas mal hechas a una API. No arregla un código que se ejecuta y calcula otra cosa, porque nada avisa del error; para eso hace falta un verificador (salidas intermedias, invariantes, tests, un paso de revisión), como con la reflexión en 5.3.

**La interfaz agente-ordenador.** SWE-agent (Yang et al., 2024) mostró que el mismo modelo resuelve muchas más tareas de ingeniería de software cuando sus herramientas se diseñan para un modelo y no para una persona: un visor de ficheros que enseña una ventana de líneas, un editor que rechaza ediciones que no compilan, feedback breve y explícito, y un mensaje claro cuando un comando termina bien sin salida en lugar de una cadena vacía. Los autores lo llaman **interfaz agente-ordenador** (*agent-computer interface*, ACI), por analogía con las interfaces persona-ordenador. Un modelo que recibe un resultado vacío tiende a inventarse uno, así que la redacción de los mensajes de error y de salida vacía de una herramienta es parte del diseño del agente.

**El código que escribe el modelo es código no fiable.** Al modelo lo puede manipular cualquier cosa que lea (inyección de prompts indirecta, [4.4](#44-patrones-de-aplicación)), así que su código se trata como código de un desconocido. El modelo de amenazas y la capa que contiene cada riesgo:

| Riesgo | Ejemplo | Contención |
|---|---|---|
| leer secretos | `open(".env")`, `os.environ` | ningún secreto en el sandbox: entorno limpio, nada montado |
| dañar ficheros | `os.remove`, escribir fuera del directorio de trabajo | un directorio de trabajo efímero, un sistema de ficheros raíz de solo lectura |
| exfiltración, ataques | peticiones HTTP, sockets | sin red, o una lista blanca de salida |
| escapar del proceso | `subprocess`, `os.system`, llamadas nativas | aislamiento del kernel: un kernel en espacio de usuario o una microVM |
| ejecutarse para siempre, agotar la memoria | `while True`, una reserva enorme | timeout de reloj, cuotas de CPU y memoria (cgroups) |
| inundar el contexto | imprimir un fichero de 1 GB | salida truncada |

**Los audit hooks no son una frontera de seguridad.** La PEP 578 (Python 3.8) añadió los audit hooks: el intérprete lanza un evento en `open`, en las llamadas a sockets, al crear procesos, al cargar librerías con `ctypes` y en muchos sitios más, y un hook instalado con `sys.addaudithook` puede vetarlo lanzando una excepción. Se diseñaron para registrar y detectar, y tanto la PEP como la documentación de CPython dicen que no son un sandbox: el código que llega a código nativo (`ctypes`, una extensión en C, un fallo de corrupción de memoria) se los salta, porque los hooks solo ven lo que el intérprete les cuenta. Dentro de un proceso aparte con timeout y entorno limpio sirven para enseñar y como cable trampa, no como capa de aislamiento.

**Qué se usa en producción.** El aislamiento tiene que venir de debajo de Python, del sistema operativo o de un hipervisor. De menos a más fuerte: un **contenedor** sin red, con el sistema de ficheros de solo lectura y sin secretos (`docker run --network none --read-only`), que aun así comparte el kernel del host, de modo que un fallo del kernel puede ser una vía de escape; **gVisor**, un kernel en espacio de usuario que intercepta las llamadas al sistema del contenedor, así que el código nunca habla directamente con el kernel del host (los sandboxes de Modal funcionan sobre él); una **microVM** como **Firecracker**, el monitor de máquinas virtuales basado en KVM que hay detrás de AWS Lambda, que arranca una VM mínima en una fracción de segundo y da a cada ejecución su propio kernel (los sandboxes de E2B son microVMs de Firecracker); y **WebAssembly** (Pyodide ejecuta CPython compilado a Wasm), donde el código solo alcanza las capacidades que el anfitrión le concede explícitamente. Las herramientas de ejecución de código gestionadas de los proveedores de modelos ejecutan el código con este tipo de aislamiento en su lado. Encima de cualquiera de ellas: cuotas de CPU, memoria, tiempo y salida, y un registro de cada ejecución.

**Acotar el daño.** Los prompts y los filtros bajan la tasa de inyecciones que funcionan, pero no la dejan en cero. La regla de diseño es suponer que tarde o temprano alguien convencerá al modelo de ejecutar el código equivocado, y asegurarse de que ese código no tenga nada a su alcance. En particular, nunca des a un mismo agente las tres cosas a la vez: acceso a datos privados, exposición a contenido no fiable y una forma de enviar datos fuera (lo que Simon Willison llama la *lethal trifecta*, 2025).

> **Notebook:** [14 · IA agéntica con LangGraph](../../notebooks/05_agents_and_mcp/14_agentic_ai.ipynb), sección 11 · **En el curso:** un sandbox con subproceso (`python -I`, un directorio de trabajo nuevo, entorno limpio, timeout, salida truncada a 2.000 caracteres y un audit hook) contiene 8 de 9 ataques de prueba; el noveno, una llamada nativa a través de `ctypes`, se cuela a propósito, porque numpy necesita `ctypes` en Windows. Un agente con pandas acierta 0/10 preguntas exactas sobre una tabla de 600 pedidos con `qwen2.5:3b`, 5/10 con `qwen2.5:7b` y una ejecución, y 6/10 con hasta cuatro ejecuciones y su feedback; los cuatro fallos se ejecutaron sin errores e imprimieron números equivocados. Con una instrucción inyectada en un fichero de datos, ninguno de los 2 secretos del proceso del notebook estaba al alcance del sandbox. El sandbox se prueba en `tests/test_agents_langgraph.py`.

### 5.9 Model Context Protocol

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

**Las funciones del cliente en la práctica.** Las tres peticiones de servidor a cliente solo llegan a un cliente que haya declarado la capacidad, y cada una lleva una decisión al sitio que le corresponde. **Sampling** (`sampling/createMessage`) permite a un servidor pedir una respuesta al modelo del host: el servidor nunca tiene modelo ni clave de API propios, y el host puede enseñar la petición al usuario, cambiar de modelo, limitar los tokens o negarse (la especificación recomienda que una persona pueda revisar tanto la petición como el resultado). **Elicitation** (`elicitation/create`, añadida en la especificación 2025-06-18) permite a un servidor pedir al *usuario* datos estructurados descritos con un esquema JSON pequeño, con tres respuestas posibles, *accept*, *decline* y *cancel*; es la versión a nivel de protocolo de una puerta de aprobación, porque al modelo se le puede convencer de llamar a una herramienta destructiva, pero no puede rellenar el formulario por la persona. Los servidores no deben usarla para pedir contraseñas ni otros secretos. **Roots** (`roots/list`) le dicen a un servidor en qué carpetas o repositorios puede trabajar. Son orientativas: un servidor cooperativo se queda dentro, uno malicioso puede ignorarlas, así que acotan a los servidores que se portan bien y no sustituyen al aislamiento del sistema operativo ([5.8](#58-ejecución-de-código-y-sandboxes)).

**Autorización en servidores remotos.** Un servidor stdio se ejecuta como el usuario y no necesita autenticación (las credenciales le llegan por su entorno). Un servidor Streamable HTTP es un endpoint al que puede llegar cualquiera, y desde la especificación 2025-03-26 (rehecha en la 2025-06-18) MCP usa OAuth 2.1 estándar, con el servidor MCP como **servidor de recursos** y un servidor de autorización aparte que emite los tokens:

1. Una petición sin token válido recibe `401 Unauthorized` con una cabecera `WWW-Authenticate: Bearer ... resource_metadata="..."`.
2. Esa URL sirve los **metadatos del recurso protegido** del servidor (RFC 9728, bajo `/.well-known/oauth-protected-resource`): en qué servidores de autorización confía y qué scopes admite.
3. El cliente descubre los metadatos del propio servidor de autorización (RFC 8414), se registra si hace falta (registro dinámico de clientes, RFC 7591) y obtiene un token con el flujo de código de autorización y PKCE. Indica el servidor MCP como audiencia con un **resource indicator** (RFC 8707).
4. Cada petición lleva `Authorization: Bearer <token>`, y el servidor lo valida cada vez: firma y caducidad si es un JWT (o una llamada de introspección), los scopes y la **audiencia**, para rechazar un token emitido para otro servicio.

Dos reglas evitan los fallos clásicos de *confused deputy*. **Nada de token passthrough**: un servidor no debe aceptar tokens que no se emitieron para él, ni reenviar el token del cliente a las APIs que llama por detrás; para esas consigue sus propias credenciales. **Validar `Origin`**: un servidor Streamable HTTP tiene que comprobar la cabecera `Origin` de las peticiones, y uno local debería escuchar solo en `127.0.0.1`; si no, una página web abierta en el navegador del usuario puede llegar a él mediante DNS rebinding.

> **Notebook:** [16 · MCP con el SDK oficial](../../notebooks/05_agents_and_mcp/16_mcp_model_context_protocol.ipynb) · **En el curso:** los 14 checks de conformidad pasan; `search_docs` acierta 6/6 en top-1 en un set fácil; el agente responde con el dato clave en 6 de 6 y cita el fichero correcto en 4 de 6; un `tools/call` cuesta ~1.6 ms en memoria, ~3.0 ms por stdio y ~5.2 ms por Streamable HTTP con validación del token, frente a ~2.4 s por llamada con el adaptador que abre una sesión nueva cada vez. La sección 9 enseña sampling (dos resúmenes hechos con el modelo del host, que ve cada petición), elicitation (decline, accept y cancel respetados) y roots (la raíz del repositorio, rechazada), y ejecuta `scripts/mcp_server.py --http --require-token`: 401 con `WWW-Authenticate` sin token o con uno equivocado, metadatos del recurso protegido con el servidor de autorización y el scope `docs:read`, y las mismas tools con el token.

---

## 6. LLMOps

Términos de esta parte: [glosario, LLMOps y LangSmith](glosario.md#llmops-y-langsmith) y [Seguridad y guardrails](glosario.md#seguridad-y-guardrails).

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

### 6.5b Frameworks de evaluación y de trazas

**Problema.** Un bucle de evaluación escrito a mano y un servicio de trazas alojado sirven para aprender, y los dos dejan de escalar en un equipo. Los bucles que escriben personas distintas toman decisiones distintas (prompt, decodificación, extracción de la respuesta, qué cuenta como error) que rara vez quedan registradas, así que sus números no se pueden comparar entre personas ni entre meses. Un servicio de trazas alojado es una dependencia, un coste, una cuota y un sitio adonde van los datos de tus usuarios. Los frameworks estandarizan las dos cosas, y los protocolos abiertos evitan quedar atado a un proveedor.

**Un framework de evaluación separa las partes de una evaluación,** para que cada una se pueda cambiar y quede registrada:

| Parte | Qué es | Ejemplos |
|---|---|---|
| dataset | muestras con entrada, objetivo y metadatos | 80 problemas de GSM8K |
| solver | lo que se hace con cada muestra | un mensaje de sistema, una generación, chain of thought, uso de tools, un bucle de agente en un sandbox |
| scorer | cómo se califica una salida, con sus métricas | exact match, una regex, una comparación numérica, un modelo calificador; la precisión y su error estándar |
| log | cada muestra con su transcripción, puntuaciones, tokens y tiempos, más la configuración de la ejecución | un fichero que se puede reabrir, comparar y compartir |
| epochs | cada muestra se ejecuta varias veces y se reduce (media, mayoría) | `--epochs 3` |

Hay dos familias. Los **harness de benchmarks** para checkpoints de modelos (el lm-evaluation-harness de EleutherAI, que alimenta los leaderboards; HELM, Liang et al., 2022) ejecutan tareas fijas, a menudo puntuadas con log-probabilidades. Los **frameworks de evaluación de sistemas** (Inspect AI, del UK AI Security Institute, OpenAI Evals, promptfoo, DeepEval) evalúan prompts, tools, agentes y sandboxes contra cualquier API. En los dos casos, un framework se adopta solo después de que reproduzca un número en el que ya confías.

**El scorer es parte de la métrica.** Las mismas generaciones calificadas con dos extractores de respuesta dan dos precisiones. Una regla de "último número" y una comparación numérica que saca otro número de una respuesta con `\boxed{}` de LaTeX discrepan justo donde el formato de la salida se aparta de lo que espera la regla. Los detalles de implementación de este tipo son una de las razones principales de que los números publicados para un mismo modelo no coincidan (Biderman et al., 2024). Una puntuación se publica con su scorer, y el scorer se fija como el código.

**Epochs y varianza.** Con un modelo que muestrea, una ejecución es una tirada. Con $n$ preguntas, una probabilidad de acierto por pregunta $\mu_i$ y $K$ muestras por pregunta, la varianza de la puntuación media se divide en dos (Miller, 2024):

$$\mathrm{Var}(\hat{\mu}) \approx \frac{1}{n}\left(\mathrm{Var}_i(\mu_i) + \frac{\mathbb{E}_i\big[\mu_i(1-\mu_i)\big]}{K}\right)$$

El primer término es *qué preguntas te tocaron*, y solo se reduce con más preguntas. El segundo es *qué muestra te tocó en cada pregunta*, y lo reducen las epochs ($K > 1$). Cuando casi todas las preguntas se aciertan siempre o se fallan siempre, $\mu_i(1-\mu_i) \approx 0$ para ellas y domina el primer término: unos $\sqrt{0.85 \cdot 0.15 / 80} \approx 4$ puntos de error estándar para un 85 % en 80 problemas ([apartado 6.5](#65-estadística-de-sets-de-evaluación-pequeños)). Así que se publica la media sobre las epochs, su dispersión y un intervalo de confianza, y los sistemas se comparan emparejados sobre las mismas preguntas.

**Trazas sobre OpenTelemetry.** El modelo de trazas y spans del [apartado 6.1](#61-trazas-spans-y-opentelemetry-genai) se vuelve portable gracias a tres piezas:

- **convenciones semánticas** para los nombres de los atributos: las `gen_ai.*` de OpenTelemetry y **OpenInference**, el conjunto que usan Phoenix y otros (`llm.model_name`, `llm.token_count.prompt`, `input.value`, `output.value`, tipos de span `LLM`, `CHAIN`, `RETRIEVER`, `TOOL`...);
- **instrumentadores**, que crean los spans de una librería automáticamente (LangChain, LlamaIndex, el SDK de OpenAI) sin tocar el código de la aplicación;
- **exportadores**, que envían los spans por OTLP (HTTP o gRPC), en lotes desde un hilo de fondo, a cualquier colector: Phoenix, Langfuse, LangSmith, Datadog o cualquier backend de OTel. Cambiar de backend es un cambio de configuración, no una reescritura.

**Las anotaciones cierran el bucle.** Una puntuación o una etiqueta pegada a un span a posteriori (por código, por un juez LLM o por una persona) convierte el almacén de trazas en un almacén de evaluaciones: las trazas fallidas se pueden filtrar, leer, etiquetar y añadir al dataset offline ([apartado 6.2](#62-evaluación-offline-y-online)). El bucle de producción es trazar, puntuar, mirar los fallos y arreglar.

**Sobrecoste.** La instrumentación corre dentro del camino de la petición, pero registra unas pocas marcas de tiempo y atributos por span y los exporta de forma asíncrona, así que cuesta milisegundos frente a llamadas al LLM que tardan segundos. Las razones para muestrear trazas son el almacenamiento y la privacidad ([apartado 6.4](#64-pii-y-muestreo-de-trazas)), no la velocidad. Las trazas guardan los prompts y las salidas tal cual, así que un almacén de trazas son datos sensibles.

**Cómo elegir.** Autoalojado y abierto (Phoenix corre como un solo proceso con SQLite; Langfuse y Opik necesitan Docker y una base de datos) cuando los datos no pueden salir de tu infraestructura; gestionado (LangSmith, Arize, Datadog LLM Observability) cuando no quieres trabajo de operaciones. Los dos lados hablan OpenTelemetry, así que el código de instrumentación es el mismo.

**Usos en producción.** Evaluaciones como código en el repositorio (dataset + solver + scorer), ejecutadas en CI con cada cambio de prompt, de modelo o de recuperación, con los logs guardados como artefactos, el scorer fijado, intervalos de confianza y epochs para todo lo que muestrea. Trazas instrumentadas a nivel de framework y exportadas en lotes a un backend que controles cuando los prompts llevan datos de usuarios, con las evaluaciones pegadas a las trazas (checks por código, jueces LLM, feedback de usuarios) y alertas de latencia, tasa de errores y deriva de las puntuaciones ([apartado 4.5b](#45b-drift-aprendizaje-online-y-actualización-continua)).

**Modos de fallo.** Una evaluación que cambia su scorer sin avisar; comparar números de extractores de respuesta distintos; una sola ejecución con muestreo publicada como "la" precisión; trazas que capturan secretos o datos personales; dashboards que nadie mira; sets de evaluación que se cuelan en los prompts o en los ejemplos few-shot.

**Papers clave.** Liang et al. (2022), *Holistic Evaluation of Language Models* (HELM); Biderman et al. (2024), *Lessons from the Trenches on Reproducible Evaluation of Language Models* (lm-evaluation-harness); Miller (2024), *Adding Error Bars to Evals: A Statistical Approach to Language Model Evaluations*; Cobbe et al. (2021), *Training Verifiers to Solve Math Word Problems* (GSM8K); Shankar et al. (2024), *Who Validates the Validators? Aligning LLM-Assisted Evaluation of LLM Outputs with Human Preferences*; Sigelman et al. (2010), *Dapper, a Large-Scale Distributed Systems Tracing Infrastructure*; y, como referencias más que como papers, la documentación de Inspect AI, la especificación de OpenInference y las convenciones semánticas GenAI de OpenTelemetry.

> **Notebook:** [29 · Frameworks de evaluación y trazas](../../notebooks/06_llmops/29_eval_and_tracing_frameworks.ipynb) (se lee después del 15) · **En el curso:** como tarea de Inspect AI, los 80 problemas de GSM8K del notebook 20 con su regla del último número dan un 85.0 % [77.5, 92.5] con `qwen2.5:3b`, exactamente el resultado del bucle escrito a mano del 20, mientras que el `match(numeric=True)` que trae Inspect puntúa las mismas generaciones con un 80.0 % (−5.0 puntos [−10.0, −1.2]; las 4 discrepancias terminan todas en `\boxed{}`); a temperatura 0.7, tres epochs dan 80.0 %, 82.5 % y 81.2 % (media 81.2 %), con el 80 % de los problemas acertados en las tres, el 17.5 % en ninguna y solo 2 problemas que cambian entre epochs; en Phoenix, cada una de las 20 peticiones de LangChain trazadas es una traza de 5 spans, el span del LLM tarda 3.585 ms de p50 y 4.641 ms de p95 y todos los demás pasos menos de 1 ms, con 97 tokens de prompt y 242 de respuesta por llamada; las 20 anotaciones puntuadas por código sobre los spans raíz se leen de vuelta como un 80 %; y con un modelo falso e instantáneo, el tracing añade 2.5 ms por petición (3.52 frente a 1.02 ms), el 0.07 % de una llamada real al LLM.

### 6.6 Seguridad: rechazos, jailbreaks y modelos guardián

**Problema.** La observabilidad te dice qué hizo el sistema; la seguridad trata de lo que debería hacer y de lo que no. Un modelo que lo responde todo es un riesgo, y uno que se niega en cuanto ve una palabra alarmante no sirve para nada. Por eso la seguridad tiene **dos tasas de error**, las peticiones dañinas que se responden y las seguras que se rechazan, y cada defensa mueve las dos.

**De dónde salen los rechazos.** Un modelo preentrenado no tiene noción de daño: continúa texto. A rechazar se le enseña en el post-training ([sección 3.4](#34-de-rlhf-a-dpo-y-grpo)): SFT con ejemplos de rechazos educados y después un ajuste por preferencias (RLHF, DPO) que convierte el rechazo a una petición dañina en la respuesta preferida. Constitutional AI (Bai et al., 2022) sustituyó la mayor parte de las etiquetas humanas de daño por un modelo que critica y corrige sus propias respuestas según una lista de principios escritos (RLAIF). Sea cual sea la receta, el resultado es una **preferencia aprendida, no una regla**: generaliza a partir de la distribución de entrenamiento y puede fallar fuera de ella.

**Por qué falla el entrenamiento de seguridad.** Wei, Haghtalab y Steinhardt (2023) reducen casi todos los jailbreaks a dos mecanismos:

- **Objetivos en competencia.** El modelo se entrenó para seguir instrucciones y también para no hacer daño, y un prompt puede enfrentar una cosa con la otra. La *inyección de prefijo* pide una respuesta que empiece de forma afirmativa; la *supresión del rechazo* prohíbe las palabras de las que están hechos los rechazos. Una vez que los primeros tokens ceden, lo más probable es seguir por ahí.
- **Generalización desajustada.** El entrenamiento de capacidades cubrió muchas más situaciones que el de seguridad. Una petición envuelta en ficción o en un juego de rol, traducida a un idioma con pocos recursos o codificada (base64, cifrados) queda fuera de lo que vio el entrenamiento de rechazo, mientras el modelo sigue sabiendo hacer la tarea.

Las plantillas escritas a mano son solo el principio. **GCG** (Zou et al., 2023) optimiza un sufijo adversario con los gradientes de modelos abiertos, y los sufijos se transfieren a modelos cerrados. **PAIR** (Chao et al., 2023) no necesita gradientes: un LLM atacante reescribe el prompt en bucle, guiándose por las respuestas del objetivo y por un juez, y a menudo lo consigue en menos de veinte consultas. Los dos son *adaptativos*, que es el atacante con el que se encuentra de verdad un sistema desplegado.

**El otro fallo: el sobre-rechazo.** El entrenamiento de rechazo también enseña atajos: "kill", "shoot" o "poison" se convierten en disparadores sea cual sea el contexto. **XSTest** (Röttger et al., 2024) mide esta *seguridad exagerada* con 250 prompts seguros pensados para activar los atajos (homónimos como matar un proceso, lenguaje figurado, contextos seguros como una novela policiaca, preguntas de privacidad sobre personajes de ficción) y 200 prompts inseguros de contraste. Rechazar de más no es una opción segura por defecto: le cuesta usuarios al producto, los empuja a reformular hasta que algo pasa y entierra los rechazos de verdad entre el ruido.

**Modelos guardián.** En lugar de fiarse solo de los rechazos del propio modelo, los sistemas en producción ponen clasificadores delante (sobre el prompt) y detrás (sobre la respuesta). **Llama Guard** (Inan et al., 2023) es un LLM ajustado para responder `safe` o `unsafe` más una categoría de riesgo; Llama Guard 3 usa la taxonomía **S1–S14** de MLCommons (delitos violentos, delitos no violentos, delitos sexuales, explotación infantil, difamación, asesoramiento especializado, privacidad, propiedad intelectual, armas de destrucción masiva, odio, autolesiones, contenido sexual, elecciones y abuso del intérprete de código). Un guardián es barato (un modelo de 1B etiqueta un prompt en decenas de milisegundos) e independiente del modelo que protege, así que las plantillas escritas para manipular al modelo objetivo no lo manipulan a él, y un mismo guardián cubre todos los modelos que tenga detrás. También es un clasificador con sus propios falsos positivos y sus fallos, más fuerte en los riesgos en los que más insisten sus datos de entrenamiento, y se le puede atacar como a cualquier modelo.

| Opción | Tipo | Notas |
|---|---|---|
| Llama Guard 3 (1B, 8B), ShieldGemma, Granite Guardian | modelos guardián abiertos | en tu propia infraestructura; las categorías y el umbral se pueden adaptar a tu política |
| Prompt Guard y otros encoders pequeños parecidos | clasificadores abiertos | pensados para patrones de jailbreak y de inyección más que para categorías de daño ([sección 1.6](#16-encoders-y-encoder-decoders)) |
| endpoint de moderación de OpenAI, Azure AI Content Safety, Amazon Bedrock Guardrails | servicios gestionados | sin modelo que mantener; la política es la del proveedor, con algo de configuración |
| NVIDIA NeMo Guardrails, Guardrails AI | frameworks | conectan comprobaciones, reglas y flujos de diálogo en una aplicación, alrededor de cualquiera de los anteriores |

**Cómo se puntúa la seguridad.**

- **Tasa de rechazo** en los prompts dañinos (cuanto más alta, más seguro) y **tasa de sobre-rechazo** en los seguros (cuanto más baja, mejor). Siempre las dos.
- **Tasa de éxito del ataque (ASR)**: la proporción de peticiones dañinas que obtienen una respuesta *dañina*. No rechazar no equivale a ayudar de verdad: una respuesta con jailbreak puede ser vaga, errónea o vacía. **StrongREJECT** (Souly et al., 2024) demostró que puntuar solo "¿ha rechazado?" exagera el éxito de los jailbreaks, y los benchmarks puntúan la nocividad con un juez: el clasificador de HarmBench (Mazeika et al., 2024), el juez de JailbreakBench (Chao et al., 2024), la rúbrica de StrongREJECT.
- **Los jueces hay que validarlos.** Un juez de rechazos o de daño es un modelo con su propia tasa de error; se mide contra etiquetas humanas antes de fiarse de las tasas que produce ([sección 4.3](#43-ragas-y-llm-as-judge)). Y tiene que ser **independiente de las defensas**: puntuar la salida de un sistema protegido con el mismo guardián da cero éxito del ataque por construcción.

**Defensa en profundidad.** Ninguna capa basta por sí sola, así que en producción se apilan varias que fallan de formas distintas:

| Capa | Qué para | Qué cuesta | Fallo típico |
|---|---|---|---|
| el entrenamiento de seguridad del modelo | las peticiones dañinas directas | nada extra | jailbreaks; sobre-rechazo |
| system prompt de seguridad | un poco de las dos cosas | algo de utilidad | gana una instrucción que compite con él más adelante en el prompt |
| guardián de entrada | la mayoría de las peticiones dañinas, disfrazadas o no | falsos positivos; decenas de ms | redacciones suaves; ataques adaptativos contra el guardián |
| guardián de salida | contenido dañino que se ha colado | latencia; falsos positivos en respuestas factuales | para evaluarlo hace falta un juez independiente |
| controles de la aplicación (autenticación, límites de tasa, nada de herramientas peligrosas al alcance de entradas no fiables, registro para revisar abusos) | el daño, haga lo que haga el modelo | trabajo de ingeniería | tratar cualquier prompt como un control de seguridad |

Cada capa añade falsos rechazos, y se suman; la pila solo se entiende tan bien como el juez con el que se mide. La inyección de prompts, el ataque a las *aplicaciones* a través de los datos que leen, está en las secciones [4.4](#44-patrones-de-aplicación) y [5.8](#58-ejecución-de-código-y-sandboxes).

**Papers clave.** Bai et al. (2022), *Constitutional AI: Harmlessness from AI Feedback*; Wei, Haghtalab y Steinhardt (2023), *Jailbroken: How Does LLM Safety Training Fail?*; Zou et al. (2023), *Universal and Transferable Adversarial Attacks on Aligned Language Models* (GCG); Chao et al. (2023), *Jailbreaking Black Box Large Language Models in Twenty Queries* (PAIR); Chao et al. (2024), *JailbreakBench*; Mazeika et al. (2024), *HarmBench*; Souly et al. (2024), *A StrongREJECT for Empty Jailbreaks*; Röttger et al. (2024), *XSTest: A Test Suite for Identifying Exaggerated Safety Behaviours in Large Language Models*; Inan et al. (2023), *Llama Guard: LLM-based Input-Output Safeguard for Human-AI Conversations*.

> **Notebook:** [22 · Seguridad y guardrails](../../notebooks/06_llmops/22_safety_guardrails.ipynb) (se lee después del 15) · **En el curso:** un juez de rechazos por palabras clave coincide con las etiquetas humanas de XSTest en el 89.7 % de 900 respuestas, y un juez LLM de 7B solo en el 70.0 %; ante la petición directa, `qwen2.5:3b` y `qwen2.5:7b` rechazan el 89 % y el 92 % de los 100 comportamientos dañinos de JailbreakBench, y rechazan sin motivo el 17 % y el 5 % de los prompts seguros de XSTest; contra el 3B, la inyección de prefijo llega a un 64 % de éxito, la supresión del rechazo a un 24 % y la ficción a un 12 %, y el 70 % de los comportamientos cae con al menos una; Llama Guard 3 1B como filtro de entrada marca el 95–100 % de los prompts dañinos, disfrazados o no, pero bloquea el 8 % de los prompts seguros de XSTest y deja pasar el 27 % de los inseguros; un system prompt de seguridad solo baja la inyección de prefijo al 44 % y sube el sobre-rechazo al 21 %; con el guardián, el éxito del ataque baja al 1 % (optimista, porque el mismo guardián hace de juez) a cambio de un 21–25 % de falsos rechazos.

---

[← Volver al índice](README.md) · [Glosario](glosario.md) · [Guía de notebooks](guia_notebooks.md)
