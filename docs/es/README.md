# LLM Edu Suite — Documentación en español

🇬🇧 [English version](../en/README.md) · [README principal](../../README.md)

Un curso práctico de ingeniería de LLMs en dos mitades, con un mapa para empezar: el notebook 00 (los transformers, en grande) se lee antes que nada. Sigue una frase a través de un modelo real y pequeño, SmolLM2-135M, desde los tokens hasta los embeddings, la atención, la predicción capa a capa (el logit lens) y la distribución del siguiente token, y enseña dónde están los parámetros, sin entrenar nada. Después:

1. **Por dentro (notebooks 01–05, 08 y 10):** construimos un modelo de lenguaje desde cero en PyTorch (tokenizador, atención, RoPE, MoE, entrenamiento, inferencia) para entender cómo funciona.
2. **Con las herramientas de la industria (notebooks 11, 13, 14, 16, 18, 19, 23, 25, 26 y 28):** fine-tuning con Hugging Face (`transformers`, `peft`, `trl`), cuantización con `optimum-quanto`, RAG con LangChain evaluado con RAGAS, agentes con LangGraph, observabilidad y evaluación con LangSmith, y el Model Context Protocol con el SDK oficial.

El notebook 06 (encoders y encoder-decoders) completa la arquitectura: un encoder desde cero con MLM, ModernBERT ajustado para clasificar, embeddings + regresión logística frente a un LLM, y Flan-T5 con su cross-attention.

El notebook 07 (ingeniería de datos) va justo antes del pre-entrenamiento (08), porque es lo que le pasa al texto antes del pre-entrenamiento: deduplicación exacta y con MinHash + LSH escrito desde cero, filtros de calidad con reglas y con un clasificador, descontaminación de benchmarks y datos sintéticos, sobre una muestra real de FineWeb, y dos modelos entrenados con datos en bruto y limpios para medir qué cambia.

El notebook 09 (sistemas) sigue al pre-entrenamiento: mide en la GPU adónde van la memoria y el tiempo de un entrenamiento (pesos, gradientes, Adam y activaciones; bf16 y activation checkpointing; FlashAttention y los kernels de atención; profiling, CUDA graphs y MFU) y cómo se reparte entre varias GPUs, con DDP real en dos procesos y una calculadora de ZeRO / FSDP para 7B y 70B.

El notebook 12 (serving bajo carga) parte del 10: el 10 abre por dentro una sola generación y este sirve los modelos como lo hace una aplicación, a través de un servidor HTTP y con muchas peticiones a la vez. Con un cliente en streaming contra Ollama mide el TTFT, el TPOT y el throughput, el prefill frente al decode según crece el prompt, lo que ahorra el prefix caching, un slot de peticiones frente a cuatro con 1 a 8 peticiones en vuelo (y la ley de Little), el mismo modelo de 1.5B en GGUF Q4_K_M, Q8_0 y FP16 (memoria, velocidad y acierto) y un gateway pequeño con fallbacks, timeouts y una cascada consciente del coste de `qwen2.5:3b` a `qwen2.5:7b`.

El notebook 15 (razonamiento y cómputo en inferencia) sigue al 14: allí GRPO es cómo se entrena el razonamiento; aquí, cómo se usa y cómo se mide. Sobre GSM8K con `qwen2.5:3b` compara la respuesta directa con la cadena de pensamiento, la self-consistency (maj@k frente a pass@k), best-of-N con un juez de 7B como verificador y el modelo de razonamiento destilado DeepSeek-R1-Distill 1.5B; los pone todos en una misma curva de coste y acierto, los ordena con Bradley-Terry y mide cuánto se contradice un juez LLM por pares cuando se cambia el orden de las respuestas.

El notebook 17 (embeddings y rerankers) va justo antes del RAG (18): abre por dentro la capa de recuperación sobre la que se construye el RAG. Sobre SciFact compara BM25 con tres modelos de embeddings (con y sin prefijos de tarea), escribe InfoNCE a mano y ajusta un bi-encoder pequeño con negativos del batch y negativos difíciles, encoge los vectores con truncado Matryoshka y cuantización int8 / binaria con rescoring, y prueba la recuperación híbrida con dos rerankers cross-encoder, midiendo tanto la calidad como la latencia.

El notebook 20 (DSPy) sigue a los patrones de aplicación (19): deja de escribir prompts a mano. Sobre la muestra de Banking77 del 06 declara el clasificador como una signature de DSPy, compara ejemplos few-shot al azar con ejemplos vecinos elegidos para cada mensaje y con la recuperación sola, ejecuta tres optimizadores de prompts (bootstrap con búsqueda aleatoria, MIPROv2 y GEPA, con `qwen2.5:7b` escribiendo instrucciones para el 3B) y mide qué aportan un modelo más grande y un router que solo llama al LLM cuando los vecinos no se ponen de acuerdo, en acierto, latencia y llamadas al LLM.

El notebook 21 (clustering y modelado de temas) parte del 17: usa los mismos embeddings para averiguar qué hay dentro de un montón de texto sin etiquetar. Sobre los 3.080 mensajes de clientes de Banking77, tratados como logs sin etiquetar, compara dos modelos de embeddings y UMAP antes de k-means, intenta elegir el número de clusters con el silhouette score, deja que HDBSCAN encuentre por su cuenta los clusters y el ruido, describe cada cluster con palabras clave de c-TF-IDF y un nombre puesto por `qwen2.5:3b` y monta un bucle de monitorización de intenciones emergentes. Las 77 etiquetas de intención solo se usan para medir cada paso (ARI, NMI, pureza).

El notebook 22 (modelos multimodales) también parte del 17: CLIP se entrena con la misma pérdida contrastiva que un modelo de embeddings de texto, solo que con imágenes en uno de los lados. Sobre 37 razas de gatos y perros (Oxford-IIIT Pet) mide CLIP en zero-shot con el nombre de la clase a secas, con una plantilla de prompt y con un ensemble de cinco, un linear probe con entre 1 y ~100 imágenes etiquetadas por raza y `qwen2.5vl:3b` en Ollama como clasificador (acierto, latencia, tokens por imagen y qué cambia con la resolución); después, extracción estructurada de imágenes de facturas sintéticas con un esquema JSON, limpias y degradadas, y un ataque tipográfico, una raza equivocada escrita sobre la imagen, contra CLIP y contra el VLM.

El notebook 24 (aprendizaje online y drift) sigue al recomendador (23): reproduce MovieLens-1M como un flujo semanal y mide cómo envejece un recomendador congelado frente a reentrenarlo cada semana, a una ventana deslizante y a actualizaciones online con decaimiento (con evaluación precuencial, primero predecir y luego aprender), si el PSI de la distribución de entrada se da cuenta del cambio y si reentrenar cuando salta el drift es mejor que hacerlo según un calendario. Después, con Banking77, se pregunta si un cambio en lo que preguntan los usuarios se puede detectar solo con embeddings, con el desplazamiento del centroide, MMD con un test de permutaciones y un test de dos muestras con clasificador.

El notebook 27 (harnesses de agentes) sigue al MCP (26): abre la capa que convierte un modelo en un agente de programación, el tipo de programa que son Claude Code y Codex CLI. Lee el código de mini-swe-agent, un harness mínimo, función a función, lo amplía con un entorno aislado y una política de permisos, y lo ejecuta con Claude Haiku 5.5 (o cualquier modelo de litellm, también locales) sobre 16 tareas de programación evaluadas con tests ocultos, 3 ejecuciones cada una. Después mide qué aporta un fichero de memoria del proyecto, `AGENTS.md`, qué hace una inyección de prompt plantada en el repositorio y qué cuesta y qué da un perfil de harness moderno.

El notebook 29 (frameworks de evaluación y trazas) sigue a LangSmith (28): lleva las mismas ideas a infraestructura de código abierto que corre en el portátil. Reproduce el resultado de GSM8K del 15 como una tarea de Inspect AI, mide cuánto cambian el número el scorer y las epochs muestreadas, y traza un pipeline de LangChain con OpenTelemetry hacia un servidor de Phoenix alojado en local: lee de vuelta la latencia y los tokens de cada paso desde el almacén de trazas, escribe evaluaciones sobre las trazas como anotaciones y mide lo que cuesta trazar.

El notebook 30 (seguridad y guardrails) cierra el curso después de la observabilidad (28 y 29): la observabilidad te dice qué hizo el sistema; este notebook trata de lo que debería hacer y de lo que no. Con `qwen2.5:3b` y `qwen2.5:7b` sobre benchmarks públicos (JailbreakBench y XSTest) mide las dos caras de la seguridad, las peticiones dañinas que rechaza y las seguras que rechaza sin motivo, y después tres plantillas de jailbreak publicadas, Llama Guard 3 (1B) como filtro de entrada, un system prompt de seguridad y todas las capas juntas, después de validar su juez de rechazos con etiquetas humanas.

Todo funciona en local: los LLMs los sirve **Ollama** (`qwen2.5:3b` y `nomic-embed-text`) y los modelos de entrenamiento son SmolLM2 de Hugging Face. Con una GPU NVIDIA va rápido (los resultados publicados se midieron en una RTX 3060 de 6 GB); en CPU también funciona, salvo el notebook 09, que necesita una GPU CUDA. LangSmith es opcional.

## Páginas

| Página | Para qué sirve |
|---|---|
| [Instalación y entorno](instalacion.md) | `.venv`, PyTorch con CUDA, Ollama y sus modelos, el `.env`, la clave de LangSmith, problemas frecuentes |
| [Guía de notebooks](guia_notebooks.md) | los 31 notebooks uno a uno: qué construyes, con qué herramientas, qué conceptos y qué resultados |
| [Fundamentos teóricos](fundamentos.md) | la teoría detrás de cada técnica y patrón: el problema que resuelve, la idea, por qué se usa hoy, cómo falla |
| [Referencia de módulos](referencia_modulos.md) | el paquete `llm_edu_suite/` (el modelo desde cero y `config.py`) y dónde se enseña cada tema con frameworks |
| [Scripts de línea de comandos](scripts_cli.md) | `mini_llm`, `train`, `sample`, `finetune` (TRL), `rag_demo` (LangChain), `mcp_server` (FastMCP), `harness_run` (mini-swe-agent), `inspect_tasks/gsm8k_cot` (Inspect AI) |
| [Caso de estudio](caso_de_estudio.md) | el proyecto como caso de portafolio: v1 desde cero, v2 con frameworks, decisiones, problemas y resultados |
| [Glosario](glosario.md) | términos ES ↔ EN: GQA, logit lens, LoRA, DPO, GRPO, RAGAS, LCEL, signature de DSPy, StateGraph, MCP, span, OpenTelemetry, SLO… |

## Mapa por tópicos

| Tópico | Carpeta | Notebooks | Herramientas |
|---|---|---|---|
| Empieza aquí | `notebooks/00_start_here/` | 00 · los transformers, en grande (se lee primero) · 01 · un mini LLM completo | `transformers` (00) · PyTorch |
| Arquitectura | `notebooks/01_architecture/` | 02 tokenización · 03 atención, RoPE y normalización · 04 atención eficiente · 05 FFN y MoE · 06 encoders y encoder-decoders | PyTorch desde cero · `transformers` y `scikit-learn` (06) |
| Pre-entrenamiento e inferencia | `notebooks/02_pretraining_and_inference/` | 07 ingeniería de datos (deduplicación, filtros, descontaminación) · 08 pre-entrenamiento · 09 sistemas (memoria, precisión, kernels, paralelismo) · 10 inferencia · 11 cuantización · 12 serving bajo carga (TTFT, batching, caché, GGUF, gateways) | PyTorch · `datasets` y `transformers` (07) · `torch.profiler` y `torch.distributed` (09) · `optimum-quanto` (11) · Ollama por HTTP (12) |
| Post-training | `notebooks/03_post_training/` | 13 SFT, LoRA, destilación y fusión de modelos · 14 DPO y GRPO · 15 razonamiento y cómputo en inferencia | `transformers` · `peft` · `trl` · LangChain (Ollama) (15) |
| Aplicaciones | `notebooks/04_applications/` | 16 tools y salida estructurada · 17 embeddings y rerankers · 18 RAG · 19 patrones de aplicación · 20 optimización de prompts con DSPy (signatures, few-shot por vecinos, MIPROv2, GEPA) · 21 clustering y modelado de temas (UMAP, HDBSCAN, c-TF-IDF) · 22 modelos multimodales (CLIP, VLMs, extracción de documentos) · 23 recomendación · 24 aprendizaje online y drift | LangChain · RAGAS · PyTorch · `sentence-transformers` (17, 20, 21) · DSPy (20) · `umap-learn` y `scikit-learn` (21) · CLIP con `transformers` y `qwen2.5vl:3b` con Ollama (22) |
| Agentes y MCP | `notebooks/05_agents_and_mcp/` | 25 IA agéntica · 26 MCP · 27 harnesses de agentes (mini-swe-agent, tareas de programación, `AGENTS.md`) | LangGraph · SDK `mcp` · mini-swe-agent con Claude Haiku 5.5 u Ollama (27) |
| LLMOps | `notebooks/06_llmops/` | 28 observabilidad y evaluación · 29 frameworks de evaluación y trazas (Inspect AI, Phoenix) · 30 seguridad: rechazos, jailbreaks y modelos guardián | LangSmith · Inspect AI y Arize Phoenix sobre OpenTelemetry (29) · Llama Guard 3 con Ollama (30) |

```mermaid
flowchart LR
    ZZ["00 · Los transformers, en grande<br/>tokens · atención · logit lens"] --> Z
    Z[01 · mini LLM] --> B[02 Tokenizador]
    B --> C[03-05 Transformer<br/>RoPE · GQA/MLA · MoE]
    C --> Q["06 Encoders y encoder-decoders<br/>BERT · ModernBERT · T5"]
    C --> P["07 Datos<br/>MinHash · filtros · descontaminación"]
    P --> D[08 Pre-training]
    D --> S["09 Sistemas<br/>memoria · kernels · paralelismo"]
    S --> E[10 Inferencia]
    E --> G["11 Cuantización<br/>optimum-quanto"]
    E --> V["12 Serving bajo carga<br/>TTFT · batching · caché · gateway"]
    D --> F["13-14 Post-training<br/>HF · peft · trl"]
    F --> R["15 Razonamiento<br/>CoT · self-consistency · verificadores"]
    E --> H[Aplicaciones con LangChain]
    H --> J[16 Tools]
    H --> T["17 Embeddings y rerankers<br/>bi-encoder · InfoNCE · Matryoshka"]
    T --> I["18 RAG + RAGAS"]
    H --> L[19 Patrones]
    L --> DS["20 DSPy<br/>signatures · few-shot por vecinos · optimizadores"]
    T --> X["21 Clustering y temas<br/>UMAP · HDBSCAN · c-TF-IDF"]
    T --> W["22 Modelos multimodales<br/>CLIP · VLM · ataques tipográficos"]
    H --> K[23 Recomendación]
    K --> Y["24 Drift y aprendizaje online<br/>PSI · MMD · reentrenamiento"]
    J --> M["25 Agentes · LangGraph"]
    J --> N["26 MCP · SDK oficial"]
    N --> AA["27 Harnesses de agentes<br/>mini-swe-agent · AGENTS.md · tests ocultos"]
    M --> O["28 LangSmith"]
    O --> PH["29 Inspect AI + Phoenix<br/>tareas · scorers · OpenTelemetry"]
    O --> U["30 Seguridad<br/>jailbreaks · sobre-rechazo · Llama Guard"]
```

## Ruta de aprendizaje sugerida

0. **Visión general:** 00, para ver qué hace un modelo real con una frase, y después 01, para construir uno entero y ver cómo encajan las piezas antes de estudiarlas por separado. Después del 00 deberías saber que un modelo lee ids de tokens, que lo que devuelve es una distribución de probabilidad que una regla de decodificación convierte en texto, y que la mayoría de sus parámetros están en las capas feed-forward.
1. **Arquitectura:** 02 → 03 → 04 → 05 → 06. Al terminar deberías poder escribir un transformer de memoria y saber cuándo conviene un encoder o un encoder-decoder en lugar de un decoder-only.
2. **Datos, entrenamiento e inferencia:** 07 → 08 → 09 → 10 → 11 → 12. Primero qué texto entra en el modelo y por qué, después cómo se entrena con él y cuánto cuesta ese entrenamiento en memoria y tiempo, en una GPU y en muchas. Después del 12 deberías saber qué espera el usuario cuando se sirve un modelo (TTFT, TPOT, percentiles), por qué ayudan el batching, el prefix caching y los pesos cuantizados, y cómo dimensionar un servidor a partir de su tasa de peticiones.
3. **Post-training con Hugging Face:** 13 → 14 → 15. Después del 15 deberías saber qué compran los tokens de razonamiento, cuándo compensa votar o usar un verificador y cómo saber si te puedes fiar de un juez.
4. **Aplicaciones con LangChain:** 16 → 17 → 18 → 19 → 20 → 21 → 22 → 23 → 24. Después del 17 deberías saber elegir, ajustar, comprimir y evaluar un recuperador antes de meterlo en un RAG. Después del 20 deberías saber escribir un paso con un LLM como una signature en lugar de como un prompt, por qué los ejemplos que metes en un prompt importan más que su redacción, qué buscan los optimizadores de prompts y cuándo no pueden ayudar, y por qué una línea base sin LLM y una cascada van antes que cualquiera de ellos. Después del 21 deberías saber encontrar los temas de unos logs sin etiquetar, por qué el número de clusters es una elección y no un descubrimiento, y cómo vigilar la aparición de intenciones nuevas. Después del 22 deberías saber cuándo clasificar imágenes con embeddings y cuándo compensa pagar un VLM, cuánto cuesta una imagen en tokens y por qué el texto que aparece en una imagen es una entrada no fiable. Después del 24 deberías saber por qué un modelo que funcionaba el mes pasado puede no funcionar ahora, cómo evaluarlo sobre un flujo y por qué vigilar solo las distribuciones de entrada puede dejar pasar el drift que importa. Regla de oro: **construye tu set de evaluación antes que el sistema**.
5. **Agentes y LLMOps:** 25 → 26 → 27, después 28 para ver cómo se observa y evalúa todo en producción, y después 29 y 30. Al terminar el 27 deberías saber qué hace el harness de un agente de programación además de llamar al modelo, por qué el "terminado" de un agente necesita tests ocultos detrás y por qué la defensa contra instrucciones escondidas en un repositorio va en el harness. Al terminar el 29 deberías saber definir una evaluación como código (dataset, solver, scorer, epochs), por qué un framework tiene que reproducir tu número de confianza antes de adoptarlo, por qué el scorer forma parte de la métrica, y cómo las trazas, los spans y las anotaciones de OpenTelemetry te dan la latencia y los tokens de cada paso y los fallos localizables en tu propia máquina. Al terminar el 30 deberías saber medir los rechazos de un modelo y también sus rechazos de más, por qué los rechazos del propio modelo no bastan como defensa y cuánto cuesta un modelo guardián en falsos positivos.

## Del notebook a producción

Los notebooks son lecciones: datos pequeños, una sola máquina y resultados medidos. Los patrones sirven para sistemas reales; la configuración, los datos y los márgenes de seguridad no. Por tópico, dónde rinde en el trabajo y qué cambiar primero:

| tópico (notebooks) | dónde se usa en el trabajo | qué cambiar, endurecer o afinar |
|---|---|---|
| **Fundamentos** (00–06) | leer model cards y papers, elegir modelo (longitud de contexto, GQA, MoE), depurar problemas de tokenización | no hay nada que desplegar: es el modelo mental. Reutiliza las comprobaciones del tokenizador (por ejemplo, contar tokens por idioma antes de presupuestar una funcionalidad) |
| **Preentrenamiento e inferencia** (07–12) | pipelines de datos (deduplicación, filtros de calidad), dimensionar GPUs, servir un modelo, elegir una cuantización | cambia el modelo de juguete por un checkpoint abierto; sirve con vLLM / SGLang / TGI en vez de un bucle; haz pruebas de carga con la forma de tu propio tráfico (12) y fija SLOs sobre la latencia p95, no sobre la media |
| **Post-entrenamiento** (13–15) | adaptar un modelo pequeño a una tarea concreta (extracción, clasificación, un estilo de la casa), destilar un modelo grande en uno barato, añadir razonamiento donde las respuestas se pueden verificar | datos etiquetados reales y un conjunto reservado; primero LoRA, y fine-tuning completo solo si LoRA se estanca; un verificador (tests, un esquema, una calculadora) antes de cualquier RL; por ejemplo, destila las respuestas de un modelo de API en un modelo de 1–3B para un endpoint de mucho volumen |
| **Aplicaciones** (16–24) | RAG sobre documentación interna, extracción estructurada, búsqueda y recomendaciones, agrupar el feedback de usuarios, optimizar prompts | tu corpus, tu chunking y un conjunto de evaluación hecho con preguntas reales; un reranker antes que un LLM más grande; salidas validadas con esquema; monitores de drift (24) en entradas y salidas; por ejemplo, apunta el 18 a tu wiki en vez de a la documentación del curso de `data/knowledge_base/`, escribe 20–50 preguntas reales y conserva las mismas métricas de recuperación y de RAGAS |
| **Agentes y MCP** (25–27) | asistentes internos que actúan (tickets, consultas), herramientas expuestas por MCP, agentes de código | herramientas con el mínimo privilegio, aprobación humana para todo lo que escribe o sale de la máquina, límites de pasos, costo y tiempo, un sandbox y un `AGENTS.md`; evalúa con tests ocultos y varias ejecuciones por tarea (27) |
| **LLMOps** (28–30) | trazar cada petición, evaluaciones en CI, guardrails | envía las trazas a un backend que controles (Phoenix, Langfuse, LangSmith); fija el scorer y reporta intervalos de confianza (29); mide el exceso de rechazos además de los rechazos (30); alerta sobre latencia, tasa de errores y deriva de las puntuaciones |

**Transversal, para cualquiera de ellos.**
- **Modelos.** Los modelos locales por defecto (`qwen2.5:3b`, SmolLM2) están elegidos para una GPU de 6 GB. En producción elige por la calidad medida en *tu* conjunto de evaluación, y después por costo y latencia. Cambiar de proveedor es un cambio en `.env` (`LLM_PROVIDER`, `LLM_MODEL`, `HARNESS_MODEL`).
- **Primero la evaluación.** Antes de cambiar un prompt, un modelo o un retriever, ten un conjunto de evaluación fijo y un número que superar. Reporta la dispersión entre ejecuciones, no una sola ejecución.
- **Secretos y datos.** Las claves en un gestor de secretos, no en un `.env` en un servidor; los prompts y las trazas tratados como datos de usuarios (retención, enmascarado de datos personales, control de acceso).
- **Costo.** Registra los tokens por petición desde el primer día; cachea lo que se repite; usa el modelo más pequeño que pase la evaluación.

## Cómo usar este repo

- **Para estudiar:** sigue los números. Cada notebook explica la idea, la ejecuta y termina con lo que se midió y cómo se usa en producción. `fundamentos.md` tiene la teoría de cada tópico.
- **En el trabajo:** copia el patrón, no el notebook. Parte del notebook más cercano, cambia los datos y el modelo, conserva su evaluación y endurécelo con la tabla de arriba.
- **Como material de práctica para otros:** todos los notebooks corren en la GPU de un portátil, y cada resultado se reproduce desde su generador. Cambia una cosa (un modelo, un tamaño de chunk, un prompt), vuelve a ejecutar y compara los números.

## Trabajar con un agente de código

Hoy la mayor parte del trabajo de ingeniería se hace junto con un agente de código (Claude Code, Codex CLI, opencode). Este repo está preparado para ello: [`AGENTS.md`](../../AGENTS.md) le explica a un agente cómo funciona el repo (comandos, cómo se generan los notebooks y las reglas: solo números medidos, documentación bilingüe, nada de secretos, un trabajo de GPU a la vez). Prompts que funcionan bien:

- *"Lee AGENTS.md y el notebook 18. Adapta el pipeline de RAG a los PDF de `mis_docs/`, conserva la misma evaluación y dime qué métricas cambiaron."*
- *"En el notebook 13, cambia la tarea de reservas por mi tarea de extracción de `data/mi_tarea.jsonl`. Entrena LoRA r=16, evalúa sobre un 20 % reservado y escribe la tabla de Results solo con las salidas ejecutadas."*
- *"Ejecuta el notebook 27 con `HARNESS_MODEL=openai/...` y compara su tasa de tareas resueltas, tokens y costo con los resultados de Claude del notebook."*
- *"Revisa `scripts/harness_run.py` buscando formas en que una tarea podría escapar del sandbox, y propón reglas de denegación con un test para cada una."*

Revisa lo que cambia el agente, ejecuta los tests y lee las salidas antes de fiarte de un número.

## Principios del proyecto

- **Entender por dentro, después usar la herramienta real.** El modelo se construye a mano porque así la atención o la KV-cache dejan de ser magia; las aplicaciones se construyen con los frameworks que usarás en el trabajo, con sus APIs actuales.
- **Todo se evalúa, con honestidad.** Cada notebook termina con la métrica que reportaría un profesional. Cuando algo no ayuda (un re-ranker LLM que empeora, DPO puro que degrada el modelo, un juez pequeño que hace ruido), el notebook lo muestra y explica por qué.
- **Local primero.** Ollama y modelos pequeños: cualquiera puede reproducir el curso en su portátil, sin coste. Para usar un proveedor en la nube basta con cambiar `LLM_PROVIDER` en el `.env`.
