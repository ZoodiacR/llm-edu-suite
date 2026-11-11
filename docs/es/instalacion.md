🇬🇧 [English version](../en/installation.md)

# Instalación y entorno de trabajo

[← Volver al índice](README.md)

Esta guía deja el proyecto listo para ejecutar **todo**: la suite de tests, los scripts de línea de comandos y los 32 notebooks.

El curso tiene dos mitades con necesidades distintas:

| Notebooks | Qué usan | Qué necesitan |
|---|---|---|
| **00** | `transformers` de Hugging Face sobre el SmolLM2-135M base, con sus pesos de atención y sus estados ocultos | SmolLM2-135M (260 MB) del Hugging Face Hub la primera vez. Sin Ollama y sin entrenar nada; CPU o GPU. Alrededor de minuto y medio. |
| **01–05, 08, 10** | El modelo PyTorch escrito desde cero (paquete `llm_edu_suite`) | Solo PyTorch. Funciona en CPU; si hay GPU NVIDIA se usa automáticamente. |
| **06** | El modelo desde cero con `causal=False`, Hugging Face `transformers` + `datasets` (ModernBERT-base, Flan-T5-small, Banking77), `scikit-learn` y LangChain | PyTorch, la descarga de ModernBERT-base (~600 MB), Flan-T5-small (~300 MB) y Banking77 desde el Hugging Face Hub la primera vez, y **Ollama** con `qwen2.5:3b` y `nomic-embed-text` para la comparación de la sección 2. Con GPU tarda unos 15 minutos. |
| **07** | El `TransformerLM` desde cero (dos entrenamientos), Hugging Face `datasets` + `transformers` (muestras de FineWeb y FineWeb-Edu, el clasificador de FineWeb-Edu, el tokenizador de SmolLM2) y LangChain | PyTorch, unos 100 MB de muestras de FineWeb que se leen en streaming una sola vez y se guardan en `data/cache/`, el clasificador `HuggingFaceFW/fineweb-edu-classifier` (~440 MB) y **Ollama** con `qwen2.5:3b` para la sección de datos sintéticos. Con GPU tarda unos 30 minutos (dos entrenamientos de 10–11 minutos y unos 3 minutos de generación). |
| **09** | El `TransformerLM` desde cero, `torch.profiler`, `torch.utils.checkpoint`, los backends de `F.scaled_dot_product_attention`, `torch.compile` y `torch.distributed` | PyTorch con CUDA y **una GPU NVIDIA** (mide la memoria y los kernels de la GPU; sin ella se detiene al empezar). No descarga nada ni usa Ollama. En Windows faltan FlashAttention-2 y Triton y `torchrun` no funciona (ver [notas sobre la GPU](#notas-sobre-la-gpu)); el notebook lo detecta y sigue. Tarda unos 5 minutos. |
| **11, 13, 14** | Hugging Face `transformers` + `peft` + `trl` + `bitsandbytes` + `optimum-quanto` sobre SmolLM2 | PyTorch y una descarga única de SmolLM2 desde el Hugging Face Hub (~1 GB). Con GPU van mucho más rápido; la sección de QLoRA del 13 necesita GPU NVIDIA (`bitsandbytes`). |
| **12** | `httpx` y `asyncio` contra la API HTTP de Ollama (`/api/chat` en streaming, `/api/ps`), `subprocess` para un segundo `ollama serve`, `datasets` de Hugging Face (ARC-Easy, GSM8K) y `config.setup` / `judge_model` | **Ollama** con `qwen2.5:3b`, `qwen2.5:7b` y los tres formatos de `qwen2.5:1.5b-instruct` (`q4_K_M`, `q8_0` y `fp16`: 986 MB, 1,6 GB y 3,1 GB); ARC-Easy (`allenai/ai2_arc`) y GSM8K (`openai/gsm8k`), que se descargan del Hugging Face Hub la primera vez. **Una GPU**: cada resultado es una latencia. El notebook arranca y para un segundo servidor de Ollama temporal en un puerto libre (con `OLLAMA_NUM_PARALLEL=4`) y saca la carpeta de modelos del `server.log` de la aplicación de escritorio de Ollama. Sin entrenamiento. Tarda unos 28 minutos en GPU, la mayor parte en la cascada 3B → 7B sobre GSM8K. |
| **15** | LangChain con Ollama (`config.get_llm` y `get_judge_llm`), `datasets` de Hugging Face (GSM8K) y Pydantic para la salida estructurada del juez | **Ollama** con `qwen2.5:3b`, `deepseek-r1:1.5b` (1,1 GB) y, como verificador y juez por pares, el `qwen2.5:7b` opcional; el test de GSM8K se descarga del Hugging Face Hub la primera vez. No entrena nada. Tarda unos 53 minutos, casi todo en llamadas al LLM. |
| **16, 18, 19, 23, 25, 26, 28** | LangChain, RAGAS, LangGraph, LangSmith, el SDK oficial de MCP + `langchain-mcp-adapters`; en el 23, además FAISS y LightGBM | Un servidor **Ollama** local con `qwen2.5:3b` y `nomic-embed-text` (y, opcional, `qwen2.5:7b` como juez del notebook 18 y para el agente que escribe código del 14). El notebook 25 usa además `pandas`, que se instala como dependencia de `datasets`. LangSmith es opcional. El notebook 23 descarga MovieLens-1M (6 MB) la primera vez. |
| **17** | `sentence-transformers` (bi-encoders, cross-encoders, entrenamiento contrastivo, cuantización de embeddings), `rank-bm25`, `datasets` de Hugging Face y el `OllamaEmbeddings` de LangChain a través de `config.get_embeddings` | SciFact y NFCorpus (`mteb/scifact`, `mteb/nfcorpus`) y cuatro modelos (all-MiniLM-L6-v2, BAAI/bge-small-en-v1.5, cross-encoder/ms-marco-MiniLM-L6-v2, BAAI/bge-reranker-base), que se descargan del Hugging Face Hub la primera vez, y **Ollama** con `nomic-embed-text`. Se recomienda GPU (tres ajustes cortos y dos cross-encoders); en CPU no se ha medido. Tarda unos 18 minutos en GPU. |
| **20** | DSPy 3.5 (signatures, módulos, `LabeledFewShot`, `BootstrapFewShotWithRandomSearch`, `MIPROv2`, `GEPA`) con su propio cliente HTTP (`engine="lm15"`) contra el endpoint `/v1` compatible con OpenAI de Ollama, Optuna (la búsqueda bayesiana de MIPROv2), `sentence-transformers` (bge-small-en-v1.5) y `datasets` de Hugging Face (Banking77) | **Ollama** con `qwen2.5:3b` y `qwen2.5:7b`; Banking77 y bge-small-en-v1.5 del Hugging Face Hub la primera vez (ya en la caché si ejecutaste el 06 y el 17). Sin litellm y sin trazas en LangSmith. Unos 57 minutos en GPU, casi todo los optimizadores y el clasificador de 7B. |
| **21** | `sentence-transformers` (all-MiniLM-L6-v2, bge-small-en-v1.5), `umap-learn`, `scikit-learn` (`KMeans`, `HDBSCAN`, `CountVectorizer`, las métricas de clustering), `datasets` de Hugging Face (Banking77) y LangChain con Ollama (`config.setup` y `get_llm`) para poner nombre a los clusters | **Ollama** con `qwen2.5:3b`; Banking77 (`PolyAI/banking77`) y los dos modelos de embeddings del Hugging Face Hub la primera vez (ya estarán en la caché si has ejecutado el 06 y el 17). La GPU acelera la codificación, pero no hace falta: los modelos son pequeños. No entrena nada. Tarda unos 196 s en GPU, 88 s de ellos en que el LLM ponga nombre a los clusters. |
| **22** | `transformers` (CLIP) y `datasets` (Oxford-IIIT Pet) de Hugging Face, `scikit-learn` para el linear probe, el cliente de Python de `ollama` para el modelo de visión, Pillow para dibujar y alterar imágenes, Pydantic para el esquema de las facturas y `config.setup` | **Ollama** con `qwen2.5vl:3b` (3,2 GB); CLIP ViT-B/32 (`openai/clip-vit-base-patch32`) y Oxford-IIIT Pet (`timm/oxford-iiit-pet`) del Hugging Face Hub la primera vez. **Se recomienda GPU** (CLIP codifica las 3.680 imágenes de entrenamiento y la muestra de test, y el VLM responde imagen a imagen). No entrena nada salvo una regresión logística. Tarda unos 12 minutos en GPU, casi todo llamadas al VLM. |
| **24** | NumPy y `llm_edu_suite.recsys` (`download_movielens_1m`, `load_movielens_1m`) para el flujo del recomendador, y `sentence-transformers`, `scikit-learn` y `datasets` de Hugging Face para los tests de drift sobre embeddings; `config.setup` en la celda de configuración | MovieLens-1M con el ayudante del propio curso (la misma descarga en `data/cache/` que el notebook 23), y Banking77 (`PolyAI/banking77`) y bge-small-en-v1.5 del Hugging Face Hub la primera vez (ya estarán en la caché si has ejecutado el 17 y el 21). Ninguna dependencia nueva y ninguna llamada a un LLM. La GPU solo acelera la codificación de Banking77. Tarda unos 90 s en GPU. |
| **27** | mini-swe-agent 2.4.6 (con litellm) en un **entorno virtual aparte**, `.venv-harness`, que lanza `scripts/harness_run.py` (que importa `scripts/harness_modern.py`); bash para los comandos del agente (Git Bash en Windows); `pytest` para los tests ocultos | El entorno del harness (ver [El entorno del harness de agentes](#el-entorno-del-harness-de-agentes-notebook-27)) y un modelo. Por defecto **Claude Haiku 5.5** por la API de Anthropic cuando hay una `ANTHROPIC_API_KEY` en el `.env`: 366 ejecuciones del agente, unos **US$ 0,42** el notebook entero (ver [Claves de API opcionales](#claves-de-api-opcionales-notebook-27)). Sin la clave recurre a **Ollama** con `qwen2.5:7b`, mucho más lento (en la v2.8 los modelos locales tardaban 64–216 s por tarea, frente a unos 7 s con Claude). `HARNESS_MODEL` elige cualquier otro modelo de litellm. Nada que descargar aparte del modelo, sin entrenamiento y sin trazas de LangSmith. |
| **29** | Inspect AI 0.3.289 y Phoenix 21.0.0 en un **entorno virtual aparte**, `.venv-eco` (sus comandos `inspect` y `phoenix`); en el entorno del curso, `arize-phoenix-otel` (un exportador de OpenTelemetry ya configurado para Phoenix) y `openinference-instrumentation-langchain`; LangChain con Ollama, y `httpx` para la API REST de Phoenix | El entorno de Inspect y Phoenix (ver [El entorno de Inspect y Phoenix](#el-entorno-de-inspect-y-phoenix-notebook-29)) y **Ollama** con `qwen2.5:3b`; GSM8K del Hugging Face Hub la primera vez (ya en la caché si ejecutaste el 15). Phoenix corre en tu máquina, en un puerto local libre. Unos 23 minutos en GPU, casi todo las cuatro pasadas de Inspect por los 80 problemas. |
| **30** | LangChain con Ollama (`config.get_llm` y `judge_model`), el cliente de Python `ollama` (se instala con `langchain-ollama`) para Llama Guard, `datasets` de Hugging Face y Pydantic para la salida estructurada del juez LLM | **Ollama** con `qwen2.5:3b`, `qwen2.5:7b` y `llama-guard3:1b` (1,6 GB); JailbreakBench (`JailbreakBench/JBB-Behaviors`), XSTest (`Paul/XSTest`) y las respuestas de XSTest etiquetadas por personas (`natolambert/xstest-v2-copy`), que se descargan del Hugging Face Hub la primera vez. Sin entrenamiento. Tarda unos 48 minutos en GPU, casi todo generación. |

No hace falta ninguna clave de API: todo puede funcionar en local. Hay una clave opcional y recomendable: con `ANTHROPIC_API_KEY` en el `.env`, el notebook 27 ejecuta su agente con Claude Haiku 5.5 (unos US$ 0,42 por ejecución completa) en lugar de con un modelo local mucho más lento (ver [Claves de API opcionales](#claves-de-api-opcionales-notebook-27)).

## Requisitos

| Componente | Mínimo | Verificado con |
|---|---|---|
| Python | 3.11 | 3.11.4 (Windows 11) |
| PyTorch | 2.3 (2.5+ para la celda opcional de FlexAttention del notebook 04) | 2.14.1+cu130 |
| GPU NVIDIA | opcional | RTX 3060 Laptop GPU, 6 GB |
| [Ollama](https://ollama.com) | cualquier versión reciente | 0.40.15 |
| Disco: entorno virtual | ~4 GB con la build CUDA de PyTorch (PyTorch solo ocupa ~2,9 GB); mucho menos con la de CPU | 4,1 GB |
| Disco: modelos de Ollama | ~2,2 GB (`qwen2.5:3b` 1,9 GB + `nomic-embed-text` 0,27 GB); 4,7 GB más con el `qwen2.5:7b` opcional (juez en el 18, agente de código en el 25), 1,1 GB de `deepseek-r1:1.5b` (notebook 15), 1,6 GB de `llama-guard3:1b` (notebook 30), unos 5,7 GB de los tres formatos de `qwen2.5:1.5b-instruct` (notebook 12), 3,2 GB del modelo de visión `qwen2.5vl:3b` (notebook 22) y, solo para repetir la comparación local de la v2.8 del notebook 27, 5,2 GB de `freehuntx/qwen3-coder:8b` (sin clave de API, el notebook 27 usa `qwen2.5:7b`) | |
| Disco: datos descargados | MovieLens-1M, 6 MB comprimido, y las muestras de FineWeb (12.000 documentos) y FineWeb-Edu (1.000), unos 45 MB en JSONL, en `data/cache/` | |
| Disco: caché de Hugging Face | ~1 GB (SmolLM2-135M-Instruct 260 MB + SmolLM2-360M-Instruct 690 MB) y ~0,9 GB más con el notebook 06 (ModernBERT-base ~600 MB + Flan-T5-small ~300 MB, más Banking77) y ~0,45 GB más con el notebook 07 (`fineweb-edu-classifier` ~440 MB y el tokenizador de SmolLM2-135M), más los cuatro modelos y los dos conjuntos de datos del notebook 17, los tres pequeños benchmarks del notebook 30 y CLIP ViT-B/32 con Oxford-IIIT Pet para el notebook 22 (los notebooks 20, 21 y 24 reutilizan Banking77 y los modelos del 17), más el SmolLM2-135M base (260 MB) para el notebook 00 | |

Las dependencias se declaran una sola vez, en `pyproject.toml` (con cotas inferiores); `requirements.txt` solo contiene `-e .[dev]`, es decir, instala el proyecto desde ahí con el extra `dev` (pytest y Jupyter). Ya no existe `requirements-dev.txt`. Los extras opcionales `openai` y `anthropic` añaden los proveedores en la nube (ver [Cambiar de proveedor](#cambiar-de-proveedor)). En v2.1 entraron dos dependencias: `lightgbm>=4.5` (el ranker LambdaMART del notebook 23) y `bitsandbytes>=0.45` (los pesos NF4 de QLoRA en el notebook 13). Con el notebook 06 entró `scikit-learn>=1.5` (la regresión logística y las métricas de clasificación), y con el 21, `sentence-transformers>=5.0` (bi-encoders, rerankers cross-encoder y entrenamiento contrastivo). Con el notebook 20 entraron `dspy>=3.5` y `optuna>=4` (la búsqueda bayesiana que hay detrás de MIPROv2), y con el 29, `arize-phoenix-otel>=0.17` y `openinference-instrumentation-langchain>=0.1.82` (la parte cliente de las trazas hacia Phoenix; Inspect AI y el servidor de Phoenix viven en su propio entorno, ver [más abajo](#el-entorno-de-inspect-y-phoenix-notebook-29)). Estas son las versiones con las que se verificó el curso:

| Paquete | Versión | Paquete | Versión |
|---|---|---|---|
| transformers | 5.22.2 | langchain | 1.4.9 |
| peft | 0.21.2 | langchain-core | 1.7.6 |
| trl | 1.17.1 | langchain-community | 0.4.2 |
| optimum-quanto | 0.2.7 | langchain-ollama | 1.1.0 |
| ragas | 0.4.3 | langgraph | 1.2.20 |
| mcp | 1.30.0 (debe seguir en `<2`) | langsmith | 0.14.17 |
| langchain-mcp-adapters | 0.3.2 | faiss-cpu | 1.15.1 |
| bitsandbytes | 0.50.2 | lightgbm | 4.7.0 |
| scikit-learn | 1.9.1 | sentence-transformers | 6.2.0 |
| umap-learn | 0.6.0 | dspy | 3.5.0 |
| optuna | 5.0.0 | arize-phoenix-otel | 0.17.3 |
| openinference-instrumentation-langchain | 0.1.82 | | |

> **Por qué un entorno virtual propio.** Si instalas en tu Python global puedes heredar combinaciones rotas
> (por ejemplo, un `matplotlib` compilado contra NumPy 1.x junto a NumPy 2.x falla con
> `ImportError: numpy.core.multiarray failed to import`). Además el curso fija `mcp<2`, porque
> `langchain-mcp-adapters` todavía no soporta el SDK 2.x. Un `.venv` aislado evita estos conflictos.

## Opción A: script de instalación (recomendado)

Instala antes Ollama (ver [Ollama](#ollama)) para que el script pueda descargar los modelos. Después:

**Windows (PowerShell):**

```powershell
cd llm-engineering-suite
powershell -ExecutionPolicy Bypass -File setup_env.ps1
```

**Linux / macOS:**

```bash
cd llm-engineering-suite
bash setup_env.sh
```

El script hace, en orden:

1. Crea `.venv/` si no existe y actualiza `pip`.
2. Instala PyTorch: la build **CUDA** (índice `cu128`) si encuentra `nvidia-smi`, o la build **CPU** (pequeña) si no. En macOS instala el wheel por defecto, que incluye soporte MPS.
3. Instala `requirements.txt`, que instala el proyecto con su extra `dev` desde `pyproject.toml` (NumPy, matplotlib, python-dotenv, LangChain, LangGraph, LangSmith, RAGAS, FAISS, rank-bm25, LightGBM, scikit-learn, umap-learn, sentence-transformers, transformers, peft, trl, datasets, accelerate, bitsandbytes, optimum-quanto, mcp, langchain-mcp-adapters, DSPy, Optuna, arize-phoenix-otel, openinference-instrumentation-langchain) más pytest, JupyterLab, ipykernel, nbclient y nbformat.
4. Registra el entorno como kernel de Jupyter con el nombre **"Python (llm-edu-suite)"**.
5. Si existe el comando `ollama`, ejecuta `ollama pull qwen2.5:3b` y `ollama pull nomic-embed-text`. Si no existe, indica cómo instalar Ollama y continúa. El juez opcional `qwen2.5:7b` no lo descarga: hazlo a mano si lo quieres (ver [Ollama](#ollama)).
6. Crea `.env` a partir de `.env.example` si todavía no hay `.env`.
7. Ejecuta los tests rápidos: `pytest -q -m "not ollama"`.
8. (Script de Windows) imprime la versión de PyTorch y la GPU encontrada.

## Opción B: paso a paso

```powershell
# Windows
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cu128    # sin GPU NVIDIA: .../whl/cpu
pip install -r requirements.txt
python -m ipykernel install --user --name llm-edu-suite --display-name "Python (llm-edu-suite)"
ollama pull qwen2.5:3b
ollama pull nomic-embed-text
Copy-Item .env.example .env
pytest -q -m "not ollama"
```

```bash
# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cu128    # sin GPU NVIDIA: .../whl/cpu ; macOS: pip install torch
pip install -r requirements.txt
python -m ipykernel install --user --name llm-edu-suite --display-name "Python (llm-edu-suite)"
ollama pull qwen2.5:3b
ollama pull nomic-embed-text
cp .env.example .env
pytest -q -m "not ollama"
```

> Si PowerShell bloquea `Activate.ps1`, ejecuta una vez `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`,
> o usa directamente `.\.venv\Scripts\python.exe` sin activar el entorno.

Para cambiar un `.venv` que ya tenía la build CPU por la build CUDA:

```bash
nvidia-smi                                   # comprueba el driver y la versión de CUDA que soporta
pip uninstall -y torch
pip install torch --index-url https://download.pytorch.org/whl/cu128     # o cu130 con drivers recientes
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## Ollama

[Ollama](https://ollama.com) sirve el modelo de chat y el de embeddings de los notebooks de aplicaciones en tu propia máquina, en la GPU si la hay.

1. **Instálalo.** Windows y macOS: descarga el instalador de [ollama.com/download](https://ollama.com/download). Linux: `curl -fsSL https://ollama.com/install.sh | sh`.
2. **Arráncalo.** La aplicación de escritorio (Windows/macOS) y el servicio de Linux arrancan el servidor solos. Si no, ejecuta `ollama serve` en otra terminal. Escucha en `http://localhost:11434`.
3. **Descarga los modelos del curso** (unos 2,2 GB en total):

   ```bash
   ollama pull qwen2.5:3b          # modelo de chat, 3.1B parámetros, Q4_K_M, admite tool calling
   ollama pull nomic-embed-text    # modelo de embeddings para recuperación y memoria
   ```

   **Opcional: un juez más fuerte.** El notebook 18 usa `qwen2.5:7b` para juzgar con RAGAS las respuestas del 3B, el 14 lo usa para el agente que responde preguntas escribiendo código pandas (el 3B casi nunca lo consigue), y `scripts/generate_item_keywords.py` lo usa para escribir las palabras clave de las películas:

   ```bash
   ollama pull qwen2.5:7b          # 4,7 GB; juez de RAGAS (notebook 18) y agente de código (notebook 25)
   ```

   **El notebook 15** necesita además un modelo de razonamiento pequeño, y usa el juez de 7B como verificador y como juez por pares:

   ```bash
   ollama pull deepseek-r1:1.5b    # 1,1 GB; DeepSeek-R1-Distill-Qwen-1.5B (notebook 15)
   ```

   **El notebook 30** necesita el modelo guardián, y usa el 7B como segundo modelo evaluado y como candidato a juez LLM:

   ```bash
   ollama pull llama-guard3:1b     # 1,6 GB; Llama Guard 3 1B, el clasificador de seguridad (notebook 30)
   ```

   **El notebook 12** compara el mismo modelo de 1.5B en tres formatos GGUF, y usa el 7B en su gateway y en su cascada 3B → 7B:

   ```bash
   ollama pull qwen2.5:1.5b-instruct-q4_K_M   # 986 MB (notebook 12)
   ollama pull qwen2.5:1.5b-instruct-q8_0     # 1,6 GB (notebook 12)
   ollama pull qwen2.5:1.5b-instruct-fp16     # 3,1 GB (notebook 12)
   ```

   **El notebook 22** necesita un modelo de visión y lenguaje, que lee imágenes además de texto:

   ```bash
   ollama pull qwen2.5vl:3b        # 3,2 GB; Qwen2.5-VL 3B (notebook 22)
   ```

   Sin el juez de 7B todo funciona: `config.judge_model()` cae al modelo de chat y el 3B se juzga a sí mismo. El notebook 18 explica por qué eso no es buena idea: con el juez de 3B aparecían diferencias entre configuraciones que el de 7B no confirma. Con el 7B, la puntuación de RAGAS tarda unos 17 minutos en una GPU de 6 GB, frente a unos 7 con el 3B. Para usar otro juez, define `JUDGE_MODEL` en `.env`.

4. **Compruébalo:**

   ```bash
   ollama list                               # deben aparecer los dos modelos
   curl http://localhost:11434/api/tags      # el servidor responde con JSON
   ollama ps                                 # modelos cargados ahora mismo, y si están en GPU o CPU
   ```

Desde Python, `llm_edu_suite.config.setup()` imprime el proveedor, los modelos, si Ollama responde y si LangSmith está activo. Todos los notebooks de aplicaciones lo llaman en su primera celda.

## El fichero `.env`

`llm_edu_suite/config.py` lee toda su configuración de variables de entorno y carga el `.env` del proyecto con `python-dotenv` (sin sobrescribir las variables que ya estén definidas en tu shell). Copia `.env.example` a `.env` (los scripts de instalación lo hacen por ti) y edítalo.

| Variable | Por defecto | Significado |
|---|---|---|
| `LLM_PROVIDER` | `ollama` | `ollama`, `openai` o `anthropic`. |
| `LLM_MODEL` | `qwen2.5:3b` (Ollama), `gpt-4o-mini` (OpenAI), `claude-haiku-4-5` (Anthropic) | El modelo de chat que devuelve `get_llm()`. |
| `JUDGE_MODEL` | `qwen2.5:7b` si está descargado en Ollama; si no, el modelo de chat | El modelo que hace de juez (LLM-as-judge) en las evaluaciones: RAGAS en el notebook 18 y `config.get_judge_llm()`. Conviene que sea más fuerte que el modelo evaluado. |
| `EMBEDDING_MODEL` | `nomic-embed-text` (Ollama), `text-embedding-3-small` (OpenAI) | El modelo de embeddings que devuelve `get_embeddings()`. Anthropic no tiene API de embeddings, así que con `anthropic` los embeddings siguen saliendo de Ollama. |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Dónde está el servidor de Ollama. |
| `LANGSMITH_TRACING` | `true` en `.env.example` | Enviar trazas a LangSmith. Solo tiene efecto si además hay `LANGSMITH_API_KEY`. |
| `LANGSMITH_API_KEY` | vacío | Tu clave de API de LangSmith. |
| `LANGSMITH_PROJECT` | `llm-edu-suite` | El proyecto de LangSmith al que van las trazas. `setup(project=...)` lo cambia por notebook, para que las trazas de cada notebook vayan a su propio proyecto. |
| `LANGSMITH_ENDPOINT` | (comentada) | Solo para cuentas de la región UE: `https://eu.api.smith.langchain.com`. |
| `OPENAI_API_KEY` | (comentada) | Solo hace falta con `LLM_PROVIDER=openai`. |
| `ANTHROPIC_API_KEY`, `GEMINI_API_KEY`, `HARNESS_MODEL`, `HARNESS_REPS`, `HARNESS_WORKERS` | (comentada / sin definir) | Opcionales, para el modelo del notebook 27; ver [Claves de API opcionales](#claves-de-api-opcionales-notebook-27). `ANTHROPIC_API_KEY` es también la clave de `LLM_PROVIDER=anthropic`. |

Con `LANGSMITH_API_KEY` vacía, el trazado queda desactivado y todo lo demás funciona igual. El notebook 28 (observabilidad) es el que más partido saca de LangSmith.

### Crear una clave de API de LangSmith

1. Inicia sesión en [smith.langchain.com](https://smith.langchain.com) (hay un plan gratuito).
2. Ve a **Settings → API Keys → Create API Key**.
3. Copia la clave (empieza por `lsv2_`) y pégala en `.env`: `LANGSMITH_API_KEY=lsv2_...`.
4. Deja `LANGSMITH_TRACING=true`. La próxima vez que ejecutes un notebook, `setup()` imprime `langsmith: on (project: ...)` y las trazas aparecen en **Projects** en la interfaz de LangSmith.

> **Nunca subas `.env` al repositorio.** Está en `.gitignore`. Pon las claves solo en `.env` o en variables
> de entorno, nunca en el código ni en celdas de notebooks. Si una clave se filtra, revócala en la misma página de Settings.

### Cambiar de proveedor

Cambiar de proveedor es un cambio en `.env`, no en el código:

```bash
# OpenAI
pip install langchain-openai
# .env:  LLM_PROVIDER=openai   OPENAI_API_KEY=sk-...   (opcional: LLM_MODEL=..., EMBEDDING_MODEL=...)

# Anthropic
pip install langchain-anthropic
# .env:  LLM_PROVIDER=anthropic   ANTHROPIC_API_KEY=sk-ant-...   (los embeddings siguen saliendo de Ollama)
```

`langchain-openai` y `langchain-anthropic` son extras opcionales (`pip install -e ".[openai]"` o `".[anthropic]"`) porque la instalación por defecto no los necesita. Un modelo en la nube cuesta dinero por llamada, y algunos resultados de los notebooks (los números comentados en el texto) se obtuvieron con `qwen2.5:3b`, así que los tuyos serán distintos. `scripts/rag_demo.py --ragas` siempre llama a su juez a través del endpoint compatible con OpenAI de Ollama, así que necesita Ollama sea cual sea el proveedor.

### Claves de API opcionales (notebook 27)

Nada del curso necesita una API de pago, pero el notebook 27 es el único sitio donde una clave de API cambia la experiencia. Su harness de agentes ejecuta 16 tareas de programación, 3 veces cada una, con varias condiciones de harness: 366 ejecuciones del agente en total. **Con `ANTHROPIC_API_KEY` en el `.env`, su modelo por defecto es Claude Haiku 5.5** (`anthropic/claude-haiku-5-5`, al que se llama con litellm desde dentro de mini-swe-agent): la ejecución final usó 2,52 M tokens de prompt y 0,33 M de respuesta, **US$ 0,42** a US$ 0,10 / 0,50 por millón de tokens de entrada / salida, unos 7 s por ejecución. **Sin clave**, el notebook recurre a `qwen2.5:7b` con Ollama y avisa de ello. Funciona, pero es mucho más lento: en la v2.8 los modelos locales tardaban 64–216 s por tarea y resolvían el 20–30 % de las tareas 01–10.

| Variable | Qué hace |
|---|---|
| `ANTHROPIC_API_KEY` | Hace de Claude Haiku 5.5 el modelo por defecto del notebook 27. Es la misma variable que lee `ChatAnthropic` con `LLM_PROVIDER=anthropic` (ver [Cambiar de proveedor](#cambiar-de-proveedor)). |
| `HARNESS_MODEL` | Elige el modelo, como id de litellm: `anthropic/claude-haiku-5-5`, `openai/...`, `gemini/...`, o el nombre de un modelo de Ollama como `qwen2.5:7b`. `scripts/harness_run.py` trata un modelo como modelo por API cuando su prefijo es `gemini`, `anthropic`, `openai`, `openrouter`, `groq`, `mistral`, `deepseek` o `xai`; todo lo demás va a Ollama. La línea de coste del notebook solo conoce el precio de Claude Haiku 5.5. |
| `HARNESS_REPS` | Ejecuciones por tarea y condición (por defecto `3`). Una ejecución de un agente es una muestra; menos ejecuciones salen más baratas y más ruidosas. |
| `HARNESS_WORKERS` | Ejecuciones en paralelo (por defecto `4` con modelos por API y `1` con un modelo local, porque dos trabajos de Ollama a la vez pueden colgar una tarjeta de 6 GB). |
| `GEMINI_API_KEY` | Solo se usa si pones en `HARNESS_MODEL` un modelo de Gemini (por ejemplo `gemini/gemini-3.6-flash`). El nivel gratuito de Gemini permite unas 20 peticiones al día por modelo, demasiado pocas para este notebook. |
| `HARNESS_MIN_INTERVAL` | Segundos de espera entre llamadas al modelo (por ejemplo `7`), para niveles gratuitos con un límite de peticiones por minuto. Por defecto `0`; el notebook pone 7 con los modelos de Gemini. |

**Dónde van las claves.** En el `.env`, que está en `.gitignore` (`.env.example` ya trae `ANTHROPIC_API_KEY`, comentada), nunca en una celda de un notebook ni en la línea de comandos. El notebook carga el `.env` por su cuenta (`ANTHROPIC_API_KEY`, `HARNESS_MODEL`) y lanza `scripts/harness_run.py` con su propio entorno, así que litellm puede leer la clave. **La clave nunca llega a la shell del agente**: cada comando que ejecuta el agente recibe un entorno limpio con solo `SYSTEMROOT`, `PATH`, `TEMP`, `TMP`, `HOME`, `USERPROFILE` y `LANG`, y la política de permisos rechaza los comandos que leen `.env`. `harness_run.py` no lee el `.env` por su cuenta: si lo ejecutas a mano con un modelo por API, la clave tiene que estar definida en la shell que lo arranca.

## El entorno del harness de agentes (notebook 27)

El notebook 27 ejecuta un harness real de agentes de programación, [mini-swe-agent](https://github.com/SWE-agent/mini-swe-agent) (MIT), con Claude Haiku 5.5 por defecto o con un modelo local a través de Ollama. mini-swe-agent vive en **su propio entorno virtual**, `.venv-harness`, junto a `.venv`: su interfaz de terminal necesita un `rich` más nuevo del que admite el entorno del curso, y tenerlo aparte evita que un entorno rompa el otro. El notebook en sí sigue corriendo en el kernel del curso; lanza `scripts/harness_run.py` con el Python del entorno del harness, un proceso por ejecución (4 a la vez con un modelo por API), y ejecuta los tests ocultos con el `pytest` de ese Python.

**1. Crea el entorno** (desde la raíz del repositorio; `.venv-*/` ya está en `.gitignore`):

```powershell
# Windows
python -m venv .venv-harness
.venv-harness\Scripts\python -m pip install mini-swe-agent==2.4.6 pytest
```

```bash
# Linux / macOS
python3 -m venv .venv-harness
.venv-harness/bin/python -m pip install mini-swe-agent==2.4.6 pytest
```

El notebook busca exactamente `.venv-harness/Scripts/python.exe` (Windows) o `.venv-harness/bin/python` (Linux/macOS) en la raíz del repositorio, y se para con `Create the harness environment first` si no lo encuentra. La versión 2.4.6 es con la que se ejecutó el notebook (trae litellm, que es quien habla con la API de Anthropic o con Ollama); el notebook imprime el código de las funciones de mini-swe-agent, así que otra versión puede enseñar un código distinto.

**2. Bash.** La única herramienta del agente es una shell bash. En Linux y macOS el runner usa el `bash` del `PATH`. En Windows usa **Git Bash**, de [Git for Windows](https://git-scm.com/download/win), en `C:\Program Files\Git\bin\bash.exe`; si el tuyo está en otro sitio, define la variable de entorno `HARNESS_BASH` con la ruta completa de `bash.exe` antes de abrir Jupyter.

**3. El modelo.** Con `ANTHROPIC_API_KEY` en el `.env` no hay nada que descargar: el notebook usa Claude Haiku 5.5 (ver [Claves de API opcionales](#claves-de-api-opcionales-notebook-27)). Sin ella, o con `HARNESS_MODEL` apuntando a un modelo local, va por Ollama (el mismo servidor que el resto del curso; la variable de entorno `OLLAMA_BASE_URL` apunta el runner a otro):

```bash
ollama pull qwen2.5:7b                  # 4,7 GB, el modelo de reserva cuando no hay clave de API
ollama pull freehuntx/qwen3-coder:8b    # 5,2 GB, opcional: uno de los tres modelos locales de la v2.8 (con qwen2.5:3b y qwen2.5:7b)
```

Con modelos locales el runner pide a Ollama un **contexto de 16k** (`num_ctx=16384`), porque el historial de un agente crece con cada paso. En una tarjeta de 6 GB eso va bien para el 3B y el 7B, pero **el modelo de 8B con 16k no cabe en la VRAM**: `ollama ps` enseña cerca del 46 % en la CPU. En la v2.8 (tareas 01–10) una ejecución tardaba 64 s con el 3B, 82 s con el 7B y 216 s con el 8B, frente a unos 7 s con Claude Haiku 5.5. Con un modelo local, ejecuta el notebook solo (ver [Notas sobre la GPU](#notas-sobre-la-gpu)).

**4. Compruébalo** sin el notebook: `-h` enseña los argumentos, y basta una carpeta con un `TASK.md` para una ejecución de verdad (ver [scripts de línea de comandos](scripts_cli.md#harness_runpy)):

```powershell
.venv-harness\Scripts\python scripts\harness_run.py -h
```

> El sandbox del runner (un entorno limpio, un timeout, una lista de comandos denegados y la carpeta de la tarea como directorio de trabajo) es una capa didáctica, como la del notebook 25, no una frontera de seguridad. Úsalo solo con carpetas de tareas desechables.

## El entorno de Inspect y Phoenix (notebook 29)

El notebook 29 usa dos herramientas de código abierto que viven fuera del entorno del curso: [Inspect AI](https://inspect.aisi.org.uk) (UK AI Security Institute, MIT), que necesita un `rich` más nuevo del que admite el entorno del curso (el mismo motivo que `.venv-harness`), y el servidor de [Phoenix](https://github.com/Arize-ai/phoenix) (Arize, Elastic License 2.0). Las dos van en **un tercer entorno virtual**, `.venv-eco`, junto a `.venv`. El notebook en sí corre en el kernel del curso: llama al comando `inspect` de `.venv-eco` para las evaluaciones y arranca su comando `phoenix` como un proceso servidor aparte, como correría Phoenix junto a una aplicación en producción. La parte cliente de las trazas, el exportador de OpenTelemetry y el instrumentador de LangChain, está en el entorno del curso.

**1. Crea el entorno** (desde la raíz del repositorio; `.venv-*/` ya está en `.gitignore`):

```powershell
# Windows
python -m venv .venv-eco
.venv-eco\Scripts\python -m pip install inspect-ai==0.3.289 arize-phoenix==21.0.0 datasets
```

```bash
# Linux / macOS
python3 -m venv .venv-eco
.venv-eco/bin/python -m pip install inspect-ai==0.3.289 arize-phoenix==21.0.0 datasets
```

`datasets` está porque la tarea de Inspect carga GSM8K por su cuenta. Son las versiones con las que se ejecutó el notebook, y su primera celda las imprime (`Inspect AI 0.3.289 | Phoenix 21.0.0`). El notebook busca exactamente `.venv-eco/Scripts/inspect.exe` (Windows) o `.venv-eco/bin/inspect` (Linux/macOS) en la raíz del repositorio, y se para con `create .venv-eco first (see scripts/inspect_tasks/gsm8k_cot.py)` si no lo encuentra.

**2. Los paquetes cliente** van en el `.venv` del curso y ya están en `pyproject.toml`: `arize-phoenix-otel` (el SDK de OpenTelemetry y un exportador ya configurado para Phoenix) y `openinference-instrumentation-langchain` (los spans de LangChain, en formato OpenInference). Si tu `.venv` es anterior al notebook 29, vuelve a ejecutar `pip install -r requirements.txt` dentro de él.

**3. Dónde van los datos.** Inspect escribe un log en JSON por ejecución en `data/cache/inspect_logs/`, y Phoenix guarda su base de datos SQLite y su `server.log` en `data/cache/phoenix/`; el notebook vacía las dos carpetas al empezar cada ejecución. Phoenix solo escucha en `127.0.0.1`, en un puerto libre que elige el notebook, con su propia telemetría desactivada (`PHOENIX_TELEMETRY_ENABLED=false`), así que ninguna traza sale de la máquina. Mientras el notebook corre, la interfaz de Phoenix está en la dirección que imprime su celda.

**4. Compruébalo** sin el notebook (el docstring de la propia tarea trae los mismos comandos; ver [scripts de línea de comandos](scripts_cli.md#inspect_tasksgsm8k_cotpy)):

```powershell
.venv-eco\Scripts\inspect eval scripts\inspect_tasks\gsm8k_cot.py --model ollama/qwen2.5:3b --model-base-url http://localhost:11434/v1 --temperature 0 --max-tokens 512 --max-connections 1 -T n=5
.venv-eco\Scripts\inspect view          # recorre los logs
```

## Qué ejecutar después

Tiempos medidos en la máquina de verificación (RTX 3060 Laptop GPU, Ollama en local).

| Qué | Comando | Tiempo |
|---|---|---|
| Tests rápidos (sin Ollama) | `pytest -q -m "not ollama"` | ~42 s (90 tests; con modelos falsos, y además lanzan código de ataque contra el sandbox del notebook 25 y comprueban que `mcp_server.py --require-token` rechaza las peticiones sin el bearer token) |
| Tests de integración con Ollama | `pytest -q -m ollama` | ~40 s (4 tests; se saltan solos si Ollama no está en marcha) |
| Todos los tests | `pytest -q` | la suma de los dos anteriores |
| Abrir los notebooks | `jupyter lab notebooks/` y elegir el kernel **Python (llm-edu-suite)** | — |
| Ejecutar todos los notebooks sin abrirlos | `python tools/run_notebooks.py` | largo; ver la [guía de notebooks](guia_notebooks.md) |
| Ejecutar algunos | `python tools/run_notebooks.py 05 14` (por número) o `python tools/run_notebooks.py 08_pretraining.ipynb` | — |
| Regenerar los notebooks desde sus generadores | `python tools/build_notebooks.py` (todos) o `python tools/build_notebooks.py 05 14`; con `--keep-outputs` conserva las salidas de las celdas de código que no cambiaron | segundos |
| Preentrenar un LM pequeño | `python scripts/train.py --steps 300 --out runs/smoke` | ver [scripts](scripts_cli.md#trainpy) |
| SFT con LoRA sobre SmolLM2 | `python scripts/finetune.py sft` | ~1,5 min en GPU |
| RAG sobre la documentación del curso | `python scripts/rag_demo.py --ask "What is GQA?"` | ~15 s |
| Servidor MCP | `python scripts/mcp_server.py` | — |

Los `.ipynb` se **generan** a partir de `tools/notebooks/nbNN_*.py` (y `scripts/mini_llm.py` a partir del generador del notebook 01). Para cambiar un notebook de forma permanente, edita su generador y regenera; lo que edites directamente en el `.ipynb` se pierde en la siguiente generación. Generar borra las salidas, así que ejecuta después `tools/run_notebooks.py`; si solo cambiaste texto (markdown o comentarios que no tocan el código), `python tools/build_notebooks.py --keep-outputs 12` mantiene las salidas ya ejecutadas de cada celda de código cuyo fuente no cambió y te ahorra volver a ejecutar el notebook ([detalles](scripts_cli.md#toolsbuild_notebookspy---keep-outputs)). Con `make` disponible (Linux/macOS/Git Bash): `make test`, `make smoke`, `make notebooks`, `make run-notebooks`, `make mini-llm`, `make mcp-server` (ver [scripts de línea de comandos](scripts_cli.md#herramientas-y-objetivos-del-makefile)).

## Datos que se descargan al ejecutar

- **SmolLM2** (notebooks 11, 13 y 14 y `finetune.py`): desde el Hugging Face Hub a la caché de Hugging Face, la primera vez.
- **ModernBERT-base, Flan-T5-small y Banking77** (notebooks 06, 21 y 24): `answerdotai/ModernBERT-base` (~600 MB), `google/flan-t5-small` (~300 MB) y el dataset `PolyAI/banking77` (CC BY 4.0) se descargan del Hugging Face Hub a su caché la primera vez que se ejecuta el notebook. El notebook 21 trata los 3.080 mensajes de test de Banking77 como logs sin etiquetar (las etiquetas solo sirven para puntuar el clustering) y usa su split de entrenamiento para los centroides de las intenciones conocidas de la sección de monitorización; usa además `sentence-transformers/all-MiniLM-L6-v2` y `BAAI/bge-small-en-v1.5`, las mismas descargas que el notebook 17. El notebook 24 codifica el split de entrenamiento de Banking77 con `BAAI/bge-small-en-v1.5` para sus tests de drift.
- **FineWeb, FineWeb-Edu y el clasificador de FineWeb-Edu** (notebook 07): el notebook lee en streaming (`datasets.load_dataset(..., streaming=True)`) los primeros 12.000 documentos de `HuggingFaceFW/fineweb` y 1.000 de `HuggingFaceFW/fineweb-edu` (configuración `sample-10BT`), unos 100 MB de descarga, y los guarda una sola vez en `data/cache/fineweb_12k.jsonl` y `data/cache/fineweb_edu_1k.jsonl` (unos 45 MB). Las siguientes ejecuciones leen esos ficheros sin conexión. FineWeb tiene licencia ODC-By y el repositorio no lo redistribuye. El clasificador `HuggingFaceFW/fineweb-edu-classifier` (~440 MB) y el tokenizador de `HuggingFaceTB/SmolLM2-135M` van a la caché de Hugging Face la primera vez.
- **GSM8K** (notebook 15): el split de test de `openai/gsm8k` (configuración `main`, licencia MIT) se descarga del Hugging Face Hub a su caché la primera vez que se ejecuta el notebook, que usa 80 de sus problemas.
- **SciFact, NFCorpus y los modelos de recuperación** (notebook 17): las copias de BEIR de SciFact y NFCorpus en el Hugging Face Hub (`mteb/scifact` y `mteb/nfcorpus`, configuraciones `corpus`, `queries` y `default` para las etiquetas de relevancia) y los modelos `sentence-transformers/all-MiniLM-L6-v2`, `BAAI/bge-small-en-v1.5`, `cross-encoder/ms-marco-MiniLM-L6-v2` y `BAAI/bge-reranker-base` se descargan a la caché de Hugging Face la primera vez que se ejecuta el notebook. `nomic-embed-text` viene de Ollama, como en los notebooks de aplicaciones. Los modelos ajustados se entrenan en memoria y no se guardan.
- **JailbreakBench y XSTest** (notebook 30): los 100 comportamientos dañinos y sus 100 gemelos benignos de `JailbreakBench/JBB-Behaviors` (configuración `behaviors`, splits `harmful` y `benign`), los prompts de XSTest de `Paul/XSTest` (el notebook toma una muestra de 100 seguros y 100 inseguros) y `natolambert/xstest-v2-copy`, las 900 respuestas de GPT-4 y Llama 2 a XSTest con sus etiquetas humanas, que sirven para validar el juez de rechazos. Los tres se descargan del Hugging Face Hub a su caché la primera vez que se ejecuta el notebook. `llama-guard3:1b` viene de Ollama. El notebook puntúa las respuestas a las peticiones dañinas, pero nunca las imprime ni las guarda.
- **ARC-Easy y GSM8K** (notebook 12): el test de ARC-Easy (`allenai/ai2_arc`, configuración `ARC-Easy`; el notebook toma 200 preguntas con una semilla fija) y los mismos 80 problemas de GSM8K que el notebook 15 (`openai/gsm8k`, configuración `main`) se descargan del Hugging Face Hub a su caché la primera vez que se ejecuta el notebook. Los documentos de políticas que hacen de system prompt cacheado son los del propio curso, en `data/sample_docs/`.
- **Oxford-IIIT Pet y CLIP** (notebook 22): la copia `timm/oxford-iiit-pet` de Oxford-IIIT Pet (Parkhi et al., 2012; 37 razas de gatos y perros, 3.680 imágenes de entrenamiento; el notebook toma 600 imágenes de test con una semilla fija) y el modelo `openai/clip-vit-base-patch32` se descargan del Hugging Face Hub a su caché la primera vez que se ejecuta el notebook. `qwen2.5vl:3b` viene de Ollama. Las facturas son sintéticas, las dibuja el propio notebook con Pillow, así que no se descarga ningún documento.
- **Notebooks 00, 20 y 29**: el 00 descarga el modelo base `HuggingFaceTB/SmolLM2-135M` (260 MB) a la caché de Hugging Face; el 20 reutiliza Banking77 (la misma muestra de test de 231 mensajes que el notebook 06) y `BAAI/bge-small-en-v1.5`, y escribe su programa optimizado en `data/cache/dspy_banking77_program.json`, que borra después de comprobar que se recarga bien; el 29 reutiliza los 80 problemas de GSM8K del notebook 15 (los carga la tarea de Inspect desde `.venv-eco` y el notebook para las 20 peticiones trazadas), y deja los logs de Inspect y la base de datos de Phoenix dentro de `data/cache/`.
- **MovieLens-1M** (notebooks 23 y 24 y `scripts/generate_item_keywords.py`): `llm_edu_suite.recsys.download_movielens_1m()` baja el zip oficial de GroupLens (6 MB) y lo descomprime en `data/cache/ml-1m/` una sola vez. `data/cache/` está en `.gitignore`: la licencia de GroupLens no permite redistribuir el dataset, así que el repositorio no lo incluye y cada uno lo descarga de la fuente original. Si estás sin conexión, copia `ml-1m.zip` en `data/cache/` y la función lo usará.
- Lo que sí está en el repositorio, porque es trabajo propio: `data/recsys/ml1m_llm_keywords.json` (las palabras clave que escribió `qwen2.5:7b` para cada película; regenerarlas lleva una hora, ver [`generate_item_keywords.py`](scripts_cli.md#generate_item_keywordspy)) y `data/recsys/conversational_eval.json` (20 peticiones con restricciones verificables).

## Notas sobre la GPU

- Nada necesita GPU salvo el notebook 09, pero el entrenamiento es mucho más rápido con ella: el mini LLM del notebook 01 tarda unos **22 s en una RTX 3060 Laptop frente a unos 5 min en CPU**. `llm_edu_suite.utils.get_device()` elige `cuda` si existe (Apple MPS solo con `allow_mps=True`). En RTX 30xx o superior, el entrenamiento usa precisión mixta **bf16**.
- **Presupuesto de VRAM en una GPU de 6 GB.** Ollama mantiene `qwen2.5:3b` cargado en la GPU unos minutos después de cada llamada, y ocupa unos **2,3 GB** de VRAM. Los notebooks de entrenamiento limitan la memoria de su propio PyTorch con `torch.cuda.set_per_process_memory_fraction` (variable de entorno `VRAM_BUDGET_GB`; p. ej. `$env:VRAM_BUDGET_GB="3"` o `export VRAM_BUDGET_GB=3` para darles más). El valor por defecto es **2 GB en el notebook 13**, que entrena once modelos seguidos y cuyo modelo de 4 bits de QLoRA deja unos 0,1 GB ocupados después, y **1,5 GB en el 08 y el 09**.
- **No ejecutes a la vez los notebooks de entrenamiento (11, 13 y 14, y el 17, que entrena y llama a Ollama) y los de Ollama (16, 18, 19, 23, 25, 26 y 28) en una GPU de 6 GB.** Lo mismo vale para el 22, que mantiene CLIP en la GPU mientras llama al modelo de visión. Cuando compiten por la VRAM, Ollama puede quedarse colgado en lugar de fallar. Ejecútalos uno detrás de otro y, si necesitas recuperar la memoria, descarga los modelos con `ollama stop qwen2.5:3b` (`ollama ps` muestra qué hay cargado).
- **El notebook 12 mide latencias, así que ejecútalo solo.** Cualquier otro trabajo en la GPU (un notebook de entrenamiento, otro cliente de Ollama) cambia todos los números. Para la sección de concurrencia descarga los modelos del servidor de Ollama por defecto, arranca un segundo servidor con `OLLAMA_NUM_PARALLEL=4` y `OLLAMA_MAX_LOADED_MODELS=1` en un puerto libre y lo para al terminar, así que la configuración del servidor por defecto no cambia nunca. El segundo servidor tiene que encontrar los mismos modelos: el notebook lee `OLLAMA_MODELS` del `%LOCALAPPDATA%\Ollama\server.log` de la aplicación de escritorio y, si no, recurre a la variable de entorno `OLLAMA_MODELS` o a `~/.ollama/models`.
- Si Ollama no consigue meter un modelo en la GPU, lo ejecuta (en parte) en CPU: sigue funcionando, solo que más lento.
- **El notebook 27 con un modelo local es una larga cadena de llamadas a Ollama, así que ejecútalo solo.** (Con Claude Haiku 5.5, su modelo por defecto cuando hay `ANTHROPIC_API_KEY`, no usa la GPU.) `freehuntx/qwen3-coder:8b` con el contexto de 16k del runner necesita más de 6 GB y va cerca de un 46 % en la CPU, y otro trabajo de Ollama a la vez puede dejar Ollama atascado. En la ejecución local de la v2.8, una tarea del 3B llegó al timeout de seguridad del notebook sin dar un solo paso, un atasco de Ollama; el notebook cuenta esa ejecución como fallida y sigue, y ejecuta los modelos locales de uno en uno (`HARNESS_WORKERS=1`).
- **Los notebooks 20 y 29 también son largas cadenas de llamadas a Ollama, así que ejecútalos de uno en uno.** El 28 hace miles de llamadas (solo los optimizadores, entre 650 y 870 cada uno) y carga `qwen2.5:7b` para sus optimizadores y su clasificador de 7B, con bge-small en la GPU al lado; el 29 manda 320 generaciones de Inspect y 20 peticiones trazadas a `qwen2.5:3b`. Inspect se ejecuta con `--max-connections 1`, una petición cada vez, por el mismo motivo.
- **El notebook 09 en Windows.** Las wheels de PyTorch para Windows no traen FlashAttention-2 (SDPA usa entonces los kernels memory-efficient o cuDNN) ni Triton, así que el backend por defecto de `torch.compile`, `inductor`, falla con `TritonMissing` y el notebook usa `backend="cudagraphs"`. Para FlashAttention-2, `torch.compile` con `inductor` o el paquete `flash-attn`, usa Linux o WSL2. El DDP del notebook corre en dos procesos de CPU con `gloo`, lanzados con `torch.multiprocessing.spawn` y conectados por un fichero compartido (`init_method="file://..."`, un `FileStore`), porque el rendezvous TCP de `torchrun` necesita libuv, que la build de Windows no incluye; NCCL, el backend de GPU, solo existe en Linux.
- **Lento en vez de sin memoria.** En Windows, cuando un proceso supera la VRAM, el driver puede desbordar a la RAM del sistema ("memoria de GPU compartida") en lugar de dar un error: en el notebook 09, el backend *math* de SDPA declaró 8,1 GB en una tarjeta de 6 GB y tardó 978 ms. Si algo se vuelve lento de repente, mira la memoria de GPU compartida en el Administrador de tareas.

## Problemas frecuentes

| Síntoma | Causa | Solución |
|---|---|---|
| `!! Ollama is not running`, `Connection refused` en el puerto 11434, o tests de Ollama saltados | El servidor de Ollama no está en marcha | Abre la aplicación de Ollama o ejecuta `ollama serve`; compruébalo con `curl http://localhost:11434/api/tags` |
| `model "qwen2.5:3b" not found, try pulling it first` (o lo mismo con `nomic-embed-text`) | No se descargaron los modelos | `ollama pull qwen2.5:3b` y `ollama pull nomic-embed-text`; compruébalo con `ollama list` |
| Un notebook o script se queda colgado en la primera llamada al LLM | Ollama se ha quedado sin VRAM porque un notebook de entrenamiento ocupa la GPU | Para el kernel de entrenamiento, `ollama stop qwen2.5:3b`, y ejecuta los notebooks uno detrás de otro (ver [notas sobre la GPU](#notas-sobre-la-gpu)) |
| `LangChainDeprecationWarning` / aviso de que `langchain-community` se va a retirar (*sunset*) | `FAISS` y `BM25Retriever` siguen viviendo en `langchain-community` | Inofensivo; los notebooks y `rag_demo.py` lo filtran. No hay que hacer nada |
| `ModuleNotFoundError: No module named 'langchain_community.chat_models.vertexai'` al importar `ragas` | ragas 0.4.3 importa un módulo que langchain-community 0.4 eliminó | Ya está resuelto: el notebook 18 y `rag_demo.py` registran un módulo vacío de sustitución antes de importar ragas. Copia ese *shim* (`_ragas_compat_shim` en `scripts/rag_demo.py`) si importas ragas en tu propio código |
| `UserWarning: huggingface_hub cache-system uses symlinks...` en Windows | Windows no permite enlaces simbólicos sin el modo de desarrollador | Inofensivo (la caché ocupa algo más de disco). Activa el modo de desarrollador o define `HF_HUB_DISABLE_SYMLINKS_WARNING=1` para ocultarlo |
| El cliente MCP por stdio falla dentro de Jupyter con un error sobre `fileno` / stderr | El `sys.stderr` de Jupyter no tiene un descriptor de fichero del sistema, y el SDK se lo pasa al subproceso del servidor | Ya está resuelto en el notebook 26, que manda el stderr del servidor a un fichero de log. En scripts normales no hace falta |
| `ModuleNotFoundError: llm_edu_suite` en un notebook | Jupyter se abrió fuera del repo, o con otro kernel | Abre con `jupyter lab notebooks/`; cada notebook añade la raíz del repo al `sys.path` |
| `ModuleNotFoundError: No module named 'umap'` en el notebook 21 | El entorno se creó antes de que `umap-learn` entrara en `pyproject.toml` | Vuelve a ejecutar `pip install -r requirements.txt` (o `pip install umap-learn`) dentro del `.venv` |
| El notebook usa otro Python | Kernel equivocado | *Kernel → Change kernel → Python (llm-edu-suite)* |
| `RuntimeWarning: Proactor event loop...` en Windows | Aviso inofensivo de `pyzmq` | Se puede ignorar; `tools/run_notebooks.py` ya lo silencia |
| La celda de FlexAttention se salta | PyTorch < 2.5 o sin soporte en CPU | Es opcional; el resto del notebook 04 funciona igual |
| `ImportError: numpy.core.multiarray failed to import` | matplotlib compilado para NumPy 1.x con NumPy 2.x instalado | Usa el `.venv` del proyecto (o `pip install -U matplotlib`) |
| Entrenamiento lento | Todo corre en CPU | Instala la build CUDA de PyTorch, o baja `--steps` / `--d-model` |
| El notebook 09 se para con `this notebook measures GPU memory and kernels; it needs a CUDA GPU` | No hay GPU NVIDIA, o PyTorch es la build de CPU | Instala la build CUDA de PyTorch; sin GPU el notebook no tiene nada que medir |
| `SDPA: FlashAttention-2: not available in this PyTorch build` o `inductor backend: unavailable here (TritonMissing)` en el notebook 09 | Las wheels de PyTorch para Windows no incluyen FlashAttention-2 ni Triton | Inofensivo: el notebook compara los kernels disponibles y usa CUDA graphs. Para probar los que faltan, ejecútalo en Linux o WSL2 |
| El notebook 12 se para con `the second Ollama server did not start` | El `ollama serve` temporal no respondió a tiempo; el notebook descarta su salida, así que no se ve el motivo | Lanza `ollama serve` a mano en un terminal con `OLLAMA_HOST=127.0.0.1:<puerto>` para ver el error, y comprueba que `ollama` está en el `PATH` del kernel de Jupyter |
| En la sección de concurrencia del notebook 12, el segundo servidor responde `model ... not found` | El segundo servidor busca los modelos en otra carpeta (por ejemplo, los moviste y no usas la aplicación de escritorio, así que no hay `server.log` que leer) | Define `OLLAMA_MODELS` con la carpeta de tus modelos antes de abrir Jupyter |
| `bind: An attempt was made to access a socket in a way forbidden by its access permissions` al arrancar un segundo `ollama serve` en un puerto fijo (p. ej. `OLLAMA_HOST=127.0.0.1:11435`) | Windows reserva rangos de puertos TCP (para Hyper-V, WSL2 o Docker) y el puerto elegido cae dentro de uno | Lista los rangos reservados con `netsh interface ipv4 show excludedportrange protocol=tcp` y elige un puerto fuera de ellos, o haz como el notebook 12 y deja que el sistema te dé uno libre (`socket.bind(("127.0.0.1", 0))`) |
| El notebook 27 se para con `Create the harness environment first` | No hay un `.venv-harness` en la raíz del repositorio, o tiene otro nombre | Créalo como en [El entorno del harness de agentes](#el-entorno-del-harness-de-agentes-notebook-27); el notebook busca exactamente esa carpeta |
| En el notebook 27 todas las ejecuciones fallan en su primer comando, con un estado de salida que empieza por `error: FileNotFoundError` | El runner no encontró bash (en Windows, Git Bash no está en `C:\Program Files\Git\bin\bash.exe`) | Instala Git for Windows, o define `HARNESS_BASH` con la ruta completa de tu `bash.exe` y reinicia el kernel |
| El notebook 27 imprime `falling back to a local model, which is much slower` | No hay `ANTHROPIC_API_KEY` en el `.env` ni `HARNESS_MODEL` | Añade la clave al `.env` (unos US$ 0,42 por ejecución completa) y reinicia el kernel, o deja que se ejecute con `qwen2.5:7b` |
| En el notebook 27 las ejecuciones locales tardan minutos cada una, o una termina como `harness: TimeoutExpired` | Un modelo local con contexto de 16k se desborda a la CPU (el qwen3-coder de 8B tardaba ~216 s por tarea en la v2.8), u otro trabajo está usando Ollama | Es lo esperado en una tarjeta de 6 GB; mira `ollama ps` y no ejecutes nada más en la GPU |
| El notebook 20 se para con `ModuleNotFoundError: No module named 'dspy'`, o el 29 con `No module named 'phoenix.otel'` o `'openinference'` | El `.venv` se creó antes de que estos paquetes entraran en `pyproject.toml` | Vuelve a ejecutar `pip install -r requirements.txt` dentro de `.venv` y reinicia el kernel |
| El notebook 29 se para con `create .venv-eco first` | No hay un `.venv-eco` en la raíz del repositorio, o tiene otro nombre | Créalo como en [El entorno de Inspect y Phoenix](#el-entorno-de-inspect-y-phoenix-notebook-29); el notebook busca exactamente esa carpeta |
| El notebook 29 se queda en `Phoenix up at ...` o falla su primera consulta de spans | El servidor de Phoenix no arrancó a tiempo, o se cerró | Lee `data/cache/phoenix/server.log`, adonde el notebook manda la salida del servidor; comprueba que `.venv-eco` tiene `arize-phoenix==21.0.0` |
| `torchrun` falla en Windows con un error sobre libuv | El rendezvous TCP de `torchrun` necesita libuv, que la build de Windows no trae | Lanza los procesos con `torch.multiprocessing.spawn` e `init_method="file://..."`, como el notebook 09, o usa Linux o WSL2 |
