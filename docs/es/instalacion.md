🇬🇧 [English version](../en/installation.md)

# Instalación y entorno de trabajo

[← Volver al índice](README.md)

Esta guía deja el proyecto listo para ejecutar **todo**: la suite de tests, los scripts de línea de comandos y los 22 notebooks.

El curso tiene dos mitades con necesidades distintas:

| Notebooks | Qué usan | Qué necesitan |
|---|---|---|
| **00–06** | El modelo PyTorch escrito desde cero (paquete `llm_edu_suite`) | Solo PyTorch. Funciona en CPU; si hay GPU NVIDIA se usa automáticamente. |
| **07–09** | Hugging Face `transformers` + `peft` + `trl` + `bitsandbytes` + `optimum-quanto` sobre SmolLM2 | PyTorch y una descarga única de SmolLM2 desde el Hugging Face Hub (~1 GB). Con GPU van mucho más rápido; la sección de QLoRA del 07 necesita GPU NVIDIA (`bitsandbytes`). |
| **10–16** | LangChain, RAGAS, LangGraph, LangSmith, el SDK oficial de MCP + `langchain-mcp-adapters`; en el 12, además FAISS y LightGBM | Un servidor **Ollama** local con `qwen2.5:3b` y `nomic-embed-text` (y, opcional, `qwen2.5:7b` como juez del notebook 10 y para el agente que escribe código del 14). El notebook 14 usa además `pandas`, que se instala como dependencia de `datasets`. LangSmith es opcional. El notebook 12 descarga MovieLens-1M (6 MB) la primera vez. |
| **17** | El modelo desde cero con `causal=False`, Hugging Face `transformers` + `datasets` (ModernBERT-base, Flan-T5-small, Banking77), `scikit-learn` y LangChain | PyTorch, la descarga de ModernBERT-base (~600 MB), Flan-T5-small (~300 MB) y Banking77 desde el Hugging Face Hub la primera vez, y **Ollama** con `qwen2.5:3b` y `nomic-embed-text` para la comparación de la sección 2. Con GPU tarda unos 15 minutos. |
| **18** | El `TransformerLM` desde cero (dos entrenamientos), Hugging Face `datasets` + `transformers` (muestras de FineWeb y FineWeb-Edu, el clasificador de FineWeb-Edu, el tokenizador de SmolLM2) y LangChain | PyTorch, unos 100 MB de muestras de FineWeb que se leen en streaming una sola vez y se guardan en `data/cache/`, el clasificador `HuggingFaceFW/fineweb-edu-classifier` (~440 MB) y **Ollama** con `qwen2.5:3b` para la sección de datos sintéticos. Con GPU tarda unos 30 minutos (dos entrenamientos de 10–11 minutos y unos 3 minutos de generación). |
| **19** | El `TransformerLM` desde cero, `torch.profiler`, `torch.utils.checkpoint`, los backends de `F.scaled_dot_product_attention`, `torch.compile` y `torch.distributed` | PyTorch con CUDA y **una GPU NVIDIA** (mide la memoria y los kernels de la GPU; sin ella se detiene al empezar). No descarga nada ni usa Ollama. En Windows faltan FlashAttention-2 y Triton y `torchrun` no funciona (ver [notas sobre la GPU](#notas-sobre-la-gpu)); el notebook lo detecta y sigue. Tarda unos 5 minutos. |
| **20** | LangChain con Ollama (`config.get_llm` y `get_judge_llm`), `datasets` de Hugging Face (GSM8K) y Pydantic para la salida estructurada del juez | **Ollama** con `qwen2.5:3b`, `deepseek-r1:1.5b` (1,1 GB) y, como verificador y juez por pares, el `qwen2.5:7b` opcional; el test de GSM8K se descarga del Hugging Face Hub la primera vez. No entrena nada. Tarda unos 53 minutos, casi todo en llamadas al LLM. |
| **21** | `sentence-transformers` (bi-encoders, cross-encoders, entrenamiento contrastivo, cuantización de embeddings), `rank-bm25`, `datasets` de Hugging Face y el `OllamaEmbeddings` de LangChain a través de `config.get_embeddings` | SciFact y NFCorpus (`mteb/scifact`, `mteb/nfcorpus`) y cuatro modelos (all-MiniLM-L6-v2, BAAI/bge-small-en-v1.5, cross-encoder/ms-marco-MiniLM-L6-v2, BAAI/bge-reranker-base), que se descargan del Hugging Face Hub la primera vez, y **Ollama** con `nomic-embed-text`. Se recomienda GPU (tres ajustes cortos y dos cross-encoders); en CPU no se ha medido. Tarda unos 18 minutos en GPU. |

No hace falta ninguna clave de API. Por defecto todo funciona en local.

## Requisitos

| Componente | Mínimo | Verificado con |
|---|---|---|
| Python | 3.11 | 3.11.4 (Windows 11) |
| PyTorch | 2.3 (2.5+ para la celda opcional de FlexAttention del notebook 03) | 2.14.1+cu130 |
| GPU NVIDIA | opcional | RTX 3060 Laptop GPU, 6 GB |
| [Ollama](https://ollama.com) | cualquier versión reciente | 0.40.15 |
| Disco: entorno virtual | ~4 GB con la build CUDA de PyTorch (PyTorch solo ocupa ~2,9 GB); mucho menos con la de CPU | 4,1 GB |
| Disco: modelos de Ollama | ~2,2 GB (`qwen2.5:3b` 1,9 GB + `nomic-embed-text` 0,27 GB); 4,7 GB más con el `qwen2.5:7b` opcional (juez en el 10, agente de código en el 14), y 1,1 GB de `deepseek-r1:1.5b` (notebook 20) | |
| Disco: datos descargados | MovieLens-1M, 6 MB comprimido, y las muestras de FineWeb (12.000 documentos) y FineWeb-Edu (1.000), unos 45 MB en JSONL, en `data/cache/` | |
| Disco: caché de Hugging Face | ~1 GB (SmolLM2-135M-Instruct 260 MB + SmolLM2-360M-Instruct 690 MB) y ~0,9 GB más con el notebook 17 (ModernBERT-base ~600 MB + Flan-T5-small ~300 MB, más Banking77) y ~0,45 GB más con el notebook 18 (`fineweb-edu-classifier` ~440 MB y el tokenizador de SmolLM2-135M), más los cuatro modelos y los dos conjuntos de datos del notebook 21 | |

Las dependencias se declaran una sola vez, en `pyproject.toml` (con cotas inferiores); `requirements.txt` solo contiene `-e .[dev]`, es decir, instala el proyecto desde ahí con el extra `dev` (pytest y Jupyter). Ya no existe `requirements-dev.txt`. Los extras opcionales `openai` y `anthropic` añaden los proveedores en la nube (ver [Cambiar de proveedor](#cambiar-de-proveedor)). En v2.1 entraron dos dependencias: `lightgbm>=4.5` (el ranker LambdaMART del notebook 12) y `bitsandbytes>=0.45` (los pesos NF4 de QLoRA en el notebook 07). Con el notebook 17 entró `scikit-learn>=1.5` (la regresión logística y las métricas de clasificación), y con el 21, `sentence-transformers>=5.0` (bi-encoders, rerankers cross-encoder y entrenamiento contrastivo). Estas son las versiones con las que se verificó el curso:

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
3. Instala `requirements.txt`, que instala el proyecto con su extra `dev` desde `pyproject.toml` (NumPy, matplotlib, python-dotenv, LangChain, LangGraph, LangSmith, RAGAS, FAISS, rank-bm25, LightGBM, scikit-learn, sentence-transformers, transformers, peft, trl, datasets, accelerate, bitsandbytes, optimum-quanto, mcp, langchain-mcp-adapters) más pytest, JupyterLab, ipykernel, nbclient y nbformat.
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

[Ollama](https://ollama.com) sirve el modelo de chat y el de embeddings de los notebooks 10–16 en tu propia máquina, en la GPU si la hay.

1. **Instálalo.** Windows y macOS: descarga el instalador de [ollama.com/download](https://ollama.com/download). Linux: `curl -fsSL https://ollama.com/install.sh | sh`.
2. **Arráncalo.** La aplicación de escritorio (Windows/macOS) y el servicio de Linux arrancan el servidor solos. Si no, ejecuta `ollama serve` en otra terminal. Escucha en `http://localhost:11434`.
3. **Descarga los modelos del curso** (unos 2,2 GB en total):

   ```bash
   ollama pull qwen2.5:3b          # modelo de chat, 3.1B parámetros, Q4_K_M, admite tool calling
   ollama pull nomic-embed-text    # modelo de embeddings para recuperación y memoria
   ```

   **Opcional: un juez más fuerte.** El notebook 10 usa `qwen2.5:7b` para juzgar con RAGAS las respuestas del 3B, el 14 lo usa para el agente que responde preguntas escribiendo código pandas (el 3B casi nunca lo consigue), y `scripts/generate_item_keywords.py` lo usa para escribir las palabras clave de las películas:

   ```bash
   ollama pull qwen2.5:7b          # 4,7 GB; juez de RAGAS (notebook 10) y agente de código (notebook 14)
   ```

   **El notebook 20** necesita además un modelo de razonamiento pequeño, y usa el juez de 7B como verificador y como juez por pares:

   ```bash
   ollama pull deepseek-r1:1.5b    # 1,1 GB; DeepSeek-R1-Distill-Qwen-1.5B (notebook 20)
   ```

   Sin el juez de 7B todo funciona: `config.judge_model()` cae al modelo de chat y el 3B se juzga a sí mismo. El notebook 10 explica por qué eso no es buena idea: con el juez de 3B aparecían diferencias entre configuraciones que el de 7B no confirma. Con el 7B, la puntuación de RAGAS tarda unos 17 minutos en una GPU de 6 GB, frente a unos 7 con el 3B. Para usar otro juez, define `JUDGE_MODEL` en `.env`.

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
| `JUDGE_MODEL` | `qwen2.5:7b` si está descargado en Ollama; si no, el modelo de chat | El modelo que hace de juez (LLM-as-judge) en las evaluaciones: RAGAS en el notebook 10 y `config.get_judge_llm()`. Conviene que sea más fuerte que el modelo evaluado. |
| `EMBEDDING_MODEL` | `nomic-embed-text` (Ollama), `text-embedding-3-small` (OpenAI) | El modelo de embeddings que devuelve `get_embeddings()`. Anthropic no tiene API de embeddings, así que con `anthropic` los embeddings siguen saliendo de Ollama. |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Dónde está el servidor de Ollama. |
| `LANGSMITH_TRACING` | `true` en `.env.example` | Enviar trazas a LangSmith. Solo tiene efecto si además hay `LANGSMITH_API_KEY`. |
| `LANGSMITH_API_KEY` | vacío | Tu clave de API de LangSmith. |
| `LANGSMITH_PROJECT` | `llm-edu-suite` | El proyecto de LangSmith al que van las trazas. `setup(project=...)` lo cambia por notebook, para que las trazas de cada notebook vayan a su propio proyecto. |
| `LANGSMITH_ENDPOINT` | (comentada) | Solo para cuentas de la región UE: `https://eu.api.smith.langchain.com`. |
| `OPENAI_API_KEY` | (comentada) | Solo hace falta con `LLM_PROVIDER=openai`. |

Con `LANGSMITH_API_KEY` vacía, el trazado queda desactivado y todo lo demás funciona igual. El notebook 15 (observabilidad) es el que más partido saca de LangSmith.

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

## Qué ejecutar después

Tiempos medidos en la máquina de verificación (RTX 3060 Laptop GPU, Ollama en local).

| Qué | Comando | Tiempo |
|---|---|---|
| Tests rápidos (sin Ollama) | `pytest -q -m "not ollama"` | ~42 s (90 tests; con modelos falsos, y además lanzan código de ataque contra el sandbox del notebook 14 y comprueban que `mcp_server.py --require-token` rechaza las peticiones sin el bearer token) |
| Tests de integración con Ollama | `pytest -q -m ollama` | ~40 s (4 tests; se saltan solos si Ollama no está en marcha) |
| Todos los tests | `pytest -q` | la suma de los dos anteriores |
| Abrir los notebooks | `jupyter lab notebooks/` y elegir el kernel **Python (llm-edu-suite)** | — |
| Ejecutar todos los notebooks sin abrirlos | `python tools/run_notebooks.py` | largo; ver la [guía de notebooks](guia_notebooks.md) |
| Ejecutar algunos | `python tools/run_notebooks.py 05 14` (por número) o `python tools/run_notebooks.py 05_pretraining.ipynb` | — |
| Regenerar los notebooks desde sus generadores | `python tools/build_notebooks.py` (todos) o `python tools/build_notebooks.py 05 14`; con `--keep-outputs` conserva las salidas de las celdas de código que no cambiaron | segundos |
| Preentrenar un LM pequeño | `python scripts/train.py --steps 300 --out runs/smoke` | ver [scripts](scripts_cli.md#trainpy) |
| SFT con LoRA sobre SmolLM2 | `python scripts/finetune.py sft` | ~1,5 min en GPU |
| RAG sobre la documentación del curso | `python scripts/rag_demo.py --ask "What is GQA?"` | ~15 s |
| Servidor MCP | `python scripts/mcp_server.py` | — |

Los `.ipynb` se **generan** a partir de `tools/notebooks/nbNN_*.py` (y `scripts/mini_llm.py` a partir del generador del notebook 00). Para cambiar un notebook de forma permanente, edita su generador y regenera; lo que edites directamente en el `.ipynb` se pierde en la siguiente generación. Generar borra las salidas, así que ejecuta después `tools/run_notebooks.py`; si solo cambiaste texto (markdown o comentarios que no tocan el código), `python tools/build_notebooks.py --keep-outputs 12` mantiene las salidas ya ejecutadas de cada celda de código cuyo fuente no cambió y te ahorra volver a ejecutar el notebook ([detalles](scripts_cli.md#toolsbuild_notebookspy---keep-outputs)). Con `make` disponible (Linux/macOS/Git Bash): `make test`, `make smoke`, `make notebooks`, `make run-notebooks`, `make mini-llm`, `make mcp-server` (ver [scripts de línea de comandos](scripts_cli.md#herramientas-y-objetivos-del-makefile)).

## Datos que se descargan al ejecutar

- **SmolLM2** (notebooks 07–09 y `finetune.py`): desde el Hugging Face Hub a la caché de Hugging Face, la primera vez.
- **ModernBERT-base, Flan-T5-small y Banking77** (notebook 17): `answerdotai/ModernBERT-base` (~600 MB), `google/flan-t5-small` (~300 MB) y el dataset `PolyAI/banking77` (CC BY 4.0) se descargan del Hugging Face Hub a su caché la primera vez que se ejecuta el notebook.
- **FineWeb, FineWeb-Edu y el clasificador de FineWeb-Edu** (notebook 18): el notebook lee en streaming (`datasets.load_dataset(..., streaming=True)`) los primeros 12.000 documentos de `HuggingFaceFW/fineweb` y 1.000 de `HuggingFaceFW/fineweb-edu` (configuración `sample-10BT`), unos 100 MB de descarga, y los guarda una sola vez en `data/cache/fineweb_12k.jsonl` y `data/cache/fineweb_edu_1k.jsonl` (unos 45 MB). Las siguientes ejecuciones leen esos ficheros sin conexión. FineWeb tiene licencia ODC-By y el repositorio no lo redistribuye. El clasificador `HuggingFaceFW/fineweb-edu-classifier` (~440 MB) y el tokenizador de `HuggingFaceTB/SmolLM2-135M` van a la caché de Hugging Face la primera vez.
- **GSM8K** (notebook 20): el split de test de `openai/gsm8k` (configuración `main`, licencia MIT) se descarga del Hugging Face Hub a su caché la primera vez que se ejecuta el notebook, que usa 80 de sus problemas.
- **SciFact, NFCorpus y los modelos de recuperación** (notebook 21): las copias de BEIR de SciFact y NFCorpus en el Hugging Face Hub (`mteb/scifact` y `mteb/nfcorpus`, configuraciones `corpus`, `queries` y `default` para las etiquetas de relevancia) y los modelos `sentence-transformers/all-MiniLM-L6-v2`, `BAAI/bge-small-en-v1.5`, `cross-encoder/ms-marco-MiniLM-L6-v2` y `BAAI/bge-reranker-base` se descargan a la caché de Hugging Face la primera vez que se ejecuta el notebook. `nomic-embed-text` viene de Ollama, como en los notebooks 10–16. Los modelos ajustados se entrenan en memoria y no se guardan.
- **MovieLens-1M** (notebook 12 y `scripts/generate_item_keywords.py`): `llm_edu_suite.recsys.download_movielens_1m()` baja el zip oficial de GroupLens (6 MB) y lo descomprime en `data/cache/ml-1m/` una sola vez. `data/cache/` está en `.gitignore`: la licencia de GroupLens no permite redistribuir el dataset, así que el repositorio no lo incluye y cada uno lo descarga de la fuente original. Si estás sin conexión, copia `ml-1m.zip` en `data/cache/` y la función lo usará.
- Lo que sí está en el repositorio, porque es trabajo propio: `data/recsys/ml1m_llm_keywords.json` (las palabras clave que escribió `qwen2.5:7b` para cada película; regenerarlas lleva una hora, ver [`generate_item_keywords.py`](scripts_cli.md#generate_item_keywordspy)) y `data/recsys/conversational_eval.json` (20 peticiones con restricciones verificables).

## Notas sobre la GPU

- Nada necesita GPU salvo el notebook 19, pero el entrenamiento es mucho más rápido con ella: el mini LLM del notebook 00 tarda unos **22 s en una RTX 3060 Laptop frente a unos 5 min en CPU**. `llm_edu_suite.utils.get_device()` elige `cuda` si existe (Apple MPS solo con `allow_mps=True`). En RTX 30xx o superior, el entrenamiento usa precisión mixta **bf16**.
- **Presupuesto de VRAM en una GPU de 6 GB.** Ollama mantiene `qwen2.5:3b` cargado en la GPU unos minutos después de cada llamada, y ocupa unos **2,3 GB** de VRAM. Los notebooks de entrenamiento limitan la memoria de su propio PyTorch con `torch.cuda.set_per_process_memory_fraction` (variable de entorno `VRAM_BUDGET_GB`; p. ej. `$env:VRAM_BUDGET_GB="3"` o `export VRAM_BUDGET_GB=3` para darles más). El valor por defecto es **2 GB en el notebook 07**, que entrena once modelos seguidos y cuyo modelo de 4 bits de QLoRA deja unos 0,1 GB ocupados después, y **1,5 GB en el 08 y el 09**.
- **No ejecutes a la vez los notebooks de entrenamiento (07–09, y el 21, que entrena y llama a Ollama) y los de Ollama (10–16) en una GPU de 6 GB.** Cuando compiten por la VRAM, Ollama puede quedarse colgado en lugar de fallar. Ejecútalos uno detrás de otro y, si necesitas recuperar la memoria, descarga los modelos con `ollama stop qwen2.5:3b` (`ollama ps` muestra qué hay cargado).
- Si Ollama no consigue meter un modelo en la GPU, lo ejecuta (en parte) en CPU: sigue funcionando, solo que más lento.
- **El notebook 19 en Windows.** Las wheels de PyTorch para Windows no traen FlashAttention-2 (SDPA usa entonces los kernels memory-efficient o cuDNN) ni Triton, así que el backend por defecto de `torch.compile`, `inductor`, falla con `TritonMissing` y el notebook usa `backend="cudagraphs"`. Para FlashAttention-2, `torch.compile` con `inductor` o el paquete `flash-attn`, usa Linux o WSL2. El DDP del notebook corre en dos procesos de CPU con `gloo`, lanzados con `torch.multiprocessing.spawn` y conectados por un fichero compartido (`init_method="file://..."`, un `FileStore`), porque el rendezvous TCP de `torchrun` necesita libuv, que la build de Windows no incluye; NCCL, el backend de GPU, solo existe en Linux.
- **Lento en vez de sin memoria.** En Windows, cuando un proceso supera la VRAM, el driver puede desbordar a la RAM del sistema ("memoria de GPU compartida") en lugar de dar un error: en el notebook 19, el backend *math* de SDPA declaró 8,1 GB en una tarjeta de 6 GB y tardó 978 ms. Si algo se vuelve lento de repente, mira la memoria de GPU compartida en el Administrador de tareas.

## Problemas frecuentes

| Síntoma | Causa | Solución |
|---|---|---|
| `!! Ollama is not running`, `Connection refused` en el puerto 11434, o tests de Ollama saltados | El servidor de Ollama no está en marcha | Abre la aplicación de Ollama o ejecuta `ollama serve`; compruébalo con `curl http://localhost:11434/api/tags` |
| `model "qwen2.5:3b" not found, try pulling it first` (o lo mismo con `nomic-embed-text`) | No se descargaron los modelos | `ollama pull qwen2.5:3b` y `ollama pull nomic-embed-text`; compruébalo con `ollama list` |
| Un notebook o script se queda colgado en la primera llamada al LLM | Ollama se ha quedado sin VRAM porque un notebook de entrenamiento ocupa la GPU | Para el kernel de entrenamiento, `ollama stop qwen2.5:3b`, y ejecuta los notebooks uno detrás de otro (ver [notas sobre la GPU](#notas-sobre-la-gpu)) |
| `LangChainDeprecationWarning` / aviso de que `langchain-community` se va a retirar (*sunset*) | `FAISS` y `BM25Retriever` siguen viviendo en `langchain-community` | Inofensivo; los notebooks y `rag_demo.py` lo filtran. No hay que hacer nada |
| `ModuleNotFoundError: No module named 'langchain_community.chat_models.vertexai'` al importar `ragas` | ragas 0.4.3 importa un módulo que langchain-community 0.4 eliminó | Ya está resuelto: el notebook 10 y `rag_demo.py` registran un módulo vacío de sustitución antes de importar ragas. Copia ese *shim* (`_ragas_compat_shim` en `scripts/rag_demo.py`) si importas ragas en tu propio código |
| `UserWarning: huggingface_hub cache-system uses symlinks...` en Windows | Windows no permite enlaces simbólicos sin el modo de desarrollador | Inofensivo (la caché ocupa algo más de disco). Activa el modo de desarrollador o define `HF_HUB_DISABLE_SYMLINKS_WARNING=1` para ocultarlo |
| El cliente MCP por stdio falla dentro de Jupyter con un error sobre `fileno` / stderr | El `sys.stderr` de Jupyter no tiene un descriptor de fichero del sistema, y el SDK se lo pasa al subproceso del servidor | Ya está resuelto en el notebook 16, que manda el stderr del servidor a un fichero de log. En scripts normales no hace falta |
| `ModuleNotFoundError: llm_edu_suite` en un notebook | Jupyter se abrió fuera del repo, o con otro kernel | Abre con `jupyter lab notebooks/`; cada notebook añade la raíz del repo al `sys.path` |
| El notebook usa otro Python | Kernel equivocado | *Kernel → Change kernel → Python (llm-edu-suite)* |
| `RuntimeWarning: Proactor event loop...` en Windows | Aviso inofensivo de `pyzmq` | Se puede ignorar; `tools/run_notebooks.py` ya lo silencia |
| La celda de FlexAttention se salta | PyTorch < 2.5 o sin soporte en CPU | Es opcional; el resto del notebook 03 funciona igual |
| `ImportError: numpy.core.multiarray failed to import` | matplotlib compilado para NumPy 1.x con NumPy 2.x instalado | Usa el `.venv` del proyecto (o `pip install -U matplotlib`) |
| Entrenamiento lento | Todo corre en CPU | Instala la build CUDA de PyTorch, o baja `--steps` / `--d-model` |
| El notebook 19 se para con `this notebook measures GPU memory and kernels; it needs a CUDA GPU` | No hay GPU NVIDIA, o PyTorch es la build de CPU | Instala la build CUDA de PyTorch; sin GPU el notebook no tiene nada que medir |
| `SDPA: FlashAttention-2: not available in this PyTorch build` o `inductor backend: unavailable here (TritonMissing)` en el notebook 19 | Las wheels de PyTorch para Windows no incluyen FlashAttention-2 ni Triton | Inofensivo: el notebook compara los kernels disponibles y usa CUDA graphs. Para probar los que faltan, ejecútalo en Linux o WSL2 |
| `torchrun` falla en Windows con un error sobre libuv | El rendezvous TCP de `torchrun` necesita libuv, que la build de Windows no trae | Lanza los procesos con `torch.multiprocessing.spawn` e `init_method="file://..."`, como el notebook 19, o usa Linux o WSL2 |
