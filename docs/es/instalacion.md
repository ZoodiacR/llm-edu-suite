🇬🇧 [English version](../en/installation.md)

# Instalación y entorno de trabajo

[← Volver al índice](README.md)

Esta guía deja el proyecto listo para ejecutar **todo**: la suite de tests, los scripts de línea de comandos y los 17 notebooks. Todo funciona en **CPU**, sin descargas de modelos y sin claves de API, y si tienes una **GPU NVIDIA** se usa automáticamente (mucho más rápido).

## Requisitos

| Componente | Versión mínima | Verificado con |
|---|---|---|
| Python | 3.10 | 3.11.4 (Windows 11) |
| PyTorch | 2.3 (2.5+ para la celda opcional de FlexAttention) | 2.15.0+cpu |
| NumPy | 1.24 | 2.4.6 |
| matplotlib | 3.7 | 3.11.2 |
| Espacio en disco | ~1 GB (casi todo es PyTorch) | |
| RAM | 4 GB | |

> **Por qué un entorno virtual propio.** Si instalas en tu Python global puedes heredar combinaciones rotas.
> Un caso real encontrado al probar este proyecto: un `matplotlib 3.7` compilado contra NumPy 1.x junto a un
> `numpy 2.4` falla al importar con `ImportError: numpy.core.multiarray failed to import`. Un `.venv`
> aislado evita este tipo de conflictos.

## Opción A — Script automático (recomendado)

**Windows (PowerShell):**

```powershell
cd llm-edu-suite
powershell -ExecutionPolicy Bypass -File setup_env.ps1
```

**Linux / macOS:**

```bash
cd llm-edu-suite
bash setup_env.sh
```

El script hace, en orden:

1. Crea `.venv/` si no existe.
2. Instala PyTorch: la variante **CUDA** si detecta una GPU NVIDIA (`nvidia-smi`), o la variante **CPU** (wheel pequeño y universal) si no.
3. Instala `requirements-dev.txt` (numpy, matplotlib, pytest, jupyterlab, ipykernel, nbclient, requests).
4. Registra el entorno como kernel de Jupyter con el nombre **"Python (llm_edu_suite)"**.
5. Ejecuta `pytest -q` para confirmar que todo funciona (deberían pasar todos; en el momento de escribir esto pytest recoge 50 tests).

## Opción B — Paso a paso

```powershell
# Windows
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements-dev.txt
python -m ipykernel install --user --name llm_edu_suite --display-name "Python (llm_edu_suite)"
pytest -q
```

```bash
# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cpu   # en macOS: pip install torch
pip install -r requirements-dev.txt
python -m ipykernel install --user --name llm_edu_suite --display-name "Python (llm_edu_suite)"
pytest -q
```

> Si PowerShell bloquea `Activate.ps1`, ejecuta una vez
> `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, o usa directamente
> `.\.venv\Scripts\python.exe` sin activar el entorno.

### ¿Tengo GPU NVIDIA?

Nada en el proyecto la necesita, pero acelera mucho el entrenamiento: el mini LLM del notebook 00 tarda unos **22 s en una RTX 3060 Laptop frente a unos 5 min en CPU**. Los scripts de setup la detectan e instalan PyTorch con CUDA automáticamente; los notebooks la usan solos (`llm_edu_suite.utils.get_device()` elige `cuda` si existe y si no `cpu`; Apple MPS solo con `allow_mps=True`). En GPU, el entrenamiento usa precisión mixta **bf16** (RTX 30xx o superior).

Para instalarla a mano o cambiar un `.venv` que ya tenía la versión CPU:

```bash
nvidia-smi                                   # comprueba el driver y la versión de CUDA que soporta
pip uninstall -y torch
pip install torch --index-url https://download.pytorch.org/whl/cu128     # o cu130 con drivers recientes
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## Qué ejecutar después

| Qué | Comando | Tiempo aprox. (CPU) |
|---|---|---|
| Tests | `pytest -q` | ~10 s |
| Abrir notebooks | `jupyter lab notebooks/` y elegir el kernel **Python (llm_edu_suite)** | — |
| Ejecutar todos los notebooks sin abrirlos | `python tools/run_notebooks.py` | ver [guía de notebooks](guia_notebooks.md) |
| Ejecutar uno solo | `python tools/run_notebooks.py 05_pretraining.ipynb` (o por número: `python tools/run_notebooks.py 05 14`) | — |
| Pre-entrenar un LM pequeño | `python scripts/train.py --steps 300 --out runs/smoke` | ~2,5 min |
| Generar tokens con él | `python scripts/sample.py --run runs/smoke --prompt "" --max-new-tokens 50` | segundos |
| Post-training SFT→DPO→GRPO | `python scripts/finetune.py --lora` | unos minutos (~5 min según el docstring del script) |
| RAG sobre documentos de ejemplo | `python scripts/rag_demo.py --docs data/sample_docs --eval data/sample_eval.json` | segundos |
| Regenerar los notebooks desde código | `python tools/build_notebooks_part1.py && python tools/build_notebooks_part2.py && python tools/build_notebooks_part3.py` | segundos |

Con `make` disponible (Linux/macOS/Git Bash): `make test`, `make smoke`, `make notebooks`, `make run-notebooks`, `make mini-llm`, `make mcp-server` (ver [scripts de línea de comandos](scripts_cli.md#herramientas-y-objetivos-del-makefile)).

> **Importante sobre los notebooks:** los `.ipynb` se **generan** a partir de `tools/build_notebooks_part*.py`.
> Si quieres cambiar un notebook de forma permanente, edita el generador y vuelve a generarlos;
> si editas el `.ipynb` directamente, perderás los cambios la próxima vez que se regeneren.

## Usar un LLM real en lugar del `MockLLM`

Los notebooks 10, 11 y 13 y `scripts/rag_demo.py` usan por defecto `MockLLM`, un modelo **determinista
basado en reglas** que permite ejecutar todo offline. Para conectar un modelo real basta con un
servidor compatible con la API de OpenAI:

```python
from llm_edu_suite.llm_client import OpenAICompatibleClient

# vLLM local:   vllm serve Qwen/Qwen2.5-7B-Instruct --enable-auto-tool-choice --tool-call-parser hermes
llm = OpenAICompatibleClient("Qwen/Qwen2.5-7B-Instruct", base_url="http://localhost:8000/v1")

# Ollama local: ollama serve  (y luego: ollama pull qwen2.5:7b)
llm = OpenAICompatibleClient("qwen2.5:7b", base_url="http://localhost:11434/v1")

# API en la nube (OpenAI, OpenRouter, ...): la clave se lee de la variable OPENAI_API_KEY
llm = OpenAICompatibleClient("gpt-4o-mini", base_url="https://api.openai.com/v1")
```

```bash
python scripts/rag_demo.py --docs data/sample_docs --ask "What is the refund policy?" \
    --model qwen2.5:7b --base-url http://localhost:11434/v1
```

> Nunca escribas claves de API en el código ni en los notebooks: usa variables de entorno
> (`OPENAI_API_KEY`) o un archivo `.env`, que ya está en `.gitignore`.

## Problemas frecuentes

| Síntoma | Causa | Solución |
|---|---|---|
| `ImportError: numpy.core.multiarray failed to import` | matplotlib compilado para NumPy 1.x con NumPy 2.x instalado | usa el `.venv` del proyecto (o `pip install -U matplotlib`) |
| `ModuleNotFoundError: llm_edu_suite` en un notebook | Jupyter se abrió fuera de `notebooks/` | abre con `jupyter lab notebooks/`; cada notebook añade la raíz del repo (dos niveles por encima de `notebooks/<tema>/`) al `sys.path` |
| El notebook usa otro Python | kernel equivocado | menú *Kernel → Change kernel → Python (llm_edu_suite)* |
| `RuntimeWarning: Proactor event loop...` en Windows | aviso inofensivo de `pyzmq` | se puede ignorar; `tools/run_notebooks.py` ya lo silencia |
| La celda de FlexAttention se salta | PyTorch < 2.5 o sin soporte en CPU | es opcional; el resto del notebook 03 funciona igual |
| Entrenamiento lento | todo corre en CPU | baja `--steps`/`--d-model`, o usa GPU |
