🇬🇧 [English version](../en/cli_scripts.md)

# Scripts de línea de comandos

> Guía de los seis scripts de `scripts/` (más `tools/build_notebooks.py`, `tools/run_notebooks.py` y los objetivos del `Makefile`): todos sus argumentos reales (extraídos de `argparse`), qué hace cada script internamente, ejemplos de uso y resultados medidos.
>
> Documentos relacionados: [README](README.md) · [Instalación](instalacion.md) · [Referencia de módulos](referencia_modulos.md) · [Guía de notebooks](guia_notebooks.md) · [Caso de estudio](caso_de_estudio.md) · [Glosario](glosario.md)

| Script | Para qué sirve | Construido sobre | Necesita |
|---|---|---|---|
| [`train.py`](#trainpy) | Preentrenar un `TransformerLM` sobre un `.txt` o un corpus sintético | `llm_edu_suite` (`data`, `tokenizer`, `model`, `training`, `evaluation`, `utils`) | PyTorch |
| [`sample.py`](#samplepy) | Generar desde un checkpoint (muestreo normal o especulativo) | `llm_edu_suite` (`model`, `tokenizer`, `inference`, `utils`) | PyTorch |
| [`mini_llm.py`](#mini_llmpy) | Un mini GPT completo en un fichero: BPE → modelo → entrenamiento → evaluación → muestreo → checkpoint | solo PyTorch (autocontenido) | PyTorch |
| [`finetune.py`](#finetunepy) | Post-training de SmolLM2-135M-Instruct: subcomandos `sft`, `dpo` o `grpo`, cada uno evaluado antes y después | Hugging Face `transformers` + `peft` + `trl` | PyTorch, descarga de ~260 MB la primera vez; muy recomendable una GPU |
| [`rag_demo.py`](#rag_demopy) | RAG híbrido con citas sobre una carpeta de documentos, con métricas de recuperación y evaluación RAGAS opcional | LangChain (FAISS + BM25 + `EnsembleRetriever`), RAGAS | Ollama (`qwen2.5:3b`, `nomic-embed-text`) |
| [`mcp_server.py`](#mcp_serverpy) | Un servidor MCP (`llm-edu-docs`) que expone `data/sample_docs` a Claude Code, Claude Desktop o cualquier cliente MCP | SDK oficial de MCP para Python (`FastMCP`) + `rank_bm25` | Nada más (ni PyTorch ni Ollama) |

---

## Antes de empezar

Todos los comandos se ejecutan desde la **raíz del repositorio** y con el intérprete del entorno virtual del proyecto (ver [instalación](instalacion.md)). Los scripts que importan `llm_edu_suite` añaden ellos mismos la raíz del repo a `sys.path`, así que no hace falta instalar el paquete con `pip install -e .`.

**Windows (PowerShell o cmd):**

```powershell
.venv\Scripts\python scripts\train.py --steps 300 --out runs\smoke
```

**Linux / macOS:**

```bash
.venv/bin/python scripts/train.py --steps 300 --out runs/smoke
```

En el resto del documento cada ejemplo aparece en las dos formas. Si un comando largo se parte en varias líneas, el carácter de continuación es `` ` `` en PowerShell, `^` en cmd y `\` en bash.

Todos los scripts con argumentos aceptan `-h` / `--help` (`mini_llm.py` no tiene: se configura con su diccionario `CFG`). `train.py`, `finetune.py` y `rag_demo.py` muestran además en la ayuda su docstring con ejemplos.

Salvo que se indique otra cosa, los números de este documento se midieron en una RTX 3060 Laptop GPU (6 GB) con Ollama en local; los de `train.py` / `sample.py` son de CPU.

---

## `train.py`

Preentrena un modelo de lenguaje *decoder-only* con predicción del siguiente token. Si no se pasa `--data`, usa un corpus sintético, lo que permite un *smoke test* sin descargas.

### Argumentos

**Datos y salida**

| Argumento | Tipo | Defecto | Descripción |
|---|---|---|---|
| `--data` | ruta | `None` | Fichero `.txt` de entrenamiento. Si se omite, usa un corpus sintético de 300 000 tokens. |
| `--out` | ruta | `runs/default` | Carpeta de salida (se crea si no existe). |
| `--vocab-size` | int | `1024` | Con `--data`, tamaño del vocabulario BPE a entrenar. Sin `--data`, número de ids del corpus sintético. |
| `--tok-train-chars` | int | `500000` | Número de caracteres iniciales del texto que se usan para *entrenar* el tokenizador (el texto completo se codifica igualmente). |

**Modelo** (se traducen a campos de [`ModelConfig`](referencia_modulos.md#4-model))

| Argumento | Tipo | Defecto | Descripción |
|---|---|---|---|
| `--d-model` | int | `256` | Dimensión del residual. |
| `--n-layers` | int | `4` | Número de bloques. |
| `--n-heads` | int | `8` | Cabezas de query. |
| `--n-kv-heads` | int | `2` | Cabezas K/V (GQA). Igual a `--n-heads` da MHA y `1` da MQA. |
| `--attn` | opción | `gqa` | `gqa`, `mla`, `swa`, `longformer`, `blocksparse` o `linear`. |
| `--hybrid-pattern` | str | `None` | Patrón por capas, p. ej. `"LLLG"`: `L` = la atención de `--attn`, `G` = GQA global. |
| `--window` | int | `64` | Ventana local para `swa`, `longformer` y `blocksparse`. Solo tiene efecto si es menor que `--block-size`. |
| `--qk-norm` | flag | off | RMSNorm sobre q y k por cabeza. |
| `--moe` | flag | off | FFN Mixture-of-Experts en todas las capas (en lugar de SwiGLU denso). |
| `--n-experts` | int | `4` | Número de expertos. |
| `--top-k` | int | `2` | Expertos activos por token. |
| `--moe-routing` | opción | `softmax` | `softmax` (pérdida auxiliar de balanceo) o `auxfree` (sesgo de balanceo, DeepSeek-V3). |

**Entrenamiento** (campos de [`TrainConfig`](referencia_modulos.md#6-training))

| Argumento | Tipo | Defecto | Descripción |
|---|---|---|---|
| `--steps` | int | `1000` | Pasos de optimización. |
| `--batch-size` | int | `32` | Secuencias por micro-batch. |
| `--block-size` | int | `128` | Longitud de contexto de entrenamiento. **También fija `max_seq_len` del modelo.** |
| `--lr` | float | `3e-3` | LR máxima (`max_lr`). |
| `--warmup` | int | `100` | Pasos de calentamiento lineal. |
| `--schedule` | opción | `cosine` | `cosine` o `wsd` (Warmup-Stable-Decay). |
| `--grad-accum` | int | `1` | Micro-batches acumulados por paso. |
| `--eval-every` | int | `100` | Frecuencia de validación y checkpoint. |
| `--compile` | flag | off | Envuelve el modelo en `torch.compile`. |
| `--seed` | int | `0` | Semilla global. |
| `--device` | str | auto | `cpu`, `cuda`, `cuda:1`, `mps`… Si se omite, elige `cuda` si está disponible y si no `cpu`. |

Algunos campos de `ModelConfig` no se exponen por CLI y toman su valor por defecto: `dropout=0`, `sandwich_norm`, los tamaños de MLA (`kv_lora_rank=32`, `qk_nope_dim=16`, `qk_rope_dim=8`, `v_dim=16`), `block=16`, `top_blocks=2`, `global_idx=(0,)`, `n_shared_experts=0`, `moe_every=1`, `aux_loss_coef=0.01`, `rope_scaling`. Lo mismo ocurre con estos campos de `TrainConfig`: `min_lr_ratio=0.1`, `weight_decay=0.1`, `grad_clip=1.0` y `eval_batches=10`. Para variarlos hay que usar la API de Python (ver los notebooks).

### Qué hace, etapa por etapa

1. **Preparación.** `seed_everything(--seed)`, elige el dispositivo y crea `--out`.
2. **Datos.**
   - *Con `--data`*: lee el fichero y entrena un `ByteBPETokenizer` sobre los primeros `--tok-train-chars` caracteres (con `verbose=True`, imprime una línea cada 100 fusiones). Guarda `<out>/tokenizer.json`, codifica **todo** el texto y muestra la relación caracteres/token. El vocabulario efectivo es el del tokenizador. El BPE está escrito en Python puro y con un texto grande este paso puede tardar.
   - *Sin `--data`*: `synthetic_structured_corpus(n_tokens=300_000, vocab=--vocab-size)`.
3. **Split** contiguo 90 % train / 10 % val (`train_val_split`).
4. **Modelo.** Construye `ModelConfig` con los argumentos y `max_seq_len = --block-size`, e imprime los parámetros totales y activos (distintos solo con `--moe`).
5. **Bucle de entrenamiento** (`train_lm`). En cada paso fija la LR según el schedule, hace `--grad-accum` micro-batches con `get_batch`, recorta el gradiente a norma 1.0 y aplica AdamW (β = (0.9, 0.95), weight decay solo en matrices). Cada `--eval-every` pasos, y en el último, estima la pérdida de validación con 10 batches, imprime una línea de log y **guarda un checkpoint** en `<out>/model.pt`. La precisión mixta bf16 se activa sola en CUDA; en CPU se entrena en fp32.
6. **Cierre.** Guarda el modelo final (`model.save`), calcula la perplejidad sobre 20 batches aleatorios de validación (`evaluation.perplexity`) y escribe `<out>/train_log.json` con la configuración del modelo, la de entrenamiento, la curva de validación, la perplejidad final y el tiempo de pared.

**Formato de la línea de log** (valores ilustrativos):

```
step   100 | train 1.012 | val 1.004 | ppl     2.7 | lr 3.00e-03 | grad 0.41 | 47.3s
```

`grad` es la norma del gradiente **antes** del clipping. Si crece de golpe, suele ser el primer síntoma de inestabilidad.

**Ficheros de salida:**

| Fichero | Contenido |
|---|---|
| `<out>/model.pt` | `{"config": dict, "state_dict": ...}`, cargable con `TransformerLM.load`. |
| `<out>/tokenizer.json` | Solo si se usó `--data`: fusiones BPE + tokens especiales. |
| `<out>/train_log.json` | `config`, `train`, `val_losses` (`[[step, loss], ...]`), `final_val_ppl`, `wall_time_s`. |

### Ejemplos

**Smoke test** (corpus sintético, modelo por defecto, CPU):

```powershell
.venv\Scripts\python scripts\train.py --steps 300 --out runs\smoke
```
```bash
.venv/bin/python scripts/train.py --steps 300 --out runs/smoke
```

**Resultado medido en CPU:** **3.03M parámetros**, perplejidad de validación final **2.65** en unos **140 s**. La pérdida de validación pasa de 6.86 (paso 0, ≈ ln 1024 = 6.93, es decir, azar) a 1.00 (paso 100), 0.978 (paso 200) y 0.974 (paso 299).

¿Por qué una perplejidad tan baja? El corpus sintético es un léxico de 60 "palabras" de 2 a 4 tokens separadas por el token 0. Solo el primer token de cada palabra es realmente incierto (≈ 1 entre 60); el resto de la palabra y el separador están determinados. Un modelo que memoriza el léxico tiene poca incertidumbre media por token. La perplejidad baja demuestra que el *pipeline* aprende, no que el modelo "sepa idioma".

¿De dónde salen los 3.03M parámetros?

| Componente | Cálculo | Parámetros |
|---|---|---|
| Embedding (atado con `lm_head`) | 1024 × 256 | 262 144 |
| Atención GQA por capa | wq 256·256 + wk 256·64 + wv 256·64 + wo 256·256 | 163 840 |
| SwiGLU por capa (`d_ff` = 688, regla de Llama) | 3 · 256 · 688 | 528 384 |
| RMSNorm por capa | 2 · 256 | 512 |
| × 4 capas + norma final | | ≈ 2.77M + 256 |
| **Total** | | **≈ 3.03M** |

Una variante más rápida es la del `Makefile` (`make smoke`):

```powershell
.venv\Scripts\python scripts\train.py --steps 100 --d-model 64 --n-layers 2 --out runs\smoke --eval-every 50
```
```bash
.venv/bin/python scripts/train.py --steps 100 --d-model 64 --n-layers 2 --out runs/smoke --eval-every 50
```

**Texto real con GQA + MoE en GPU.** `data/input.txt` no viene en el repo: aporta tu propio `.txt`, por ejemplo *tiny shakespeare*.

```powershell
.venv\Scripts\python scripts\train.py --data data\input.txt --vocab-size 2048 --d-model 384 --n-layers 6 --n-heads 6 --n-kv-heads 2 --moe --n-experts 4 --steps 3000 --batch-size 32 --block-size 256 --out runs\shakespeare
```
```bash
.venv/bin/python scripts/train.py --data data/input.txt --vocab-size 2048 --d-model 384 --n-layers 6 \
    --n-heads 6 --n-kv-heads 2 --moe --n-experts 4 --steps 3000 --batch-size 32 --block-size 256 \
    --out runs/shakespeare
```

**Comparar arquitecturas** con todo lo demás igual. Las perplejidades se leen en cada `train_log.json`.

```bash
.venv/bin/python scripts/train.py --steps 300 --attn mla                         --out runs/mla
.venv/bin/python scripts/train.py --steps 300 --attn swa --window 32             --out runs/swa32
.venv/bin/python scripts/train.py --steps 300 --attn linear --hybrid-pattern LLLG --out runs/hybrid
.venv/bin/python scripts/train.py --steps 300 --moe --moe-routing auxfree        --out runs/moe_auxfree
.venv/bin/python scripts/train.py --steps 300 --schedule wsd                     --out runs/wsd
```

En Windows basta con cambiar `.venv/bin/python scripts/train.py` por `.venv\Scripts\python scripts\train.py`.

### Notas y trampas

- **`--window` debe ser menor que `--block-size` para importar.** Por defecto vale 64 frente a un `--block-size` de 128, así que `--attn swa` ya restringe la atención. Si subes `--window` hasta `--block-size` (o más), la ventana cubre todo el contexto y la atención equivale a una causal completa.
- `--block-size` limita también la generación posterior: `sample.py` no puede producir más de `max_seq_len` tokens en total.
- Solo `gqa` y `mla` tienen KV-cache. Con `swa`, `longformer`, `blocksparse`, `linear` o un patrón híbrido que las incluya, `sample.py` recalcula la secuencia completa en cada token y la generación es más lenta.
- `--compile` necesita un *toolchain* de compilación funcional (Triton en GPU; un compilador de C++ en CPU). En Windows y en CPU suele fallar o no compensar para modelos tan pequeños.
- El checkpoint intermedio y el final se escriben en el mismo `<out>/model.pt`, así que el último guardado sobrescribe a los anteriores.

---

## `sample.py`

Carga un checkpoint de `train.py` y genera tokens, con muestreo normal (temperatura, top-k, top-p, penalización de repetición) o con **decodificación especulativa** usando un modelo *draft*.

### Argumentos

| Argumento | Tipo | Defecto | Descripción |
|---|---|---|---|
| `--run` | ruta | **obligatorio** | Carpeta con `model.pt` (y `tokenizer.json` si se entrenó con texto). |
| `--prompt` | str | `"\n"` | Texto inicial. **Se ignora si la carpeta no tiene `tokenizer.json`** (ver abajo). |
| `--max-new-tokens` | int | `200` | Tokens nuevos como máximo (limitado también por `max_seq_len`). |
| `--temperature` | float | `0.8` | `<= 0` da decodificación greedy (argmax). |
| `--top-k` | int | `50` | Solo muestrea entre los k logits más altos. |
| `--top-p` | float | `None` | *Nucleus sampling*: conserva la masa de probabilidad `p`. |
| `--repetition-penalty` | float | `1.0` | Penalización CTRL a tokens ya generados (`1.0` = sin penalización). |
| `--speculative` | ruta | `None` | Carpeta de un modelo *draft* más pequeño, entrenado con el mismo tokenizador. Activa la decodificación especulativa. |
| `--device` | str | auto | Igual que en `train.py`. |

### Qué hace, etapa por etapa

1. **Carga** `TransformerLM.load(<run>/model.pt)`. Si existe `<run>/tokenizer.json`, carga también el `ByteBPETokenizer`.
2. **Prompt.** Con tokenizador, `ids = tok.encode(--prompt)`. Sin tokenizador, o si el prompt codificado queda vacío (`--prompt ""`), `ids = [0]`: el token 0, que en el corpus sintético es el separador de "palabras".
3. **Generación.**
   - *Normal*: `model.generate(...)` con los parámetros de muestreo. Si todas las capas lo permiten (`gqa`/`mla`), usa la KV-cache: un *prefill* del prompt y después un forward de un solo token por paso. Se detiene en `eos_id` (si hay tokenizador), al llegar a `--max-new-tokens` o al llenar `max_seq_len`.
   - *Especulativa*: carga `<speculative>/model.pt` como draft y llama a `speculative_decode(target, draft, ids, max_new_tokens, k=4, temperature=max(T, 1e-3))`. Solo usa la temperatura: `--top-k`, `--top-p` y `--repetition-penalty` se ignoran. Imprime la **tasa de aceptación** y los **tokens por pasada del target**.
4. **Salida.** Imprime `[N tokens in X s = Y tok/s]` y después el texto decodificado (con tokenizador) o la **lista de ids** (sin tokenizador). El texto incluye el prompt.

### ¿Por qué el smoke test devuelve números y no texto?

```powershell
.venv\Scripts\python scripts\sample.py --run runs\smoke --max-new-tokens 50
```
```bash
.venv/bin/python scripts/sample.py --run runs/smoke --max-new-tokens 50
```

**Resultado medido en CPU:** unos **200 tok/s**. La salida es una lista de enteros con la forma

```
[50 tokens in 0.25s = 200.0 tok/s]

[0, 734, 12, 0, 455, 91, 803, 0, ...]
```

(los ids concretos dependen del muestreo). Esto es lo esperado, no un error:

- `runs/smoke` se entrenó **sin `--data`**, sobre el corpus sintético. Ese corpus ya está formado por ids enteros, sin ningún texto detrás, así que `train.py` no entrena ni guarda un tokenizador.
- Sin `tokenizer.json`, `sample.py` no puede codificar el prompt (por eso lo ignora y empieza en `[0]`) ni decodificar la salida (por eso imprime los ids).
- Lo que sí se puede comprobar es que el modelo aprendió la estructura: aparecen grupos de 2 a 4 ids que se repiten (las "palabras" del léxico) separados por `0`.

Para obtener texto, entrena con `--data <fichero.txt>`: así se genera `tokenizer.json` y `sample.py` decodifica automáticamente.

### Ejemplos

**Texto con muestreo nucleus y penalización de repetición:**

```powershell
.venv\Scripts\python scripts\sample.py --run runs\shakespeare --prompt "ROMEO:" --max-new-tokens 200 --temperature 0.8 --top-k 50 --top-p 0.95 --repetition-penalty 1.1
```
```bash
.venv/bin/python scripts/sample.py --run runs/shakespeare --prompt "ROMEO:" --max-new-tokens 200 \
    --temperature 0.8 --top-k 50 --top-p 0.95 --repetition-penalty 1.1
```

**Greedy (determinista):**

```bash
.venv/bin/python scripts/sample.py --run runs/shakespeare --prompt "ROMEO:" --temperature 0
```

**Decodificación especulativa.** Primero entrena un draft pequeño con **los mismos datos y el mismo `--vocab-size`**. El BPE es determinista, así que el tokenizador resultante es idéntico.

```powershell
.venv\Scripts\python scripts\train.py --data data\input.txt --vocab-size 2048 --d-model 128 --n-layers 2 --n-heads 4 --block-size 256 --steps 1500 --out runs\small_draft
.venv\Scripts\python scripts\sample.py --run runs\shakespeare --prompt "ROMEO:" --speculative runs\small_draft
```
```bash
.venv/bin/python scripts/train.py --data data/input.txt --vocab-size 2048 --d-model 128 --n-layers 2 \
    --n-heads 4 --block-size 256 --steps 1500 --out runs/small_draft
.venv/bin/python scripts/sample.py --run runs/shakespeare --prompt "ROMEO:" --speculative runs/small_draft
```

La salida tiene la forma `[speculative] acceptance NN%, X.XX tokens/target pass` (sin medición de referencia en este repositorio). Cuanto más se parezcan draft y target, mayor es la aceptación y más tokens se obtienen por cada pasada del modelo grande.

### Notas y trampas

- **Longitud máxima:** el total (prompt + generado) no supera `max_seq_len`, que es el `--block-size` del entrenamiento. Con `runs/smoke` (128), el `--max-new-tokens 200` por defecto se queda en unos 128 tokens nuevos.
- **Modo especulativo:** `--top-k`, `--top-p` y `--repetition-penalty` **no se aplican**. Solo se usa la temperatura, y `k = 4` propuestas por ronda está fijado en el código. La implementación recalcula sin caché y es para batch 1, así que en CPU puede ser *más lenta* que la generación normal con KV-cache. Su objetivo es didáctico: comprobar la tasa de aceptación y que la distribución se preserva.
- **Prompt vacío:** `--prompt ""` (como en el ejemplo del README) funciona también en una carpeta con `tokenizer.json`: si el prompt codificado queda vacío, la generación arranca desde el id 0. Ten en cuenta que, con un tokenizador BPE real, el id 0 es el byte `\x00`, no un token especial; para texto suele ser mejor un prompt de al menos un carácter (el valor por defecto `"\n"` funciona).
- La medida de tok/s incluye el *prefill* y la primera llamada (calentamiento). Para medir TTFT y *throughput* con rigor, usa `evaluation.benchmark_generation` (notebook 06).

---

## `mini_llm.py`

Un **mini LLM completo en un solo fichero** que solo depende de PyTorch (no importa nada de `llm_edu_suite`): tokenizador BPE a nivel de bytes → GPT con RoPE, RMSNorm y SwiGLU → entrenamiento con AdamW y schedule coseno → evaluación frente a la pérdida irreducible del corpus → muestreo → guardado y recarga del checkpoint. Es el notebook 00 (`notebooks/00_start_here/00_mini_llm_end_to_end.ipynb`) exportado como script: el pipeline entero de un vistazo, antes de que los notebooks desmonten cada pieza y el paquete la escale.

> **Fichero generado.** `scripts/mini_llm.py` se genera a partir del notebook 00 con su generador, `tools/notebooks/nb00_mini_llm_end_to_end.py`. Para cambiarlo de forma permanente, **edita el generador**, no el script: el siguiente `make notebooks` lo sobrescribe.

### Configuración

No hay `argparse`. Todo se controla con el diccionario `CFG` del principio del fichero:

| Clave | Por defecto | Descripción |
|---|---|---|
| `text_path` | `None` | Un fichero `.txt` (p. ej. *tiny shakespeare*). Con `None`, o con una ruta que no existe, se usa el corpus sintético incorporado. |
| `vocab_size` | `512` | Vocabulario BPE objetivo (256 bytes + fusiones). El entrenamiento se detiene antes si ya no queda nada que fusionar. |
| `block_size` | `128` | Longitud de contexto en tokens. |
| `d_model`, `n_layers`, `n_heads` | `192`, `4`, `6` | Tamaño del modelo. |
| `dropout` | `0.0` | Dropout en la atención. |
| `batch_size`, `steps` | `32`, `800` | Secuencias por lote y pasos de optimización. |
| `max_lr`, `min_lr`, `warmup` | `3e-3`, `3e-4`, `80` | Calentamiento lineal y después caída en coseno hasta `min_lr`. |
| `weight_decay`, `grad_clip` | `0.1`, `1.0` | Weight decay de AdamW (solo matrices) y recorte de la norma del gradiente. |
| `eval_every`, `eval_batches` | `100`, `20` | Frecuencia de evaluación y número de lotes por estimación. |
| `seed` | `0` | Semilla de `torch` y `random`. |

### Qué hace, etapa por etapa

1. **Corpus.** Sin `text_path`, 30 000 frases sintéticas "*sujeto verbo objeto lugar.*", cada hueco elegido entre 8 opciones (p. ej. "The old sailor found a golden key near the river.").
2. **Tokenizador.** Un BPE a nivel de bytes escrito desde cero (pretokenizador estilo GPT-2, fusiones aplicadas en el orden aprendido) entrenado con los primeros 200 000 caracteres. Imprime el número de fusiones, el tamaño real del vocabulario, los caracteres por token, una comprobación de ida y vuelta y cómo se trocea una frase de ejemplo.
3. **Datos.** División contigua 90 / 10; ventanas aleatorias de `block_size` tokens con el objetivo desplazado una posición (predicción del siguiente token).
4. **Modelo.** `MiniGPT`: embeddings atados, bloques pre-norm con RMSNorm, atención causal con RoPE (`scaled_dot_product_attention`), FFN SwiGLU (regla 8/3 de Llama redondeada a 64), inicialización GPT-2 con las proyecciones residuales escaladas por 1/√(2L). Imprime el número de parámetros y la pérdida inicial junto a ln V (lo que daría una apuesta uniforme).
5. **Entrenamiento.** AdamW (β = (0.9, 0.95), decay solo en matrices), calentamiento + coseno, recorte. Una línea de log cada `eval_every` pasos: `step | train | val | ppl | lr | tiempo`.
6. **Evaluación.** Pérdida y perplejidad finales en validación. Con el corpus sintético calcula además la **pérdida irreducible**: cada frase contiene exactamente 4 · ln 8 nats de incertidumbre real (cuatro huecos, ocho opciones cada uno), así que el suelo por token es `4 · ln 8 · n_frases / n_tokens`. Imprime cuánto se queda el modelo por encima de ese suelo. Es la vara de medir honesta: ningún modelo puede bajar de ahí.
7. **Muestreo.** Tres generaciones a partir de "The old sailor": greedy, T = 0.8 con top-k 40 y T = 1.0 con top-p 0.9.
8. **Checkpoint.** Guarda `{config, state_dict, fusiones del tokenizador}` en `mini_gpt.pt` dentro de la carpeta temporal del sistema (`tempfile.gettempdir()`), lo recarga en un modelo y un tokenizador nuevos y comprueba que los logits y las codificaciones son idénticos.

### Ejemplo

```powershell
.venv\Scripts\python scripts\mini_llm.py
```
```bash
.venv/bin/python scripts/mini_llm.py        # o bien: make mini-llm
```

En una GPU NVIDIA el script entrena automáticamente con **autocast bf16** (Ampere o posterior) y deja que las multiplicaciones de matrices usen **TF32** (`torch.set_float32_matmul_precision("high")`); en CPU funciona en fp32 normal. Con los 800 pasos por defecto tarda unos **3–5 minutos en CPU**. Para probar con tu propio texto, define `text_path` en `CFG` (la línea de pérdida irreducible solo aparece con el corpus sintético).

**Medido en la ejecución por defecto de 800 pasos** (RTX 3060 Laptop GPU): **1.86M parámetros**, vocabulario de **455** tras **199 fusiones BPE** (el corpus sintético se queda sin pares que fusionar antes de llegar a 512), **~22 s** de entrenamiento, pérdida de validación **0.778** (perplejidad 2.18) frente a un suelo del corpus de **0.774** (perplejidad 2.17): **un 0.6% por encima del suelo**.

¿De dónde salen los 1.86M parámetros? Embedding (atado) 455 × 192 = 87 360; por capa, atención 4 · 192² = 147 456 + SwiGLU 3 · 192 · 512 = 294 912 + normas 384; × 4 capas + norma final ≈ 1.86M.

### Notas y trampas

- El vocabulario que se imprime es el **real** (`len(tok.vocab)`), que puede ser menor que `vocab_size` con un corpus pequeño o repetitivo. El modelo se construye con ese tamaño real.
- El checkpoint va a la carpeta temporal, no a `runs/`. Cópialo si quieres conservarlo.
- No hay KV-cache: la generación recalcula la ventana (recortada a `block_size`) en cada paso. Para la versión con caché, consulta [`model`](referencia_modulos.md#4-model) y `sample.py`.

---

## `finetune.py`

*Post-training* con el stack de Hugging Face (`transformers` + `peft` + `trl`) sobre las dos tareas sintéticas pequeñas de los notebooks 07–08, con **SmolLM2-135M-Instruct** (se descarga del Hugging Face Hub la primera vez, ~260 MB). Un subcomando por técnica:

| Subcomando | Tarea | Qué hace |
|---|---|---|
| `sft` | Reserva → extracción a JSON | SFT con LoRA (o completo) con el `SFTTrainer` de TRL |
| `dpo` | Reserva → extracción a JSON | Primero SFT con LoRA, después DPO (+ NLL por defecto) con el `DPOTrainer` de TRL y un segundo adaptador LoRA |
| `grpo` | Aritmética de dos cifras | GRPO con el `GRPOTrainer` de TRL, un adaptador LoRA y una recompensa verificable |

Cada subcomando evalúa sobre un conjunto **reservado** (*held-out*) antes y después de entrenar. Los generadores de las tareas, la puntuación y las funciones de recompensa son las mismas que definen los notebooks 07–08, y `tests/test_finetune_hf.py` las prueba.

### Argumentos

```
finetune.py {sft,dpo,grpo} [opciones]
```

| Argumento | Tipo | Por defecto | Lo usa | Descripción |
|---|---|---|---|---|
| `stage` (posicional) | opción | **obligatorio** | — | `sft`, `dpo` o `grpo`. |
| `--model` | str | `HuggingFaceTB/SmolLM2-135M-Instruct` | todos | Cualquier LM causal de Hugging Face con plantilla de chat. |
| `--full` | flag | off | `sft`, `dpo` | Fine-tuning completo en lugar de LoRA en la etapa SFT (necesita más VRAM; usa el optimizador Adafactor). |
| `--rank` | int | `16` | todos | Rango `r` de LoRA; `lora_alpha = 2 * rank`, `target_modules="all-linear"`. |
| `--n-train` | int | `300` | `sft`, `dpo` | Ejemplos de entrenamiento de SFT. |
| `--epochs` | float | `2` | `sft`, `dpo` | Épocas de SFT. |
| `--lr` | float | `None` | todos | Learning rate. Por defecto: SFT `1e-3` (LoRA) o `1e-4` (`--full`); DPO `2e-4`; GRPO `1e-4`. En el subcomando `dpo`, `--lr` solo afecta a la etapa DPO, y la etapa SFT conserva su valor por defecto. |
| `--nll-weight` | float | `1.0` | `dpo` | Peso de la pérdida SFT (NLL) sobre la respuesta elegida. `0` = DPO puro. |
| `--beta` | float | `0.1` | `dpo` | β de DPO (cuánto se ata la política a la referencia). |
| `--reward` | opción | `exact` | `grpo` | `exact` (dispersa: 1 si el último número de la respuesta es correcto) o `partial` (densa: hasta 0.8 por acercarse, +0.2 por responder solo con un número). |
| `--grpo-steps` | int | `40` | `grpo` | Pasos de optimización de GRPO. |
| `--seed` | int | `0` | todos | `transformers.set_seed`. |
| `--out` | ruta | `<carpeta temporal>/llm_edu_finetune` | todos | Carpeta de trabajo de los *trainers*. No se guarda nada en ella (`save_strategy="no"`). |

### Qué hace internamente

**Común a todos los subcomandos.** Carga el tokenizador (*padding* a la izquierda, para generar por lotes) y el modelo en **bf16 en CUDA** o fp32 en CPU. Todos los *trainers* usan micro-batches de 4 con 4 pasos de acumulación de gradiente (16 secuencias por paso; GRPO usa 8 × 4), *gradient checkpointing*, log cada 5 pasos, `report_to="none"` y sin checkpoints. La evaluación usa decodificación greedy con la plantilla de chat.

**`sft`**

1. **Datos.** `booking_examples(--n-train, seed)` genera frases como *"We are six. Could we get a table at Casa Pepe on Friday around 8:30 pm? Booking for Ana."* con el objetivo `{"name", "restaurant", "people" (int), "day" (minúsculas), "time" (24 h HH:MM)}`. Cada una se convierte en una fila conversacional **prompt-completion** (system + user → JSON del assistant). Los 40 ejemplos de test usan nombres y restaurantes que nunca aparecen en entrenamiento.
2. **Antes.** Puntúa el modelo base: `valid` (la respuesta contiene JSON parseable), `exact` (todos los campos bien) y `fields` (fracción de campos bien).
3. **Entrenamiento.** `SFTTrainer` con `SFTConfig(lr_scheduler_type="cosine", warmup_steps=5, max_length=256)` y un `LoraConfig` (o ninguno con `--full`). Con datos prompt-completion, TRL calcula la pérdida **solo sobre la respuesta** (*completion-only loss*): los tokens del prompt reciben la etiqueta `-100`.
4. **Después.** Fusiona los pesos LoRA en el modelo (`merge_and_unload`) y lo vuelve a puntuar.

**`dpo`**

1. Ejecuta la etapa `sft` anterior e imprime las puntuaciones `SFT:`. Con los valores por defecto el modelo SFT queda solo parcialmente entrenado, así que DPO tiene margen para mejorar.
2. Construye 400 **pares de preferencias**: *elegido* = el JSON correcto, *rechazado* = el mismo JSON con un error realista (las 8 pm dejadas como `08:00`, `people` como cadena, el día en mayúscula o un nombre equivocado).
3. Entrena un **nuevo adaptador LoRA encima del modelo SFT fusionado** con `DPOTrainer`. El modelo de referencia es la misma red con el adaptador desactivado, así que no se guarda una segunda copia en memoria. `loss_type=["sigmoid", "sft"]` con `loss_weights=[1, --nll-weight]` es DPO + NLL; con `--nll-weight 0` queda `["sigmoid"]` (DPO puro). Una época, schedule coseno, 3 pasos de calentamiento.
4. Imprime las log-probabilidades de las respuestas elegida y rechazada en el primer y el último paso de log, y después las puntuaciones tras DPO. Con `--nll-weight 0` se ve cómo baja también la log-prob de la elegida (*likelihood displacement*).

**`grpo`** (parte del modelo base, no del SFT)

1. **Datos.** 2000 prompts de entrenamiento y 100 de test del tipo *"What is 47 - 85?"*, con el system prompt *"Answer with the number only."*.
2. **Antes.** Exactitud del modelo base en los prompts de test.
3. **Entrenamiento.** `GRPOTrainer` con un adaptador LoRA: 8 respuestas por prompt (`num_generations=8`), 32 respuestas por paso de optimización (4 prompts × 8), `max_completion_length=32`, `temperature=1.0`, coeficiente KL `beta=0.04`, learning rate constante. La recompensa es `reward_exact` o `reward_partial`.
4. **Después.** La exactitud otra vez.

### Ejemplos

```powershell
.venv\Scripts\python scripts\finetune.py sft                          # SFT con LoRA
.venv\Scripts\python scripts\finetune.py sft --full                   # fine-tuning completo
.venv\Scripts\python scripts\finetune.py dpo                          # SFT y después DPO + NLL
.venv\Scripts\python scripts\finetune.py dpo --nll-weight 0           # DPO puro: observa cómo baja la log-prob de la elegida
.venv\Scripts\python scripts\finetune.py grpo --reward exact          # recompensa dispersa y verificable
.venv\Scripts\python scripts\finetune.py grpo --reward partial        # recompensa densa con crédito parcial
.venv\Scripts\python scripts\finetune.py grpo --grpo-steps 10         # una ejecución rápida
```
```bash
.venv/bin/python scripts/finetune.py sft
.venv/bin/python scripts/finetune.py sft --full
.venv/bin/python scripts/finetune.py dpo
.venv/bin/python scripts/finetune.py dpo --nll-weight 0
.venv/bin/python scripts/finetune.py grpo --reward exact
.venv/bin/python scripts/finetune.py grpo --reward partial
.venv/bin/python scripts/finetune.py grpo --grpo-steps 10
```

**Medido** (RTX 3060 Laptop GPU, 6 GB, con los modelos de Ollama descargados de la GPU):

| Comando | Antes | Después | Tiempo |
|---|---|---|---|
| `finetune.py sft` | valid 0.00, exact 0.00, fields 0.00 | valid **1.00**, exact **0.375**, fields **0.855** | 73 s de entrenamiento, 94 s en total |
| `finetune.py grpo --grpo-steps 10` | exactitud 0.04 | exactitud **0.42** | ~5 s por paso, 69 s en total |

Cómo leerlos:

- **SFT.** El modelo base nunca responde con JSON a este prompt. Tras 38 pasos de optimización sobre 300 ejemplos, todas las respuestas son JSON válido y el 85 % de los campos están bien, pero solo el 37,5 % de las respuestas tiene *todos* los campos bien (la conversión de la hora es lo más difícil). Ese hueco es el que ataca `dpo`.
- **GRPO.** Casi todo el salto en 10 pasos viene del formato: el modelo base suele razonar con palabras y se queda sin sus 32 tokens (el 75 % de las respuestas se cortaban en los cinco primeros pasos), y la recompensa le enseña a responder solo con el número. En esos primeros pasos, entre el 65 y el 70 % de los grupos tenían la misma recompensa en las 8 muestras (`frac_reward_zero_std` en el log): esos grupos no dan gradiente, que es la debilidad de una recompensa dispersa que `--reward partial` intenta resolver.
- Las ejecuciones completas por defecto (`dpo`, y `grpo` con 40 pasos) no se cronometraron para este documento. Cuenta con unos minutos cada una en una GPU parecida, y mucho más en CPU.

### Notas y trampas

- **No lo ejecutes mientras Ollama ocupa la GPU** en una tarjeta de 6 GB (ver [instalación, notas sobre la GPU](instalacion.md#notas-sobre-la-gpu)). Si has usado Ollama en los últimos minutos, ejecuta antes `ollama stop qwen2.5:3b`.
- `--full` solo cambia la etapa SFT. DPO y GRPO siempre entrenan un adaptador LoRA.
- `--epochs` y `--n-train` no afectan a `grpo`; `--grpo-steps` y `--reward` no afectan a `sft` / `dpo`.
- En Windows, la primera descarga puede imprimir un aviso de `huggingface_hub` sobre enlaces simbólicos. Es inofensivo (ver [problemas frecuentes](instalacion.md#problemas-frecuentes)).
- No se guarda nada: el script es una demostración que imprime puntuaciones. Para conservar un modelo, mira el notebook 07, que guarda el adaptador LoRA con `save_pretrained`, lo recarga y lo fusiona.

---

## `rag_demo.py`

RAG sobre una carpeta de ficheros `.md` / `.txt` con **LangChain**: recuperación híbrida (búsqueda densa con FAISS + BM25, fusionadas con Reciprocal Rank Fusion por un `EnsembleRetriever`) y una respuesta con citas numeradas. Es el pipeline del notebook 10 empaquetado como herramienta de línea de comandos. Los modelos salen de `llm_edu_suite.config` (Ollama `qwen2.5:3b` y `nomic-embed-text` por defecto; edita `.env` para cambiarlos).

### Argumentos

| Argumento | Tipo | Por defecto | Descripción |
|---|---|---|---|
| `--docs` | ruta | `data/knowledge_base` | Carpeta con ficheros `.md` / `.txt` (solo el primer nivel, no recursivo). |
| `--ask` | str | `None` | Pregunta a responder con citas. |
| `--eval` | ruta (valor opcional) | `None` | Un JSON de evaluación. Si se pasa **sin** valor, usa `data/eval/rag_eval.json`. |
| `--ragas` | flag | off | Con `--eval`: además genera una respuesta para cada pregunta y las puntúa con RAGAS. Lento: el LLM hace de juez. |
| `--retriever` | opción | `hybrid` | `hybrid`, `dense` o `bm25`: el recuperador que usa la cadena (y RAGAS). |
| `--k` | int | `4` | Chunks por respuesta. |
| `--chunk-size` | int | `800` | Caracteres por chunk; el solapamiento es `chunk_size // 8`. |

Se pueden combinar `--ask` y `--eval`. Sin ninguno de los dos, el script solo indexa e imprime la línea de resumen.

### Qué hace internamente

1. **Preparación.** `config.setup(project="llm-edu-suite-rag-demo")` carga `.env` (así que con una clave de LangSmith cada ejecución se traza en ese proyecto). Si el proveedor es Ollama y el servidor no responde, el script termina con un mensaje que explica cómo arrancarlo.
2. **Carga.** Cada `*.md` y `*.txt` del primer nivel de `--docs` se convierte en un `Document` de LangChain con el nombre del fichero en `metadata["source"]`.
3. **Troceado.** `RecursiveCharacterTextSplitter(chunk_size, chunk_overlap=chunk_size // 8, add_start_index=True)`.
4. **Indexado.**
   - *Denso*: `FAISS` sobre `get_embeddings()`, con embeddings calculados en lotes de 64 (una única petición enorme puede saturar un servidor de embeddings local).
   - *Léxico*: `BM25Retriever` con un tokenizador de palabras en minúsculas (el de por defecto solo separa por espacios, así que `Llama-3?` no coincidiría con `llama-3`).
   - *Híbrido*: `EnsembleRetriever([bm25, dense], weights=[0.5, 0.5])` (RRF ponderado), recortado a los `k` primeros.
   Imprime `indexed N files -> M chunks in X s | retriever ... | llm ...`.
5. **La cadena** (LCEL): `RunnableParallel(docs=retriever, question=RunnablePassthrough()).assign(answer=prompt | llm | StrOutputParser())`. Los documentos viajan junto a la respuesta, para poder citarlos. El system prompt pide de una a tres frases, citas como `[1]` o `[2][3]`, y "I don't know" si el contexto no contiene la respuesta.
6. **`--ask`.** Imprime `ANSWER` y después `SOURCES` como `[n]* fichero: primeros 100 caracteres`; el `*` marca los pasajes que la respuesta citó de verdad.
7. **`--eval`.** Puntúa **los tres recuperadores** (diga lo que diga `--retriever`) en cada pregunta:
   - *Nivel de fichero*: `file_hit@1`, `file_hit@k` y `file_mrr`: ¿hay un chunk de un fichero correcto en el top 1 / top k?, y la media del inverso de la posición del primero.
   - *Nivel de evidencia* (preguntas con lista `evidence`): `evidence_hit@1`, `evidence_hit@k`, `evidence_mrr`: ¿algún chunk recuperado contiene de verdad una de las cadenas de evidencia? Es más estricto: no basta con el fichero correcto si el chunk es la parte equivocada.
8. **`--ragas`.** Ejecuta la cadena con `--retriever` sobre cada pregunta (`batch`, 4 a la vez), quita las marcas `[n]` para que el juez no las cuente como afirmaciones sin respaldo, y puntúa las respuestas con cuatro métricas de RAGAS: **faithfulness**, **answer relevancy**, **context precision** y **context recall**. El juez es el modelo de chat configurado, llamado a través del endpoint compatible con OpenAI de Ollama (`OLLAMA_BASE_URL + /v1`). Si una métrica falla en una pregunta, se descarta (NaN) en lugar de detener la ejecución. Antes de importar ragas, el script registra un módulo vacío de sustitución para `langchain_community.chat_models.vertexai`, que ragas 0.4.3 importa y langchain-community 0.4 eliminó.

### Formato del fichero de evaluación

```json
[
  {"question": "What does Reciprocal Rank Fusion sum to combine several rankings?",
   "answer": "For each document RRF sums 1/(k + rank) over the rankings, with k = 60.",
   "sources": ["glossary.md", "module_reference.md"],
   "evidence": ["1/(60 + rank)"]}
]
```

La fuente se puede dar como `sources` (lista), `source` o `doc` (un único nombre de fichero), y debe coincidir con los nombres de fichero de `--docs`. `answer` es la respuesta de referencia (la necesitan context precision y context recall); `evidence` es opcional.

El repositorio incluye dos corpus:

- `data/knowledge_base/` (el de por defecto): cinco documentos largos sobre el propio curso, con `data/eval/rag_eval.json` (20 preguntas con fuentes y cadenas de evidencia).
- `data/sample_docs/`: cuatro documentos cortos (`refund_policy.md`, `security.md`, `shipping.md`, `support.md`), con `data/sample_eval.json` (6 preguntas en el formato `doc`).

### Ejemplos

**Hacer una pregunta** (corpus por defecto, recuperación híbrida):

```powershell
.venv\Scripts\python scripts\rag_demo.py --ask "Why must an MCP stdio server never print to stdout?"
```
```bash
.venv/bin/python scripts/rag_demo.py --ask "Why must an MCP stdio server never print to stdout?"
```

Salida medida (~16 s, de los que ~6 s son el indexado de 358 chunks):

```
indexed 5 files -> 358 chunks in 6.4s | retriever hybrid | llm qwen2.5:3b

ANSWER: Because stdout is reserved for the protocol and a single stray `print()` corrupts the stream and the client fails to parse the next message [1].

SOURCES:
  [1]* module_reference.md: '- *stdio framing*. On the stdio transport the host launches the server as a subprocess, and each mes'
  [2]  cli_scripts.md: '- **Never print to stdout.** In an stdio server, stdout carries the protocol; a single stray `print('
  ...
```

**Evaluación de la recuperación** (rápida, sin generación):

```powershell
.venv\Scripts\python scripts\rag_demo.py --eval
```
```bash
.venv/bin/python scripts/rag_demo.py --eval
```

Medido (20 preguntas, `k = 4`, ~17 s en total):

| Recuperador | file hit@1 | file hit@4 | file MRR | evidence hit@1 | evidence hit@4 | evidence MRR |
|---|---|---|---|---|---|---|
| dense | 0.95 | 1.00 | 0.97 | 0.65 | 0.85 | 0.73 |
| bm25 | 0.90 | 1.00 | 0.94 | 0.70 | **0.95** | 0.80 |
| **hybrid** | 0.95 | 1.00 | 0.97 | **0.80** | 0.90 | **0.84** |

A nivel de fichero los tres son casi perfectos: con cinco documentos, encontrar el *fichero* correcto es fácil. Las columnas de evidencia son las informativas: el recuperador híbrido pone primero un chunk que contiene de verdad la respuesta en el 80 % de las preguntas, frente al 65 % de la búsqueda densa sola.

**Recuperación + RAGAS** sobre el corpus pequeño de ejemplo:

```powershell
.venv\Scripts\python scripts\rag_demo.py --docs data\sample_docs --eval data\sample_eval.json --ragas
```
```bash
.venv/bin/python scripts/rag_demo.py --docs data/sample_docs --eval data/sample_eval.json --ragas
```

Medido (6 preguntas, juez `qwen2.5:3b`, la parte de RAGAS 47 s, ~55 s en total): recuperación 1.00 en todo (cada documento de ejemplo es un único chunk, así que con `k = 4` siempre se recuperan los cuatro), **faithfulness 1.00, answer relevancy 0.75, context precision 1.00, context recall 1.00**. Sobre el conjunto por defecto de 20 preguntas `--ragas` tarda bastante más, porque cada pregunta necesita una generación y varias llamadas al juez.

**Otras opciones:**

```bash
.venv/bin/python scripts/rag_demo.py --docs data/sample_docs --ask "What is the refund policy?" --retriever dense
.venv/bin/python scripts/rag_demo.py --docs ./mis_docs --ask "..." --k 6 --chunk-size 1200
```

### Notas y trampas

- Las respuestas salen de un modelo de 3B en local: en este corpus suelen ser correctas, pero pueden parafrasear, citar el número de pasaje equivocado o responder de más (el ejemplo `--ask` con `dense` sobre `sample_docs` devolvió el documento de devoluciones entero). Para eso están las métricas de RAGAS.
- `--ragas` usa Ollama para el juez aunque `LLM_PROVIDER` sea `openai` o `anthropic`.
- Un juez de 3B es ruidoso. Usa sus puntuaciones RAGAS como señal relativa (comparar recuperadores o tamaños de chunk con el mismo juez), no como calidad absoluta.
- El índice FAISS vive en memoria y se reconstruye en cada ejecución (unos segundos para este corpus).
- `langchain-community` muestra un aviso de *sunset* para `FAISS` / `BM25Retriever`; el script lo filtra.

---

## `mcp_server.py`

Un **servidor MCP** construido con el SDK oficial de MCP para Python (`mcp.server.fastmcp.FastMCP`). Se llama **`llm-edu-docs`** y expone la pequeña base de conocimiento de `data/sample_docs/` (o cualquier carpeta con ficheros `.md` / `.txt`) a cualquier cliente MCP: Claude Code, Claude Desktop, el MCP Inspector, un cliente Python del SDK, o un agente de LangChain a través de `langchain-mcp-adapters`. No necesita ni PyTorch ni Ollama.

### Argumentos

| Argumento | Tipo | Por defecto | Descripción |
|---|---|---|---|
| `--docs` | ruta | `data/sample_docs` | Carpeta con los ficheros `.md` / `.txt` que se exponen (solo el primer nivel). |
| `--http` | flag | off | Servir **Streamable HTTP** en lugar de stdio. |
| `--host` | str | `127.0.0.1` | Host HTTP (solo con `--http`). |
| `--port` | int | `8000` | Puerto HTTP (solo con `--http`). El endpoint es `http://<host>:<port>/mcp`. |
| `--log-level` | str | `WARNING` | Nivel de log en stderr. `INFO` registra cada petición. |

### Qué expone

| Primitiva | Nombre | Descripción |
|---|---|---|
| tool | `search_docs(query, k=3)` | Búsqueda BM25 sobre los documentos (`rank_bm25`). Cada línea de los documentos de ejemplo (salvo los títulos) es una frase autocontenida, así que cada línea es un pasaje; el nombre del fichero se añade al pasaje para la búsqueda. Devuelve los `k` mejores (limitado a 1–10) como líneas `[fichero] pasaje`, o `no results`. Anotada con `readOnlyHint: true`. |
| tool | `calculate(expression)` | Aritmética segura: la expresión se analiza con `ast` y solo se evalúan números, `+ - * / ** %` y paréntesis; cualquier otra cosa es un error de herramienta (`isError: true`). Anotada con `readOnlyHint: true`. |
| tool | `create_ticket(title, priority="normal")` | Acción de escritura simulada (`priority` ∈ `low`, `normal`, `high`). Devuelve el ticket (`id`, `title`, `priority`, `created`) como JSON. Anotada con `destructiveHint: true`, así que un host bien hecho **pide confirmación**. |
| resource | `docs://index` | Lista JSON de todos los documentos con su URI y su título. |
| resource template | `docs://{name}` | El texto completo de un documento, p. ej. `docs://shipping.md`. |
| prompt | `answer_with_sources(question)` | Prompt RAG: usar `search_docs`, responder solo con los pasajes, citar los nombres de fichero y decirlo si la respuesta no está. |

El servidor envía además unas `instructions` en su respuesta a `initialize` (*"Use search_docs before answering policy questions and cite the file names."*). El SDK deriva el JSON Schema de cada herramienta de sus anotaciones de tipo, y su descripción del docstring.

**Medido** con un cliente del SDK por stdio (`ClientSession` + `stdio_client`): la sesión completa (arrancar el subproceso, `initialize`, listar tools / resources / templates / prompts, llamar a las tres herramientas, leer un recurso) tarda **unos 2 s**. `search_docs("express shipping", k=2)` devuelve las dos líneas de `shipping.md`; `calculate("(17 * 23) + 4")` devuelve `395`; `calculate("__import__('os')")` devuelve un error de herramienta (`only numbers and + - * / ** % ( ) are allowed`). El servidor anuncia la versión de protocolo `2025-11-25` con el SDK 1.30.0, y con `--http` responde a `initialize` en `/mcp` como *event stream* con una cabecera `mcp-session-id`.

### Ejecutarlo

En modo stdio (el de por defecto) el servidor espera mensajes JSON-RPC por stdin y no imprime nada hasta que se conecta un cliente; se para con Ctrl+C.

```powershell
.venv\Scripts\python scripts\mcp_server.py                                # stdio
.venv\Scripts\python scripts\mcp_server.py --docs C:\ruta\a\tus\docs
.venv\Scripts\python scripts\mcp_server.py --http --port 8000             # http://127.0.0.1:8000/mcp
```
```bash
.venv/bin/python scripts/mcp_server.py                                    # o bien: make mcp-server
.venv/bin/python scripts/mcp_server.py --docs ruta/a/tus/docs
.venv/bin/python scripts/mcp_server.py --http --port 8000 --log-level INFO
```

### Conectarlo a Claude Code

```bash
claude mcp add llm-edu-docs -- python /ABSOLUTE/PATH/scripts/mcp_server.py
```

Todo lo que va después de `--` es el comando que lanza Claude Code. Usa el intérprete del proyecto si el `python` de tu `PATH` no tiene los paquetes `mcp` y `rank_bm25`:

```powershell
claude mcp add llm-edu-docs -- D:\ruta\a\llm-engineering-suite\.venv\Scripts\python.exe D:\ruta\a\llm-engineering-suite\scripts\mcp_server.py
```
```bash
claude mcp add llm-edu-docs -- /ruta/a/llm-engineering-suite/.venv/bin/python /ruta/a/llm-engineering-suite/scripts/mcp_server.py
```

Después, `claude mcp list` muestra el servidor y su estado, y `/mcp` dentro de una sesión de Claude Code lista sus herramientas. Pregunta, por ejemplo, *"How long does express shipping take? Use the llm-edu-docs server."*

Para conectarte al modo HTTP, arranca el servidor con `--http` y ejecuta `claude mcp add --transport http llm-edu-docs http://127.0.0.1:8000/mcp`.

### Conectarlo a Claude Desktop

Añade el servidor a `claude_desktop_config.json` (*Settings → Developer → Edit Config*), con rutas absolutas, y reinicia Claude Desktop:

```json
{
  "mcpServers": {
    "llm-edu-docs": {
      "command": "/ABSOLUTE/PATH/.venv/bin/python",
      "args": ["/ABSOLUTE/PATH/scripts/mcp_server.py"]
    }
  }
}
```

En Windows usa `"command": "D:\\ruta\\a\\llm-engineering-suite\\.venv\\Scripts\\python.exe"` (en JSON las barras invertidas van duplicadas).

### Probarlo con el MCP Inspector

La interfaz web oficial para probar servidores (necesita Node.js):

```bash
npx @modelcontextprotocol/inspector python scripts/mcp_server.py
```

Abre una página en el navegador donde puedes listar y llamar a las herramientas, leer los recursos y renderizar el prompt.

### Desde Python

Con el cliente oficial del SDK:

```python
import asyncio, sys
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def main():
    params = StdioServerParameters(command=sys.executable, args=["scripts/mcp_server.py"])
    async with stdio_client(params) as (read, write), ClientSession(read, write) as session:
        await session.initialize()
        res = await session.call_tool("search_docs", {"query": "express shipping", "k": 1})
        print(res.content[0].text)          # [shipping.md] Express shipping delivers in 1 to 2 business days ...

asyncio.run(main())
```

Como herramientas de LangChain para un agente, con `langchain-mcp-adapters`:

```python
from langchain_mcp_adapters.client import MultiServerMCPClient

client = MultiServerMCPClient({"llm-edu-docs": {"transport": "stdio", "command": sys.executable,
                                                "args": ["scripts/mcp_server.py"]}})
tools = await client.get_tools()            # search_docs, calculate, create_ticket como tools de LangChain
```

El notebook 16 recorre las dos formas y construye un agente encima de las herramientas con `create_agent` de LangChain.

### Notas y trampas

- **Nunca imprimas en stdout.** En un servidor stdio, stdout transporta el protocolo; un solo `print()` perdido corrompe el flujo y el cliente no consigue parsear el siguiente mensaje. Todo lo que registra el servidor (la dirección HTTP, los tickets creados) va a `sys.stderr`.
- Usa **rutas absolutas** en la configuración de Claude Code y Claude Desktop: el host lanza el proceso desde su propio directorio de trabajo.
- Los tickets viven en memoria y desaparecen al terminar el proceso.
- `FastMCP` es aquí la clase del SDK 1.x. El curso fija `mcp<2` porque `langchain-mcp-adapters` todavía no soporta el SDK 2.x; en 2.x la misma clase se llama `mcp.server.mcpserver.MCPServer` y los decoradores funcionan igual.
- Dentro de Jupyter, un cliente stdio necesita un apaño porque el `sys.stderr` de Jupyter no tiene un descriptor de fichero del sistema; el notebook 16 ya lo aplica. En scripts normales no hace falta.

---

## Herramientas y objetivos del Makefile

**`tools/build_notebooks.py`** regenera los `.ipynb` a partir de sus generadores en `tools/notebooks/nbNN_*.py` (el generador del notebook 00 escribe además `scripts/mini_llm.py`). Regenerar **borra las salidas**.

```bash
python tools/build_notebooks.py            # todos los notebooks
python tools/build_notebooks.py 05 14      # solo algunos, por número
```

**`tools/run_notebooks.py`** ejecuta notebooks sin interfaz y **en el sitio** (conservan sus salidas), cada uno con su propia carpeta como directorio de trabajo, igual que Jupyter. Busca en `notebooks/` de forma **recursiva**, así que encuentra los notebooks de todas las carpetas temáticas. Los argumentos pueden ser nombres de fichero o prefijos numéricos:

```bash
python tools/run_notebooks.py                       # todos los notebooks, por orden de número
python tools/run_notebooks.py 05_pretraining.ipynb  # por nombre de fichero, esté donde esté
python tools/run_notebooks.py 05 10 14              # por número (prefijo)
```

Imprime `[segundos] nombre: OK | FAIL` por notebook y `N/M notebooks OK` al final, con un límite de 1 hora por notebook, y sale con código 1 si alguno falla, de modo que sirve para CI. Un argumento que coincide con cero o con varios notebooks detiene la ejecución con un error. Los notebooks 10–16 necesitan Ollama en marcha; en una GPU de 6 GB, evita ejecutar los notebooks de entrenamiento (07–09) mientras Ollama tiene un modelo cargado (ver [instalación](instalacion.md#notas-sobre-la-gpu)).

**Objetivos del Makefile** (Linux/macOS o Git Bash con `make`):

| Objetivo | Comando | Qué hace |
|---|---|---|
| `make setup` | `bash setup_env.sh` | Crea `.venv` e instala todo (en Windows, ejecuta `setup_env.ps1`). |
| `make test` | `pytest -q` | Todos los tests. Los 4 tests `ollama` se saltan solos si Ollama no está en marcha. Usa `pytest -q -m "not ollama"` para solo los rápidos (82 tests, ~35 s medidos). |
| `make notebooks` | `python tools/build_notebooks.py` | Regenera todos los `.ipynb` (y `scripts/mini_llm.py`) desde los generadores. **Borra las salidas.** |
| `make run-notebooks` | `python tools/run_notebooks.py` | Ejecuta todos los notebooks en el sitio. |
| `make smoke` | `train.py --steps 100 --d-model 64 --n-layers 2 ...` | Prueba de humo corta de preentrenamiento. |
| `make mini-llm` | `python scripts/mini_llm.py` | Entrena el mini GPT autocontenido (el notebook 00 como script). |
| `make mcp-server` | `python scripts/mcp_server.py` | Arranca el servidor MCP por stdio. |

---

## Resumen de resultados medidos

| Comando | Hardware | Resultado |
|---|---|---|
| `train.py --steps 300 --out runs/smoke` | CPU | 3.03M parámetros, perplejidad de validación **2.65**, ≈ 140 s |
| `sample.py --run runs/smoke --max-new-tokens 50` | CPU | ≈ **200 tok/s** (salida en ids porque el corpus es sintético) |
| `mini_llm.py` (800 pasos por defecto) | RTX 3060 Laptop GPU (~3–5 min en CPU) | 1.86M parámetros, vocabulario 455 (199 fusiones), ~22 s de entrenamiento, pérdida de validación **0.778** frente a un suelo del corpus de **0.774** (un 0.6 % por encima) |
| `finetune.py sft` | RTX 3060 Laptop GPU | exact 0.00 → **0.375**, fields 0.00 → **0.855**, JSON válido 0.00 → **1.00**; 94 s |
| `finetune.py grpo --grpo-steps 10` | RTX 3060 Laptop GPU | exactitud 0.04 → **0.42**; ~5 s por paso, 69 s |
| `rag_demo.py --eval` | RTX 3060 Laptop GPU + Ollama | híbrido: file hit@1 **0.95**, evidence hit@1 **0.80**, evidence MRR **0.84** (denso: 0.65 / 0.73); ~17 s |
| `rag_demo.py --docs data/sample_docs --eval data/sample_eval.json --ragas` | RTX 3060 Laptop GPU + Ollama | faithfulness **1.00**, answer relevancy **0.75**, context precision **1.00**, context recall **1.00**; ~55 s |
| `rag_demo.py --ask "..."` | RTX 3060 Laptop GPU + Ollama | ~10–16 s incluido el indexado |
| `mcp_server.py` con un cliente stdio del SDK | CPU | sesión completa (arranque, `initialize`, listar todo, 4 llamadas a herramientas, 1 lectura de recurso) ≈ 2 s |
| `pytest -q -m "not ollama"` / `pytest -q -m ollama` | RTX 3060 Laptop GPU + Ollama | 82 pasan en ~35 s / 4 pasan en ~40 s |

Para el contexto pedagógico de cada etapa, consulta la [guía de notebooks](guia_notebooks.md). Para un recorrido de principio a fin con interpretación de resultados, el [caso de estudio](caso_de_estudio.md).
