🇬🇧 [English version](../en/cli_scripts.md)

# Scripts de línea de comandos

> Guía de los seis scripts de `scripts/` (más `tools/run_notebooks.py` y los objetivos del `Makefile`): todos sus argumentos reales (extraídos de `argparse`), qué hace cada etapa internamente, ejemplos de uso y resultados medidos en CPU.
>
> Documentos relacionados: [README](README.md) · [Instalación](instalacion.md) · [Referencia de módulos](referencia_modulos.md) · [Guía de notebooks](guia_notebooks.md) · [Caso de estudio](caso_de_estudio.md) · [Glosario](glosario.md)

| Script | Para qué sirve | Módulos que usa | Necesita `torch` |
|---|---|---|---|
| [`train.py`](#trainpy) | Preentrenar un `TransformerLM` sobre un `.txt` o un corpus sintético | `data`, `tokenizer`, `model`, `training`, `evaluation`, `utils` | sí |
| [`sample.py`](#samplepy) | Generar desde un checkpoint (muestreo normal o especulativo) | `model`, `tokenizer`, `inference`, `utils` | sí |
| [`finetune.py`](#finetunepy) | Demo de post-training: SFT (completo o LoRA) → DPO y GRPO, con evaluación | `data`, `model`, `training`, `finetune`, `evaluation`, `utils` | sí |
| [`rag_demo.py`](#rag_demopy) | Indexar una carpeta de documentos, responder preguntas y evaluar el RAG | `rag`, `llm_client`, `evaluation` | no |
| [`mini_llm.py`](#mini_llmpy) | Un mini GPT completo en un fichero: BPE → modelo → entrenamiento → evaluación → muestreo → checkpoint | ninguno (autocontenido) | sí |
| [`mcp_server.py`](#mcp_serverpy) | Un servidor MCP real por stdio que expone `data/sample_docs` a Claude Code, Claude Desktop o cualquier cliente MCP | `mcp`, `rag` | no |

---

## Antes de empezar

Todos los comandos se ejecutan desde la **raíz del repositorio** y con el intérprete del entorno virtual del proyecto (ver [instalación](instalacion.md)). Cada script añade la raíz del repo a `sys.path`, así que no hace falta instalar el paquete con `pip install -e .`.

**Windows (PowerShell o cmd):**

```powershell
.venv\Scripts\python scripts\train.py --steps 300 --out runs\smoke
```

**Linux / macOS:**

```bash
.venv/bin/python scripts/train.py --steps 300 --out runs/smoke
```

En el resto del documento cada ejemplo aparece en las dos formas. Si un comando largo se parte en varias líneas, el carácter de continuación es `` ` `` en PowerShell, `^` en cmd y `\` en bash.

Todos los scripts con argumentos aceptan `-h` / `--help` (`mini_llm.py` no tiene: se configura con su diccionario `CFG`). `train.py` además muestra en la ayuda su docstring con ejemplos.

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

## `finetune.py`

Demo de *post-training* de principio a fin sobre la tarea de juguete **"invertir la secuencia"**: dado `[5, 6, 7]`, el asistente debe responder `[7, 6, 5]`. Muestra cómo cambian el *exact match* y la perplejidad de la respuesta tras cada técnica.

### Argumentos

| Argumento | Tipo | Defecto | Descripción |
|---|---|---|---|
| `--steps` | int | `900` | Pasos de SFT. 900 deja el modelo *parcialmente* entrenado, con margen para que DPO y GRPO mejoren algo. |
| `--lora` | flag | off | SFT con LoRA (r = 8, α = 16) **+ embeddings entrenables** (`train_embeddings=True`) en lugar de fine-tuning completo. |
| `--dpo-steps` | int | `300` | Pasos de DPO. `0` desactiva la etapa. |
| `--nll-weight` | float | `1.0` | Peso del ancla NLL sobre la respuesta elegida (DPO + NLL / RPO). `0` = DPO puro, para observar el *likelihood displacement*. |
| `--grpo-iters` | int | `150` | Iteraciones de GRPO. `0` desactiva la etapa. |
| `--device` | str | auto | Igual que en `train.py`. |

El resto de hiperparámetros está fijo en el script (ver la tabla siguiente).

### Qué hace, etapa por etapa

| Etapa | Qué ocurre | Hiperparámetros fijos |
|---|---|---|
| 0. Formato | `V = 512`; `ChatFormat.for_vocab(512)` reserva los ids 507–511 como `SYS, USER, ASSIST, END, PAD`. | — |
| 1. Modelo base | Preentrena un `TransformerLM` (d_model 128, 4 capas, 8 cabezas, 2 KV, `max_seq_len` 128, ≈ 0.76M parámetros) durante **200 pasos** sobre un corpus sintético de 100 000 tokens con vocabulario `V − 5`, para no usar los ids de rol. Sin log. | batch 32, block 64, lr 3e-3, warmup 20 |
| 2. Datos | **8000** ejemplos SFT de entrenamiento (`seed=0`) y 200 de test (`seed=99`) para la perplejidad de la respuesta. Aparte, 150 casos de evaluación generados con otra semilla (longitud 2–5, tokens 1–400). | system = `[10]` |
| 3. Evaluación `base` | `exact_match_rate` con `greedy_answer` sobre los 150 casos + `perplexity_on_examples` sobre el test. | — |
| 4. SFT | Se copia la base. *Sin `--lora`*: `train_sft` sobre todos los parámetros. *Con `--lora`*: `apply_lora(r=8, alpha=16, train_embeddings=True)` y `train_sft` sobre `trainable_parameters` (adaptadores + `tok_emb`/`lm_head` atado: los tokens de rol nunca aparecieron en el preentrenamiento, así que sus embeddings hay que aprenderlos). Después `merge_lora` y **descongela** todos los pesos para que DPO y GRPO puedan actualizar el modelo fusionado. Se evalúa. Log cada 300 pasos. | full: lr 1e-3; LoRA: lr 2e-3; batch 32 |
| 5. DPO (si `--dpo-steps > 0`) | `train_dpo` desde el modelo SFT (la referencia es una copia congelada) con **3000** pares (`seed=1`) y `nll_weight=--nll-weight`. *Elegido* = inversión correcta, *rechazado* = correcta salvo el último token (sustituido por otro distinto). Se evalúa como `SFT+DPO+NLL` (o `SFT+DPO` con `--nll-weight 0`). Log cada 100 pasos (`loss`, `acc`, `margin`). | lr 5e-5, β 0.1, batch 16 |
| 6. GRPO (si `--grpo-iters > 0`) | `train_grpo` sobre **una copia del modelo SFT** (no del DPO). En cada iteración, 8 prompts (longitud 2–5, tokens 1–400) × 8 respuestas muestreadas. Recompensa **densa con crédito parcial**: fracción de posiciones correctas respecto a la inversión, `− 0.1 ×` la diferencia de longitud y `+0.2` si termina con `END`. Se evalúa como `SFT+GRPO`. Log cada 50 iteraciones (`mean reward`, `pg`, `kl`). | lr 1e-4, β (KL) 0.04, max_new 8 |

```
base --SFT (completo | LoRA)--> sft --DPO(+NLL)--> sft+dpo
                                    +--GRPO-------> sft+grpo     (dos ramas independientes desde SFT)
```

¿Por qué una recompensa densa? Con una recompensa binaria (1 solo si la inversión es exacta), un modelo parcialmente entrenado obtiene a menudo la misma recompensa en las 8 respuestas del grupo, la ventaja relativa es 0 y GRPO no recibe gradiente. El crédito parcial por posición diferencia respuestas "casi buenas" de malas y da señal en casi todos los grupos.

**Ejemplo de salida** (formato; los números dependen de la máquina y los argumentos):

```
base model 758.91K params on cpu
base           exact match   0.0%  reply ppl   ...
sft step    0  loss ...
SFT (LoRA)     exact match  ...%  reply ppl   ...
dpo step    0  loss ...  acc 0.00  margin +0.00
SFT+DPO+NLL    exact match  ...%  reply ppl   ...
grpo iter   0  mean reward ...  pg ...  kl 0.0000
SFT+GRPO       exact match  ...%  reply ppl   ...
```

En el primer paso de DPO, política y referencia son idénticas, así que `margin` es 0 y `acc` 0. Con `--nll-weight 0` la pérdida es exactamente `0.693` (= ln 2), una buena comprobación de cordura; con el valor por defecto (`1.0`) es ln 2 más la NLL de la respuesta elegida.

Todavía no hay números medidos para este script con los valores por defecto actuales.

### Ejemplos

**Lo que propone el README** (SFT con LoRA → DPO y GRPO, versión corta):

```powershell
.venv\Scripts\python scripts\finetune.py --lora --dpo-steps 200 --grpo-iters 40
```
```bash
.venv/bin/python scripts/finetune.py --lora --dpo-steps 200 --grpo-iters 40
```

**La receta completa por defecto** (SFT completo → DPO+NLL y GRPO; unos 5 minutos en CPU según el docstring):

```powershell
.venv\Scripts\python scripts\finetune.py
```
```bash
.venv/bin/python scripts/finetune.py
```

**Solo SFT completo** (la ejecución más rápida):

```bash
.venv/bin/python scripts/finetune.py --dpo-steps 0 --grpo-iters 0
```

**Comparar LoRA y fine-tuning completo** con el mismo presupuesto:

```powershell
.venv\Scripts\python scripts\finetune.py --dpo-steps 0 --grpo-iters 0
.venv\Scripts\python scripts\finetune.py --dpo-steps 0 --grpo-iters 0 --lora
```

**Ver el *likelihood displacement*** con DPO puro (sin ancla NLL) y comparar con la salida por defecto:

```powershell
.venv\Scripts\python scripts\finetune.py --nll-weight 0 --grpo-iters 0
```
```bash
.venv/bin/python scripts/finetune.py --nll-weight 0 --grpo-iters 0
```

### Notas y trampas

- **DPO y GRPO no se encadenan.** Son dos ramas independientes que parten del modelo SFT. La salida reporta `SFT+DPO(+NLL)` y `SFT+GRPO`, no `SFT+DPO+GRPO`. Para encadenarlos habría que pasar el modelo devuelto por `train_dpo` a `train_grpo` (ver [notebook 08](guia_notebooks.md)).
- **¿Por qué DPO + NLL por defecto?** En esta tarea elegida y rechazada difieren en un solo token. DPO puro puede aumentar el margen bajando la probabilidad de *las dos* respuestas (*likelihood displacement*), y el *exact match* empeora aunque `acc` suba. El ancla NLL (`--nll-weight 1.0`) mantiene alta la probabilidad de la respuesta correcta. Ver [referencia §7.3](referencia_modulos.md#7-finetune).
- **¿Por qué `train_embeddings=True` con `--lora`?** Los ids 507–511 (tokens de rol) no aparecen en el corpus de preentrenamiento, así que sus embeddings siguen siendo aleatorios. LoRA solo modifica las proyecciones lineales y no puede arreglarlos; por eso el script entrena también `tok_emb` (y el `lm_head` atado), como `modules_to_save` en `peft`.
- Es el script más pesado en CPU, sobre todo GRPO: cada iteración genera 64 respuestas token a token sin KV-cache. Reduce `--grpo-iters` para iterar rápido.- En esta implementación GRPO hace una única actualización por batch muestreado ($\mu = 1$), así que el *ratio* de importancia vale 1 y el *clipping* no llega a actuar (ver [referencia, §7.4](referencia_modulos.md#7-finetune)).
- Los 150 casos de evaluación se generan al azar en el mismo rango de tokens que el entrenamiento. Miden si se aprendió la *habilidad* (invertir), no si se memorizaron ejemplos concretos, porque la probabilidad de coincidencia exacta con un ejemplo de entrenamiento es despreciable.

---

## `rag_demo.py`

Indexa una carpeta de documentos `.txt` / `.md`, responde preguntas con **recuperación híbrida (denso + BM25 + RRF)** y un LLM, y opcionalmente evalúa la recuperación y la generación con un conjunto de preguntas de referencia. **No necesita PyTorch.** Sin `--model` usa `MockLLM`, que es offline y determinista.

### Argumentos

| Argumento | Tipo | Defecto | Descripción |
|---|---|---|---|
| `--docs` | ruta | **obligatorio** | Carpeta con ficheros `.txt` y/o `.md` (solo el primer nivel, no recursivo). |
| `--ask` | str | `None` | Pregunta a responder: muestra las fuentes y la respuesta. |
| `--eval` | ruta | `None` | JSON con una lista de `{"question", "doc", "answer"}`. |
| `--k` | int | `4` | Chunks recuperados que se meten en el prompt. |
| `--chunk-size` | int | `120` | Tamaño del chunk en palabras. El solapamiento es `chunk_size // 4`. |
| `--model` | str | `None` | Nombre del modelo en un servidor compatible con OpenAI. Si se omite, se usa `MockLLM`. |
| `--base-url` | str | `http://localhost:8000/v1` | URL del servidor (vLLM por defecto; Ollama: `http://localhost:11434/v1`). |

Se pueden pasar `--ask` y `--eval` a la vez. Si no se pasa ninguno, el script solo indexa e imprime el resumen.

### Qué hace, etapa por etapa

1. **Carga.** Busca `*.txt` y `*.md` en `--docs`. El `doc_id` de cada documento es el **nombre del fichero** (sin la ruta).
2. **Chunking.** `chunk_documents(docs, size=--chunk-size, overlap=--chunk-size // 4)`: ventanas de palabras con solapamiento.
3. **Índice.** `Retriever(chunks)` con la configuración por defecto: `HashedTfidfEmbedder` ajustado a los chunks y `VectorIndex` (coseno exacto) para la parte densa, más `BM25` para la léxica, sin *reranker*.
4. **LLM.** Usa `OpenAICompatibleClient(--model, base_url=--base-url)` si se pasa `--model`; si no, `MockLLM()`. Imprime `indexed N docs -> M chunks | llm = ...`.
5. **`--ask`.** `retriever.retrieve(q, k=--k)` (top-20 denso y top-20 BM25, fusión RRF, top-k) → `build_rag_prompt` (contexto numerado + instrucciones de citar) → `llm.complete`. Muestra `SOURCES` (`[n] doc#chunk: primeros 100 caracteres...`) y `ANSWER`.
6. **`--eval`.** Para cada pregunta:
   - *Recuperación*: la lista ordenada de `doc_id` **únicos** del top-k frente al conjunto relevante `{doc}`. Se agrega con `evaluate_ranking(ks=(1, 3))`, que da `recall@1`, `hit@1`, `ndcg@1`, `recall@3`, `hit@3`, `ndcg@3` y `mrr`.
   - *Generación*: `token_f1(respuesta, answer)` y `context_support(respuesta, contexto recuperado)`, un proxy de fidelidad que mide qué parte de la respuesta está respaldada por el contexto.
   - Imprime las métricas de recuperación (redondeadas a 3 decimales) y la media de token-F1 y de *context support*.

### Ejemplos con los documentos de muestra

El repositorio incluye `data/sample_docs/` (cuatro documentos cortos: `refund_policy.md`, `security.md`, `shipping.md`, `support.md`) y `data/sample_eval.json` (seis preguntas con su documento y respuesta de referencia).

**Hacer una pregunta:**

```powershell
.venv\Scripts\python scripts\rag_demo.py --docs data\sample_docs --ask "What is the refund policy?"
```
```bash
.venv/bin/python scripts/rag_demo.py --docs data/sample_docs --ask "What is the refund policy?"
```

**Evaluar:**

```powershell
.venv\Scripts\python scripts\rag_demo.py --docs data\sample_docs --eval data\sample_eval.json
```
```bash
.venv/bin/python scripts/rag_demo.py --docs data/sample_docs --eval data/sample_eval.json
```

**Resultado medido (con `MockLLM`):** `recall@1 = 1.0`, `MRR = 1.0`, `token-F1 = 0.887`.

Cómo leer estos números:

- **recall@1 = 1.0 y MRR = 1.0**: en las 6 preguntas, el documento correcto sale *primero*. Cada documento de muestra tiene unas 60 palabras (menos que `--chunk-size 120`), así que forma un único chunk y el índice tiene 4. Con `--k 4` se recuperan siempre los cuatro documentos, por lo que `recall@3` es trivialmente 1. La métrica informativa aquí es la **posición** (recall@1, MRR, nDCG).
- **token-F1 = 0.887 (y no 1.0)**: `MockLLM` responde con la frase del contexto que más palabras comparte con la pregunta, seguida de la cita `[n]`. Esa frase suele coincidir casi exactamente con la referencia, pero la cita añade un token y algunas frases son más largas que la respuesta esperada (p. ej. "… 8am to 8pm *Central European Time*."). Ambos efectos bajan la precisión del F1.
- Con un LLM real, el F1 suele *bajar* porque el modelo parafrasea. Por eso conviene complementar con `context_support` y con un juez LLM ([notebook 10](guia_notebooks.md)).

**Usar tus propios documentos:**

```bash
.venv/bin/python scripts/rag_demo.py --docs ./mis_docs --ask "¿Cuál es la política de devoluciones?" --k 3 --chunk-size 200
```

El formato de `--eval` es:

```json
[
  {"question": "How long does express shipping take?", "doc": "shipping.md",
   "answer": "Express shipping delivers in 1 to 2 business days."}
]
```

`doc` debe coincidir exactamente con el nombre de fichero dentro de `--docs`. `answer` es opcional: si falta, el F1 de esa pregunta es 0.

**Con un LLM real** a través de un servidor compatible con OpenAI. Requiere `pip install requests`, que viene comentado en `requirements.txt` e incluido en el extra `rag` de `pyproject.toml`.

```powershell
# vLLM (en otra terminal / máquina con GPU):  vllm serve Qwen/Qwen2.5-7B-Instruct
.venv\Scripts\python scripts\rag_demo.py --docs data\sample_docs --eval data\sample_eval.json --model Qwen/Qwen2.5-7B-Instruct --base-url http://localhost:8000/v1

# Ollama:  ollama serve  (y antes: ollama pull llama3.1)
.venv\Scripts\python scripts\rag_demo.py --docs data\sample_docs --ask "Are gift cards refundable?" --model llama3.1 --base-url http://localhost:11434/v1
```
```bash
.venv/bin/python scripts/rag_demo.py --docs data/sample_docs --eval data/sample_eval.json \
    --model Qwen/Qwen2.5-7B-Instruct --base-url http://localhost:8000/v1
```

Para un proveedor remoto, la clave se lee de la variable de entorno `OPENAI_API_KEY`. Por ejemplo, `$env:OPENAI_API_KEY="sk-..."` en PowerShell o `export OPENAI_API_KEY=sk-...` en bash, con `--base-url https://api.openai.com/v1`.

### Notas y trampas

- La carpeta del ejemplo del README (`./my_docs`) **no existe** en el repositorio. Usa `data/sample_docs` o tu propia carpeta.
- El embedder por defecto (`HashedTfidfEmbedder`) es léxico: no entiende sinónimos ni otros idiomas. Una pregunta en español sobre documentos en inglés recuperará mal. Para embeddings reales, usa `SentenceTransformerEmbedder` desde Python (ver [referencia §10](referencia_modulos.md#10-rag)), que el script no expone por CLI.
- `MockLLM` busca la frase con más solapamiento léxico, no razona. Sirve para comprobar que el *pipeline* (recuperar → prompt → citar → evaluar) funciona de extremo a extremo, no para medir la calidad de generación.

---

## `mini_llm.py`

Un **mini LLM completo en un solo fichero** que solo depende de PyTorch (no importa nada de `llm_edu_suite`): tokenizador BPE a nivel de bytes → GPT con RoPE, RMSNorm y SwiGLU → entrenamiento con AdamW y schedule coseno → evaluación frente a la pérdida irreducible del corpus → muestreo → guardado y recarga del checkpoint. Es el notebook 00 (`notebooks/00_start_here/00_mini_llm_end_to_end.ipynb`) exportado como script: el pipeline entero de un vistazo, antes de que los notebooks desmonten cada pieza y el paquete la escale.

> **Fichero generado.** `scripts/mini_llm.py` se genera a partir del notebook 00 con `tools/build_notebooks_part3.py`. Para cambiarlo de forma permanente, **edita el generador**, no el script: el siguiente `make notebooks` lo sobrescribe.

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

## `mcp_server.py`

Un **servidor MCP real sobre stdio**, listo para ejecutar, construido con [`llm_edu_suite.mcp`](referencia_modulos.md#18-mcp) y sin SDK. Expone la base de conocimiento de ejemplo de `data/sample_docs/` (o cualquier carpeta con ficheros `.md` / `.txt`) a cualquier cliente MCP: Claude Code, Claude Desktop, el MCP Inspector, o `MCPClient` + `StdioTransport` desde Python. No necesita PyTorch.

### Argumentos

| Argumento | Tipo | Por defecto | Descripción |
|---|---|---|---|
| `--docs` | ruta | `data/sample_docs` | Carpeta con los ficheros `.md` / `.txt` que se exponen (solo el primer nivel). |

### Qué expone

| Primitiva | Nombre | Descripción |
|---|---|---|
| tool | `search_docs(query, k=3)` | Recuperación híbrida BM25 + densa (`rag.Retriever`, fragmentos de 80 palabras con 20 de solapamiento). Devuelve líneas `[fichero] pasaje`. `readOnlyHint: true`. |
| tool | `calculate(expression)` | Aritmética segura: solo se aceptan dígitos, `+ - * / ( ) .` y espacios; cualquier otra cosa es un error de herramienta. `readOnlyHint: true`. |
| tool | `create_ticket(title, priority="normal")` | Acción de escritura simulada (`priority` ∈ `low \| normal \| high`); devuelve el ticket como JSON (también en `structuredContent`). Anotada con `destructiveHint: true`, así que un cliente bien hecho debería **pedir confirmación**. |
| resources | `docs://<fichero>` | Un recurso por fichero (`text/markdown`), descrito por su primera línea. |
| prompt | `answer_with_sources(question)` | Plantilla RAG: usar `search_docs`, responder solo con los pasajes, citar los nombres de fichero y decirlo si la respuesta no está. |

El servidor envía además unas `instructions` en su respuesta a `initialize` ("usa `search_docs` antes de responder preguntas sobre políticas y cita los nombres de fichero").

### Ejecutarlo y conectarlo

El servidor espera mensajes JSON-RPC por stdin (Ctrl+C para pararlo). Al arrancar escribe una línea en **stderr**: `[mcp_server] 3 tools, 4 resources, 1 prompts — waiting on stdin`.

```powershell
.venv\Scripts\python scripts\mcp_server.py
.venv\Scripts\python scripts\mcp_server.py --docs C:\ruta\a\tus\docs
```
```bash
.venv/bin/python scripts/mcp_server.py          # o bien: make mcp-server
.venv/bin/python scripts/mcp_server.py --docs ruta/a/tus/docs
```

**Claude Code:**

```bash
claude mcp add llm_edu_suite-docs -- python /ABSOLUTE/PATH/scripts/mcp_server.py
```

**Claude Desktop** (`claude_desktop_config.json`):

```json
{"mcpServers": {"llm_edu_suite-docs": {"command": "python", "args": ["/ABSOLUTE/PATH/scripts/mcp_server.py"]}}}
```

**MCP Inspector** (interfaz web oficial para probar servidores; necesita Node.js):

```bash
npx @modelcontextprotocol/inspector python scripts/mcp_server.py
```

**Desde Python**, con el cliente del propio paquete:

```python
import sys
from llm_edu_suite.mcp import MCPClient, StdioTransport, result_text

client = MCPClient(StdioTransport([sys.executable, "scripts/mcp_server.py"]))
client.initialize()
print(result_text(client.call_tool("search_docs", {"query": "express shipping", "k": 1})))   # [shipping.md] ...
print(result_text(client.call_tool("calculate", {"expression": "6*7"})))                     # 42
client.close()
```

### Notas y trampas

- **Nunca imprimas en stdout.** En un servidor stdio, stdout transporta el protocolo; un solo `print()` perdido corrompe el flujo y el cliente no consigue parsear el siguiente mensaje. Todo lo que registra el servidor (arranque, tickets creados) va a `sys.stderr`.
- Usa **rutas absolutas** en la configuración de Claude Code y Claude Desktop: el host lanza el proceso desde su propio directorio de trabajo. Usa el intérprete del entorno virtual del proyecto (p. ej. `/ABSOLUTE/PATH/.venv/bin/python` o `.venv\Scripts\python.exe`) si el `python` del `PATH` no tiene las dependencias (NumPy).
- Los tickets viven en memoria y desaparecen al terminar el proceso. `create_ticket` hace de sustituto de una escritura real.
- El recuperador es léxico (`HashedTfidfEmbedder` + BM25), como en [`rag_demo.py`](#rag_demopy): no entiende sinónimos ni otros idiomas.

---

## Herramientas y objetivos del Makefile

**`tools/run_notebooks.py`** ejecuta notebooks sin interfaz y **en el sitio** (conservan sus salidas), cada uno con su propia carpeta como directorio de trabajo, igual que Jupyter. Busca en `notebooks/` de forma **recursiva**, así que encuentra todos los notebooks de las carpetas temáticas. Los argumentos pueden ser nombres de fichero o prefijos numéricos:

```bash
python tools/run_notebooks.py                       # todos los notebooks, por orden de número
python tools/run_notebooks.py 05_pretraining.ipynb  # por nombre de fichero, esté donde esté
python tools/run_notebooks.py 05 10 14              # por número (prefijo)
```

Imprime `[segundos] nombre: OK | FAIL` por notebook y `N/M notebooks OK` al final, y sale con código 1 si alguno falla, de modo que sirve para CI. Un argumento que coincide con cero o con varios notebooks detiene la ejecución con un error.

**Objetivos del Makefile** (Linux/macOS o Git Bash con `make`):

| Objetivo | Comando | Qué hace |
|---|---|---|
| `make setup` | `bash setup_env.sh` | Crea `.venv` e instala todo (en Windows, ejecuta `setup_env.ps1`). |
| `make test` | `pytest -q` | Tests unitarios y de comportamiento (~15 s). |
| `make notebooks` | `build_notebooks_part1.py`, `part2`, `part3` | Regenera todos los `.ipynb` (y `scripts/mini_llm.py`) desde los generadores. **Borra las salidas.** |
| `make run-notebooks` | `python tools/run_notebooks.py` | Ejecuta todos los notebooks en el sitio (~40 min en CPU). |
| `make smoke` | `train.py --steps 100 --d-model 64 --n-layers 2 ...` | Prueba de humo de preentrenamiento de 30 segundos. |
| `make mini-llm` | `python scripts/mini_llm.py` | Entrena el mini GPT autocontenido (el notebook 00 como script). |
| `make mcp-server` | `python scripts/mcp_server.py` | Arranca el servidor MCP de demostración por stdio. |

---

## Resumen de resultados medidos (CPU salvo indicación)

| Comando | Resultado |
|---|---|
| `train.py --steps 300 --out runs/smoke` | 3.03M parámetros, perplejidad de validación **2.65**, ≈ 140 s |
| `sample.py --run runs/smoke --max-new-tokens 50` | ≈ **200 tok/s** (salida en ids porque el corpus es sintético) |
| `rag_demo.py --docs data/sample_docs --eval data/sample_eval.json` | recall@1 **1.0**, MRR **1.0**, token-F1 **0.887** (`MockLLM`) |
| `mini_llm.py` (800 pasos por defecto, RTX 3060 Laptop GPU; ~3–5 min en CPU) | 1.86M parámetros, vocabulario 455 (199 fusiones), ~22 s de entrenamiento, pérdida de validación **0.778** frente a un suelo del corpus de **0.774** (un 0.6% por encima) |

Para el contexto pedagógico de cada etapa, consulta la [guía de notebooks](guia_notebooks.md). Para un recorrido de principio a fin con interpretación de resultados, el [caso de estudio](caso_de_estudio.md).
