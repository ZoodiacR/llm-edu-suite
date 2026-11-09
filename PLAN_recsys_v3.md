# Plan: sistema de recomendación v3 (LLM + arquitectura industrial)

> Tareas para la próxima sesión. Objetivo: rehacer el notebook 12 para que el LLM **sume**, no reste, usando la arquitectura que usan Netflix, YouTube y Spotify, en versión toy pero funcional y medida.

## Por qué el actual falla con el LLM (diagnóstico)

- El catálogo es **sintético**: títulos inventados y "gustos" que son clusters ocultos. El LLM no tiene ningún conocimiento del mundo que aportar, así que su re-ranking solo mete ruido (hit@10 0.20 → 0.16).
- El LLM **reemplaza** el ranking de SASRec en lugar de combinarse con él.
- Solo medimos hit@10 global; no medimos donde un LLM sí ayuda (usuarios nuevos, ítems sin historial, explicaciones, diversidad).

## Regla de oro (para que no salga peor)

Cada etapa nueva solo se activa si **mejora nDCG@10 en validación** respecto a la etapa anterior. Si no mejora, el pipeline usa la anterior (fallback) y el notebook lo reporta con honestidad. Las señales del LLM entran como **features** de un ranker aprendido, nunca como sustituto ciego.

## Arquitectura objetivo (embudo multi-etapa, como YouTube / Netflix)

```
catálogo + interacciones
   │
   ├─ 1. Generación de candidatos (varias fuentes, ~200 por usuario)
   │     · two-tower (usuario/ítem) + FAISS          ← YouTube, Spotify
   │     · SASRec secuencial (el actual)              ← Netflix/Alibaba (transformers sobre secuencias)
   │     · embeddings de contenido con nomic-embed    ← cold start
   │     · popularidad / tendencias
   ├─ 2. Ranking: LightGBM LambdaMART con features de todas las fuentes + features LLM
   ├─ 3. Re-ranking: diversidad (MMR) + reglas de negocio + exploración (bandit)  ← Spotify
   └─ 4. Capa LLM visible: explicaciones y recomendación conversacional  ← Netflix/Spotify AI DJ
```

## Tareas

### Fase 0: datos reales (clave para que el LLM aporte)
- [ ] Usar **MovieLens `ml-latest-small`** (100k ratings, 9k películas con título real, año y géneros, ~1 MB). Descargarlo en tiempo de ejecución a `data/cache/` (la licencia de GroupLens no permite redistribuirlo: **no commitearlo**, añadir a `.gitignore`).
- [ ] Interacciones implícitas (rating ≥ 4 = positivo), secuencias por usuario ordenadas por timestamp.
- [ ] **Split temporal** (train/valid/test por tiempo, no aleatorio) y un segmento **cold-start** (usuarios con ≤ 5 interacciones; ítems nuevos en test).
- [ ] Mantener el catálogo sintético actual solo para tests rápidos de CI.

### Fase 1: baselines y métricas
- [ ] Baselines: popularidad, ItemKNN (co-ocurrencia), SASRec actual adaptado.
- [ ] Métricas: hit@10, nDCG@10, MRR, **coverage** del catálogo, **diversidad** intra-lista, **novedad**; todas por segmento (todos / cold-start / heavy users).
- [ ] Tabla de referencia guardada para comparar cada fase.

### Fase 2: generación de candidatos
- [ ] **Two-tower** en PyTorch (torre usuario = media/atención sobre historial; torre ítem = id + géneros + embedding de contenido), loss in-batch softmax con corrección de popularidad. Índice **FAISS** para ANN.
- [ ] **Embeddings de contenido** con `OllamaEmbeddings(nomic-embed-text)` sobre "título + año + géneros (+ sinopsis generada, fase 4)".
- [ ] Unión de candidatos y medir **recall@200** de la unión frente a cada fuente sola (el ranker no puede recuperar lo que el retrieval no trajo).

### Fase 3: ranker aprendido
- [ ] **LightGBM `LGBMRanker` (lambdarank)** con features: score/rank de cada fuente, popularidad, afinidad de género, recencia, similitud de contenido con el historial.
- [ ] Ablation: quitar cada grupo de features y ver cuánto cae nDCG@10.
- [ ] Criterio: el ranker debe superar a la mejor fuente individual. Si no, investigar antes de seguir.

### Fase 4: el LLM donde sí ayuda (`qwen2.5:3b` vía LangChain)
- [ ] **Enriquecer ítems**: el LLM genera para cada película un perfil corto (tono, temas, público) con `with_structured_output`; cachearlo en disco. Embeddings de esos perfiles como feature → medir impacto en **cold-start**.
- [ ] **Perfil de usuario en lenguaje natural** ("le gusta la ciencia ficción de los 90...") generado desde su historial → similitud con perfiles de ítems como feature del ranker.
- [ ] **Re-ranking LLM como feature**, solo sobre top-20, y combinado (blend) con el score del ranker; validar el peso en valid. Se activa solo si mejora.
- [ ] **Recomendación conversacional**: "algo como Interstellar pero más corta" → el LLM extrae filtros/intención (structured output) → retrieval + ranker → respuesta. Pequeño eval de 20 consultas con criterios verificables.
- [ ] **Explicaciones** fundamentadas ("porque viste X y Y") y medir con un juez que la explicación cita ítems reales del historial (groundedness).
- [ ] Trazar todo con LangSmith.

### Fase 5: re-ranking final y exploración (estilo Spotify)
- [ ] **MMR** para diversidad: curva nDCG vs diversidad según lambda.
- [ ] **Bandit** contextual simple (Thompson sampling / epsilon-greedy) simulado offline para explorar ítems nuevos; medir regret en una simulación.

### Fase 6: cierre
- [ ] Tabla final por etapa y segmento, con gráficas (embudo, ablation, curva diversidad).
- [ ] Notebook regenerado desde `tools/notebooks/nb12_*.py`, ejecutado con la GPU, prosa contrastada con las salidas (`check_numbers.py`).
- [ ] Mover lógica reutilizable a `llm_edu_suite/recsys.py` + tests (sin Ollama en CI, con modelos falsos).
- [ ] Actualizar README, guía de notebooks, caso de estudio (ES + EN).
- [ ] Opcional avanzado: **semantic IDs / generative retrieval** (tipo TIGER de Google) en versión mínima.

## Expectativas realistas

- En MovieLens, SASRec bien ajustado ya es un baseline fuerte; la mejora global del ranker será **moderada** (unos puntos de nDCG).
- Donde el LLM debería notarse de verdad: **cold-start**, **explicaciones**, **búsqueda conversacional** y **diversidad**. El notebook lo medirá por segmento y reportará lo que salga, aunque alguna parte no ayude.
- Presupuesto: GPU de 6 GB (two-tower y SASRec caben de sobra); enriquecer 9k ítems con el LLM son varios minutos → cachear y permitir usar un subconjunto.

## Referencias para inspirarse

- Covington et al., *Deep Neural Networks for YouTube Recommendations* (2016): embudo candidatos → ranking.
- Yi et al., *Sampling-Bias-Corrected Neural Modeling* (two-tower, Google 2019).
- Kang & McAuley, *SASRec* (2018); Sun et al., *BERT4Rec* (2019).
- Netflix Tech Blog: *Foundation Model for Personalized Recommendation* (2025).
- Spotify Research: *BaRT* (bandits for recommendations) y su AI DJ.
- Rajput et al., *Recommender Systems with Generative Retrieval* (TIGER, 2023).
