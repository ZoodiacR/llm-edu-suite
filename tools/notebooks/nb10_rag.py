"""Notebook 10 - RAG with LangChain, evaluated with RAGAS

Builds 10_rag.ipynb. Run: python tools/build_notebooks.py 10
"""
from _common import *  # noqa: F401,F403

nb = NB()
nb.md(r'''
# 10 · Retrieval-Augmented Generation with LangChain, evaluated with RAGAS

A language model only knows what was in its training data, and it cannot tell you where an answer came from. **RAG** keeps the knowledge in a searchable store you control and lets the model do the reasoning over what you retrieve:

```
offline:  documents ─► load ─► split into chunks ─► embed ─► vector index (FAISS)
                                                 └─────────► keyword index (BM25)
online:   question ─► dense search ──┐
                   ─► BM25 search ───┴─► RRF fusion ─► top-k chunks ─► prompt with numbered sources ─► LLM ─► answer [1][3]
```

We build it with the tools most teams use today:

| piece | tool |
|---|---|
| loading and splitting | `langchain_community` loaders, `langchain_text_splitters.RecursiveCharacterTextSplitter` |
| embeddings and LLM | `OllamaEmbeddings(nomic-embed-text)` and `ChatOllama(qwen2.5:3b)`, through `llm_edu_suite.config` |
| vector store | FAISS (`langchain_community.vectorstores.FAISS`) |
| keyword search and fusion | `BM25Retriever` and `EnsembleRetriever` (Reciprocal Rank Fusion) |
| the pipeline | LCEL (LangChain Expression Language) runnables, traced in LangSmith when a key is set |
| evaluation | our own labelled set for retrieval, and **RAGAS** for generation |

**The corpus** is the course's own documentation (`data/knowledge_base/*.md`, a snapshot of the docs: installation, glossary, CLI scripts, module reference and notebooks guide, about 190 KB). It is a realistic target: technical prose, tables, code blocks and many exact strings such as flag names and numbers.

**What you will learn:** what each component does and how to tune it, and above all how to **evaluate retrieval and generation separately**, so that when the system is wrong you know which half to fix.
''')
nb.code(setup_apps("llm-edu-suite-10-rag"))
nb.md(r'''
## 1. Loading documents

A LangChain `Document` is just `page_content` (a string) plus a `metadata` dict. Loaders exist for PDFs, web pages, Notion, S3 and hundreds of other sources; they all produce the same object, so the rest of the pipeline never changes. The metadata is what lets us **cite sources** later, so we keep the file name in `metadata["source"]`.
''')
nb.code(r'''
from langchain_community.document_loaders import DirectoryLoader, TextLoader

KB_DIR = os.path.join(ROOT, "data", "knowledge_base")
docs = DirectoryLoader(KB_DIR, glob="*.md", loader_cls=TextLoader, loader_kwargs={"encoding": "utf-8"}).load()
for d in docs:
    d.metadata["source"] = os.path.basename(d.metadata["source"])     # keep only the file name, it is what we cite
docs.sort(key=lambda d: d.metadata["source"])
for d in docs:
    print(f"{d.metadata['source']:22s} {len(d.page_content):7,d} characters")
print("\nfirst 200 characters of", docs[2].metadata["source"], ":\n", docs[2].page_content[:200])
''')
nb.md(r'''
## 2. Splitting into chunks

We cannot embed a 110 KB file as one vector: an embedding is a fixed-size summary, and the longer the text, the more detail it averages away. We also cannot put the whole corpus in the prompt of a 3B model with a 4k-token window. So we split documents into **chunks** and retrieve only the few that matter.

**How `RecursiveCharacterTextSplitter` works.** It has a list of separators, from the most to the least meaningful: `"\n\n"` (paragraphs), `"\n"` (lines), `" "` (words), `""` (characters). It splits on the first one, and any piece still longer than `chunk_size` is split again with the next separator, then neighbouring pieces are merged back up to `chunk_size`. The effect is that chunks break at paragraph boundaries when they can and mid-sentence only when they must. `chunk_overlap` repeats the end of one chunk at the start of the next, so a fact that straddles a boundary is complete in at least one of them.

**The chunk-size trade-off.**
- *Small chunks* (a few sentences) give precise embeddings and short prompts, but a chunk may lose the context that makes it meaningful (a table row without its header, a pronoun without its noun).
- *Large chunks* keep context but dilute the embedding (one vector for many topics) and spend the prompt budget on irrelevant text.

There is no universal right size: it depends on the documents and the questions, so we will **measure** three sizes in the evaluation. `chunk_size` here counts characters (about 4 per token for English).
''')
nb.code(r'''
from langchain_text_splitters import RecursiveCharacterTextSplitter

def split(docs, size):
    splitter = RecursiveCharacterTextSplitter(chunk_size=size, chunk_overlap=size // 8, add_start_index=True)
    return splitter.split_documents(docs)

CHUNK_SIZES = (300, 800, 1500)
chunk_sets = {size: split(docs, size) for size in CHUNK_SIZES}
for size, chunks in chunk_sets.items():
    lens = [len(c.page_content) for c in chunks]
    print(f"chunk_size {size:5d}: {len(chunks):4d} chunks, mean length {np.mean(lens):6.0f} chars, median {np.median(lens):5.0f}")

chunks = chunk_sets[800]                         # the default configuration for the rest of the notebook
print("\nexample chunk, metadata:", chunks[40].metadata)
print(chunks[40].page_content[:400])
''')
nb.md(r'''
Many chunks are shorter than `chunk_size`: Markdown has many blank lines, so the splitter often stops at a paragraph or table boundary before reaching the limit. That is the behaviour we want.

## 3. Embeddings

An **embedding model** maps a text to a vector (768 numbers for `nomic-embed-text`) so that texts with similar meaning point in similar directions. It is trained contrastively: pairs that belong together (a question and the passage that answers it) are pulled together, random pairs are pushed apart. Similarity is the **cosine** of the angle between two vectors, which for unit-length vectors is just the dot product.

Notice in the output that a paraphrase with no word in common still scores high. That is what keyword search cannot do.
''')
nb.code(r'''
emb = get_embeddings()
sentences = ["How do I lower the memory used by the attention cache?",
             "Grouped-query attention shares each K/V head across several query heads to shrink the KV-cache.",
             "Run Set-ExecutionPolicy once if PowerShell blocks the activation script."]
V = np.array(emb.embed_documents(sentences))
print("embedding shape:", V.shape, "| vector norms:", np.round(np.linalg.norm(V, axis=1), 3))
Vn = V / np.linalg.norm(V, axis=1, keepdims=True)
print("cosine(question, GQA sentence)        =", round(float(Vn[0] @ Vn[1]), 3))
print("cosine(question, PowerShell sentence) =", round(float(Vn[0] @ Vn[2]), 3))
''')
nb.md(r'''
## 4. A vector store: FAISS

A vector store keeps the chunk vectors and answers "which k vectors are closest to this query vector?". **FAISS** (Meta) is the standard library for it. LangChain's `FAISS` wrapper uses an exact `IndexFlatL2` index by default: it compares the query with every vector, which is perfectly fast for a few thousand chunks. With millions of vectors you would switch to an approximate index (HNSW, IVF) or a database such as pgvector or Qdrant; the retriever interface stays the same.

FAISS returns an **L2 distance** (lower = closer). The vectors are unit length, so `L2² = 2 − 2·cosine` and the ranking is identical to ranking by cosine.

One practical detail: we embed in batches of 64. Sending all chunks in one request can overload a local embedding server (it crashed Ollama in our tests), and hosted APIs cap the batch size anyway.
''')
nb.code(r'''
from langchain_community.vectorstores import FAISS

def build_faiss(chunks, embeddings, batch=64):
    store = FAISS.from_documents(chunks[:batch], embeddings)
    for i in range(batch, len(chunks), batch):
        store.add_documents(chunks[i:i + batch])
    return store

t0 = time.time()
stores = {size: build_faiss(cs, emb) for size, cs in chunk_sets.items()}
print(f"embedded {sum(map(len, chunk_sets.values()))} chunks for the three chunk sizes in {time.time() - t0:.1f}s")
store = stores[800]

q = "How much KV-cache memory does Llama-3-8B need per token?"
for doc, dist in store.similarity_search_with_score(q, k=3):
    print(f"L2 {dist:.3f}  {doc.metadata['source']:20s} {doc.page_content[:110]!r}")
''')
nb.md(r'''
## 5. Keyword search: BM25

Dense retrieval is weak on **exact strings**: a flag name like `--chunk-size`, an error message, a version number. Embeddings blur them into "something about configuration". **BM25**, the classic search-engine ranking function, scores a chunk by the query terms it contains:

$$\text{BM25}(q, d) = \sum_{t \in q} \text{IDF}(t)\cdot\frac{f(t,d)\,(k_1+1)}{f(t,d) + k_1\,(1 - b + b\,|d|/\text{avgdl})}$$

Rare terms (high IDF) count much more than common ones, repeated terms saturate (controlled by $k_1$), and long chunks are penalised (controlled by $b$). It needs no model and no GPU.

`BM25Retriever` splits text on whitespace by default, so `"Llama-3-8B?"` and `"llama-3-8b"` would be different tokens. A tokenizer that lowercases and splits on word characters fixes that; it is the kind of small detail that changes lexical retrieval quality noticeably.
''')
nb.code(r'''
import re
from langchain_community.retrievers import BM25Retriever

def tokenize(text):
    return re.findall(r"\w+", text.lower())

K = 4                                        # chunks per answer; four 800-character chunks are ~800 tokens of context
bm25 = BM25Retriever.from_documents(chunks, k=K, preprocess_func=tokenize)
dense = store.as_retriever(search_kwargs={"k": K})

q_exact = "What does chunk_size // 4 control?"
print("BM25 :", [(d.metadata["source"], d.page_content[:60]) for d in bm25.invoke(q_exact)[:2]])
print("dense:", [(d.metadata["source"], d.page_content[:60]) for d in dense.invoke(q_exact)[:2]])
''')
nb.md(r'''
## 6. Hybrid retrieval with Reciprocal Rank Fusion

Dense and BM25 fail on different questions, so we run both and merge the rankings. Their scores are on incomparable scales (an L2 distance versus an unbounded BM25 sum), so instead of adding scores we add **ranks**:

$$\text{RRF}(d) = \sum_{r \in \text{retrievers}} w_r \cdot \frac{1}{c + \text{rank}_r(d)}, \qquad c = 60$$

A chunk ranked high by either retriever rises; a chunk ranked high by both rises most. The constant $c$ damps the difference between rank 1 and rank 2 so that one retriever cannot dominate. LangChain's `EnsembleRetriever` implements exactly this (it lives in `langchain_classic` in LangChain 1.x). It returns the union of both lists, so we keep the first `K`.

Below, a short "under the hood" cell re-implements RRF in six lines and checks it gives the same order.
''')
nb.code(r'''
from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.runnables import RunnableLambda

ensemble = EnsembleRetriever(retrievers=[bm25, dense], weights=[0.5, 0.5])
hybrid = ensemble | RunnableLambda(lambda ds: ds[:K])        # an LCEL pipe: retriever, then keep the top K

# Under the hood: weighted Reciprocal Rank Fusion, keyed on the chunk text.
def rrf(rankings, weights=None, c=60):
    weights = weights or [1.0] * len(rankings)
    scores, by_text = {}, {}
    for w, ranking in zip(weights, rankings):
        for rank, d in enumerate(ranking, start=1):
            scores[d.page_content] = scores.get(d.page_content, 0.0) + w / (c + rank)
            by_text[d.page_content] = d
    return [by_text[t] for t in sorted(scores, key=scores.get, reverse=True)]

mine = rrf([bm25.invoke(q), dense.invoke(q)], [0.5, 0.5])
theirs = ensemble.invoke(q)
print("same order as EnsembleRetriever:", [d.page_content for d in mine] == [d.page_content for d in theirs])
for d in hybrid.invoke(q):
    print(f"  {d.metadata['source']:20s} {d.page_content[:90]!r}")
''')
nb.md(r'''
## 7. The RAG chain in LCEL, with citations

**LCEL** composes *runnables* (prompts, models, retrievers, parsers, plain functions) with `|`. Every runnable has the same interface (`invoke`, `batch`, `stream` and their async versions), so the whole chain gets batching, streaming and LangSmith tracing for free. Two building blocks matter here:
- `RunnableParallel` (a dict of runnables) runs branches on the same input and returns a dict;
- `.assign(...)` adds a key computed from the keys already there, so the retrieved documents travel with the answer and we can show the sources.

The prompt numbers each chunk and asks for `[n]` citations, and tells the model to say so when the context does not contain the answer. Citations make an answer **checkable**, which is the point of RAG.
''')
nb.code(r'''
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

llm = get_llm()

RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You answer questions about a software course using ONLY the numbered context passages. "
               "Cite the passages you used with their numbers in square brackets, like [1] or [2][3]. "
               "Be concise: one to three sentences. If the context does not contain the answer, say \"I don't know\"."),
    ("human", "Context:\n{context}\n\nQuestion: {question}"),
])

def format_docs(docs):
    return "\n\n".join(f"[{i}] (source: {d.metadata['source']})\n{d.page_content}" for i, d in enumerate(docs, start=1))

def make_rag_chain(retriever):
    answer = ({"context": lambda x: format_docs(x["docs"]), "question": lambda x: x["question"]}
              | RAG_PROMPT | llm | StrOutputParser())
    return RunnableParallel(docs=retriever, question=RunnablePassthrough()).assign(answer=answer)

rag = make_rag_chain(hybrid)

def show(result):
    print("Q:", result["question"]); print("A:", result["answer"])
    cited = sorted({int(n) for n in re.findall(r"\[(\d+)\]", result["answer"])})
    for i, d in enumerate(result["docs"], start=1):
        print(f"   [{i}] {'*' if i in cited else ' '} {d.metadata['source']:20s} {d.page_content[:70]!r}")

show(rag.invoke("Why must an MCP server that uses the stdio transport never print to stdout?"))
print("\nThe same question without retrieval, for comparison:")
print("A:", llm.invoke("Why must an MCP server that uses the stdio transport never print to stdout? Answer in one or two sentences.").content)
''')
nb.md(r'''
The starred sources are the ones the answer cites. When LangSmith is enabled, each `invoke` appears as one trace with a child span for the retrievers, the prompt and the model call, so you can see exactly which chunks the model was shown.

## 8. Query rewriting: multi-query (RAG-Fusion)

Users phrase questions differently from documents. **Multi-query** retrieval asks the LLM for a few rephrasings, retrieves for each, and fuses all the rankings with RRF ("RAG-Fusion"). It costs one extra LLM call per question. LangChain ships a ready-made `MultiQueryRetriever` (in `langchain_classic.retrievers`), but it concatenates the results instead of fusing them, so we build the fused version in a few lines of LCEL, reusing `rrf` from section 6. We will see in the evaluation whether it is worth the extra call here.
''')
nb.code(r'''
REWRITE_PROMPT = ChatPromptTemplate.from_template(
    "Write 3 different search queries that would help find documentation passages answering this question. "
    "Use different wording and keywords. One query per line, no numbering, nothing else.\n\nQuestion: {question}")
rewrite = REWRITE_PROMPT | llm | StrOutputParser() | (lambda text: [l.strip(" -*0123456789.") for l in text.splitlines() if l.strip()][:3])

def multi_query_retrieve(question):
    queries = [question] + rewrite.invoke({"question": question})
    rankings = ensemble.batch(queries)                 # one hybrid search per query, run in parallel
    return rrf(rankings)[:K]

multi_query = RunnableLambda(multi_query_retrieve)
print(rewrite.invoke({"question": "Why does get_device not select Apple MPS by default?"}))
''')
nb.md(r'''
## 9. The evaluation set

You cannot improve what you do not measure, and public benchmarks say little about *your* documents. So the first real step of a RAG project is a small labelled set. Ours has 20 questions written by reading the knowledge base (`data/eval/rag_eval.json`), four per document. Each item has:
- `answer`: a reference answer, correct according to the docs;
- `source` and `sources`: the file that answers it, plus any other file that also states the fact;
- `evidence`: one or more short verbatim strings that a chunk must contain to actually answer the question.

The questions mix concepts ("how does RMSNorm differ from LayerNorm?"), exact strings (`chunk_size // 4`, `Set-ExecutionPolicy`) and numbers from results tables. 20 items is enough to see large differences and to catch regressions; it is not enough to tell apart systems that differ by a few points.
''')
nb.code(r'''
EVAL = json.load(open(os.path.join(ROOT, "data", "eval", "rag_eval.json"), encoding="utf-8"))
from collections import Counter
print(len(EVAL), "questions per source:", dict(Counter(it["source"] for it in EVAL)))
print(json.dumps(EVAL[8], indent=1))
''')
nb.md(r'''
## 10. Evaluating retrieval

Retrieval is evaluated without the LLM, which makes it fast and deterministic. We report two levels:

- **File level**: is a correct *file* among the retrieved chunks' sources? (`hit@1`, `hit@4`, and MRR, the mean of 1/rank of the first correct file.) This is the metric you can compute with only a "which document answers this?" label.
- **Evidence level**: does one of the top-k *chunks* contain the evidence string? This is stricter and closer to what the LLM needs: retrieving the right file but the wrong section of a 110 KB reference is a miss.

The question to answer: which retriever, and which chunk size, puts the evidence in front of the model most often?
''')
nb.code(r'''
def norm(text):
    return re.sub(r"\s+", " ", text.replace("*", "").replace("`", "")).lower()

def first_rank(docs, is_relevant):
    return next((i for i, d in enumerate(docs, start=1) if is_relevant(d)), None)

def retrieval_metrics(retrieve, k=K):
    """retrieve(question) -> ranked list of Documents. Returns file-level and evidence-level hit rates and MRR."""
    file_ranks, ev_ranks = [], []
    for it in EVAL:
        ds = retrieve(it["question"])[:k]
        file_ranks.append(first_rank(ds, lambda d: d.metadata["source"] in it["sources"]))
        ev_ranks.append(first_rank(ds, lambda d: any(e in norm(d.page_content) for e in it["evidence"])))
    def summary(ranks, prefix):
        return {f"{prefix}hit@1": np.mean([r == 1 for r in ranks]), f"{prefix}hit@{k}": np.mean([r is not None for r in ranks]),
                f"{prefix}mrr": np.mean([1 / r if r else 0 for r in ranks])}
    return {**summary(file_ranks, "file_"), **summary(ev_ranks, "evidence_")}, ev_ranks

def make_retrievers(size):
    cs, st = chunk_sets[size], stores[size]
    b = BM25Retriever.from_documents(cs, k=K, preprocess_func=tokenize)
    d = st.as_retriever(search_kwargs={"k": K})
    e = EnsembleRetriever(retrievers=[b, d], weights=[0.5, 0.5])
    return {"dense": d.invoke, "bm25": b.invoke, "hybrid": lambda q: e.invoke(q)[:K]}

retrieval_results, evidence_ranks = {}, {}
for name, fn in make_retrievers(800).items():
    retrieval_results[name], evidence_ranks[name] = retrieval_metrics(fn)
t0 = time.time()
retrieval_results["hybrid + multi-query"], evidence_ranks["hybrid + multi-query"] = retrieval_metrics(multi_query_retrieve)
print(f"(multi-query: {time.time() - t0:.0f}s for {len(EVAL)} questions, one extra LLM call each)\n")

cols = ["file_hit@1", "file_hit@4", "file_mrr", "evidence_hit@1", "evidence_hit@4", "evidence_mrr"]
print(f"{'chunk 800':22s}" + "".join(f"{c:>16s}" for c in cols))
for name, m in retrieval_results.items():
    print(f"{name:22s}" + "".join(f"{m[c]:16.2f}" for c in cols))
''')
nb.code(r'''
# The same three retrievers at three chunk sizes (evidence level only, the stricter metric).
sweep = {}
for size in CHUNK_SIZES:
    for name, fn in make_retrievers(size).items():
        sweep[(size, name)] = retrieval_metrics(fn)[0]
print(f"{'':16s}" + "".join(f"{s:>22d}" for s in CHUNK_SIZES))
for name in ("dense", "bm25", "hybrid"):
    print(f"{name:16s}" + "".join(f"   hit@4 {sweep[(s, name)]['evidence_hit@4']:.2f} mrr {sweep[(s, name)]['evidence_mrr']:.2f}" for s in CHUNK_SIZES))
''')
nb.code(r'''
COLORS = {"dense": "#2a78d6", "bm25": "#eb6834", "hybrid": "#1baf7a", "hybrid + multi-query": "#eda100"}
fig, ax = plt.subplots(1, 2, figsize=(13, 4))
names = list(retrieval_results)
metrics = ["file_hit@1", "evidence_hit@1", "evidence_hit@4", "evidence_mrr"]
x = np.arange(len(metrics)); w = 0.8 / len(names)
for i, n in enumerate(names):
    vals = [retrieval_results[n][m] for m in metrics]
    ax[0].bar(x + (i - (len(names) - 1) / 2) * w, vals, w * 0.92, color=COLORS[n], label=n)
ax[0].set_xticks(x); ax[0].set_xticklabels(metrics); ax[0].set_ylim(0, 1.05); ax[0].set_title("retrievers at chunk size 800 (20 questions)")
ax[0].legend(loc="lower right", fontsize=8)
for n in ("dense", "bm25", "hybrid"):
    ys = [sweep[(s, n)]["evidence_mrr"] for s in CHUNK_SIZES]
    ax[1].plot(CHUNK_SIZES, ys, marker="o", lw=2, ms=8, color=COLORS[n], label=n)
ax[1].set_xticks(CHUNK_SIZES); ax[1].set_xlabel("chunk_size (characters)"); ax[1].set_ylabel("evidence MRR"); ax[1].set_ylim(0, 1.05)
ax[1].set_title("chunk size sweep"); ax[1].legend(fontsize=8)
for a in ax:
    a.grid(axis="y", alpha=.3); a.spines[["top", "right"]].set_visible(False)
plt.tight_layout(); plt.show()
''')
nb.md(r'''
**Reading the retrieval results.**
- At file level every retriever finds the right document in the top 4 (hit@4 = 1.00): the metric is saturated, which is exactly why we also score whether the chunk that actually *contains the evidence* was retrieved.
- At evidence level the differences appear. **Hybrid retrieval is the best at the top of the ranking** (evidence hit@1 0.80, MRR 0.84 vs 0.73 for dense alone), and **BM25 is surprisingly strong** (hit@4 0.95): our corpus is technical documentation full of exact identifiers (`--chunk-size`, `kv_lora_rank`, error messages) that lexical search matches and embeddings blur. Fusing both with RRF keeps the strengths of each.
- **Multi-query did not help** here (MRR 0.82, with one extra LLM call per question, ~15 s for 20 questions): paraphrasing pays off when users phrase questions very differently from the documents, not on precise technical questions.
- **Chunk size matters, and not in the same direction for every retriever**: dense retrieval gets worse with bigger chunks (MRR 0.77 → 0.67, the embedding of a long chunk averages too many topics), while BM25 and hybrid improve (up to 0.82 and 0.89). Two caveats: a bigger chunk contains the evidence string more easily, and every retrieved chunk costs prompt tokens. 800 characters is a reasonable middle ground for this corpus.
''')
nb.md(r'''
## 11. Evaluating generation with RAGAS

Good retrieval is necessary, not sufficient: the model can still ignore the context, add facts that are not there, or answer a different question. **RAGAS** scores generation with an LLM acting as a judge, broken into four questions:

| metric | question it answers | how it is computed | needs a reference? |
|---|---|---|---|
| **faithfulness** | is every claim in the answer supported by the retrieved context? | the judge splits the answer into atomic statements, then checks each one against the context (NLI); score = supported / total | no |
| **answer relevancy** | does the answer address the question? | the judge writes the question the answer seems to respond to; score = cosine similarity between that and the real question (embeddings) | no |
| **context precision** | are the relevant chunks ranked at the top? | the judge marks each retrieved chunk useful or not for the reference answer; score = average precision over the ranking | yes |
| **context recall** | does the context contain everything the reference answer needs? | the judge checks each sentence of the reference against the context | yes |

Faithfulness and answer relevancy can run on live traffic (no reference needed); context precision and recall need a labelled set like ours.

**The judge.** RAGAS 0.4 builds its metrics on structured output: `llm_factory(model, client=...)` wraps an OpenAI-style client with the Instructor library so every judge step returns a validated Pydantic object. (Its older `LangchainLLMWrapper` still exists but is deprecated in 0.4.) Ollama serves an OpenAI-compatible API at `/v1`, so the same local `qwen2.5:3b` and `nomic-embed-text` act as the judge. In production you would use a stronger judge than the model being judged; keep that in mind when reading the numbers.

Two practical details in the next cell:
- ragas 0.4.3 still imports a module (`langchain_community.chat_models.vertexai`) that langchain-community 0.4 removed. Until ragas fixes it, registering an empty stand-in module before the import is enough; we never use Vertex AI.
- We strip the `[n]` citation markers from the answers before scoring: the judge otherwise treats "[2]" as a claim that the context does not support, and faithfulness drops for a formatting reason.
''')
nb.code(r'''
import sys, types, asyncio
warnings.filterwarnings("ignore", message="IProgress not found")    # tqdm's notebook-widget warning, harmless
try:                                             # compatibility shim, see the note above
    import langchain_community.chat_models.vertexai  # noqa: F401
except ImportError:
    _stub = types.ModuleType("langchain_community.chat_models.vertexai")
    _stub.ChatVertexAI = type("ChatVertexAI", (), {})
    sys.modules[_stub.__name__] = _stub

from openai import AsyncOpenAI
from ragas.llms import llm_factory
from ragas.embeddings import OpenAIEmbeddings as RagasEmbeddings
from ragas.metrics.collections import Faithfulness, AnswerRelevancy, ContextPrecision, ContextRecall

judge_client = AsyncOpenAI(base_url=os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/") + "/v1", api_key="ollama",
                           timeout=180, max_retries=1)        # a stuck judge call becomes a missing score instead of a hung notebook
judge = llm_factory(info["chat_model"], client=judge_client, temperature=0)
judge_emb = RagasEmbeddings(client=judge_client, model=info["embedding_model"])

METRICS = {
    "faithfulness": Faithfulness(llm=judge),
    "answer_relevancy": AnswerRelevancy(llm=judge, embeddings=judge_emb, strictness=1),   # 1 generated question instead of 3, to save time
    "context_precision": ContextPrecision(llm=judge),
    "context_recall": ContextRecall(llm=judge),
}

def strip_citations(text):
    return re.sub(r"\s*\[\d+\](\[\d+\])*", "", text).strip()

async def ragas_scores(item, result, sem):
    ctx = [d.page_content for d in result["docs"]]
    ans = strip_citations(result["answer"])
    args = {"faithfulness": dict(user_input=item["question"], response=ans, retrieved_contexts=ctx),
            "answer_relevancy": dict(user_input=item["question"], response=ans),
            "context_precision": dict(user_input=item["question"], reference=item["answer"], retrieved_contexts=ctx),
            "context_recall": dict(user_input=item["question"], reference=item["answer"], retrieved_contexts=ctx)}
    async def one(name):
        async with sem:
            try:
                return name, (await METRICS[name].ascore(**args[name])).value
            except Exception as e:            # a 3B judge sometimes returns JSON that does not validate; count it as missing
                return name, float("nan")
    return dict(await asyncio.gather(*(one(n) for n in METRICS)))
print("judge:", info["chat_model"], "| embeddings:", info["embedding_model"])
''')
nb.md(r'''
We compare two configurations that differ only in the retriever: **dense-only** and **hybrid**, both with 800-character chunks and k = 4. `chain.batch(..., max_concurrency=4)` sends the 20 questions in parallel; Ollama queues what the GPU cannot run at once.
''')
nb.code(r'''
CONFIGS = {"dense": make_rag_chain(dense), "hybrid": make_rag_chain(hybrid)}
questions = [it["question"] for it in EVAL]
outputs, timings = {}, {}
for name, chain in CONFIGS.items():
    t0 = time.time()
    outputs[name] = chain.batch(questions, config={"max_concurrency": 4})
    timings[name] = time.time() - t0
    print(f"{name:7s}: {len(questions)} answers in {timings[name]:.0f}s")
show(outputs["hybrid"][0])
''')
nb.code(r'''
t0 = time.time()
sem = asyncio.Semaphore(4)                     # requests in flight; Ollama runs what it can in parallel and queues the rest
memo = {}                                      # when both configs retrieved the same chunks they gave the same answer: score it once

async def scored(item, out):
    key = (item["question"], out["answer"], tuple(d.page_content for d in out["docs"]))
    if key not in memo:
        memo[key] = asyncio.ensure_future(ragas_scores(item, out, sem))
    return await memo[key]

names = list(CONFIGS)
rows = await asyncio.gather(*(scored(it, out) for name in names for it, out in zip(EVAL, outputs[name])))
ragas_rows = {name: rows[i * len(EVAL):(i + 1) * len(EVAL)] for i, name in enumerate(names)}
print(f"RAGAS scoring: {time.time() - t0:.0f}s for {len(memo)} distinct (question, context, answer) triples "
      f"out of {len(names) * len(EVAL)}, {len(METRICS)} metrics each\n")

ragas_summary = {name: {m: float(np.nanmean([r[m] for r in rows])) for m in METRICS} for name, rows in ragas_rows.items()}
missing = {name: sum(np.isnan(r[m]) for r in rows for m in METRICS) for name, rows in ragas_rows.items()}
print(f"{'config':8s}" + "".join(f"{m:>19s}" for m in METRICS) + "   unscored")
for name, s in ragas_summary.items():
    print(f"{name:8s}" + "".join(f"{s[m]:19.2f}" for m in METRICS) + f"   {missing[name]:5d}")
''')
nb.code(r'''
fig, ax = plt.subplots(figsize=(9, 3.8))
x = np.arange(len(METRICS)); w = 0.38
for i, name in enumerate(CONFIGS):
    vals = [ragas_summary[name][m] for m in METRICS]
    bars = ax.bar(x + (i - 0.5) * w, vals, w * 0.92, color=COLORS[name], label=name)
    ax.bar_label(bars, fmt="%.2f", fontsize=8, padding=2)
ax.set_xticks(x); ax.set_xticklabels(list(METRICS)); ax.set_ylim(0, 1.12); ax.set_ylabel("mean score (0-1)")
ax.set_title("RAGAS, judged by the local qwen2.5:3b (20 questions)"); ax.legend(loc="lower right")
ax.grid(axis="y", alpha=.3); ax.spines[["top", "right"]].set_visible(False); plt.tight_layout(); plt.show()
''')
nb.md(r'''
### Looking at individual questions

Averages hide the interesting part. Below, every question with its evidence rank (did retrieval find it?) next to the RAGAS scores of the hybrid configuration. Read the low rows: is the failure in retrieval (no evidence in the top 4), in generation (evidence there, answer unfaithful), or in the judge?
''')
nb.code(r'''
print(f"{'question':58s} {'ev.rank':>7s} {'faith':>6s} {'relev':>6s} {'c.prec':>6s} {'c.rec':>6s}")
for it, r, rank in zip(EVAL, ragas_rows["hybrid"], evidence_ranks["hybrid"]):
    print(f"{it['question'][:57]:58s} {str(rank or '-'):>7s} " + " ".join(f"{r[m]:6.2f}" for m in METRICS))
worst = min(range(len(EVAL)), key=lambda i: np.nan_to_num(ragas_rows["hybrid"][i]["faithfulness"], nan=1.0))
print("\nlowest faithfulness (hybrid):")
show(outputs["hybrid"][worst]); print("reference:", EVAL[worst]["answer"])
''')
nb.md(r'''
**Reading the RAGAS results — and the judge.**
- The retrieval-side metrics move in the same direction as our own retrieval metrics: with hybrid retrieval, **context precision rises from 0.76 to 0.84 and context recall from 0.57 to 0.70**, and faithfulness from 0.38 to 0.54. Better context, more grounded answers. (The judge is an LLM, so the exact values move a little between runs; in every run we made, hybrid came out ahead on these three.)
- Answer relevancy is lower for hybrid (0.30 vs 0.44), and it is the metric that moved most between our runs. Before reading anything into it, look at the per-question table: the local 3B judge gives **relevancy 0 to some correct answers** (it flags them as "noncommittal") and **context recall 0 to an answer whose context plainly contains the fact**. With a judge this small, single scores are noisy; trust the *direction* of large differences, not the second decimal.
- That is the most important lesson of LLM-as-judge evaluation: **validate the judge** before trusting it. In practice you use a stronger model as judge (GPT-4o-class or a large open model), spot-check its verdicts against a few dozen human labels, and keep the judge fixed while you compare system versions.
- Cost matters too: scoring 40 answers × 4 metrics took about 7 minutes (440 s) on a local GPU — most of this notebook's runtime. Evaluation is a real budget line; run it on a fixed, small, representative set on every change, not on everything.
''')
nb.md(r'''
## Takeaways

- A RAG system is a **retriever plus a generator, and you evaluate them separately**. Retrieval metrics need only labels and run in seconds; RAGAS needs an LLM judge and runs in minutes.
- **Hybrid retrieval (BM25 + dense with RRF) is the default to beat.** Technical documentation is full of exact strings that embeddings blur.
- **Chunk size is a hyperparameter**: measure it on your own questions instead of copying a default.
- **Query rewriting is not free**: an extra LLM call per question, and it only pays off when queries and documents use different words.
- **LLM judges are models too.** A 3B judge is noisy and sometimes fails to return valid JSON; use a stronger judge than the system under test, spot-check its verdicts by hand, and compare configurations on the same judge rather than reading absolute numbers.

**Failure modes and fixes**

| symptom | likely cause | fix |
|---|---|---|
| the answer is in the docs but not retrieved | chunks too large or too small, wording mismatch | measure chunk sizes, hybrid search, multi-query / HyDE |
| the right chunk is retrieved but the model ignores it | too much context, weak instructions, small model | fewer and better chunks, a reranker, "answer only from context", require citations |
| the question needs facts from several places | single-shot retrieval | iterative or agentic retrieval (notebook 14), parent-document retrieval |
| wrong on names, codes, versions | dense-only retrieval | add BM25 |
| slow | big context, brute-force index | smaller k, ANN index (HNSW), cache embeddings |

In production the same structure scales up: a hosted or larger embedding model, a vector database (pgvector, Qdrant, Weaviate), a cross-encoder reranker on the top 20-50 candidates, and RAGAS or LangSmith evaluations run on every change. `scripts/rag_demo.py` packages this notebook's pipeline as a command-line tool.

**Next:** `11_agents_tools_structured_output.ipynb`, where the model stops only reading context and starts calling tools.
''')
nb.save(notebook_path("10_rag.ipynb"))
print("10_rag.ipynb written")
