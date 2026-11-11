"""Notebook 29 - Open-source evaluation and tracing frameworks: Inspect AI and Arize Phoenix

Builds 29_eval_and_tracing_frameworks.ipynb. Run: python tools/build_notebooks.py 29
Read it after notebook 28 (observability). Result commentary cells contain {{R_...}} markers filled after execution.
Needs the isolated environment described in scripts/inspect_tasks/gsm8k_cot.py (.venv-eco).
"""
from _common import *  # noqa: F401,F403

nb = NB()
nb.md(r'''
# 29 · Evaluation and tracing frameworks: Inspect AI and Phoenix

Up to here every evaluation in the course was a hand-written loop, and the traces went to LangSmith (notebook 28), a hosted service whose free plan ran out of traces while this course was being built. Both choices are fine for learning and both stop scaling in a team: hand-written loops are hard to compare across people and months, and a hosted tracer is a dependency, a cost and a place your data goes.

This notebook replaces each with the open-source tool teams actually use, chosen from Good AI List's *AI evaluation and observability* category for running locally on this machine:

| tool | layer | what we measure |
|---|---|---|
| **Inspect AI** (UK AI Security Institute, MIT) | evaluation framework: datasets, solvers, scorers, logs, epochs | notebook 15's GSM8K result reproduced; how much the scorer changes the score; variance over epochs |
| **Arize Phoenix** (Elastic License 2.0, self-hosted) | tracing and observability on OpenTelemetry | spans from a LangChain pipeline; latency and tokens per step from the trace store; correctness written back as annotations; tracing overhead |

Read it **after notebook 28**, whose tracing and evaluation ideas it puts on open infrastructure. It reuses notebook 15's GSM8K setup.

**Why these two and not the alternatives** (as of November 2026):
- **lm-evaluation-harness** (EleutherAI) is the standard for *benchmarking model checkpoints* (it powers leaderboards) and needs log-probabilities, which Ollama's API does not expose. **Inspect** evaluates *systems*: prompts, tools, agents and sandboxes, against any API, including Ollama.
- **Langfuse** and **Opik** cover the same ground as Phoenix and are popular. Self-hosting them means running Docker with a database (Postgres, ClickHouse), while Phoenix runs as one Python process with SQLite, which is what fits a laptop. All three ingest OpenTelemetry, so the instrumentation code is the same.

**Sources.** Huyen, *AI Engineering*, ch. 3–4 (evaluation) and 10 (observability); the Inspect AI documentation (inspect.aisi.org.uk); the OpenInference semantic conventions; OpenTelemetry's GenAI semantic conventions.
''')
nb.md(r'''
## Background: what a framework adds to a loop

**An evaluation framework** separates the parts of an evaluation so each can be swapped and recorded:
- **dataset** of samples (input, target, metadata);
- **solver**: what is done to each sample (a system message, a generation, a chain of thought, tool use, an agent loop in a sandbox);
- **scorer**: how an output is graded (exact match, a regex, a model grader), with **metrics** (accuracy and its standard error);
- **log**: every sample with its full transcript, tokens, timings, scores, model and generation settings, in one file you can reopen, diff and share;
- **epochs**: each sample run several times and reduced (mean, majority), because one sampled run is one draw.

**Tracing** records what a running application did. OpenTelemetry represents each request as a **trace**, a tree of **spans** (one per step: a chain, a prompt, an LLM call, a retrieval, a tool call), each with start/end times, a status and attributes. **OpenInference** is the set of attribute names for LLM work (`llm.model_name`, `llm.token_count.prompt`, `input.value`, `output.value`, span kinds `LLM`, `CHAIN`, `RETRIEVER`, `TOOL`...). An **instrumentor** creates the spans for a library automatically; an **exporter** ships them (OTLP over HTTP or gRPC) to a collector: Phoenix, Langfuse, LangSmith, Datadog or any OTel backend. Because the protocol is open, changing backend is a configuration change, not a rewrite.

**Annotations** close the loop: a score or a label attached to a span (by code, an LLM judge or a person), which turns production traces into evaluation data.
''')
nb.code(setup_apps("llm-edu-suite-29-frameworks"))
nb.code(r'''
warnings.filterwarnings("ignore")
import re, socket, subprocess, glob, shutil, statistics
import httpx
from collections import defaultdict
from llm_edu_suite.evaluation import bootstrap_ci, paired_bootstrap_diff

ECO = os.path.join(ROOT, ".venv-eco", "Scripts" if os.name == "nt" else "bin")
INSPECT, PHOENIX = os.path.join(ECO, "inspect"), os.path.join(ECO, "phoenix")
assert os.path.exists(INSPECT + (".exe" if os.name == "nt" else "")), "create .venv-eco first (see scripts/inspect_tasks/gsm8k_cot.py)"
TASK = os.path.join(ROOT, "scripts", "inspect_tasks", "gsm8k_cot.py")
LOGS = os.path.join(ROOT, "data", "cache", "inspect_logs"); shutil.rmtree(LOGS, ignore_errors=True); os.makedirs(LOGS)
OLLAMA_V1 = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/") + "/v1"
ver = subprocess.run([os.path.join(ECO, "python"), "-c", "import inspect_ai, phoenix; print(inspect_ai.__version__, phoenix.__version__)"],
                     capture_output=True, text=True).stdout.split()
print(f"Inspect AI {ver[0]} | Phoenix {ver[1]} | both in {os.path.relpath(ECO, ROOT)}")
''')
nb.md(r'''
## 1. Inspect AI: the GSM8K evaluation of notebook 15, as a task

The task file is `scripts/inspect_tasks/gsm8k_cot.py`, about 40 lines: notebook 15's 80 problems, its chain-of-thought system message, and **two scorers on the same generations**, our "last number" rule and Inspect's built-in `match(numeric=True)`. It runs from the command line like any Inspect evaluation; we call it from Python and read the JSON log it writes.
''')
nb.code(r'''
print(open(TASK, encoding="utf-8").read().split('"""', 2)[2].strip())
''')
nb.code(r'''
def inspect_eval(*args, timeout=3600):
    env = {**os.environ, "INSPECT_LOG_DIR": LOGS, "PYTHONIOENCODING": "utf-8", "NO_COLOR": "1"}
    before = set(glob.glob(os.path.join(LOGS, "*.json")))
    t0 = time.time()
    p = subprocess.run([INSPECT, "eval", TASK, "--model", "ollama/qwen2.5:3b", "--model-base-url", OLLAMA_V1,
                        "--max-tokens", "512", "--max-connections", "1", "--log-format", "json", "--display", "none", *args],
                       capture_output=True, text=True, encoding="utf-8", errors="replace", env=env, timeout=timeout)
    new = sorted(set(glob.glob(os.path.join(LOGS, "*.json"))) - before)
    assert new, p.stdout[-2000:] + p.stderr[-2000:]
    log = json.load(open(new[-1], encoding="utf-8"))
    print(f"status {log['status']} | {len(log['samples'])} samples | {time.time() - t0:.0f}s | log {os.path.basename(new[-1])} "
          f"({os.path.getsize(new[-1]) / 1024:.0f} KB)")
    return log

greedy = inspect_eval("--temperature", "0")
for s in greedy["results"]["scores"]:
    print(f"  scorer {s['name']:12s} " + " | ".join(f"{k} {v['value']:.3f}" for k, v in s["metrics"].items()))
''')
nb.code(r'''
# What the log holds for one sample: the whole transcript, the scores of both scorers, tokens and time
s0 = greedy["samples"][0]
print("keys:", list(s0))
print("scores:", {k: (v["value"], v.get("answer")) for k, v in s0["scores"].items()})
print("usage:", s0.get("model_usage"), "| total time", s0.get("total_time"))
print("answer:", s0["output"]["choices"][0]["message"]["content"][-300:])
print("\nrun settings recorded in the log:", {k: greedy["eval"][k] for k in ("model", "task", "created")}, greedy["plan"]["config"])
''')
nb.md(r'''
### 1.1 Does the framework reproduce the hand-written loop?

Notebook 15 measured **85.0 %** for greedy chain of thought on these 80 problems with its own loop. Same model, prompt, decoding and extraction should give the same number, and we can check problem by problem. Where the two scorers disagree, we print the answers to see which rule is right.
''')
nb.code(r'''
def per_sample(log, scorer):
    return np.array([1.0 if s["scores"][scorer]["value"] == "C" else 0.0 for s in sorted(log["samples"], key=lambda s: s["id"])])
ours, builtin = per_sample(greedy, "last_number"), per_sample(greedy, "match")
for name, v in (("Inspect, last-number scorer", ours), ("Inspect, built-in match(numeric)", builtin)):
    m, lo, hi = bootstrap_ci(v); print(f"{name:34s} {m:.1%} [{lo:.1%}, {hi:.1%}]")
print(f"{'notebook 15, hand-written loop':34s} 85.0%")
d, lo, hi = paired_bootstrap_diff(ours, builtin)
print(f"\nscorer effect on the same outputs: {d:+.1%} [{lo:+.1%}, {hi:+.1%}] | disagreements: {int((ours != builtin).sum())} of {len(ours)}")
by_id = {s["id"]: s for s in greedy["samples"]}
for i in np.where(ours != builtin)[0][:4]:
    s = by_id[int(i)]
    print(f"\n-- problem {i}: reference {s['target']} | last-number {s['scores']['last_number']['value']} ({s['scores']['last_number'].get('answer')}) "
          f"| match {s['scores']['match']['value']} ({s['scores']['match'].get('answer')})")
    print("   ..." + s["output"]["choices"][0]["message"]["content"][-200:].replace("\n", " "))
''')
nb.md(r'''
**What Inspect reproduces, and what it adds.**
- **The same number as the hand-written loop.** With notebook 15's last-number rule as a custom scorer, Inspect gives **85.0%** [77.5%, 92.5%] on the same 80 problems. That is exactly notebook 15's result, and the greedy runs also match in length (246 output tokens per problem in both).

  A framework is worth adopting only if it can reproduce your trusted numbers first. Here it does.
- **The scorer is part of the metric.** Inspect's built-in `match(numeric=True)` scored the *same generations* at **80.0%**, which is −5.0 points [−10.0, −1.2] on a paired comparison. The four disagreements all end with a LaTeX `\boxed{}` answer, and the built-in scorer read another number of the text (10 instead of 7, 12 instead of 144). Two "evaluations of the same model" that differ only in answer extraction can be 5 points apart. **Report the scorer with the score.**
- **What the log gives you for free.** Every sample is stored as one JSON log (1.9 MB for 80 samples) with:
  - the input, the target and the full message list;
  - the output, the score of every scorer and the extracted answer;
  - token usage per model and timings.

  The run settings (model, temperature 0, max tokens, connections) are recorded too. That is the audit trail notebook 15 had to build by hand, and `inspect view` browses it.

### 1.2 Epochs: how much does one sampled run tell you?

At temperature 0.7 (the setting of notebook 15's self-consistency), each problem is answered 3 times (`--epochs 3`). Inspect reports the mean over epochs with its standard error; from the log we also get every epoch separately, which shows how far a single run can land from the average.
''')
nb.code(r'''
sampled = inspect_eval("--temperature", "0.7", "--epochs", "3", "--seed", "0")
for s in sampled["results"]["scores"]:
    if s["name"] == "last_number":
        print("mean over 3 epochs:", " | ".join(f"{k} {v['value']:.3f}" for k, v in s["metrics"].items()))
E = defaultdict(dict)
for s in sampled["samples"]:
    E[s["epoch"]][s["id"]] = 1.0 if s["scores"]["last_number"]["value"] == "C" else 0.0
per_epoch = {e: np.mean(list(v.values())) for e, v in sorted(E.items())}
print("each epoch alone:", {e: f"{a:.1%}" for e, a in per_epoch.items()}, f"| spread {max(per_epoch.values()) - min(per_epoch.values()):.1%}")
grid = np.array([[E[e][i] for e in sorted(E)] for i in sorted(E[1])])
print(f"problems right in all 3 epochs {np.mean(grid.sum(1) == 3):.1%} | in none {np.mean(grid.sum(1) == 0):.1%} | "
      f"unstable (right in 1 or 2) {np.mean((grid.sum(1) > 0) & (grid.sum(1) < 3)):.1%}")
tok = [s["model_usage"][next(iter(s["model_usage"]))]["output_tokens"] for s in greedy["samples"]]
print(f"greedy run: {np.mean(tok):.0f} output tokens per problem (notebook 15: 246)")
''')
nb.md(r'''
**What sampling variance looks like.**
- At temperature 0.7, three epochs of the 80 problems give **81.2%** on average. The three epochs alone give 80.0%, 82.5% and 81.2%, a spread of 2.5 points.
- **Most problems are stable:**
  - 80% were solved in all three epochs;
  - 17.5% in none;
  - only 2.5% (2 problems) changed between epochs.

  The uncertainty is mostly *which problems you drew* (the ±4 points of the 80-problem standard error), not *which sample you drew per problem*.
- **Sampling costs accuracy here:** greedy decoding scored 85.0% and sampling 81.2%. For a single-answer task, temperature 0 is the right default. Sampling pays off when you take several samples and combine them (self-consistency, notebook 15).
- `--epochs` makes this measurement a flag instead of a loop. With a sampled model, a single run is one draw: report the mean over epochs and its spread.

## 2. Phoenix: tracing on OpenTelemetry, on your own machine

Phoenix runs as a separate server process, like it would in production (there, in a container next to the application). The notebook only gets the light client side: `arize-phoenix-otel` (an OpenTelemetry exporter preconfigured for Phoenix) and the OpenInference instrumentor for LangChain.
''')
nb.code(r'''
def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0)); return s.getsockname()[1]
PORT, GRPC = free_port(), free_port()
PHX_DIR = os.path.join(ROOT, "data", "cache", "phoenix"); shutil.rmtree(PHX_DIR, ignore_errors=True); os.makedirs(PHX_DIR)
phx_env = {**os.environ, "PHOENIX_PORT": str(PORT), "PHOENIX_GRPC_PORT": str(GRPC), "PHOENIX_WORKING_DIR": PHX_DIR,
           "PHOENIX_HOST": "127.0.0.1", "PHOENIX_TELEMETRY_ENABLED": "false", "PYTHONIOENCODING": "utf-8"}
phx_log = open(os.path.join(PHX_DIR, "server.log"), "w", encoding="utf-8")
server = subprocess.Popen([PHOENIX, "serve"], env=phx_env, stdout=phx_log, stderr=subprocess.STDOUT)
API = f"http://127.0.0.1:{PORT}"
for _ in range(120):
    try:
        if httpx.get(API + "/healthz", timeout=2).status_code == 200: break
    except httpx.HTTPError: pass
    time.sleep(1)
print(f"Phoenix up at {API} (UI in a browser at the same address), data in {os.path.relpath(PHX_DIR, ROOT)}")
''')
nb.code(r'''
from phoenix.otel import register
from openinference.instrumentation.langchain import LangChainInstrumentor
from langchain_core._api.deprecation import LangChainDeprecationWarning
warnings.filterwarnings("ignore", category=LangChainDeprecationWarning)   # openinference still calls message.text() as a method
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda

PROJECT = "gsm8k-cot"
tracer_provider = register(project_name=PROJECT, endpoint=f"{API}/v1/traces", protocol="http/protobuf",
                           batch=True, set_global_tracer_provider=False, verbose=False)
instrumentor = LangChainInstrumentor(); instrumentor.instrument(tracer_provider=tracer_provider)

import datasets; datasets.disable_progress_bars()
gsm = datasets.load_dataset("openai/gsm8k", "main", split="test").shuffle(seed=0).select(range(20))
COT = "Solve the problem step by step, then give the final answer as a number on the last line."
def final_number(text):
    m = re.findall(r"-?\d+(?:\.\d+)?", text.replace("$", "").replace(",", ""))
    return float(m[-1]) if m else None
chain = (ChatPromptTemplate.from_messages([("system", COT), ("user", "{question}")]) | get_llm(num_predict=512)
         | StrOutputParser() | RunnableLambda(final_number).with_config(run_name="extract_answer"))

gold, preds = [], []
t0 = time.time()
for q, a in zip(gsm["question"], gsm["answer"]):
    preds.append(chain.invoke({"question": q}, config={"run_name": "solve_gsm8k", "metadata": {"dataset": "gsm8k"}}))
    gold.append(float(a.split("####")[-1].replace(",", "")))
tracer_provider.force_flush()
print(f"20 traced requests in {time.time() - t0:.0f}s | accuracy {np.mean([p is not None and abs(p - g) < 1e-6 for p, g in zip(preds, gold)]):.0%}")
''')
nb.md(r'''
### 2.1 Reading the traces back

Everything the instrumentor recorded is now in Phoenix, queryable through its REST API: the same data the UI shows. Production dashboards (latency per step, tokens per request, error rate) are computed from exactly this.
''')
nb.code(r'''
def get_spans(project, **params):
    out, cursor = [], None
    while True:
        r = httpx.get(f"{API}/v1/projects/{project}/spans", params={"limit": 1000, **params, **({"cursor": cursor} if cursor else {})}, timeout=30)
        r.raise_for_status(); body = r.json(); out += body["data"]; cursor = body.get("next_cursor")
        if not cursor: return out
def attr(span, key):
    a = span["attributes"]
    if key in a: return a[key]
    for part in key.split("."):                     # attributes may come flat or nested
        if not isinstance(a, dict) or part not in a: return None
        a = a[part]
    return a
def ms(span):
    from datetime import datetime
    f = lambda t: datetime.fromisoformat(t.replace("Z", "+00:00"))
    return (f(span["end_time"]) - f(span["start_time"])).total_seconds() * 1000

time.sleep(2)
spans = get_spans(PROJECT)
traces = {s["context"]["trace_id"] for s in spans}
print(f"{len(spans)} spans in {len(traces)} traces ({len(spans) / len(traces):.0f} spans per request)")
by = defaultdict(list)
for s in spans: by[(s["span_kind"], s["name"])].append(ms(s))
print(f"\n{'kind':8s} {'span':22s} {'count':>5s} {'p50 ms':>9s} {'p95 ms':>9s}")
for (kind, name), v in sorted(by.items(), key=lambda kv: -np.median(kv[1])):
    print(f"{kind:8s} {name:22s} {len(v):5d} {np.median(v):9.1f} {np.percentile(v, 95):9.1f}")
llm_spans = [s for s in spans if s["span_kind"] == "LLM"]
pt = [attr(s, "llm.token_count.prompt") or 0 for s in llm_spans]; ct = [attr(s, "llm.token_count.completion") or 0 for s in llm_spans]
print(f"\nLLM spans: model {attr(llm_spans[0], 'llm.model_name')} | prompt tokens {np.mean(pt):.0f} / completion tokens {np.mean(ct):.0f} per call (means)")
print("one LLM span's input, as recorded:", str(attr(llm_spans[0], "input.value"))[:200])
''')
nb.md(r'''
### 2.2 Writing evaluations back onto the traces

A trace store becomes an evaluation store when scores are attached to spans. We grade each request with code (is the extracted number right?) and post the result as an annotation on its root span, then read the annotations back and recompute the accuracy from Phoenix alone. With real traffic there is no reference answer, and the annotator is an LLM judge (notebooks 18 and 28) or a person reviewing a sample, but the mechanics are the same.
''')
nb.code(r'''
roots = sorted([s for s in spans if s["parent_id"] is None], key=lambda s: s["start_time"])
assert len(roots) == len(preds)
payload = {"data": [{"span_id": r["context"]["span_id"], "name": "correct", "annotator_kind": "CODE",
                     "result": {"label": "correct" if ok else "wrong", "score": float(ok)}}
                    for r, ok in zip(roots, [p is not None and abs(p - g) < 1e-6 for p, g in zip(preds, gold)])]}
r = httpx.post(f"{API}/v1/span_annotations", json=payload, params={"sync": "true"}, timeout=30); r.raise_for_status()
print(f"posted {len(r.json()['data'])} annotations")
ann = httpx.get(f"{API}/v1/projects/{PROJECT}/span_annotations", params=[("span_ids", r_["context"]["span_id"]) for r_ in roots] + [("limit", 1000)], timeout=30).json()["data"]
print(f"read back {len(ann)} | accuracy computed from Phoenix: {np.mean([a['result']['score'] for a in ann]):.0%}")
''')
nb.md(r'''
### 2.3 What does tracing cost?

Instrumentation runs inside the request path. We time the same chain with a fake, instant chat model (so only LangChain and tracing remain) 300 times with tracing on and 300 times off, after warming up.
''')
nb.code(r'''
from langchain_core.language_models.fake_chat_models import FakeListChatModel
fake = (ChatPromptTemplate.from_messages([("system", COT), ("user", "{question}")]) | FakeListChatModel(responses=["The answer is\n42"])
        | StrOutputParser() | RunnableLambda(final_number))
def bench(n=300):
    for _ in range(20): fake.invoke({"question": "q"})
    t = []
    for _ in range(n):
        t0 = time.perf_counter(); fake.invoke({"question": "q"}); t.append((time.perf_counter() - t0) * 1000)
    return np.median(t), np.percentile(t, 95)
on = bench(); instrumentor.uninstrument(); off = bench(); instrumentor.instrument(tracer_provider=tracer_provider)
print(f"tracing on : median {on[0]:.2f} ms, p95 {on[1]:.2f} ms per request (4 spans, batched export)")
print(f"tracing off: median {off[0]:.2f} ms, p95 {off[1]:.2f} ms")
llm_ms = np.median([ms(s) for s in llm_spans])
print(f"overhead {on[0] - off[0]:.2f} ms per request, against a median LLM call of {llm_ms:.0f} ms above "
      f"({(on[0] - off[0]) / llm_ms:.3%} of it)")
tracer_provider.force_flush(); instrumentor.uninstrument()
server.terminate(); server.wait(timeout=30); phx_log.close()
print("Phoenix stopped")
''')
nb.md(r'''
**What the trace store shows.**
- **Every request became a trace of 5 spans.** The pipeline (`solve_gsm8k`) contains the prompt template, the LLM call, the output parser and the answer extractor. 20 requests gave 100 spans, with no change to the pipeline's code: the OpenInference instrumentor hooks into LangChain.
- **Latency per step, read back from Phoenix's REST API:**
  - the LLM span is the whole cost (p50 3,585 ms, p95 4,641 ms);
  - every other step takes under 1 ms.

  In a real application this table is how you find the slow step: retrieval, a tool call, or the model.
- **Tokens per LLM call:** 97 prompt tokens and 242 completion tokens on average, recorded on the span. Cost per request is tokens × price, computed from the traces.
- **Evaluations live next to the traces.** We posted the correctness of each answer (computed by code) as an annotation on each root span and read it back: **80% on these 20 problems**, now filterable in the UI (for example, show me the failed traces). This is the production loop:
  1. trace;
  2. score (by code, a judge or users);
  3. look at the failures;
  4. fix.
- **Tracing is cheap.** With a fake LLM, so that only the instrumentation is measured, a traced request took 3.52 ms against 1.02 ms untraced: **2.5 ms of overhead** for 4 spans with batched export. That is 0.07% of one real LLM call. There is no performance reason not to trace every request; the reasons to sample are storage and privacy.
- **A caveat:** the recorded input is LangChain's serialized message format, verbatim. Traces store prompts and outputs in full, so treat the trace store as sensitive data (PII masking, retention, access control).

## Results

| question | measured | lesson |
|---|---|---|
| does Inspect reproduce the hand-written evaluation? | 85.0% on notebook 15's 80 GSM8K problems with the same rule; same output length | adopt a framework after it reproduces your trusted number |
| does the scorer matter? | built-in `match` 80.0% on the same generations (−5.0 points [−10.0, −1.2]); 4 disagreements, all `\boxed{}` answers | the answer extractor is part of the metric: report it |
| how much does one sampled run tell you? | T = 0.7, 3 epochs: 80.0 / 82.5 / 81.2%; 2.5% of problems unstable; greedy 85.0% | variance is mostly the sample of problems; sampling cost ~4 points here |
| what does tracing show? | 5 spans per request; LLM p50 3.6 s, everything else < 1 ms; 97 / 242 tokens per call | per-step latency and tokens come from the trace store, not from guesses |
| evaluations on traces | 20 code-scored annotations written and read back (80%) | scores next to traces make failures searchable |
| tracing overhead | 2.5 ms per request (0.07% of an LLM call) | trace everything; sample for storage and privacy, not speed |

## In production

**Evaluation frameworks** (Inspect, lm-evaluation-harness, OpenAI Evals, promptfoo, DeepEval):
- Define each evaluation as code (dataset + solver + scorer) in the repository.
- Run it in CI on every prompt, model or retrieval change.
- Keep the logs as artifacts.
- Pin the scorer, report confidence intervals, and use epochs for anything sampled.
- Inspect is a good fit for agentic and safety evaluations (tools, sandboxes, multi-turn). It is what the UK AI Security Institute uses for its own model evaluations.

**Tracing** (Phoenix, Langfuse, LangSmith of notebook 28, OpenLLMetry, any OpenTelemetry backend):
- Instrument at the framework level (LangChain, LlamaIndex, OpenAI SDK instrumentors).
- Export with batching, and send traces to a backend you control when prompts contain user data.
- Attach evaluations to traces: code checks, LLM judges (notebook 28) and user feedback.
- Alert on latency, error rate and score drift (notebook 24).

**Choosing.**
- Self-hosted and open (Phoenix, Langfuse) when data cannot leave your infrastructure.
- Managed (LangSmith, Arize, Datadog LLM Observability) when you want no operations work.
- Because both sides speak OpenTelemetry, switching backends is a configuration change, not a rewrite.

**Failure modes.**
- An evaluation that silently changes its scorer.
- Comparing numbers from different answer extractors.
- A single sampled run reported as "the" accuracy.
- Traces that capture secrets or personal data.
- Dashboards nobody looks at.
- Evaluation sets that leak into prompts or few-shot examples.

**Next:** notebook 30 (safety and guardrails), or notebook 27 for evaluating agents with hidden tests.
''')
nb.save(notebook_path("29_eval_and_tracing_frameworks.ipynb"))
print("29_eval_and_tracing_frameworks.ipynb written")
