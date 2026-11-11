"""Notebook 16 - Tools, function calling and structured output with LangChain

Builds 16_agents_tools_structured_output.ipynb. Run: python tools/build_notebooks.py 16

The code cells that tests/test_agents_langgraph.py also exercises (with fake chat models, no Ollama)
are kept in module-level strings (TOOLS_CELL, BY_HAND_CELL, EVAL_HELPERS_CELL) so the test runs
exactly the code the notebook runs.
"""
from _common import *  # noqa: F401,F403

# The four tools the notebook uses. search_docs needs embeddings, so the vector store is built lazily
# (the tests replace it with a fake retriever and never call Ollama).
TOOLS_CELL = r'''
import os, json
from typing import Literal
from pydantic import BaseModel, Field
from langchain_core.tools import tool

WEATHER_C = {"lima": 19, "cusco": 12, "madrid": 24, "london": 11}          # a fake weather service
ORDERS = {"A100": {"status": "shipped", "total": 120.0},                   # a fake orders database
          "B200": {"status": "processing", "total": 45.5},
          "C300": {"status": "delivered", "total": 80.0}}

@tool
def get_weather(city: str, unit: Literal["C", "F"] = "C") -> dict:
    """Get the current temperature for a city, in Celsius (C) or Fahrenheit (F)."""
    c = WEATHER_C.get(city.strip().lower())
    if c is None:
        return {"error": f"no weather data for {city}"}
    return {"city": city, "temperature": c if unit == "C" else round(c * 9 / 5 + 32), "unit": unit}

class CalculateArgs(BaseModel):
    expression: str = Field(description="An arithmetic expression using only digits, + - * / ** and parentheses, e.g. '(17 * 23) + 4'")

@tool(args_schema=CalculateArgs)
def calculate(expression: str) -> str:
    """Evaluate an arithmetic expression and return the numeric result. Use it for any arithmetic."""
    if not all(ch in "0123456789+-*/(). " for ch in expression):
        raise ValueError(f"unsupported characters in {expression!r}; use only digits, + - * / ** and parentheses")
    return str(round(eval(expression, {"__builtins__": {}}, {}), 4))   # a toy sandbox; never eval untrusted code in production

class OrderArgs(BaseModel):
    order_id: str = Field(pattern=r"^[A-Z]\d{3}$", description="The order id: one capital letter followed by three digits, e.g. A100")

@tool(args_schema=OrderArgs)
def order_status(order_id: str) -> dict:
    """Look up the status and the total amount (in dollars) of a customer order."""
    return ORDERS.get(order_id, {"error": f"order {order_id} not found"})

RETRIEVER = None          # built on first use: a LangChain vector store over data/sample_docs

def _retriever():
    global RETRIEVER
    if RETRIEVER is None:
        from langchain_core.documents import Document
        from langchain_core.vectorstores import InMemoryVectorStore
        folder = os.path.join(ROOT, "data", "sample_docs")
        docs = [Document(open(os.path.join(folder, f), encoding="utf-8").read(), metadata={"source": f}) for f in sorted(os.listdir(folder))]
        RETRIEVER = InMemoryVectorStore.from_documents(docs, get_embeddings()).as_retriever(search_kwargs={"k": 1})
    return RETRIEVER

@tool
def search_docs(query: str) -> str:
    """Search the company documentation: refund policy, shipping times and costs, account security, support hours, API limits."""
    return "\n\n".join(d.page_content for d in _retriever().invoke(query))

TOOLS = [get_weather, calculate, order_status, search_docs]
'''

# The agent loop written by hand, in a dozen lines.
BY_HAND_CELL = r'''
from langchain_core.messages import HumanMessage, ToolMessage

def run_by_hand(model, tools, question, max_steps=5, verbose=True):
    """The tool-calling loop without any framework: call the model, run the tools it asks for, repeat."""
    by_name = {t.name: t for t in tools}
    bound = model.bind_tools(tools)               # attaches the JSON schemas to every request
    messages = [HumanMessage(question)]
    for step in range(1, max_steps + 1):
        ai = bound.invoke(messages)
        messages.append(ai)
        if not ai.tool_calls:                     # no tool requested: this is the final answer
            break
        for call in ai.tool_calls:
            if call["name"] not in by_name:       # models do invent tool names; tell them instead of crashing
                result = ToolMessage(f"Error: there is no tool named {call['name']!r}", tool_call_id=call["id"], status="error")
            else:
                result = by_name[call["name"]].invoke(call)   # invoking a tool with the whole ToolCall returns a ToolMessage
            messages.append(result)
            if verbose:
                print(f"step {step}: {call['name']}({call['args']}) -> {str(result.content)[:90]}")
    return messages
'''

# Scoring of one agent run against a labelled case; also used by the tests.
EVAL_HELPERS_CELL = r'''
from langchain_core.messages import AIMessage, ToolMessage

def trajectory(messages):
    """The tool calls an agent made, the ToolMessages it got back, the token count and the final answer."""
    calls = [tc for m in messages if isinstance(m, AIMessage) for tc in m.tool_calls]
    results = [m for m in messages if isinstance(m, ToolMessage)]
    tokens = sum((m.usage_metadata or {}).get("total_tokens", 0) for m in messages if isinstance(m, AIMessage))
    final = messages[-1].content if isinstance(messages[-1], AIMessage) else ""
    return calls, results, tokens, final

def score_case(messages, expected_tools, facts, arg_check=None):
    """
    tool_ok    : the set of tools called equals the expected set (an empty set means "answer directly").
    args_ok    : no tool call came back as an error and, if given, arg_check(calls) holds.
    success    : every expected fact appears in the final answer (case-insensitive) and the tools were right.
    """
    calls, results, tokens, final = trajectory(messages)
    tool_ok = {c["name"] for c in calls} == set(expected_tools)
    args_ok = all(r.status != "error" for r in results) and (arg_check(calls) if arg_check and calls else True)
    success = tool_ok and all(f.lower() in final.lower() for f in facts)
    return {"tool_ok": tool_ok, "args_ok": args_ok, "success": success, "n_calls": len(calls),
            "n_errors": sum(r.status == "error" for r in results), "tokens": tokens, "answer": final}
'''


def build():
    nb = NB()
    nb.md(r'''
# 16 · Tools, function calling and structured output with LangChain

A language model only produces text. It cannot look up an order, read a database or do exact arithmetic. **Tool calling** (also called *function calling*) is the convention that lets it ask *you* to do those things:

1. You describe each tool to the model with a name, a description and a **JSON schema** for its arguments.
2. The model answers either with text or with a **tool call**: a structured request `{"name": ..., "args": {...}, "id": ...}`.
3. Your code validates the arguments, runs the function, and sends the result back as a **tool message** that carries the same `id`.
4. The model reads the result and either calls another tool or writes the final answer.

The model never executes anything; your code always does. An **agent** is this exchange in a loop with a goal and some limits. **Structured output** uses the same machinery for a different purpose: instead of calling a function, the model fills in a schema (a Pydantic model) so your program gets typed data instead of prose.

**What you will learn**
- How LangChain turns a Python function into the JSON schema the model sees (`@tool`, type hints, docstrings, Pydantic `args_schema`)
- `bind_tools`, the `AIMessage.tool_calls` the model returns, and `ToolMessage`
- The agent loop by hand (a dozen lines), then the same loop with `langchain.agents.create_agent`
- Error handling: invalid arguments and failing tools are returned to the model so it can correct itself
- Structured output with `with_structured_output`: the three methods, validation, `include_raw`, and retries that feed the error back
- An evaluation on labelled requests: tool-selection accuracy, argument validity, task success, latency and tokens

**Stack:** `langchain-core` (tools, messages), `langchain-ollama` (`ChatOllama` running `qwen2.5:3b` locally), `langchain` (`create_agent`), `pydantic`. If `.env` has a LangSmith key, every call below is traced to the project `llm-edu-suite-11-tools`; without one, nothing changes.
''')
    nb.md(r'''
## Background: how a text generator learned to use tools

### The problem tool calling solves

A pretrained language model has four limits that no amount of prompting removes. Its knowledge stops at the training cut-off. It has never seen your private data (orders, tickets, internal docs). It is unreliable at exact symbolic work such as arithmetic, counting or date maths, because it predicts tokens rather than executing algorithms. And it cannot *act*: it cannot send an email or write to a database. Retrieval (notebook 18) addresses the first two for read-only knowledge. Tools address all four, as long as the model can express a request that a program can execute.

That last condition is the real difficulty. The model emits a stream of tokens; a program needs a function name and typed arguments. Everything in this notebook is about building a reliable bridge between the two.

### A short history of the bridge

| year | work | idea |
|---|---|---|
| 2022 | MRKL (Karpas et al., *MRKL Systems*) | a language model routes sub-problems to external "expert" modules (calculator, APIs) |
| 2022 | ReAct (Yao et al., *ReAct: Synergizing Reasoning and Acting in Language Models*) | prompt the model to write `Thought / Action / Observation` lines; the program parses the `Action` line with string matching |
| 2023 | Toolformer (Schick et al., *Toolformer: Language Models Can Teach Themselves to Use Tools*) | self-supervised: the model inserts candidate API calls into text, and a call is kept for training only if its result makes the following tokens easier to predict |
| 2023 | Gorilla (Patil et al., *Gorilla: Large Language Model Connected with Massive APIs*) | fine-tune on API documentation so the model writes correct calls for thousands of APIs |
| 2023 | OpenAI function calling (June 2023) | the product form used today: JSON schemas in the request, a structured `tool_calls` field in the response, models fine-tuned for it |
| 2023-24 | ToolLLM (Qin et al.), Berkeley Function-Calling Leaderboard (Yan et al.) | large tool-use datasets and benchmarks that measure tool selection and argument correctness separately |

The early approaches parsed free text, and they broke whenever the model added a word, forgot a bracket or wrote the action in a slightly different format. The industry converged on *native* tool calling: the format is part of the model's training, and the serving layer turns it into structured data.

### Why tool calling is trained into the model

A model that has only been prompted to "reply with JSON when you need a tool" spreads its probability over many plausible continuations, and some of them are not valid calls. Fine-tuning on thousands of conversations that contain tool schemas, tool calls and tool results concentrates that probability on the exact format. The chat template marks the parts with special tags; Qwen2.5, the model we use, lists the tools inside `<tools>` tags in the system prompt and writes each call as JSON inside `<tool_call>` tags, which Ollama parses into `tool_calls`.

Training teaches four distinct skills, and it helps to keep them apart because they fail independently:

1. **Deciding** whether a tool is needed at all (or whether to answer directly).
2. **Selecting** the right tool among the ones offered.
3. **Filling the arguments**: extracting values from the conversation and mapping them onto the schema, essentially the slot-filling task of classic dialogue systems.
4. **Using the result**: reading the tool output and either calling another tool or writing an answer grounded in it.

Section 7 measures the first three separately (tool selection, argument validity) and the fourth through task success.

### Schemas are the contract

The interface between model and code is a **JSON Schema** per tool. It plays two roles at once. For the model it is *documentation*: names, descriptions, enums and field descriptions are the only information it has about what a tool does. For your code it is a *validator*: arguments that do not match are rejected before the function runs. This double role is why a Pydantic model is a convenient source of truth: one class produces the schema the model reads and the validator that checks what it wrote. Providers support different subsets of JSON Schema (keywords such as `pattern`, `minLength` or nested `oneOf` are not honoured everywhere), so do not rely on the model respecting a constraint just because it appears in the schema; validate on your side.

### Three levels of guarantee for structured output

When you want *data* back instead of an action, there are three increasingly strong ways to get it, and they matter again in section 6:

| level | mechanism | what is guaranteed |
|---|---|---|
| prompting | "answer in JSON with keys ..." in the prompt, then parse | nothing; the model usually complies |
| JSON mode | the decoder only allows tokens that keep the output valid JSON | syntactically valid JSON, any keys |
| schema-constrained decoding | the decoder only allows tokens that keep the output valid *for your schema* | valid JSON with the right keys, types and enum values |

Whatever the level, validate the result in your own code. Business rules ("at most 12 words", "the refund cannot exceed the order total") are beyond what any decoder enforces.

### How constrained decoding enforces a schema

At every step a language model produces a vector of logits $z_t \in \mathbb{R}^{|V|}$, one per token of the vocabulary $V$, and samples the next token from $\mathrm{softmax}(z_t)$. Constrained decoding adds a **mask** computed from a formal grammar:

$$
p(y_t \mid y_{<t}) \;\propto\; \exp\big(z_{t,y_t} + m_t(y_t)\big), \qquad
m_t(v) = \begin{cases} 0 & \text{if } y_{<t}\,v \text{ is a valid prefix of the language} \\ -\infty & \text{otherwise} \end{cases}
$$

Tokens that would break the grammar get probability zero, so whatever the model samples is, by construction, a prefix of a valid document. When generation stops, the output is valid. A JSON schema is first compiled into a grammar (a regular expression or a context-free grammar), and the decoder tracks the grammar state as tokens are produced. A small example, partway through a `Ticket`:

```
generated so far:   {"category": "refund", "urgency": "
grammar says:       the next characters must spell low, medium or high, then a closing quote
allowed tokens:     "low", "lo", "l", "medium", "med", "m", "high", "h", ...   (prefixes of the enum)
masked (-inf):      "urgent", "very", "}", "\n", digits, every other token
```

The hard part is efficiency. The grammar works on characters, the model on tokens that can span several grammar symbols (a single token might be `": "` or `"},`), and the vocabulary has on the order of 150,000 entries. Checking every token at every step would be slow. Willard and Louf (2023, *Efficient Guided Generation for Large Language Models*, the Outlines library) showed how to precompute, for each state of a finite-state machine, the set of tokens that are allowed, turning the mask into a table lookup. XGrammar (Dong et al., 2024) extends this to context-free grammars, which JSON needs for nesting, by precomputing the masks for the tokens that do not depend on the parser stack. llama.cpp, the engine underneath Ollama, implements the same idea with its GBNF grammar format; when you pass a schema as Ollama's `format`, it is converted into such a grammar. Earlier work applied the same principle to SQL (PICARD, Scholak et al., 2021) and to code (Synchromesh, Poesia et al., 2022).

With precomputation the overhead per token is small, and some engines are even *faster* with a grammar, because they can emit forced tokens (a closing quote, a fixed key name) without running the model ("jump-forward" decoding in SGLang).

### What constrained decoding does not guarantee

- **Content.** The mask guarantees the shape. A field can be perfectly typed and still wrong, or `null` when a value was present; section 6 shows exactly this.
- **An unbiased distribution.** The mask is applied greedily, token by token. The model is pushed down paths it may consider unlikely, and the result is not the same as sampling from the model's distribution restricted to valid outputs. Park et al. (2024, *Grammar-Aligned Decoding*) analyse this distortion.
- **The same quality of reasoning.** Tam et al. (2024, *Let Me Speak Freely?*) reported that strict format restrictions can hurt reasoning tasks; other practitioners argued that the effect came from prompt differences and disappears with good prompts. This is still debated. A practical habit that both sides agree on: put a free-text `reasoning` field *before* the answer fields, because the model generates keys in schema order and can then "think" before committing.
- **Every schema feature.** Recursive schemas, `pattern`, numeric bounds or length limits may be ignored or unsupported, depending on the engine.

Note also that, in most open-source serving stacks including Ollama, *tool calls* are not grammar-constrained by default: the model writes the call as text and the server parses it. Several hosted providers offer a "strict" option that applies schema-constrained decoding to tool arguments as well.

### An agent is a loop, and loops have a cost structure

An agent repeats: call the model, run the requested tools, append the results, call the model again. Every model call re-sends the whole conversation, including the tool schemas. If the fixed part of the prompt (system prompt and schemas) is $P$ tokens and each step adds about $g$ tokens of calls and results, then $k$ steps cost roughly

$$
\text{input tokens} \approx kP + g\,\frac{k(k-1)}{2},
$$

linear in the fixed prompt and quadratic in the history. That is why the number of steps, not the length of the final answer, dominates the cost and latency of an agent, and why limits on steps are not optional (notebook 25). Prompt caching on hosted APIs reduces the price of the repeated prefix, but not the fact that it is processed.

### Common failure modes

| failure | what it looks like | usual mitigation |
|---|---|---|
| wrong decision | the model answers from memory when it should call a tool, or calls a tool for chit-chat | clearer descriptions, a system prompt that says when tools are required, evaluation |
| wrong selection | a similar-sounding tool is picked | fewer tools per prompt, distinct names, routing (notebook 25) |
| bad arguments | missing fields, wrong types, invented ids | schema validation, errors returned to the model, normalisation inside the tool |
| hallucinated tool | a call to a tool that does not exist | reject with a clear message |
| loops | the same call repeated | step and spend limits |
| prompt injection through tool output | a retrieved document says "ignore your instructions and ..." | treat tool output as data, gate actions behind confirmation (Greshake et al., 2023, *Not what you've signed up for*; notebook 19) |
| excessive agency | a tool can do far more than the task needs | least privilege: narrow tools, read-only by default |

### Key references

- Yao et al., 2022. *ReAct: Synergizing Reasoning and Acting in Language Models.*
- Karpas et al., 2022. *MRKL Systems: A modular, neuro-symbolic architecture.*
- Schick et al., 2023. *Toolformer: Language Models Can Teach Themselves to Use Tools.*
- Patil et al., 2023. *Gorilla: Large Language Model Connected with Massive APIs.*
- Qin et al., 2023. *ToolLLM: Facilitating Large Language Models to Master 16000+ Real-world APIs.*
- Willard and Louf, 2023. *Efficient Guided Generation for Large Language Models.*
- Dong et al., 2024. *XGrammar: Flexible and Efficient Structured Generation Engine for Large Language Models.*
- Tam et al., 2024. *Let Me Speak Freely? A Study on the Impact of Format Restrictions on Performance of Large Language Models.*
- Greshake et al., 2023. *Not what you've signed up for: Compromising Real-World LLM-Integrated Applications with Indirect Prompt Injection.*
''')
    nb.code(setup_apps("llm-edu-suite-11-tools"))
    nb.md(r'''
## 1. Describing tools

The `@tool` decorator reads three things from a function: its **name**, its **docstring** (which becomes the description) and its **type hints** (which become the argument schema). When you need richer constraints, such as per-field descriptions or a regex pattern, give it a Pydantic model as `args_schema`. We define four tools for a small online shop: a weather lookup, a calculator, an orders lookup and a search over the company documentation (a LangChain vector store with `nomic-embed-text` embeddings; notebook 18 covers retrieval in depth).
''')
    nb.code(TOOLS_CELL + r'''
llm = get_llm()          # ChatOllama(qwen2.5:3b, temperature=0)
print([t.name for t in TOOLS])
''')
    nb.md(r'''
This is exactly what the model receives for two of the tools: the OpenAI-style function schema that every major provider (and Ollama) accepts. Note where each piece came from: the description is the docstring, `unit` became an `enum` because of the `Literal` hint and is optional because it has a default, and `order_id` carries the description and the regex pattern from the Pydantic field.
''')
    nb.code(r'''
from langchain_core.utils.function_calling import convert_to_openai_tool
for t in (get_weather, order_status):
    print(json.dumps(convert_to_openai_tool(t), indent=1))
''')
    nb.md(r'''
The description is a prompt. The model chooses tools by reading these few lines, so a vague description ("does stuff with orders") produces wrong calls just as a vague prompt produces wrong answers. Treat tool descriptions as part of your prompt engineering and evaluate changes to them (section 6).

## 2. `bind_tools` and the tool-call message

`llm.bind_tools(TOOLS)` returns a model that sends those schemas with every request. The reply is an `AIMessage`; when the model wants a tool, its `content` is usually empty and `tool_calls` holds the parsed requests. When no tool is useful, the model answers in text and `tool_calls` is empty.
''')
    nb.code(r'''
llm_with_tools = llm.bind_tools(TOOLS)
for question in ["What is (17 * 23) + 4?", "What's the weather in Lima, in Fahrenheit?", "Hello! Who wrote Don Quixote?"]:
    t0 = time.time()
    ai = llm_with_tools.invoke(question)
    print(f"Q: {question}  ({time.time() - t0:.1f}s, {ai.usage_metadata['input_tokens']} prompt tokens)")
    print("   content   :", repr(ai.content[:80]))
    print("   tool_calls:", ai.tool_calls)
''')
    nb.md(r'''
The first two questions produced a tool call with well-formed arguments (note `unit: "F"` taken from the enum). The third shows a typical small-model weakness: with tools attached, `qwen2.5:3b` declined to answer a general-knowledge question it could have answered on its own. Tools change the model's behaviour even when they are not called; the evaluation in section 7 measures this.

Look at the prompt-token counts: the tool schemas are sent as part of the prompt on every call, so each tool you add costs tokens on every request whether or not it is used. That is one reason not to give one agent dozens of tools (notebook 25 routes requests to specialists instead).

Under the hood, Ollama's chat template for Qwen2.5 writes the schemas into the system prompt, the model generates a tagged JSON block, and the server parses it back into `tool_calls`. Models are fine-tuned on this format, which is why tool calling works far better with models trained for it than with prompting an arbitrary model to "answer in JSON". You can also force a tool with `bind_tools(TOOLS, tool_choice="calculate")` or force *some* tool with `tool_choice="any"`.

## 3. The loop by hand

To answer, the model needs the tool result. We run the tool and append a `ToolMessage` whose `tool_call_id` matches the call, then call the model again with the whole conversation. Calling `tool.invoke(tool_call)` with the full tool-call dict does the bookkeeping for us: it validates the arguments against the schema, runs the function and returns a `ToolMessage` with the right id.

**Why every call carries an id.** A model can request several tools in a single turn (*parallel tool calling*), for example the weather in two cities at once. The id pairs each result with its request, so the results can arrive in any order and the model still knows which is which. Most providers reject a conversation in which a tool call has no matching result, which is why even a failed call must be answered with a `ToolMessage`.

**The message list is the whole state.** The model keeps nothing between calls; it is a stateless function from a list of messages to a new message. Everything the agent "knows" during a run (the question, its own earlier calls, every tool result) lives in that list, which you re-send each time. This is what makes agents easy to log, replay and test, and also what makes them expensive as runs get longer.
''')
    nb.code(BY_HAND_CELL + r'''
messages = run_by_hand(llm, TOOLS, "What is the total of order A100 plus 21% tax?")
print("\nfinal answer:", messages[-1].content)
print("message types:", [type(m).__name__ for m in messages])
''')
    nb.md(r'''
That loop is all an agent is. The production versions add the parts we left out: parallel tool execution, error handling, limits on steps and spend, streaming, persistence, human approval, tracing.

## 4. The same loop with `create_agent`

`langchain.agents.create_agent` (LangChain 1.x; it replaces the deprecated `langgraph.prebuilt.create_react_agent`) builds this loop as a LangGraph graph: a `model` node and a `tools` node with an edge back. Its input and output are a dict with the list of `messages`.

The reason to express a dozen-line loop as a graph is everything around it. Once the loop is a graph, each step is a node with a recorded input and output, so the framework can save the state after every step (and resume after a crash or a human approval), stream intermediate steps, enforce a step limit, and let **middleware** hook in before and after each model or tool call. Middleware is the extension point used below for error handling and in notebook 25 for limits, approvals and memory. The design idea is the same as web-framework middleware: cross-cutting policies live in small composable wrappers instead of being mixed into the loop.
''')
    nb.code(r'''
from langchain.agents import create_agent

agent = create_agent(llm, TOOLS, system_prompt="You are a helpful assistant for an online shop. Use the tools for facts and arithmetic.")
t0 = time.time()
result = agent.invoke({"messages": [("user", "What's the weather in Madrid, and what is order C300's total minus 15 dollars?")]})
for m in result["messages"]:
    m.pretty_print()
print(f"\n{time.time() - t0:.1f}s")
''')
    nb.code(r'''
print(agent.get_graph().draw_mermaid())       # paste into https://mermaid.live to see it; notebook 25 builds graphs like this by hand
''')
    nb.md(r'''
## 5. Errors go back to the model

Two different things can go wrong, and the agent treats them differently.

**Errors as observations.** In the ReAct view of an agent, an error is simply another observation: the model acted, the world answered "that did not work, because ...", and the next step can take that into account. This works for a specific reason. Language models are poor at finding their own mistakes without outside help (Huang et al., 2024, *Large Language Models Cannot Self-Correct Reasoning Yet*), but they are reasonably good at fixing a mistake once an external checker has pointed at it (Gou et al., 2024, *CRITIC*, uses tools as that checker). A schema validator or a failing function is exactly such a checker. The quality of the error message is therefore part of the design: "order_id must match ^[A-Z]\d{3}$, e.g. A100" gives the model something to act on; "invalid input" does not.

There is also a security side. Exception text can contain file paths, SQL, stack traces or secrets, and whatever you send to the model may end up in its answer to the user. That is why LangChain returns validation errors automatically (they only describe the schema) but makes you opt in for execution errors.

**Invalid arguments.** The arguments do not match the schema (a missing field, a wrong type, here the `order_id` pattern). The tool node catches the validation error *before* running the function and returns it as a `ToolMessage` with `status="error"`, so the model can read what was wrong and try again. To show it deterministically, we replace the model with a scripted fake (`GenericFakeChatModel` from `langchain_core`, the same one used in the tests) that first sends a bad id and then a corrected one.
''')
    nb.code(r'''
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

class ScriptedModel(GenericFakeChatModel):
    """Replays a fixed list of AIMessages. bind_tools is a no-op because the script already contains the tool calls."""
    def bind_tools(self, tools, **kwargs):
        return self

script = [AIMessage("", tool_calls=[{"name": "order_status", "args": {"order_id": "a-100"}, "id": "call_1"}]),
          AIMessage("", tool_calls=[{"name": "order_status", "args": {"order_id": "A100"}, "id": "call_2"}]),
          AIMessage("Order A100 has shipped.")]
out = create_agent(ScriptedModel(messages=iter(script)), TOOLS).invoke({"messages": [("user", "status of order a-100?")]})
for m in out["messages"]:
    if isinstance(m, ToolMessage):
        print(f"ToolMessage status={m.status}: {m.content}\n")
''')
    nb.md(r'''
**The tool itself fails.** The arguments are valid but the function raises (a timeout, a missing record, here an unsupported operator). By default the exception propagates and the run stops: LangChain does not send arbitrary exception text to the model unless you opt in, because it can leak internal details. You opt in with `ToolErrorMiddleware`, choosing which exceptions become a message and what that message says.
''')
    nb.code(r'''
from langchain.agents.middleware import ToolErrorMiddleware

bad = [AIMessage("", tool_calls=[{"name": "calculate", "args": {"expression": "2^10"}, "id": "call_1"}]), AIMessage("done")]
try:
    create_agent(ScriptedModel(messages=iter(bad)), TOOLS).invoke({"messages": [("user", "2^10?")]})
except ValueError as e:
    print("without middleware the run crashes:", e)

def tool_error_to_message(exc, request):
    """Only ValueErrors (bad input) go back to the model; anything else still stops the run."""
    if isinstance(exc, ValueError):
        return f"{request.tool_call['name']} failed: {exc}. Fix the input and try again."
    return None

safe_agent = create_agent(llm, TOOLS, middleware=[ToolErrorMiddleware(tool_error_to_message)])
out = safe_agent.invoke({"messages": [("user", "Use the calculator to compute 2^10, writing the expression exactly as 2^10.")]})
for m in out["messages"][1:]:
    m.pretty_print()
''')
    nb.md(r'''
Read the transcript above: the error message reached the model and its next turn reacts to it. A 3B model does not always recover the way you would like (it may ask the user instead of retrying with `**`); a precise error message ("use `**` for powers") is what gives it the best chance. Whatever the model does, the run no longer crashes and the failure is visible in the trace.

## 6. Structured output

Often you do not want an action but **data**: the fields of a support ticket, a classification, a plan. `llm.with_structured_output(Schema)` returns a runnable whose output is an instance of your Pydantic model. `ChatOllama` offers three ways to get there:

| method | how it works | guarantees |
|---|---|---|
| `json_schema` (default) | the schema is sent as Ollama's `format`; the server constrains decoding with a grammar so only tokens that keep the JSON valid can be sampled | valid JSON with the right keys and types |
| `function_calling` | the schema is offered as a tool the model must call | as good as the model's tool calling |
| `json_mode` | the server only guarantees *some* valid JSON; the schema has to be described in the prompt | valid JSON, nothing more |

**Trade-offs between the methods.** `json_schema` relies on the grammar-masked decoding described in the Background: the shape is guaranteed, at the price of the distortions discussed there and of depending on what subset of JSON Schema the server compiles. `function_calling` reuses the model's tool-calling training, which for many models is the format they have seen most often in fine-tuning, but on Ollama the call is parsed from free text, so the shape is not guaranteed. `json_mode` is the most portable and the weakest: it works with any server that has a JSON grammar, and everything about the schema depends on the prompt. None of the three changes the cost much; the prompt (schema plus instructions) dominates the tokens, and the grammar adds little latency per token.

Whatever the method, **Pydantic validates the result on the client**: types, enums (`Literal`), and any custom validators with business rules. Below, the ticket summary must be at most 12 words, a rule the decoder cannot enforce.
''')
    nb.code(r'''
from typing import Optional
from pydantic import field_validator

class Ticket(BaseModel):
    """A customer support ticket extracted from an email."""
    customer_name: str
    email: str = Field(description="the customer's email address")
    order_id: Optional[str] = Field(None, description="the order id, like A100, if one is mentioned")
    category: Literal["refund", "shipping", "account", "other"]
    urgency: Literal["low", "medium", "high"]
    summary: str = Field(description="one short sentence, at most 12 words")

    @field_validator("summary")
    @classmethod
    def summary_is_short(cls, v):
        if len(v.split()) > 12:
            raise ValueError(f"summary has {len(v.split())} words but the limit is 12")
        return v

email = ("Hi, this is Maria Lopez (maria.lopez@example.com). My order C300 arrived broken and I want my money back "
         "as soon as possible, it was a birthday present and the party is tomorrow.")
extractor = llm.with_structured_output(Ticket)               # method="json_schema"
ticket = extractor.invoke(email)
print(type(ticket).__name__, ticket.model_dump())
''')
    nb.md(r'''
With `include_raw=True` the runnable does not raise when validation fails. It returns a dict with the `raw` `AIMessage`, the `parsed` object (or `None`) and the `parsing_error`, which is what you want in production: log the raw output, then decide what to do. Here we use `json_mode` with a prompt that lists the keys but, as happens in real code, forgets to mention the 12-word rule: the output is valid JSON, but it can still break the business rule.
''')
    nb.code(r'''
from pydantic import ValidationError

KEYS_PROMPT = ("Return a JSON object with the keys customer_name, email, order_id, category (refund, shipping, account or other), "
               "urgency (low, medium or high) and summary (a summary of the email).")

def validation_message(parsing_error):
    """The useful part of a parsing error: Pydantic's per-field messages (the parser keeps the ValidationError as __cause__)."""
    cause = parsing_error.__cause__
    if isinstance(cause, ValidationError):
        return "; ".join(f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in cause.errors())
    return str(parsing_error)[:300]                     # not even valid JSON

loose = llm.with_structured_output(Ticket, method="json_mode", include_raw=True)
r = loose.invoke([("system", KEYS_PROMPT), ("user", email)])
print("raw output :", r["raw"].content)
print("parsed     :", r["parsed"])
print("error      :", validation_message(r["parsing_error"]) if r["parsing_error"] else None)
''')
    nb.md(r'''
**Retrying.** `runnable.with_retry()` simply repeats the call, and at temperature 0 the same input tends to give the same output, so a blind retry rarely fixes a validation error. What works is to **send the error back**: append the model's bad output and the validation message to the conversation and ask again, the same idea as tool errors in section 5.
''')
    nb.code(r'''
def extract_with_feedback(text, system=KEYS_PROMPT, schema=Ticket, method="json_mode", max_attempts=3):
    """Structured output with validation feedback: on failure, show the model its output and the error, and ask again."""
    structured = llm.with_structured_output(schema, method=method, include_raw=True)
    messages = [("system", system), ("user", text)]
    for attempt in range(1, max_attempts + 1):
        r = structured.invoke(messages)
        if r["parsing_error"] is None:
            return r["parsed"], attempt
        error = validation_message(r["parsing_error"])
        print(f"attempt {attempt} failed: {error}")
        messages += [r["raw"], ("user", f"Your JSON failed validation: {error}. Return the corrected JSON object only.")]
    return None, max_attempts

parsed, attempts = extract_with_feedback(email)
print(f"after {attempts} attempt(s):", parsed)
''')
    nb.md(r'''
Each failed attempt printed the exact validation message, and the feedback did move the model: the summary shrank from 22 words to 13. But 13 is still over the limit, the third attempt repeated it, and after 3 attempts the function gave up and returned `None`. Feedback improves the odds; it is not a guarantee, especially with a 3B model and a rule that needs counting (language models count words badly, because they see tokens). So cap the attempts (here 3), always have a fallback (enforce the rule in code, for example by truncating the summary; escalate to a bigger model; or send the raw output to review), and log how often retries are needed: a rising retry rate is an early sign that a prompt or model change hurt.
''')
    nb.md(r'''
### Which method should you use? A small measurement

Eight labelled emails, the three methods, the same `Ticket` schema, and this time the full JSON schema (with the field descriptions) in the system prompt for every method: `json_mode` needs it, and for the other two it keeps the comparison fair. We report the share of outputs that pass validation and, among all eight, the share of correctly extracted fields (name, email, order id, category, urgency; an invalid output counts as all fields wrong).
''')
    nb.code(r'''
EMAILS = [
    ("Hi, this is Maria Lopez (maria.lopez@example.com). My order C300 arrived broken and I want my money back as soon as possible, it was a birthday present and the party is tomorrow.",
     {"customer_name": "Maria Lopez", "email": "maria.lopez@example.com", "order_id": "C300", "category": "refund", "urgency": "high"}),
    ("Hello, I can't log into my account since yesterday, it says my password is wrong. Not urgent. - Tom Becker, tom@becker.dev",
     {"customer_name": "Tom Becker", "email": "tom@becker.dev", "order_id": None, "category": "account", "urgency": "low"}),
    ("Where is my parcel? Order A100 was supposed to arrive last week and the tracking hasn't moved. Please check today. Jane Doe <jane.doe@mail.com>",
     {"customer_name": "Jane Doe", "email": "jane.doe@mail.com", "order_id": "A100", "category": "shipping", "urgency": "high"}),
    ("Dear team, could you tell me if you plan to sell gift wrapping? Just curious, no rush. Best, Li Wei (li.wei@example.org)",
     {"customer_name": "Li Wei", "email": "li.wei@example.org", "order_id": None, "category": "other", "urgency": "low"}),
    ("I was charged twice for order B200. I'd like the duplicate payment refunded. Thanks, Omar Haddad, omar.h@example.net",
     {"customer_name": "Omar Haddad", "email": "omar.h@example.net", "order_id": "B200", "category": "refund", "urgency": "medium"}),
    ("Someone changed my account email without my permission!!! Fix this immediately. Anna Schmidt anna.schmidt@example.de",
     {"customer_name": "Anna Schmidt", "email": "anna.schmidt@example.de", "order_id": None, "category": "account", "urgency": "high"}),
    ("Hi, how many days does standard shipping to Canada take for order C300? Thanks, Paul Martin - paul.martin@example.ca",
     {"customer_name": "Paul Martin", "email": "paul.martin@example.ca", "order_id": "C300", "category": "shipping", "urgency": "low"}),
    ("Good morning. I returned the shoes from order A100 two weeks ago and still have no refund. Sofia Rossi (sofia.rossi@example.it)",
     {"customer_name": "Sofia Rossi", "email": "sofia.rossi@example.it", "order_id": "A100", "category": "refund", "urgency": "medium"}),
]
FIELDS = ["customer_name", "email", "order_id", "category", "urgency"]
SCHEMA_PROMPT = ("Extract a support ticket from the email. Reply with a JSON object that follows this JSON schema:\n"
                 + json.dumps(Ticket.model_json_schema()))
so_results, so_misses = {}, {}
for method in ["json_schema", "function_calling", "json_mode"]:
    structured = llm.with_structured_output(Ticket, method=method, include_raw=True)
    valid, correct, seconds = 0, 0, 0.0
    for text, gold in EMAILS:
        t0 = time.time()
        r = structured.invoke([("system", SCHEMA_PROMPT), ("user", text)])
        seconds += time.time() - t0
        if r["parsed"] is not None:
            valid += 1
            correct += sum(getattr(r["parsed"], f) == gold[f] for f in FIELDS)
            so_misses.setdefault(method, []).extend((f, gold[f], getattr(r["parsed"], f)) for f in FIELDS if getattr(r["parsed"], f) != gold[f])
        else:
            so_misses.setdefault(method, []).append(("invalid output", "", validation_message(r["parsing_error"])[:100]))
    so_results[method] = {"valid": valid / len(EMAILS), "field_acc": correct / (len(EMAILS) * len(FIELDS)), "sec": seconds / len(EMAILS)}
    print(f"{method:17s} valid {valid}/{len(EMAILS)}   field accuracy {so_results[method]['field_acc']:.0%}   {so_results[method]['sec']:.2f}s per email")
for method, misses in so_misses.items():
    print(f"\n{method} mistakes (field, expected, got):", misses)
''')
    nb.code(r'''
fig, ax = plt.subplots(figsize=(7, 3.2))
methods = list(so_results); xs = np.arange(len(methods))
ax.bar(xs - 0.2, [so_results[m]["valid"] for m in methods], 0.4, label="passes validation")
ax.bar(xs + 0.2, [so_results[m]["field_acc"] for m in methods], 0.4, label="field accuracy")
for x, m in zip(xs, methods):
    ax.text(x - 0.2, so_results[m]["valid"] + 0.02, f"{so_results[m]['valid']:.0%}", ha="center")
    ax.text(x + 0.2, so_results[m]["field_acc"] + 0.02, f"{so_results[m]['field_acc']:.0%}", ha="center")
ax.set_xticks(xs); ax.set_xticklabels(methods); ax.set_ylim(0, 1.15); ax.legend(loc="lower right"); ax.grid(axis="y", alpha=.3)
ax.set_title(f"structured output with qwen2.5:3b ({len(EMAILS)} emails)"); plt.tight_layout(); plt.show()
''')
    nb.md(r'''
In our run every method produced a valid `Ticket` for all eight emails (with the full schema in the prompt, the 12-word rule was respected), but they were not equally *correct*. `json_schema` returned `order_id = null` for every email that mentions an order, while `function_calling` and `json_mode` extracted the ids and each missed a single urgency label. Constrained decoding guarantees the **shape** of the output, not its **content**; which method works best depends on the model and the schema, and the only way to know is to measure field by field as above. Urgency is also the most subjective label (is a double charge "medium" or "high"?), so part of that disagreement is label noise.

## 7. Evaluation: does the agent pick the right tool with the right arguments?

Agents are evaluated on the **trajectory**, not just on the final text. For each labelled request we record:

- **tool-selection accuracy**: the set of tools called equals the expected set (including "no tool" for chit-chat);
- **argument validity**: no tool call came back with an error (schema violations count) and the arguments pass a check;
- **task success**: right tools and every expected fact in the final answer;
- **latency** and **tokens** (summed over all model calls in the run, from `usage_metadata`).

Run this after every change to a prompt, a tool description or the model: these numbers move with small wording changes.

### How tool use is evaluated in the literature

Our metrics follow the same decomposition used by public benchmarks. The **Berkeley Function-Calling Leaderboard** (Yan et al., 2024) checks calls in two ways: by comparing the call's abstract syntax tree with the expected one (right function, right argument names and values, independent of formatting) and by executing it and comparing results. It also has a *relevance* category in which no tool should be called, the equivalent of our chit-chat cases. **ToolLLM / ToolBench** (Qin et al., 2023) evaluates multi-step tool use over thousands of real APIs. **τ-bench** (Yao et al., 2024) simulates a user and checks the final state of a database, and it reports *pass^k*: the probability that the agent succeeds on **all** of $k$ independent attempts at the same task. Pass^k exposes inconsistency that a single run hides, which matters for agents because users repeat similar requests many times.

Two points from that literature are worth keeping in mind for our small set. First, **trajectory and outcome are different things**: an agent can reach the right answer by the wrong path (guessing a number instead of computing it), and an outcome-only metric rewards that. Second, **variance is large**: with 14 cases, one case is about 7 percentage points, and a small model's behaviour shifts between runs, so treat differences of one or two cases as noise.
''')
    nb.code(EVAL_HELPERS_CELL + r'''
city_is = lambda name: (lambda calls: any(c["args"].get("city", "").lower() == name for c in calls))
CASES = [  # (request, expected tools, facts the answer must contain, optional check on the arguments)
    ("What's the weather in Lima right now?", ["get_weather"], ["19"], city_is("lima")),
    ("What is the temperature in Madrid in Fahrenheit?", ["get_weather"], ["75"], lambda calls: any(c["args"].get("unit") == "F" for c in calls)),
    ("What is 17 * 23 + 4?", ["calculate"], ["395"], None),
    ("Compute (120 - 45.5) / 2", ["calculate"], ["37.25"], None),
    ("What is the status of order A100?", ["order_status"], ["shipped"], None),
    ("Has order B200 shipped yet?", ["order_status"], ["processing"], None),
    ("Look up order c300 for me", ["order_status"], ["delivered"], None),
    ("How long does express shipping take?", ["search_docs"], ["1 to 2"], None),
    ("Within how many days can I get a full refund?", ["search_docs"], ["30"], None),
    ("What is the rate limit of the public API?", ["search_docs"], ["100"], None),
    ("Hi! Tell me one fun fact about octopuses.", [], [], None),
    ("Translate 'good morning' into Spanish.", [], ["buenos"], None),
    ("What's the weather in Cusco, and what is 12 * 3?", ["get_weather", "calculate"], ["12", "36"], city_is("cusco")),
    ("What is the total of order C300 plus 15 dollars for express shipping?", ["order_status", "calculate"], ["95"], None),
]
eval_agent = create_agent(llm, TOOLS, system_prompt="You are a helpful assistant for an online shop. Use the tools for facts and arithmetic; answer directly when no tool is needed.",
                          middleware=[ToolErrorMiddleware(tool_error_to_message)])
rows = []
for text, tools_expected, facts, check in CASES:
    t0 = time.time()
    out = eval_agent.invoke({"messages": [("user", text)]}, {"recursion_limit": 12})
    s = score_case(out["messages"], tools_expected, facts, check)
    s["latency"] = time.time() - t0; s["request"] = text
    s["called"] = [c["name"] for c in trajectory(out["messages"])[0]]
    rows.append(s)
    print(f"{'OK ' if s['success'] else 'BAD'} tools={'ok' if s['tool_ok'] else 'WRONG'} args={'ok' if s['args_ok'] else 'ERR'} "
          f"{s['latency']:4.1f}s {s['tokens']:5d} tok  {text[:55]:55s} called={s['called']}")
n = len(rows)
summary = {k: sum(r[k] for r in rows) / n for k in ("tool_ok", "args_ok", "success")}
print(f"\ntool selection {summary['tool_ok']:.0%} | argument validity {summary['args_ok']:.0%} | task success {summary['success']:.0%} "
      f"| median latency {np.median([r['latency'] for r in rows]):.1f}s | mean tokens {np.mean([r['tokens'] for r in rows]):.0f}")
''')
    nb.code(r'''
for r in rows:
    if not r["success"]:
        print(f"FAILED: {r['request']}\n   called {r['called']}, {r['n_errors']} tool errors\n   answer: {r['answer'][:220]!r}\n")
''')
    nb.code(r'''
fig, ax = plt.subplots(1, 3, figsize=(15, 4), gridspec_kw={"width_ratios": [1, 2, 2]})
names = ["tool selection", "argument validity", "task success"]
vals = [summary["tool_ok"], summary["args_ok"], summary["success"]]
ax[0].bar(names, vals, color=["tab:blue", "tab:orange", "tab:green"]); ax[0].set_ylim(0, 1.1)
for i, v in enumerate(vals): ax[0].text(i, v + 0.02, f"{v:.0%}", ha="center")
ax[0].tick_params(axis="x", rotation=20); ax[0].set_title(f"{n} labelled requests")
idx = np.arange(n); colors = ["tab:green" if r["success"] else "tab:red" for r in rows]
ax[1].bar(idx, [r["latency"] for r in rows], color=colors); ax[1].set_ylabel("seconds"); ax[1].set_title("latency per request (red = failed)")
ax[2].bar(idx, [r["tokens"] for r in rows], color=colors); ax[2].set_ylabel("tokens (all model calls)"); ax[2].set_title("tokens per request")
for a in ax[1:]:
    a.set_xticks(idx); a.set_xticklabels([str(i + 1) for i in idx]); a.set_xlabel("request #")
for a in ax: a.grid(axis="y", alpha=.3)
plt.tight_layout(); plt.show()
''')
    nb.md(r'''
**Reading the results.** Most single-tool lookups (weather, orders, documentation) were reliable. The failures printed above are instructive (the exact set changes a little between runs, which is itself a lesson about evaluating small models):

- **`Compute (120 - 45.5) / 2` called no tool**: the model did the arithmetic "in its head". It happened to get 37.25 right, but the trajectory is wrong, and on harder expressions this is exactly how wrong numbers reach users. Scoring the trajectory catches it; scoring only the final text would not.

- **`c300` in lower case** broke the `order_id` pattern. The validation error went back to the model, which asked the user for a correctly formatted id instead of upper-casing it. The schema protected the tool, but the friendlier design is to normalise the id inside the tool and keep strict validation for truly malformed input (notebook 25 does this).
- **The chit-chat request called `search_docs`**: small models over-use tools when tools are available.
- **The translation needed no tool and got none**, but the answer was wrong: a plain model-quality failure that no amount of tool plumbing fixes.
- **The order-plus-shipping request** skipped the order lookup and sent `calculate` an expression it could not evaluate. Multi-step requests are where a 3B model struggles most; notebook 25 measures that directly.

Latency depends heavily on how busy the GPU is (other processes shared it during our run), so compare latencies only within one run. Tokens are dominated by the tool schemas, which are re-sent with every model call (about 390 prompt tokens per call here before any conversation), so a run with two model calls costs roughly twice a run with one.

## Takeaways

- **Tool calling is a protocol, not magic:** schemas go in, `tool_calls` come out, your code runs the function and returns a `ToolMessage` with the matching id. `create_agent` is that loop as a LangGraph graph.
- **Descriptions and schemas are prompts.** Type hints, `Literal` enums, Pydantic field descriptions and patterns both guide the model and validate its output before anything runs.
- **Return errors to the model on purpose:** validation errors come back automatically; execution errors only when you opt in (`ToolErrorMiddleware`), with messages that say how to fix the call.
- **Structured output:** prefer the constrained `json_schema` method when the server supports it, keep business rules in Pydantic validators, use `include_raw=True` in production and retry *with the error*, not blindly.
- **Evaluate trajectories:** tool selection, argument validity, success, latency and tokens on a labelled set, every time a prompt, tool description or model changes. Production practice: sandbox code execution, log every trajectory (LangSmith), cap steps and spend, and treat tool output as untrusted data (notebook 19 covers prompt injection).
''')
    nb.md(r'''
## Results

| question | measured | lesson |
|---|---|---|
| does the model pick the right tool? | tool selection 79 % on 14 labelled requests | a 3B model both skips tools (did arithmetic "in its head") and over-uses them (chit-chat → `search_docs`) |
| are the arguments valid, and does the task succeed? | argument validity 86 %, task success 64 %; median latency 0.8 s, 864 tokens per request on average | score the trajectory, not only the final text |
| which structured-output method? | 8/8 valid outputs for all three; field accuracy `json_schema` 85 %, `function_calling` 98 %, `json_mode` 98 % | valid JSON is not correct JSON; `json_schema` dropped `order_id` 5 times |
| does retrying with the error fix a rule? | summary length 22 → 13 → 13 words against a limit of 12, still failing after 3 attempts | retries help with format, not with rules the model cannot follow; enforce those in code |
| what do the tool schemas cost? | about 390 prompt tokens per call with four tools, before any conversation | schemas are re-sent on every model call |

## In production

**How tool use is wired today (late 2026).**
- Native tool calling is part of every major model API and open serving engine; strict modes and constrained decoding (vLLM and SGLang with grammar backends such as XGrammar or Outlines) make schema-valid arguments the norm, so the remaining errors are semantic.
- MCP is the standard way to expose tools to any agent (notebook 26).
- Common frameworks: LangChain `create_agent` / LangGraph, the OpenAI Agents SDK, the Claude Agent SDK, Pydantic AI; Instructor for structured output alone.

**Engineering practice.**
- Few tools with clear descriptions; normalise inputs inside the tool and keep strict validation for real errors.
- Idempotent tools, timeouts, step and cost caps, and human approval for actions with side effects (notebook 25).
- Sandbox code execution; log every trajectory (notebook 28).
- Choose models with public tool-use benchmarks (BFCL, tau-bench), then confirm on your own labelled trajectories after every prompt, tool or model change.

**Failure modes.**
- Wrong or missing tool calls, invented arguments, loops that never stop.
- Selection gets worse as the number of tools grows.
- Tool output used as an injection channel (notebooks 19 and 30).
- Token cost of large tool schemas on every call.

**Next:** `17_embeddings_and_rerankers.ipynb`. The agents track continues in `25_agentic_ai.ipynb` (LangGraph graphs: guardrails, human approval, memory, planning, reflection, multi-agent routing) and `26_mcp_model_context_protocol.ipynb` (the standard way to expose tools to any agent).
''')
    return nb


if __name__ == "__main__":
    nb = build()
    nb.save(notebook_path("16_agents_tools_structured_output.ipynb"))
    print("16_agents_tools_structured_output.ipynb written")
