"""Notebook 11 - Tools, structured output and agents

Builds 11_agents_tools_structured_output.ipynb. Run: python tools/build_notebooks.py 11
"""
from _common import *  # noqa: F401,F403

# Notebook 11 - Tools, structured output and agents
nb = NB()
nb.md(r'''
# 11 · Tool use, structured output and agents

The model never *executes* anything. It emits a **tool call** (name + JSON arguments); your code validates and runs it, returns the result as a `tool` message, and the model continues. An **agent** is this in a loop with a goal and limits.

**What you will learn**
- Describing tools with JSON schemas; validating arguments *before* execution
- The agent loop: steps, observations, error feedback, max-steps, human confirmation gates
- Structured output: robust JSON parsing + schema validation + retry (and when to use constrained decoding instead)
- Evaluation: tool-selection accuracy, argument correctness, task success, steps and cost per task

Runs offline with `MockLLM`; swap in `OpenAICompatibleClient` (vLLM with `--enable-auto-tool-choice`) for a real model.

**Source:** `llm_edu_suite/agents.py`, `llm_edu_suite/llm_client.py`.
''')
nb.code(SETUP_NOTORCH)
nb.code(r'''
from llm_edu_suite.agents import ToolRegistry, Agent, validate_args, parse_json_output, structured_output
from llm_edu_suite.llm_client import MockLLM
from llm_edu_suite.rag import chunk_documents, Retriever

registry = ToolRegistry()

@registry.register("calculate", "Evaluate an arithmetic expression and return the numeric result",
                   {"type": "object", "properties": {"expression": {"type": "string"}}, "required": ["expression"]})
def calculate(expression: str) -> str:
    if not all(ch in "0123456789+-*/(). " for ch in expression):        # a real sandbox would be much stricter than this
        raise ValueError("only arithmetic is allowed")
    return str(eval(expression, {"__builtins__": {}}, {}))

docs = {"policy.md": "Refunds are accepted within 30 days of purchase with a receipt. Shipping costs are not refunded. Digital products cannot be refunded once downloaded.",
        "hours.md": "Support is available Monday to Friday from 9am to 6pm. Weekend requests are answered on the next business day."}
retriever = Retriever(chunk_documents(docs, size=20, overlap=5))

@registry.register("search_docs", "Search the company documentation knowledge base for policies, hours and procedures",
                   {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]})
def search_docs(query: str) -> str:
    return " | ".join(c.text for c in retriever.retrieve(query, k=2))

@registry.register("get_weather", "Get the current temperature for a city",
                   {"type": "object", "properties": {"city": {"type": "string"}, "unit": {"type": "string", "enum": ["C", "F"]}}, "required": ["city"]})
def get_weather(city: str, unit: str = "C") -> dict:
    return {"city": city, "temp": 19 if unit == "C" else 66, "unit": unit, "condition": "cloudy"}

print(json.dumps(registry.to_openai()[0], indent=1))     # this is what the model sees
''')
nb.md(r'''
## 1. Validation before execution
Never trust generated arguments blindly: validate types/required/enums, and return errors **as text** so the model can correct itself instead of crashing the loop.
''')
nb.code(r'''
schema = registry.tools["get_weather"].parameters
print(validate_args({"city": "Lima"}, schema))                       # []
print(validate_args({"city": 5, "unit": "K"}, schema))                # wrong type and a value outside the enum
print(validate_args({"unit": "C"}, schema))                          # a required argument is missing
print(registry.execute("calculate", {"expression": "import os"}))    # the error comes back as a string instead of raising
print(registry.execute("nope", {}))
''')
nb.md(r'''
## 2. The agent loop
`Agent.run` alternates model ↔ tools until the model answers in text or `max_steps` is hit. Observations are truncated (`max_obs_chars`) so the context does not explode. A `confirm` callback gates dangerous actions (human-in-the-loop).
''')
nb.code(r'''
llm = MockLLM()
agent = Agent(llm, registry, max_steps=4)
for task in ["Please calculate (17 * 23) + 4 and give me the arithmetic result",
             "search the documentation: can I get a refund after 30 days?",
             "What is the weather temperature in Lima?"]:
    run = agent.run(task, verbose=True)
    print(f"USER: {task}\nAGENT: {run.answer}\n   ({len(run.steps)} steps, stopped: {run.stopped_reason})\n")
''')
nb.code(r'''
# A human confirmation gate: only the read-only tools are approved, everything else is denied.
guarded = Agent(llm, registry, confirm=lambda name, args: name in ("search_docs", "get_weather"))
run = guarded.run("Please calculate 2 + 2 arithmetic", verbose=True)
print("AGENT:", run.answer)
''')
nb.md(r'''
## 3. Structured output
Asking for JSON in the prompt fails some percent of the time. Two robust options: (a) **constrained decoding** on the server (`response_format={"type":"json_schema", ...}` in vLLM/SGLang via XGrammar/Outlines) — guaranteed valid, free; (b) client-side **parse → validate → retry with the error**, implemented in `structured_output`.
''')
nb.code(r'''
print(parse_json_output('Sure! Here it is:\n```json\n{"a": 1, "b": [1, 2]}\n```\nLet me know.'))
print(parse_json_output('{"nested": {"x": 1}} trailing text'))

schema = {"type": "object", "properties": {"email": {"type": "string"}, "amount": {"type": "number"}, "id": {"type": "string"}},
          "required": ["email", "amount"]}
text = "Order #B77-120 from Carlos Ruiz, total $2,480.00, contact carlos.ruiz@example.com, 3 units"
print(structured_output(llm, f"Extract the fields.\n\nText:\n{text}", schema))
''')
nb.md(r'''
## 4. Evaluation

Agents are evaluated on **trajectories**, not just final answers:
1. **Tool-selection accuracy** — did it call the right tool (or none) for each test input?
2. **Argument correctness** — schema-valid and semantically right?
3. **Task success** — final answer correct (exact match / judge).
4. **Efficiency** — steps and tokens per task; **safety** — no unconfirmed dangerous calls.

Build a test set of (input → expected tool, expected answer) and run it after every prompt/tool-description change: tool descriptions are the agent's "API docs" and small wording changes move these numbers.
''')
nb.code(r'''
test_cases = [
    ("Please calculate 12 * 12 arithmetic", "calculate", "144"),
    ("Compute the arithmetic expression (3 + 5) * 2", "calculate", "16"),
    ("search the documentation: what are the support hours?", "search_docs", "Monday to Friday"),
    ("search the documentation for the refund policy on digital products", "search_docs", "cannot be refunded"),
    ("weather temperature in Cusco", "get_weather", "19"),
]
tool_ok = arg_ok = ans_ok = 0; total_steps = 0
for text, exp_tool, exp_substr in test_cases:
    run = agent.run(text)
    called = [s.tool for s in run.steps if s.tool]
    tool_ok += bool(called) and called[0] == exp_tool
    arg_ok += bool(called) and not str(run.steps[0].result).startswith("ERROR")
    ans_ok += exp_substr.lower() in run.answer.lower()
    total_steps += len(run.steps)
n = len(test_cases)
print(f"tool selection {tool_ok}/{n} | valid arguments {arg_ok}/{n} | answer contains expected {ans_ok}/{n} | avg steps {total_steps/n:.1f}")
print(f"token usage: {llm.usage.calls} calls, {llm.usage.prompt_tokens} prompt tokens, est. cost ${llm.usage.cost():.5f}")
''')
nb.md(r'''
**Production notes:** use the model's native tool-calling format via the chat template; sandbox code execution (containers, timeouts, allow-lists); log every trajectory; cap steps and spend; put retrieved/tool text in delimiters and tell the model it is *data* (see notebook 13 on prompt injection). The Model Context Protocol (MCP) standardizes how tools are exposed to models.

**Next:** `12_recommender_systems.ipynb`.
''')
nb.save(notebook_path("11_agents_tools_structured_output.ipynb"))
print("11_agents_tools_structured_output.ipynb written")
