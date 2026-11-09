"""
One place to configure the models and services every application notebook uses.

The course runs locally by default: Ollama serves the chat model and the embedding model on your own
machine (GPU if you have one), and LangSmith records traces when you put an API key in `.env`.
Everything is read from environment variables, so switching to another provider is a one-line change
in `.env`, not a code change.

    from llm_edu_suite.config import get_llm, get_embeddings, setup
    setup()                       # loads .env, prints what is configured, checks Ollama
    llm = get_llm()               # a LangChain chat model (ChatOllama by default)
    emb = get_embeddings()        # a LangChain embeddings model

Environment variables (all optional):
    LLM_PROVIDER        ollama (default) | openai | anthropic
    LLM_MODEL           default: qwen2.5:3b for Ollama
    JUDGE_MODEL         the LLM-as-judge for evaluations; default: qwen2.5:7b if it is pulled, else LLM_MODEL
    EMBEDDING_MODEL     default: nomic-embed-text for Ollama
    OLLAMA_BASE_URL     default: http://localhost:11434
    LANGSMITH_TRACING   true to send traces to LangSmith (needs LANGSMITH_API_KEY)
    LANGSMITH_PROJECT   the LangSmith project the traces go to
"""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DEFAULTS = {
    "ollama": {"chat": "qwen2.5:3b", "embeddings": "nomic-embed-text", "judge": "qwen2.5:7b"},
    "openai": {"chat": "gpt-4o-mini", "embeddings": "text-embedding-3-small"},
    "anthropic": {"chat": "claude-haiku-4-5", "embeddings": None},
}


def load_env() -> None:
    """Load the project's .env file (if there is one) without overriding variables already set."""
    try:
        from dotenv import load_dotenv
    except ImportError:          # python-dotenv is optional; plain environment variables still work
        return
    load_dotenv(ROOT / ".env", override=False)


def provider() -> str:
    return os.environ.get("LLM_PROVIDER", "ollama").lower()


def get_llm(model: str | None = None, temperature: float = 0.0, **kwargs):
    """
    The chat model for the course. Temperature 0 by default so notebook results are as repeatable as a
    real LLM allows. Extra keyword arguments go straight to the LangChain class.
    """
    load_env()
    p = provider()
    model = model or os.environ.get("LLM_MODEL") or DEFAULTS[p]["chat"]
    if p == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(model=model, temperature=temperature,
                          base_url=os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434"), **kwargs)
    if p == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=model, temperature=temperature, **kwargs)
    if p == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model=model, temperature=temperature, **kwargs)
    raise ValueError(f"unknown LLM_PROVIDER {p!r} (use ollama, openai or anthropic)")


def ollama_models(timeout: float = 2.0) -> list[str]:
    """Names of the models pulled into the local Ollama server (empty if it does not answer)."""
    import json
    import urllib.request
    url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/") + "/api/tags"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return [m["name"] for m in json.load(r).get("models", [])]
    except Exception:
        return []


def judge_model() -> str:
    """
    The model used as LLM-as-judge (RAGAS, LangSmith evaluators). A judge should be stronger than the
    system it grades, so with Ollama the default is qwen2.5:7b when it is pulled; otherwise the chat model.
    """
    load_env()
    if os.environ.get("JUDGE_MODEL"):
        return os.environ["JUDGE_MODEL"]
    p = provider()
    chat = os.environ.get("LLM_MODEL") or DEFAULTS[p]["chat"]
    if p == "ollama":
        want = DEFAULTS["ollama"]["judge"]
        return want if any(m == want or m == want + ":latest" for m in ollama_models()) else chat
    return chat


def get_judge_llm(**kwargs):
    """A chat model for grading: `judge_model()` at temperature 0."""
    return get_llm(model=judge_model(), temperature=0.0, **kwargs)


def get_embeddings(model: str | None = None):
    """The embedding model used for retrieval and memory. Anthropic has no embeddings API, so it falls back to Ollama."""
    load_env()
    p = provider()
    if p == "openai":
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(model=model or os.environ.get("EMBEDDING_MODEL") or DEFAULTS["openai"]["embeddings"])
    from langchain_ollama import OllamaEmbeddings
    return OllamaEmbeddings(model=model or os.environ.get("EMBEDDING_MODEL") or DEFAULTS["ollama"]["embeddings"],
                            base_url=os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434"))


def ollama_available(timeout: float = 2.0) -> bool:
    """True if an Ollama server answers. Notebooks and tests use it to fail early with a helpful message."""
    import urllib.request
    url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/") + "/api/tags"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return r.status == 200
    except Exception:
        return False


def langsmith_enabled() -> bool:
    load_env()
    return os.environ.get("LANGSMITH_TRACING", "").lower() == "true" and bool(os.environ.get("LANGSMITH_API_KEY"))


def setup(project: str | None = None, quiet: bool = False) -> dict:
    """
    Call once at the top of a notebook: loads .env, optionally sets the LangSmith project for this
    notebook (so each notebook's traces land in their own project), and reports what is configured.
    """
    load_env()
    if project and langsmith_enabled():
        os.environ["LANGSMITH_PROJECT"] = project
    info = {
        "provider": provider(),
        "chat_model": os.environ.get("LLM_MODEL") or DEFAULTS[provider()]["chat"],
        "judge_model": judge_model(),
        "embedding_model": os.environ.get("EMBEDDING_MODEL") or DEFAULTS["ollama" if provider() == "anthropic" else provider()]["embeddings"],
        "ollama_running": ollama_available() if provider() == "ollama" else None,
        "langsmith": f"on (project: {os.environ.get('LANGSMITH_PROJECT', 'default')})" if langsmith_enabled() else "off (add LANGSMITH_API_KEY to .env to enable)",
    }
    if not quiet:
        for k, v in info.items():
            print(f"{k:16s}: {v}")
        if info["ollama_running"] is False:
            print("\n!! Ollama is not running. Start it (the Ollama app, or `ollama serve`) and pull the models:\n"
                  "   ollama pull qwen2.5:3b && ollama pull nomic-embed-text")
    return info
