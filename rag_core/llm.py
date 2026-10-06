"""
LLM wrapper around a locally running Ollama server.

Uses `langchain_ollama.ChatOllama`, the current, non-deprecated integration
(replaces the old `langchain_community.llms.Ollama`).
"""

from functools import lru_cache

from langchain_ollama import ChatOllama

from rag_core.config import OLLAMA_MODEL, OLLAMA_BASE_URL, LLM_TEMPERATURE


@lru_cache(maxsize=1)
def get_llm(
    model: str = OLLAMA_MODEL,
    base_url: str = OLLAMA_BASE_URL,
    temperature: float = LLM_TEMPERATURE,
) -> ChatOllama:
    """Return a cached ChatOllama client pointed at the local Ollama server."""
    return ChatOllama(model=model, base_url=base_url, temperature=temperature)


@lru_cache(maxsize=1)
def get_json_llm(
    model: str = OLLAMA_MODEL,
    base_url: str = OLLAMA_BASE_URL,
    temperature: float = 0.3,
) -> ChatOllama:
    """
    Client that constrains Ollama to emit valid JSON (used by the Quiz
    Generator). `num_predict` caps the output so a runaway generation cannot
    block the app for minutes on CPU.
    """
    return ChatOllama(
        model=model,
        base_url=base_url,
        temperature=temperature,
        format="json",
        num_predict=1200,
    )
