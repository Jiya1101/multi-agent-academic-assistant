"""
rag_core
========
Shared retrieval layer for the "Multi-Agent AI Framework for Personalized
Academic Assistance" project.

Every future agent (Concept Explainer, Quiz Generator, Progress Tracker,
Doubt Resolver, ...) should import from this package instead of talking to
FAISS / Ollama / HuggingFace directly. That keeps ingestion, embeddings and
retrieval logic in exactly one place.

Typical usage from another agent:

    from rag_core import load_vectorstore, answer_question

    vectorstore = load_vectorstore()
    result = answer_question("Explain gradient descent", vectorstore)
    print(result.grounded, result.answer, result.source_documents)

`answer_question` is score-gated: if nothing in the index is actually
relevant to the query, it returns `grounded=False` and a fixed "not
covered" message without calling the LLM at all -- agents built on top of
it get that safety net for free instead of re-implementing it.
"""

from rag_core.config import (
    DATA_DIR,
    DB_DIR,
    EMBEDDING_MODEL_NAME,
    OLLAMA_MODEL,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    TOP_K,
    RELEVANCE_SCORE_THRESHOLD,
)

_LAZY_EXPORTS = {
    "get_embeddings": ("rag_core.embeddings", "get_embeddings"),
    "load_pdf": ("rag_core.loader", "load_pdf"),
    "load_all_pdfs": ("rag_core.loader", "load_all_pdfs"),
    "normalize_page_text": ("rag_core.normalize", "normalize_page_text"),
    "extract_section_title": ("rag_core.normalize", "extract_section_title"),
    "split_documents": ("rag_core.splitter", "split_documents"),
    "build_vectorstore": ("rag_core.vectorstore", "build_vectorstore"),
    "save_vectorstore": ("rag_core.vectorstore", "save_vectorstore"),
    "load_vectorstore": ("rag_core.vectorstore", "load_vectorstore"),
    "vectorstore_exists": ("rag_core.vectorstore", "vectorstore_exists"),
    "get_retriever": ("rag_core.vectorstore", "get_retriever"),
    "get_llm": ("rag_core.llm", "get_llm"),
    "answer_question": ("rag_core.chain", "answer_question"),
}


def __getattr__(name):
    """Load optional-heavy helpers only when a caller actually asks for them."""
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module 'rag_core' has no attribute {name!r}")
    module_name, attr_name = _LAZY_EXPORTS[name]
    from importlib import import_module

    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value

__all__ = [
    "DATA_DIR",
    "DB_DIR",
    "EMBEDDING_MODEL_NAME",
    "OLLAMA_MODEL",
    "CHUNK_SIZE",
    "CHUNK_OVERLAP",
    "TOP_K",
    "RELEVANCE_SCORE_THRESHOLD",
    "get_embeddings",
    "load_pdf",
    "load_all_pdfs",
    "normalize_page_text",
    "extract_section_title",
    "split_documents",
    "build_vectorstore",
    "save_vectorstore",
    "load_vectorstore",
    "vectorstore_exists",
    "get_retriever",
    "get_llm",
    "answer_question",
]
