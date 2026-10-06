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
from rag_core.embeddings import get_embeddings
from rag_core.loader import load_pdf, load_all_pdfs
from rag_core.normalize import normalize_page_text, extract_section_title
from rag_core.splitter import split_documents
from rag_core.vectorstore import (
    build_vectorstore,
    save_vectorstore,
    load_vectorstore,
    vectorstore_exists,
    get_retriever,
)
from rag_core.llm import get_llm
from rag_core.chain import answer_question

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
