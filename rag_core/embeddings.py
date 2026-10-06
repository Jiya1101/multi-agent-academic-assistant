"""
Embedding model wrapper.

Uses `langchain_huggingface.HuggingFaceEmbeddings` (the modern, non-deprecated
package) around `sentence-transformers/all-MiniLM-L6-v2` — a small, fast,
CPU-friendly model well suited to a local student project.
"""

from functools import lru_cache

from langchain_huggingface import HuggingFaceEmbeddings

from rag_core.config import EMBEDDING_MODEL_NAME


@lru_cache(maxsize=1)
def get_embeddings() -> HuggingFaceEmbeddings:
    """Return a cached embedding model instance (loaded once per process)."""
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
