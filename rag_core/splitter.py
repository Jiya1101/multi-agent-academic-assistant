"""
Document chunking.

RecursiveCharacterTextSplitter tries to split on paragraph/sentence/word
boundaries (in that order) before falling back to a hard character cut, so
chunks stay semantically coherent for embedding + retrieval.
"""

from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag_core.config import CHUNK_SIZE, CHUNK_OVERLAP


def split_documents(documents: List[Document]) -> List[Document]:
    """Split raw page-level Documents into smaller, embedding-sized chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(documents)
