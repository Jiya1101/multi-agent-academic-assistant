"""
FAISS vector store management: build, persist, load, and expose a retriever.

Keeping this behind a small function API (rather than letting callers touch
FAISS directly) is what lets every future agent share one on-disk index
without duplicating load/save logic.
"""

from pathlib import Path
from typing import Iterable, List, Tuple

from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from rag_core.config import DB_DIR, FACULTY_SOURCE_NAME, FAISS_INDEX_NAME, TOP_K
from rag_core.embeddings import get_embeddings


def _index_path(db_dir: str | Path = DB_DIR) -> Path:
    return Path(db_dir) / FAISS_INDEX_NAME


def vectorstore_exists(db_dir: str | Path = DB_DIR) -> bool:
    """Check whether a persisted FAISS index already exists on disk."""
    path = _index_path(db_dir)
    return (path / "index.faiss").exists() and (path / "index.pkl").exists()


def index_version(db_dir: str | Path = DB_DIR) -> int:
    """
    Modification time of the on-disk index (0 if none). Callers can use it as
    a cache key so every session reloads the index after anyone rebuilds it or
    adds a faculty answer.
    """
    index_file = _index_path(db_dir) / "index.faiss"
    return index_file.stat().st_mtime_ns if index_file.exists() else 0


def build_vectorstore(
    chunks: List[Document],
    embeddings: HuggingFaceEmbeddings | None = None,
) -> FAISS:
    """Embed `chunks` and build a fresh in-memory FAISS index."""
    embeddings = embeddings or get_embeddings()
    return FAISS.from_documents(chunks, embeddings)


def save_vectorstore(vectorstore: FAISS, db_dir: str | Path = DB_DIR) -> None:
    """Persist a FAISS index to disk so it can be reloaded without re-embedding."""
    path = _index_path(db_dir)
    path.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(path))


def load_vectorstore(
    db_dir: str | Path = DB_DIR,
    embeddings: HuggingFaceEmbeddings | None = None,
) -> FAISS:
    """Load a previously persisted FAISS index from disk."""
    embeddings = embeddings or get_embeddings()
    path = _index_path(db_dir)
    return FAISS.load_local(
        str(path),
        embeddings,
        # Safe here: the index is only ever written by our own ingest.py.
        allow_dangerous_deserialization=True,
    )


def get_retriever(vectorstore: FAISS, k: int = TOP_K) -> VectorStoreRetriever:
    """Return a top-k similarity retriever for `vectorstore`."""
    return vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": k})


def faculty_documents(qa_pairs: Iterable[Tuple[str, str]]) -> List[Document]:
    """
    Turn professor-written (question, answer) pairs into indexable Documents.

    The text is stored as "Q: ... A: ..." so its embedding sits close to the
    wording of future student questions on the same topic.
    """
    return [
        Document(
            page_content=f"Q: {question}\nA: {answer}",
            metadata={
                "source": FACULTY_SOURCE_NAME,
                "page": 0,
                "section": question[:80],
            },
        )
        for question, answer in qa_pairs
    ]


def add_faculty_answer(
    vectorstore: FAISS,
    question: str,
    answer: str,
    db_dir: str | Path = DB_DIR,
) -> None:
    """Add one professor answer to the live index and persist it to disk."""
    vectorstore.add_documents(faculty_documents([(question, answer)]))
    save_vectorstore(vectorstore, db_dir)


def list_indexed_topics(vectorstore: FAISS, source_filenames: Iterable[str] | None = None) -> List[str]:
    """
    Slide/section titles in the index, in document order, without duplicates.

    Used to offer a pick-list of topics for note generation. Professor answers
    are excluded because they are not part of the uploaded material.
    """
    from rag_core.normalize import short_section_title

    wanted = set(source_filenames) if source_filenames is not None else None
    found = []
    for doc in vectorstore.docstore._dict.values():
        name = Path(doc.metadata.get("source", "")).name
        section = doc.metadata.get("section")
        if not section or name == FACULTY_SOURCE_NAME or (wanted is not None and name not in wanted):
            continue
        found.append((name, doc.metadata.get("page") or 0, short_section_title(section)))
    found.sort()
    topics, seen = [], set()
    for _, _, title in found:
        if title not in seen:
            seen.add(title)
            topics.append(title)
    return topics


def list_indexed_sources(vectorstore: FAISS) -> List[str]:
    """
    Return the sorted, deduplicated list of source filenames present in
    `vectorstore`, so callers (e.g. the UI) can offer per-file selection
    without re-reading the PDFs.

    LangChain's FAISS wrapper has no public "list all documents" API, so
    this reaches into the underlying in-memory docstore directly.
    """
    names = {
        Path(doc.metadata["source"]).name
        for doc in vectorstore.docstore._dict.values()
        if "source" in doc.metadata
    }
    return sorted(names)
