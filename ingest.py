"""
ingest.py
=========
Build (or rebuild) the FAISS vector index from every PDF in `data/`.

Run this whenever new course notes are added to `data/` and you are not
using the Streamlit "Rebuild index" button.

Usage:
    python ingest.py
    python ingest.py --data-dir data --db-dir db
"""

import argparse
import sys

from rag_core.config import DATA_DIR, DB_DIR
from rag_core.loader import load_all_pdfs
from rag_core.splitter import split_documents
from rag_core.embeddings import get_embeddings
from rag_core.query_log import resolved_answers
from rag_core.vectorstore import build_vectorstore, faculty_documents, save_vectorstore


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest PDFs into a FAISS index.")
    parser.add_argument("--data-dir", default=str(DATA_DIR), help="Folder containing PDF files.")
    parser.add_argument("--db-dir", default=str(DB_DIR), help="Folder to write the FAISS index to.")
    args = parser.parse_args()

    # 1. Extract text from every PDF in the data directory.
    print(f"[1/4] Loading PDFs from '{args.data_dir}' ...")
    documents = load_all_pdfs(args.data_dir)
    if not documents:
        print(f"No PDF files found in '{args.data_dir}'. Add some notes and re-run.")
        sys.exit(1)
    print(f"      Loaded {len(documents)} page(s).")

    # 2. Chunk documents for embedding.
    print("[2/4] Splitting documents into chunks ...")
    chunks = split_documents(documents)
    print(f"      Produced {len(chunks)} chunk(s).")
    if not chunks:
        print(
            "No extractable text was found. This usually means a PDF is "
            "scanned/image-based rather than text-based (OCR is not part "
            "of this module). Try a different file."
        )
        sys.exit(1)

    # Professor-written answers live in the question log, not in the PDFs, so
    # a rebuild must re-add them or they would silently disappear.
    faculty_docs = faculty_documents(resolved_answers(args.db_dir))
    if faculty_docs:
        print(f"      Re-adding {len(faculty_docs)} professor answer(s) from the log.")
        chunks = chunks + faculty_docs

    # 3. Embed + build the FAISS index.
    print("[3/4] Generating embeddings and building the FAISS index ...")
    embeddings = get_embeddings()
    vectorstore = build_vectorstore(chunks, embeddings)

    # 4. Persist the index to disk.
    print(f"[4/4] Saving index to '{args.db_dir}' ...")
    save_vectorstore(vectorstore, args.db_dir)

    print("Done. The vector index is ready for querying.")


if __name__ == "__main__":
    main()
