"""
PDF loading / text extraction.

Uses the modern `langchain_community.document_loaders.PyPDFLoader`, which
returns one LangChain `Document` per page, each carrying `source` and
`page` metadata. That metadata is what lets the UI show *where* an answer
came from.

Each page is also run through `normalize_page_text` (fixes bullet
artifacts, hyphenated line-wraps, glued words) and tagged with a
best-effort `section` title (the page's first substantial line) so
retrieved chunks can be cited by slide/section, not just page number.
"""

from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader

from rag_core.config import DATA_DIR
from rag_core.normalize import normalize_page_text, extract_section_title


def load_pdf(file_path: str | Path) -> List[Document]:
    """Extract text from a single PDF file, one cleaned Document per page."""
    loader = PyPDFLoader(str(file_path))
    pages = loader.load()

    for page in pages:
        raw_text = page.page_content
        section = extract_section_title(raw_text)
        if section:
            page.metadata["section"] = section
        page.page_content = normalize_page_text(raw_text)

    return pages


def load_all_pdfs(data_dir: str | Path = DATA_DIR) -> List[Document]:
    """Extract text from every PDF found in `data_dir`."""
    data_dir = Path(data_dir)
    documents: List[Document] = []

    for pdf_path in sorted(data_dir.glob("*.pdf")):
        documents.extend(load_pdf(pdf_path))

    return documents
