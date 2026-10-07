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
from rag_core.normalize import normalize_page_text, page_heading


def load_pdf(file_path: str | Path) -> List[Document]:
    """Extract text from a single PDF file, one cleaned Document per page."""
    loader = PyPDFLoader(str(file_path))
    pages = loader.load()

    found = []
    for page in pages:
        raw_text = page.page_content
        found.append(page_heading(raw_text))
        page.page_content = normalize_page_text(raw_text)

    # Notes that number their sections ("3. OSI / CMIP") have many smaller unnumbered lines that look like
    # titles (diagram labels, sub-points). When numbering is the document's way of marking sections, only
    # numbered headings count; other pages continue the previous topic.
    numbered_pages = sum(1 for title, numbered in found if title and numbered)
    numbering_is_structure = numbered_pages >= max(5, 0.25 * len(pages))
    for page, (title, numbered) in zip(pages, found):
        if title and (numbered or not numbering_is_structure):
            page.metadata["section"] = title

    return pages


def load_all_pdfs(data_dir: str | Path = DATA_DIR) -> List[Document]:
    """Extract text from every PDF found in `data_dir`."""
    data_dir = Path(data_dir)
    documents: List[Document] = []

    for pdf_path in sorted(data_dir.glob("*.pdf")):
        documents.extend(load_pdf(pdf_path))

    return documents
