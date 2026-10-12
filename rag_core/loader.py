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

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from pypdf import PdfReader

from rag_core.config import DATA_DIR
from rag_core.normalize import normalize_page_text, page_heading

MIN_READABLE_CHARACTERS = 40   # a PDF with fewer characters of text than this has nothing to study from
IMAGE_COUNT_PAGES = 10         # how many pages are looked at when counting the pictures in an unreadable PDF


@dataclass(frozen=True)
class PdfStatus:
    """Whether a PDF has text the app can read, and if not, what is in it instead."""

    pages: int
    characters: int       # characters of text found (stops counting once the PDF is clearly readable)
    images: int           # pictures on the pages, counted only when no text was found
    readable: bool


def inspect_pdf(file_path: str | Path) -> PdfStatus:
    """
    Check a PDF for text without loading it into the index.

    A scanned PDF, or one made only of pictures (diagrams, screenshots), has no text layer, so nothing can be
    indexed from it however often the library is updated. The app has no OCR; this tells the user so, instead of
    leaving the file looking as if it only "needs an update".
    """
    try:
        reader = PdfReader(str(file_path))
        pages = list(reader.pages)
    except Exception:  # corrupt, password-protected, or not a PDF at all
        return PdfStatus(0, 0, 0, False)
    characters = 0
    for page in pages:
        try:
            characters += len((page.extract_text() or "").strip())
        except Exception:
            continue
        if characters >= MIN_READABLE_CHARACTERS:
            return PdfStatus(len(pages), characters, 0, True)
    # Count DISTINCT pictures on the first few pages: slide templates often share the same images across pages,
    # and a long scan should not be unpacked in full just to describe it.
    distinct = set()
    for page in pages[:IMAGE_COUNT_PAGES]:
        try:
            distinct.update(hashlib.md5(image.data).hexdigest() for image in page.images)
        except Exception:
            continue
    return PdfStatus(len(pages), characters, len(distinct), False)


def unreadable_pdfs(data_dir: str | Path = DATA_DIR) -> Dict[str, PdfStatus]:
    """The PDFs in `data_dir` that have no readable text, by file name."""
    found = {}
    for path in sorted(Path(data_dir).glob("*.pdf")):
        status = inspect_pdf(path)
        if not status.readable:
            found[path.name] = status
    return found


def describe_unreadable(status: PdfStatus) -> str:
    """One plain sentence on why a PDF cannot be used, for the library page."""
    if status.pages == 0:
        return "This file could not be opened as a PDF (it may be damaged or password-protected)."
    pictures = f" and {status.images} picture(s)" if status.images else ""
    return (
        f"No readable text: this PDF has {status.pages} page(s){pictures} but no text layer, so it is a scan or "
        "made of images. Updating the library cannot fix that. Use a text version of these notes instead."
    )


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
