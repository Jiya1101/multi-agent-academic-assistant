"""
Export generated study notes as a PDF (ReportLab).

Each note section is laid out as:
    Heading
      Definition       short paragraph
      Key points       bullets
      Use case         short paragraph
      Source           file and pages the section was written from

Text is cleaned for the PDF's built-in Helvetica font: typographic quotes and
dashes become plain ones, and characters Helvetica cannot draw become "?"
instead of black boxes.
"""

from collections import defaultdict
from datetime import date
from io import BytesIO
from typing import Any, Dict, List
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    HRFlowable,
    KeepTogether,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
)

_REPLACEMENTS = {
    "‘": "'", "’": "'", "“": '"', "”": '"', "–": "-", "—": "-",
    "•": "-", " ": " ", "→": "->", "…": "...", "−": "-",
}

_ACCENT = colors.HexColor("#1f4e79")
_MUTED = colors.HexColor("#666666")


def pdf_safe(text: str) -> str:
    """Make `text` drawable in Helvetica and safe inside ReportLab's mini-markup."""
    for old, new in _REPLACEMENTS.items():
        text = text.replace(old, new)
    text = text.encode("cp1252", "replace").decode("cp1252")
    return escape(text)


def _styles() -> Dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle("NoteTitle", parent=base["Title"], fontSize=22, leading=26,
                                textColor=_ACCENT, alignment=TA_LEFT, spaceAfter=4),
        "meta": ParagraphStyle("NoteMeta", parent=base["Normal"], fontSize=9, textColor=_MUTED, spaceAfter=10),
        "h1": ParagraphStyle("NoteH1", parent=base["Heading1"], fontSize=16, leading=20,
                             textColor=_ACCENT, spaceBefore=14, spaceAfter=4),
        "h2": ParagraphStyle("NoteH2", parent=base["Heading2"], fontSize=11.5, leading=14,
                             textColor=colors.black, spaceBefore=8, spaceAfter=2),
        "body": ParagraphStyle("NoteBody", parent=base["Normal"], fontSize=10.5, leading=15, spaceAfter=2),
        "bullet": ParagraphStyle("NoteBullet", parent=base["Normal"], fontSize=10.5, leading=15, spaceAfter=2,
                                 leftIndent=16, bulletIndent=3, bulletFontName="ZapfDingbats", bulletFontSize=6),
        "source": ParagraphStyle("NoteSource", parent=base["Normal"], fontSize=8.5, leading=11,
                                 textColor=_MUTED, spaceBefore=6),
    }


def source_line(sources: List[Dict[str, Any]]) -> str:
    pages: Dict[str, set] = defaultdict(set)
    for source in sources:
        pages[source.get("file", "unknown")].add(source.get("page"))
    parts = []
    for file, page_set in pages.items():
        numbers = sorted(p for p in page_set if p is not None)
        parts.append(f"{file}, page{'s' if len(numbers) > 1 else ''} {', '.join(map(str, numbers))}" if numbers else file)
    return "Source: " + "; ".join(parts) if parts else ""


CHECK_MARK = " *"
CHECK_LEGEND = "* Contains wording that is not in the source slides. Check it before relying on it."


def _section_flowables(section: Dict[str, Any], styles: Dict[str, ParagraphStyle]) -> list:
    check = section.get("check", {})
    head = [
        Paragraph(pdf_safe(section["heading"]), styles["h1"]),
        HRFlowable(width="100%", thickness=0.6, color=_ACCENT, spaceAfter=4),
        Paragraph("Definition", styles["h2"]),
        Paragraph(pdf_safe(section["definition"]), styles["body"]),
    ]
    flowables: list = [KeepTogether(head)]
    if section.get("key_points"):
        flowables.append(Paragraph("Key points", styles["h2"]))
        for index, point in enumerate(section["key_points"]):
            mark = CHECK_MARK if index in check.get("key_points", []) else ""
            # ZapfDingbats is a standard PDF font with a real bullet glyph. ReportLab's
            # ListFlowable bullet was written as an undefined byte in Helvetica.
            flowables.append(Paragraph(pdf_safe(point + mark), styles["bullet"], bulletText="l"))
    if section.get("use_case"):
        flowables.append(Paragraph("Use case", styles["h2"]))
        mark = CHECK_MARK if check.get("use_case") else ""
        flowables.append(Paragraph(pdf_safe(section["use_case"] + mark), styles["body"]))
    if check.get("key_points") or check.get("use_case"):
        flowables.append(Paragraph(pdf_safe(CHECK_LEGEND), styles["source"]))
    source = source_line(section.get("sources", []))
    if source:
        flowables.append(Paragraph(pdf_safe(source), styles["source"]))
    return flowables


def notes_to_pdf(sections: List[Dict[str, Any]], title: str = "Study Notes", source_files: List[str] | None = None) -> bytes:
    """Render note sections to PDF bytes."""
    styles = _styles()
    buffer = BytesIO()

    def footer(canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(_MUTED)
        canvas.drawString(20 * mm, 12 * mm, "Generated by AI from your course material. Check against the source slides.")
        canvas.drawRightString(A4[0] - 20 * mm, 12 * mm, f"Page {doc.page}")
        canvas.restoreState()

    document = SimpleDocTemplate(
        buffer, pagesize=A4, leftMargin=20 * mm, rightMargin=20 * mm, topMargin=18 * mm, bottomMargin=20 * mm,
        title=title, author="Academic Assistant",
    )
    meta = f"Generated {date.today().isoformat()}"
    if source_files:
        meta += " from " + ", ".join(source_files)
    story: list = [Paragraph(pdf_safe(title), styles["title"]), Paragraph(pdf_safe(meta), styles["meta"])]
    for section in sections:
        story.extend(_section_flowables(section, styles))
        story.append(Spacer(1, 6))
    if not sections:
        story.append(Paragraph("No notes could be generated from the selected material.", styles["body"]))
    document.build(story, onFirstPage=footer, onLaterPages=footer)
    return buffer.getvalue()
