"""
Tests for telling a user why a PDF cannot be used. A PDF made only of pictures has no text layer, so updating the
library can never make it studyable; the app used to leave it looking as if it merely "needed an update".

Run:  python -m unittest discover -s tests -v
"""

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas

from rag_core.loader import (
    MIN_READABLE_CHARACTERS,
    PdfStatus,
    describe_unreadable,
    inspect_pdf,
    unreadable_pdfs,
)


def text_pdf(path: Path, pages: int = 2, text: str = "Mutual exclusion means one process at a time enters the critical section.") -> None:
    pdf = canvas.Canvas(str(path))
    for _ in range(pages):
        pdf.drawString(72, 750, text)
        pdf.showPage()
    pdf.save()


def picture_pdf(path: Path, pages: int = 3, pictures_per_page: int = 2) -> None:
    """Pages that hold only drawn pictures, like a scan or an exported diagram: no text layer at all."""
    pdf = canvas.Canvas(str(path))
    for _ in range(pages):
        for i in range(pictures_per_page):
            pdf.drawImage(ImageReader(Image.new("RGB", (60, 40), (200 - 40 * i, 120, 80))), 72, 600 - 150 * i, 120, 80)
        pdf.showPage()
    pdf.save()


class InspectPdfTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_a_pdf_with_text_is_readable(self):
        text_pdf(self.dir / "notes.pdf")
        status = inspect_pdf(self.dir / "notes.pdf")
        self.assertTrue(status.readable)
        self.assertEqual(status.pages, 2)
        self.assertGreaterEqual(status.characters, MIN_READABLE_CHARACTERS)

    def test_a_pdf_of_pictures_is_not_readable_and_says_what_it_holds(self):
        picture_pdf(self.dir / "scan.pdf", pages=3, pictures_per_page=2)
        status = inspect_pdf(self.dir / "scan.pdf")
        self.assertFalse(status.readable)
        self.assertEqual((status.pages, status.characters), (3, 0))
        self.assertGreaterEqual(status.images, 1)

    def test_the_same_picture_on_every_page_is_counted_once(self):
        pdf = canvas.Canvas(str(self.dir / "shared.pdf"))
        same = ImageReader(Image.new("RGB", (60, 40), (10, 120, 80)))
        for _ in range(4):
            pdf.drawImage(same, 72, 600, 120, 80)
            pdf.showPage()
        pdf.save()
        self.assertEqual(inspect_pdf(self.dir / "shared.pdf").images, 1)

    def test_a_pdf_with_almost_no_text_is_not_readable(self):
        text_pdf(self.dir / "tiny.pdf", pages=1, text="Page 1")
        self.assertFalse(inspect_pdf(self.dir / "tiny.pdf").readable)

    def test_a_file_that_is_not_a_pdf_is_not_readable_and_does_not_crash(self):
        (self.dir / "broken.pdf").write_bytes(b"this is not a pdf at all")
        status = inspect_pdf(self.dir / "broken.pdf")
        self.assertEqual((status.readable, status.pages), (False, 0))

    def test_only_the_unreadable_files_are_listed(self):
        text_pdf(self.dir / "good.pdf")
        picture_pdf(self.dir / "scan.pdf")
        (self.dir / "ignored.txt").write_text("not a pdf")
        self.assertEqual(list(unreadable_pdfs(self.dir)), ["scan.pdf"])

    def test_an_empty_folder_has_nothing_unreadable(self):
        self.assertEqual(unreadable_pdfs(self.dir), {})


class DescriptionTests(unittest.TestCase):
    def test_the_message_says_why_and_that_updating_will_not_help(self):
        message = describe_unreadable(PdfStatus(pages=3, characters=0, images=4, readable=False))
        self.assertIn("3 page(s)", message)
        self.assertIn("4 picture(s)", message)
        self.assertIn("Updating the library cannot fix that", message)

    def test_a_file_that_cannot_be_opened_gets_its_own_message(self):
        message = describe_unreadable(PdfStatus(pages=0, characters=0, images=0, readable=False))
        self.assertIn("could not be opened", message)
        self.assertNotIn("scan", message)


class LibraryPageTests(unittest.TestCase):
    """The library shows the reason for an unreadable file, not 'needs an update'."""

    def test_the_real_page_explains_an_image_only_pdf(self):
        from streamlit.testing.v1 import AppTest

        root = Path(__file__).resolve().parent.parent
        src = (root / "app.py").read_text(encoding="utf-8")
        cut = src.rfind("# ----", 0, src.index("# Sidebar + routing"))
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            text_pdf(tmp / "good.pdf")
            picture_pdf(tmp / "scan.pdf")
            page = root / "_tmp_library_page.py"
            page.write_text(
                "import rag_core.config as _c\nfrom pathlib import Path\n_c.DATA_DIR = Path(r'" + str(tmp) + "')\n_c.DB_DIR = Path(r'"
                + str(tmp) + "')\n" + src[:cut] + "\nrender_student_material()\n",
                encoding="utf-8",
            )
            try:
                at = AppTest.from_file(str(page), default_timeout=120)
                at.run()
                warnings = [w.value for w in at.warning]
                captions = [c.value for c in at.caption]
            finally:
                page.unlink(missing_ok=True)
        self.assertEqual(at.exception, [])
        self.assertTrue(any("No readable text" in w and "Updating the library cannot fix that" in w for w in warnings), warnings)
        self.assertIn("Needs an update before you can use it", captions)   # the readable one, not yet indexed
        self.assertEqual(sum("No readable text" in w for w in warnings), 1)


if __name__ == "__main__":
    unittest.main()
