import fitz
import pytest

from src import pdf_processor
from src.pdf_processor import extract_pdf_pages


def _create_pdf(path, text=None):
    document = fitz.open()
    page = document.new_page()
    if text:
        page.insert_text((72, 72), text)
    document.save(path)
    document.close()


def test_extracts_text_and_preserves_page_numbers(tmp_path, monkeypatch):
    path = tmp_path / "sample.pdf"
    document = fitz.open()
    first = document.new_page()
    first.insert_text((72, 72), "First page text")
    document.new_page()
    third = document.new_page()
    third.insert_text((72, 72), "Third page text")
    document.save(path)
    document.close()

    # Short text triggers OCR; keep this unit test independent of Tesseract.
    monkeypatch.setattr(pdf_processor, "_extract_ocr_text", lambda page: "")
    pages = extract_pdf_pages(path)
    assert [page["page"] for page in pages] == [1, 3]
    assert pages[0]["source"] == "sample.pdf"
    assert "First page text" in pages[0]["text"]


def test_uses_ocr_for_scanned_page_and_preserves_metadata(tmp_path, monkeypatch):
    path = tmp_path / "scanned.pdf"
    _create_pdf(path)
    monkeypatch.setattr(
        pdf_processor,
        "_extract_ocr_text",
        lambda page: "Recognized text from scanned document.",
    )

    pages = extract_pdf_pages(path)
    assert pages == [
        {
            "text": "Recognized text from scanned document.",
            "source": "scanned.pdf",
            "page": 1,
        }
    ]


def test_rejects_blank_pdf_when_ocr_finds_no_text(tmp_path, monkeypatch):
    path = tmp_path / "blank.pdf"
    _create_pdf(path)
    monkeypatch.setattr(pdf_processor, "_extract_ocr_text", lambda page: "")

    with pytest.raises(ValueError, match="No readable text"):
        extract_pdf_pages(path)


def test_reports_missing_tesseract_for_scanned_pdf(tmp_path, monkeypatch):
    path = tmp_path / "scanned.pdf"
    _create_pdf(path)

    def missing_tesseract(page):
        raise pdf_processor.pytesseract.TesseractNotFoundError()

    monkeypatch.setattr(pdf_processor, "_extract_ocr_text", missing_tesseract)
    with pytest.raises(ValueError, match="Tesseract OCR was not found"):
        extract_pdf_pages(path)


def test_rejects_non_pdf_file(tmp_path):
    path = tmp_path / "not-a-pdf.pdf"
    path.write_text("not a PDF", encoding="utf-8")
    with pytest.raises(ValueError, match="readable PDF"):
        extract_pdf_pages(path)
