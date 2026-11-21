import fitz
import pytest

from src.pdf_processor import extract_pdf_pages


def test_extracts_text_and_preserves_page_numbers(tmp_path):
    path = tmp_path / "sample.pdf"
    document = fitz.open()
    first = document.new_page()
    first.insert_text((72, 72), "First page text")
    document.new_page()
    third = document.new_page()
    third.insert_text((72, 72), "Third page text")
    document.save(path)
    document.close()

    pages = extract_pdf_pages(path)
    assert [page["page"] for page in pages] == [1, 3]
    assert pages[0]["source"] == "sample.pdf"
    assert "First page text" in pages[0]["text"]


def test_rejects_scanned_or_empty_pdf(tmp_path):
    path = tmp_path / "blank.pdf"
    document = fitz.open()
    document.new_page()
    document.save(path)
    document.close()

    with pytest.raises(ValueError, match="OCR"):
        extract_pdf_pages(path)


def test_rejects_non_pdf_file(tmp_path):
    path = tmp_path / "not-a-pdf.pdf"
    path.write_text("not a PDF", encoding="utf-8")
    with pytest.raises(ValueError, match="readable PDF"):
        extract_pdf_pages(path)
