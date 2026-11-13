from pathlib import Path
import fitz


def extract_pdf_pages(file_path: str | Path) -> list[dict]:
    """Extract embedded text page by page, preserving 1-based page numbers."""
    path = Path(file_path)
    pages: list[dict] = []
    with fitz.open(path) as document:
        if document.is_encrypted:
            raise ValueError(f"{path.name} is password-protected.")
        for page_number, page in enumerate(document, start=1):
            text = page.get_text("text").strip()
            if text:
                pages.append({"text": text, "source": path.name, "page": page_number})
    return pages
