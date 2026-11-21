from pathlib import Path
import fitz


def extract_pdf_pages(file_path: str | Path) -> list[dict]:
    """Extract embedded text page by page, preserving 1-based page numbers."""
    path = Path(file_path)
    try:
        with fitz.open(path) as document:
            if document.is_encrypted:
                raise ValueError(f"{path.name} is password-protected.")
            pages = []
            for page_number, page in enumerate(document, start=1):
                text = page.get_text("text").strip()
                if text:
                    pages.append({"text": text, "source": path.name, "page": page_number})
    except (fitz.FileDataError, fitz.EmptyFileError) as exc:
        raise ValueError(f"{path.name} is not a readable PDF file.") from exc
    except RuntimeError as exc:
        raise ValueError(f"Could not open {path.name}. The PDF may be damaged or password-protected.") from exc

    if not pages:
        raise ValueError(
            f"No selectable text was found in {path.name}. Scanned PDFs require OCR before indexing."
        )
    return pages
