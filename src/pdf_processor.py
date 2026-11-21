from io import BytesIO
import os
from pathlib import Path
import shutil

import fitz
import pytesseract
from PIL import Image


# Prefer an explicit environment override, then the standard Windows install,
# then an executable available on PATH.
_tesseract_override = os.getenv("TESSERACT_CMD")
_windows_tesseract = Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe")

if _tesseract_override:
    pytesseract.pytesseract.tesseract_cmd = _tesseract_override
elif _windows_tesseract.exists():
    pytesseract.pytesseract.tesseract_cmd = str(_windows_tesseract)
elif shutil.which("tesseract"):
    pytesseract.pytesseract.tesseract_cmd = shutil.which("tesseract")


def _extract_ocr_text(page) -> str:
    """Render a PDF page and extract English text with Tesseract OCR."""
    pixmap = page.get_pixmap(dpi=300, alpha=False)
    image = Image.open(BytesIO(pixmap.tobytes("png"))).convert("RGB")
    return pytesseract.image_to_string(image, lang="eng").strip()


def extract_pdf_pages(file_path: str | Path) -> list[dict]:
    """
    Extract PDF text page by page, using OCR when embedded text is absent
    or unusually short. Preserve 1-based page numbers and source filenames.
    """
    path = Path(file_path)
    try:
        with fitz.open(path) as document:
            if document.is_encrypted:
                raise ValueError(f"{path.name} is password-protected.")

            pages = []
            for page_number, page in enumerate(document, start=1):
                text = page.get_text("text").strip()
                if len(text) < 40:
                    try:
                        text = _extract_ocr_text(page) or text
                    except pytesseract.TesseractNotFoundError as exc:
                        raise ValueError(
                            "Tesseract OCR was not found. Install Tesseract and "
                            "set TESSERACT_CMD to its executable path if needed."
                        ) from exc
                    except pytesseract.TesseractError as exc:
                        raise ValueError(
                            "Tesseract OCR could not process this page. Check that "
                            "the English language data (eng.traineddata) is installed."
                        ) from exc

                if text:
                    pages.append(
                        {"text": text, "source": path.name, "page": page_number}
                    )
    except (fitz.FileDataError, fitz.EmptyFileError) as exc:
        raise ValueError(f"{path.name} is not a readable PDF file.") from exc
    except RuntimeError as exc:
        raise ValueError(
            f"Could not open {path.name}. The PDF may be damaged or password-protected."
        ) from exc

    if not pages:
        raise ValueError(
            f"No readable text could be extracted from {path.name}. "
            "The PDF may be blank or OCR could not recognize its scanned text."
        )
    return pages
