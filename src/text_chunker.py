from collections.abc import Iterable


def chunk_pages(pages: Iterable[dict], chunk_size: int = 800, overlap: int = 120) -> list[dict]:
    """Split page text into overlapping chunks while retaining source metadata."""
    if not isinstance(chunk_size, int) or isinstance(chunk_size, bool) or chunk_size < 100:
        raise ValueError("chunk_size must be an integer of at least 100 characters.")
    if not isinstance(overlap, int) or isinstance(overlap, bool) or overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be an integer >= 0 and smaller than chunk_size.")

    chunks = []
    step = chunk_size - overlap
    for page in pages:
        if not isinstance(page, dict) or not all(key in page for key in ("text", "source", "page")):
            raise ValueError("Each page must contain text, source, and page fields.")
        if not isinstance(page["text"], str):
            raise ValueError("Page text must be a string.")
        text = " ".join(page["text"].split())
        for start in range(0, len(text), step):
            piece = text[start:start + chunk_size].strip()
            if piece:
                chunks.append({
                    "text": piece,
                    "source": str(page["source"]),
                    "page": int(page["page"]),
                })
    return chunks
