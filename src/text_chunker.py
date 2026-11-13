from collections.abc import Iterable


def chunk_pages(pages: Iterable[dict], chunk_size: int = 800, overlap: int = 120) -> list[dict]:
    """Split page text into overlapping chunks while retaining source metadata."""
    if chunk_size < 100:
        raise ValueError("chunk_size must be at least 100 characters.")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be >= 0 and smaller than chunk_size.")
    chunks = []
    step = chunk_size - overlap
    for page in pages:
        text = " ".join(page["text"].split())
        for start in range(0, len(text), step):
            piece = text[start:start + chunk_size].strip()
            if piece:
                chunks.append({"text": piece, "source": page["source"], "page": int(page["page"])})
    return chunks
