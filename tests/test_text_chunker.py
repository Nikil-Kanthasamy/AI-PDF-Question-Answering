import pytest
from src.text_chunker import chunk_pages


def test_chunks_keep_source_and_page():
    pages = [{"text": "A " * 500, "source": "notes.pdf", "page": 3}]
    chunks = chunk_pages(pages, chunk_size=400, overlap=80)
    assert len(chunks) > 1
    assert all(c["source"] == "notes.pdf" and c["page"] == 3 for c in chunks)
    assert all(len(c["text"]) <= 400 for c in chunks)


def test_rejects_invalid_overlap():
    with pytest.raises(ValueError):
        chunk_pages([], chunk_size=400, overlap=400)
