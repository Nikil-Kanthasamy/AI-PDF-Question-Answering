import pytest

from src.text_chunker import chunk_pages


def test_chunks_keep_source_and_page():
    pages = [{"text": "A " * 500, "source": "notes.pdf", "page": 3}]
    chunks = chunk_pages(pages, chunk_size=400, overlap=80)
    assert len(chunks) > 1
    assert all(c["source"] == "notes.pdf" and c["page"] == 3 for c in chunks)
    assert all(len(c["text"]) <= 400 for c in chunks)


def test_overlap_is_preserved_between_chunks():
    text = "x" * 500
    chunks = chunk_pages([{"text": text, "source": "a.pdf", "page": 1}], 200, 50)
    assert chunks[0]["text"][-50:] == chunks[1]["text"][:50]


@pytest.mark.parametrize(
    ("chunk_size", "overlap"),
    [(99, 0), (200, -1), (200, 200), (True, 0), (200, 1.5)],
)
def test_rejects_invalid_chunk_parameters(chunk_size, overlap):
    with pytest.raises(ValueError):
        chunk_pages([], chunk_size=chunk_size, overlap=overlap)


def test_rejects_missing_page_metadata():
    with pytest.raises(ValueError, match="text, source, and page"):
        chunk_pages([{"text": "hello"}])


def test_rejects_non_string_page_text():
    with pytest.raises(ValueError, match="string"):
        chunk_pages([{"text": 42, "source": "a.pdf", "page": 1}])


def test_empty_pages_return_no_chunks():
    assert chunk_pages([]) == []
