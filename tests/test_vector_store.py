import json

import numpy as np
import pytest

from src.vector_store import LocalVectorStore


def test_add_search_and_reload(tmp_path):
    store = LocalVectorStore(tmp_path)
    records = [
        {"text": "first", "source": "a.pdf", "page": 1},
        {"text": "second", "source": "b.pdf", "page": 2},
    ]
    store.add(records, np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32))

    results = store.search(np.array([0.9, 0.1]), top_k=1)
    assert len(results) == 1
    assert results[0]["text"] == "first"
    assert results[0]["score"] == pytest.approx(0.9)

    restored = LocalVectorStore(tmp_path)
    assert restored.records == records
    assert restored.vectors.shape == (2, 2)


def test_rejects_mismatched_embedding_dimensions(tmp_path):
    store = LocalVectorStore(tmp_path)
    store.add([{"text": "one"}], np.array([[1.0, 0.0]]))
    with pytest.raises(ValueError, match="dimensions differ"):
        store.add([{"text": "two"}], np.array([[1.0, 0.0, 0.0]]))


def test_rejects_invalid_search_arguments(tmp_path):
    store = LocalVectorStore(tmp_path)
    with pytest.raises(ValueError, match="top_k"):
        store.search(np.array([1.0]), top_k=0)


def test_detects_incomplete_index(tmp_path):
    store = LocalVectorStore(tmp_path)
    store.vectors_path.write_bytes(b"not a complete index")
    with pytest.raises(ValueError, match="incomplete"):
        LocalVectorStore(tmp_path)


def test_detects_record_vector_count_mismatch(tmp_path):
    store = LocalVectorStore(tmp_path)
    store.vectors = np.array([[1.0, 0.0]], dtype=np.float32)
    store.records = []
    store.save()
    with pytest.raises(ValueError, match="inconsistent"):
        LocalVectorStore(tmp_path)
