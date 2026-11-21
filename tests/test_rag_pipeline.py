import numpy as np
import pytest

from src.rag_pipeline import RAGPipeline
from src.vector_store import LocalVectorStore


class FakeEmbeddings:
    def encode(self, texts):
        return np.array([[1.0, 0.0] for _ in texts], dtype=np.float32)


def make_pipeline(tmp_path):
    store = LocalVectorStore(tmp_path)
    store.add(
        [{"text": "The sky is blue.", "source": "facts.pdf", "page": 4}],
        np.array([[1.0, 0.0]], dtype=np.float32),
    )
    return RAGPipeline(store, embedding_model=FakeEmbeddings())


def test_retrieve_returns_page_aware_source(tmp_path):
    pipeline = make_pipeline(tmp_path)
    results = pipeline.retrieve("What color is the sky?")
    assert results[0]["source"] == "facts.pdf"
    assert results[0]["page"] == 4


def test_rejects_blank_question(tmp_path):
    pipeline = make_pipeline(tmp_path)
    with pytest.raises(ValueError, match="question"):
        pipeline.retrieve("  ")


def test_answer_returns_guidance_when_index_is_empty(tmp_path):
    pipeline = RAGPipeline(LocalVectorStore(tmp_path), embedding_model=FakeEmbeddings())
    answer, sources = pipeline.answer("A question")
    assert "index a PDF" in answer
    assert sources == []


def test_answer_calls_ollama_and_returns_sources(tmp_path, monkeypatch):
    pipeline = make_pipeline(tmp_path)

    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {"response": "The sky is blue [facts.pdf, page 4]."}

    def fake_post(url, json, timeout):
        assert url.endswith("/api/generate")
        assert json["stream"] is False
        assert "facts.pdf, page 4" in json["prompt"]
        return FakeResponse()

    monkeypatch.setattr("src.rag_pipeline.requests.post", fake_post)
    answer, sources = pipeline.answer("What color is the sky?")
    assert answer.startswith("The sky is blue")
    assert sources[0]["page"] == 4


def test_empty_ollama_response_is_reported(tmp_path, monkeypatch):
    pipeline = make_pipeline(tmp_path)

    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {"response": ""}

    monkeypatch.setattr("src.rag_pipeline.requests.post", lambda *args, **kwargs: FakeResponse())
    with pytest.raises(RuntimeError, match="empty answer"):
        pipeline.answer("What color is the sky?")
