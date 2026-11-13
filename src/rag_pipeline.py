import os
import requests
from src.embeddings import EmbeddingModel
from src.vector_store import LocalVectorStore


class RAGPipeline:
    def __init__(self, store: LocalVectorStore, model_name: str | None = None,
                 base_url: str | None = None, embedding_model: EmbeddingModel | None = None):
        self.store = store
        self.model_name = model_name or os.getenv("OLLAMA_MODEL", "qwen2.5:0.5b")
        self.base_url = (base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")).rstrip("/")
        self.embeddings = embedding_model or EmbeddingModel()

    def index(self, chunks: list[dict]) -> int:
        if not chunks:
            return 0
        self.store.add(chunks, self.embeddings.encode([item["text"] for item in chunks]))
        return len(chunks)

    def retrieve(self, question: str, top_k: int = 4) -> list[dict]:
        return self.store.search(self.embeddings.encode([question])[0], top_k=top_k)

    def answer(self, question: str, top_k: int = 4) -> tuple[str, list[dict]]:
        sources = self.retrieve(question, top_k)
        if not sources:
            return "I cannot find any indexed PDF passages. Upload and index a PDF first.", []
        context = "\n\n".join(
            f"[Source: {s['source']}, page {s['page']}]\n{s['text']}" for s in sources
        )
        prompt = f"""You answer questions using only the PDF excerpts below.
If the answer is not present, say you cannot find it in the indexed documents.
Do not invent facts or citations. Cite claims using the exact source and page labels.

EXCERPTS:
{context}

QUESTION:
{question}

Give a clear, concise answer with source/page citations."""
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={"model": self.model_name, "prompt": prompt, "stream": False,
                      "options": {"temperature": 0.1, "num_ctx": 2048}},
                timeout=180,
            )
            response.raise_for_status()
            return response.json().get("response", "").strip(), sources
        except requests.ConnectionError as exc:
            raise RuntimeError("Cannot connect to Ollama. Start Ollama and check that it is running locally.") from exc
        except requests.Timeout as exc:
            raise RuntimeError("Ollama timed out. Try a shorter question or smaller model.") from exc
        except requests.HTTPError as exc:
            detail = exc.response.text[:400] if exc.response is not None else str(exc)
            raise RuntimeError(f"Ollama request failed: {detail}") from exc
