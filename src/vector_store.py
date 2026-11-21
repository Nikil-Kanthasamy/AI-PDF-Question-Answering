from pathlib import Path
import json
import numpy as np


class LocalVectorStore:
    """Persistent cosine-similarity index backed by local NumPy files."""

    def __init__(self, directory: str | Path):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.vectors_path = self.directory / "vectors.npy"
        self.records_path = self.directory / "records.json"
        self.vectors = np.empty((0, 0), dtype=np.float32)
        self.records: list[dict] = []
        self.load()

    def load(self):
        if not self.vectors_path.exists() and not self.records_path.exists():
            return
        if not self.vectors_path.exists() or not self.records_path.exists():
            raise ValueError("The local index is incomplete. Clear it and index your PDFs again.")
        try:
            vectors = np.load(self.vectors_path, allow_pickle=False).astype(np.float32)
            records = json.loads(self.records_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError("The local index is corrupted. Clear it and index your PDFs again.") from exc
        if vectors.ndim != 2 or not isinstance(records, list) or len(records) != len(vectors):
            raise ValueError("The local index has inconsistent data. Clear it and index your PDFs again.")
        if len(records) and vectors.shape[1] == 0:
            raise ValueError("The local index contains empty embeddings.")
        self.vectors = vectors
        self.records = records

    def save(self):
        np.save(self.vectors_path, self.vectors)
        self.records_path.write_text(
            json.dumps(self.records, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def add(self, records: list[dict], vectors: np.ndarray):
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim != 2 or len(records) != len(vectors):
            raise ValueError("Each record must have exactly one 1-dimensional embedding.")
        if not records:
            return
        if not np.isfinite(vectors).all():
            raise ValueError("Embeddings must contain only finite values.")
        if self.vectors.size and self.vectors.shape[1] != vectors.shape[1]:
            raise ValueError("Embedding dimensions differ. Clear the index before changing models.")
        self.vectors = vectors if not self.vectors.size else np.vstack([self.vectors, vectors])
        self.records.extend(records)
        self.save()

    def search(self, query_vector: np.ndarray, top_k: int = 4) -> list[dict]:
        if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 1:
            raise ValueError("top_k must be a positive integer.")
        query = np.asarray(query_vector, dtype=np.float32).reshape(-1)
        if not np.isfinite(query).all():
            raise ValueError("Query embedding must contain only finite values.")
        if not self.records or self.vectors.size == 0:
            return []
        if query.shape[0] != self.vectors.shape[1]:
            raise ValueError("Query embedding dimension does not match the indexed embeddings.")
        scores = self.vectors @ query
        best = np.argsort(scores)[::-1][:top_k]
        return [{**self.records[int(i)], "score": float(scores[int(i)])} for i in best]

    def clear(self):
        self.vectors = np.empty((0, 0), dtype=np.float32)
        self.records = []
        for path in (self.vectors_path, self.records_path):
            if path.exists():
                path.unlink()
