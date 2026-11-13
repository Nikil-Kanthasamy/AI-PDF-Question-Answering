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
        if self.vectors_path.exists() and self.records_path.exists():
            self.vectors = np.load(self.vectors_path).astype(np.float32)
            self.records = json.loads(self.records_path.read_text(encoding="utf-8"))

    def save(self):
        np.save(self.vectors_path, self.vectors)
        self.records_path.write_text(json.dumps(self.records, ensure_ascii=False, indent=2), encoding="utf-8")

    def add(self, records: list[dict], vectors: np.ndarray):
        if len(records) != len(vectors):
            raise ValueError("Each record must have exactly one embedding.")
        if not records:
            return
        vectors = np.asarray(vectors, dtype=np.float32)
        if self.vectors.size and self.vectors.shape[1] != vectors.shape[1]:
            raise ValueError("Embedding dimensions differ. Clear the index before changing models.")
        self.vectors = vectors if not self.vectors.size else np.vstack([self.vectors, vectors])
        self.records.extend(records)
        self.save()

    def search(self, query_vector: np.ndarray, top_k: int = 4) -> list[dict]:
        if not self.records or self.vectors.size == 0:
            return []
        scores = self.vectors @ np.asarray(query_vector, dtype=np.float32).reshape(-1)
        best = np.argsort(scores)[::-1][:top_k]
        return [{**self.records[int(i)], "score": float(scores[int(i)])} for i in best]

    def clear(self):
        self.vectors = np.empty((0, 0), dtype=np.float32)
        self.records = []
        for path in (self.vectors_path, self.records_path):
            if path.exists():
                path.unlink()
