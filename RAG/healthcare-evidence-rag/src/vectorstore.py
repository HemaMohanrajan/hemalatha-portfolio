import os
import json
import numpy as np
import faiss

class FaissStore:
    def __init__(self, dim: int):
        self.index = faiss.IndexFlatIP(dim)
        self.meta = []  # aligned with vectors

    def add(self, vectors: np.ndarray, metadatas: list[dict]):
        self.index.add(vectors)
        self.meta.extend(metadatas)

    def search(self, query_vec: np.ndarray, k: int = 5):
        q = np.array([query_vec], dtype="float32")
        scores, ids = self.index.search(q, k)
        results = []
        for score, idx in zip(scores[0].tolist(), ids[0].tolist()):
            if idx == -1:
                continue
            m = self.meta[idx]
            results.append({"score": float(score), "id": int(idx), "meta": m})
        return results

    def save(self, index_dir: str):
        os.makedirs(index_dir, exist_ok=True)
        faiss.write_index(self.index, os.path.join(index_dir, "faiss.index"))
        with open(os.path.join(index_dir, "meta.jsonl"), "w", encoding="utf-8") as f:
            for m in self.meta:
                f.write(json.dumps(m, ensure_ascii=False) + "\n")

    @staticmethod
    def load(index_dir: str):
        idx_path = os.path.join(index_dir, "faiss.index")
        meta_path = os.path.join(index_dir, "meta.jsonl")
        if not (os.path.exists(idx_path) and os.path.exists(meta_path)):
            raise FileNotFoundError("Index not found. Run ingest.py first.")
        index = faiss.read_index(idx_path)
        store = FaissStore(index.d)
        store.index = index
        meta = []
        with open(meta_path, "r", encoding="utf-8") as f:
            for line in f:
                meta.append(json.loads(line))
        store.meta = meta
        return store