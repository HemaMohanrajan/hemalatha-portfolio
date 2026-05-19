import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

class DenseRetriever:
    def __init__(self, texts):
        self.model = SentenceTransformer("sentence-transformers/paraphrase-MiniLM-L3-v2")

        embeddings = self.model.encode(texts, normalize_embeddings=True)
        self.index = faiss.IndexFlatIP(embeddings.shape[1])
        self.index.add(embeddings.astype("float32"))

    def search(self, query, top_k=20):
        q = self.model.encode([query], normalize_embeddings=True)
        scores, idx = self.index.search(q, top_k)
        return idx[0], scores[0]