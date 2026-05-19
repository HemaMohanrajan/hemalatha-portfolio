import numpy as np
from sentence_transformers import CrossEncoder

class Reranker:
    def __init__(self):
        self.model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    def rerank(self, query, texts, ids, top_k=10):
        pairs = [(query, texts[i]) for i in ids]
        scores = self.model.predict(pairs)
        order = np.argsort(scores)[::-1]
        return [ids[i] for i in order[:top_k]]