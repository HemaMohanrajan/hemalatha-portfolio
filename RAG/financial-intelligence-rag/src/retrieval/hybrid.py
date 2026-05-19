import numpy as np
from .bm25 import BM25Retriever
from .dense import DenseRetriever

def minmax(x):
    x = np.array(x)
    if x.max() == x.min():
        return np.zeros_like(x)
    return (x - x.min()) / (x.max() - x.min())

class HybridRetriever:
    def __init__(self, corpus_df):
        self.texts = corpus_df["text"].tolist()
        self.df = corpus_df

        self.bm25 = BM25Retriever(self.texts)
        self.dense = DenseRetriever(self.texts)

    def search(self, query, top_k=20):
        b_idx, b_scores = self.bm25.search(query, top_k)
        d_idx, d_scores = self.dense.search(query, top_k)

        b_norm = minmax(b_scores)
        d_norm = minmax(d_scores)

        scores = {}

        for i, s in zip(b_idx, b_norm):
            scores[i] = scores.get(i, 0) + 0.5 * s

        for i, s in zip(d_idx, d_norm):
            scores[i] = scores.get(i, 0) + 0.5 * s

        sorted_ids = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [i for i, _ in sorted_ids[:top_k]]