import re
from rank_bm25 import BM25Okapi

def tokenize(text):
    return re.sub(r"[^a-z0-9 ]", " ", text.lower()).split()

class BM25Retriever:
    def __init__(self, texts):
        self.tokens = [tokenize(t) for t in texts]
        self.bm25 = BM25Okapi(self.tokens)

    def search(self, query, top_k=20):
        scores = self.bm25.get_scores(tokenize(query))
        import numpy as np
        idx = np.argsort(scores)[::-1][:top_k]
        return idx, scores[idx]