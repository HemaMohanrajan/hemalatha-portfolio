from sentence_transformers import CrossEncoder


class Reranker:
    def __init__(self):
        self.model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    def rerank(self, query, docs, top_k=5):
        pairs = [[query, doc.text] for doc in docs]
        scores = self.model.predict(pairs)

        scored = list(zip(docs, scores))
        scored.sort(key=lambda x: x[1], reverse=True)

        return [d for d, s in scored[:top_k]]