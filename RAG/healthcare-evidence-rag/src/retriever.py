from dataclasses import dataclass
from rank_bm25 import BM25Okapi
import re

def _tokenize(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9]+", text.lower())

@dataclass(frozen=True)
class RetrievedChunk:
    text: str
    doc_id: str
    page: int
    chunk_id: int
    score: float
    source: str  # "dense" | "bm25"

class HybridRetriever:
    def __init__(self, metadatas: list[dict]):
        self.texts = [m["text"] for m in metadatas]
        self.tokens = [_tokenize(t) for t in self.texts]
        self.bm25 = BM25Okapi(self.tokens)
        self.meta = metadatas

    def bm25_search(self, query: str, k: int = 5) -> list[RetrievedChunk]:
        qtok = _tokenize(query)
        scores = self.bm25.get_scores(qtok)
        top = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        out = []
        for i in top:
            m = self.meta[i]
            out.append(RetrievedChunk(
                text=m["text"], doc_id=m["doc_id"], page=m["page"], chunk_id=m["chunk_id"],
                score=float(scores[i]), source="bm25"
            ))
        return out

def merge_results(dense: list[dict], bm25: list[RetrievedChunk], metadatas: list[dict], k: int = 6):
    # Normalize dense scores already cosine-ish; bm25 varies => do rank-based fusion
    fused = {}
    # dense list: {"score", "meta"}
    for rank, item in enumerate(dense, start=1):
        m = item["meta"]
        key = (m["doc_id"], m["page"], m["chunk_id"])
        fused.setdefault(key, {"meta": m, "rrf": 0.0})
        fused[key]["rrf"] += 1.0 / (60 + rank)

    for rank, item in enumerate(bm25, start=1):
        key = (item.doc_id, item.page, item.chunk_id)
        # get canonical meta
        if key not in fused:
            # find meta in metadatas by exact match (fast enough for demo)
            m = next((x for x in metadatas if (x["doc_id"], x["page"], x["chunk_id"]) == key), None)
            if not m:
                continue
            fused[key] = {"meta": m, "rrf": 0.0}
        fused[key]["rrf"] += 1.0 / (60 + rank)

    top = sorted(fused.values(), key=lambda x: x["rrf"], reverse=True)[:k]
    chunks = []
    for item in top:
        m = item["meta"]
        chunks.append(RetrievedChunk(
            text=m["text"], doc_id=m["doc_id"], page=m["page"], chunk_id=m["chunk_id"],
            score=float(item["rrf"]), source="hybrid"
        ))
    return chunks