import json
from src.embeddings import Embedder
from src.vectorstore import FaissStore
from src.retriever import HybridRetriever, merge_results
from src.config import settings


def load_eval_dataset(path):
    data = []
    with open(path, "r") as f:
        for line in f:
            data.append(json.loads(line))
    return data


def evaluate_retrieval():
    store = FaissStore.load(settings.index_dir)
    embedder = Embedder(settings.embedding_model)
    hybrid = HybridRetriever(store.meta)

    dataset = load_eval_dataset("data/eval_questions.jsonl")

    recall_at_k = 0
    total = len(dataset)

    for item in dataset:
        question = item["question"]
        expected_keyword = item["expected"]

        qv = embedder.embed_query(question)

        dense = store.search(qv, k=10)
        bm25 = hybrid.bm25_search(question, k=10)

        merged = merge_results(dense, bm25, store.meta, k=5)

        retrieved_text = " ".join([m.text.lower() for m in merged])

        if expected_keyword.lower() in retrieved_text:
            recall_at_k += 1

    recall = recall_at_k / total

    print("Retrieval Recall@5:", recall)


if __name__ == "__main__":
    evaluate_retrieval()