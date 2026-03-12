import json
from pathlib import Path
from src.config import settings
from src.vectorstore import FaissStore
from src.embeddings import Embedder
from src.llm_providers import OpenAILLM, OllamaLLM
from src.rag import RAGPipeline
from src.eval_metrics import semantic_similarity, faithfulness_score
from src.tracking import RunLogger

def load_eval(path: str):
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            items.append(json.loads(line))
    return items

def main():
    store = FaissStore.load(settings.index_dir)
    embedder = Embedder(settings.embedding_model)

    if settings.llm_provider == "openai":
        llm = OpenAILLM(settings.openai_api_key, settings.openai_model)
    else:
        llm = OllamaLLM(settings.ollama_base_url, settings.ollama_model)

    pipe = RAGPipeline(store, embedder, llm)
    eval_items = load_eval("data/eval_questions.jsonl")

    run = RunLogger()
    run.log_params(
        eval_items=len(eval_items),
        llm_provider=settings.llm_provider,
        llm_model=(settings.openai_model if settings.llm_provider == "openai" else settings.ollama_model),
        embedding_model=settings.embedding_model,
    )

    sem_scores = []
    faith_scores = []
    latencies = []

    for it in eval_items:
        q = it["question"]
        expected = it["expected"]

        ans = pipe.ask(q, k=6, mode="grounded", temperature=0.2, use_cache=False)
        context = "\n\n".join([r["preview"] for r in ans.retrieval])

        sem = semantic_similarity(embedder, ans.answer, expected)
        faith = 0 
        # Temporarily skip LLM-as-judge for low-memory local evaluation
        #faith, rationale = faithfulness_score(llm, q, ans.answer, context)

        sem_scores.append(sem)
        faith_scores.append(faith)
        latencies.append(ans.latency_ms)

    run.log_metrics(
        semantic_similarity_avg=sum(sem_scores)/len(sem_scores),
        faithfulness_avg=sum(faith_scores)/len(faith_scores),
        latency_ms_avg=sum(latencies)/len(latencies),
    )
    run.end()

    print("✅ Eval complete")
    print("Semantic avg:", sum(sem_scores)/len(sem_scores))
    #print("Faithfulness avg:", sum(faith_scores)/len(faith_scores))
    print("Latency ms avg:", sum(latencies)/len(latencies))

if __name__ == "__main__":
    main()