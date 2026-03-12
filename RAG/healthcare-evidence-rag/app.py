from fastapi import FastAPI
from fastapi import HTTPException
from pydantic import BaseModel, Field
from src.config import settings
from src.vectorstore import FaissStore
from src.embeddings import Embedder
from src.llm_providers import OpenAILLM, OllamaLLM
from src.rag import RAGPipeline
from src.tracking import RunLogger

app = FastAPI(title="Healthcare Evidence RAG", version="1.0.0")

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3)
    k: int = 6
    mode: str = "grounded"   # grounded | cot | self_consistency
    temperature: float = 0.2
    use_cache: bool = True

class QueryResponse(BaseModel):
    answer: str
    citations: list[str]
    latency_ms: float
    tokens_in: int
    tokens_out: int
    estimated_cost: float
    safety_note: str
    retrieval: list[dict]

def build_pipeline():
    store = FaissStore.load(settings.index_dir)
    embedder = Embedder(settings.embedding_model)

    if settings.llm_provider == "openai":
        if not settings.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY missing.")
        llm = OpenAILLM(settings.openai_api_key, settings.openai_model)
    elif settings.llm_provider == "ollama":
        llm = OllamaLLM(settings.ollama_base_url, settings.ollama_model)
    else:
        raise RuntimeError("LLM_PROVIDER must be openai or ollama.")

    return RAGPipeline(store=store, embedder=embedder, llm=llm)

PIPELINE = None

@app.on_event("startup")
def _startup():
    global PIPELINE
    PIPELINE = build_pipeline()

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    global PIPELINE

    run = RunLogger()
    try:
        run.log_params(
            llm_provider=settings.llm_provider,
            llm_model=(settings.openai_model if settings.llm_provider == "openai" else settings.ollama_model),
            embedding_model=settings.embedding_model,
            k=req.k,
            mode=req.mode,
            temperature=req.temperature,
            cache=req.use_cache,
        )

        allowed_modes = {"grounded", "cot", "self_consistency"}
        if req.mode not in allowed_modes:
            return QueryResponse(
                answer=f"Invalid mode. Choose one of: {sorted(list(allowed_modes))}",
                citations=[],
                latency_ms=0.0,
                tokens_in=0,
                tokens_out=0,
                estimated_cost=0.0,
                safety_note="invalid_request",
                retrieval=[]
            )
        
        if PIPELINE is None:
            raise RuntimeError("Pipeline not initialized.")

        ans = PIPELINE.ask(
            question=req.question,
            k=req.k,
            mode=req.mode,
            temperature=req.temperature,
            use_cache=req.use_cache,
        )

        run.log_metrics(
            latency_ms=ans.latency_ms,
            tokens_in=ans.tokens_in,
            tokens_out=ans.tokens_out,
            estimated_cost=ans.estimated_cost,
            citations=len(ans.citations),
        )
        run.log_text(req.question, "question.txt")
        run.log_text(ans.answer, "answer.txt")

        return QueryResponse(**ans.__dict__)
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        run.end()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=settings.host, port=settings.port)