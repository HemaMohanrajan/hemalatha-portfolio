# Healthcare Evidence RAG (Production-Style)

A healthcare-focused, evidence-grounded Retrieval-Augmented Generation (RAG) system that answers questions ONLY from provided medical PDFs, returns citations, applies safety guardrails, and includes evaluation + LLMOps tracking.

## Why this project
Healthcare is high-stakes. This project emphasizes:
- Grounded answers with citations (hallucination mitigation)
- Hybrid retrieval (dense + BM25)
- Systematic evaluation (semantic similarity + faithfulness judge)
- LLMOps via MLflow (metrics, prompts, params)
- Production API via FastAPI + Docker


## Architecture

The system implements an evidence-grounded Retrieval Augmented Generation (RAG) pipeline for healthcare question answering.

Ingestion: PDFs → chunking → embeddings → FAISS  
Query: Safety → hybrid retrieval → grounded prompt → answer + citations  
Ops: caching + logging + MLflow experiment tracking

```mermaid
flowchart TB
  %% ============ Ingestion ============
  subgraph ING[Offline Ingestion (run ingest.py)]
    A[Public Medical PDFs<br/>data/docs/] --> B[PDF Text Extractor<br/>(pypdf)]
    B --> C[Medical-aware Chunking<br/>(overlap chunks)]
    C --> D[Embeddings<br/>(Sentence-Transformers)]
    D --> E[FAISS Vector Index<br/>data/index/faiss.index]
    C --> M1[Chunk Metadata<br/>data/index/meta.jsonl]
  end

  %% ============ Serving ============
  subgraph SRV[Online Serving (FastAPI)]
    U[User / Client] -->|POST /query| API[FastAPI<br/>app.py]

    API --> S[Safety Guardrails<br/>emergency + dosing caution]
    API --> CACHE[TTL Cache<br/>(query→answer)]

    CACHE -->|hit| OUT[Answer + Citations + Latency]
    CACHE -->|miss| RAG[RAG Pipeline<br/>src/rag.py]
  end

  %% ============ Retrieval ============
  subgraph RET[Hybrid Retrieval]
    Q[Question] --> QEMB[Query Embedding]
    QEMB -->|dense| FAISS[FAISS Search]
    Q -->|sparse| BM25[BM25 Search]
    FAISS --> FUSE[Rank Fusion (RRF)]
    BM25 --> FUSE
    FUSE --> CTX[Top-k Context Chunks<br/>with citations]
  end

  %% ============ Generation ============
  subgraph GEN[Grounded Generation]
    CTX --> PROMPT[Prompt Builder<br/>grounded / cot / self_consistency]
    PROMPT --> LLM[OpenAI LLM<br/>(Responses API)]
    LLM --> ANS[Draft Answer]
    PROMPT -->|self_consistency: 3 samples| LLM
  end

  %% ============ Judge ============
  subgraph JUDGE[Faithfulness Judge (LLM-as-Judge)]
    CTX --> JP[Judge Prompt<br/>(faithfulness 1-5)]
    ANS --> JP
    JP --> LLMJ[OpenAI LLM<br/>(temp=0)]
    LLMJ --> PICK[Pick Best Candidate<br/>(max score)]
  end

  %% ============ Tracking ============
  subgraph OPS[LLMOps + Evaluation]
    API --> LOG[Logging + Metrics]
    LOG --> MLF[MLflow Tracking<br/>(params, metrics, artifacts)]
    EVAL[evaluate.py<br/>offline eval] --> MLF
  end

  %% ============ Wiring ============
  E --> RAG
  M1 --> RAG
  Q --> RAG
  RAG --> QEMB
  CTX --> PROMPT
  ANS -->|grounded/cot| OUT
  PICK --> OUT
  OUT --> CACHE

## Features
- FastAPI endpoint: POST /query
- Hybrid retrieval: FAISS (dense) + BM25 + rank-fusion
- Prompt modes: grounded, cot (optional)
- Safety guardrails: emergency detection + dosing caution messaging
- Evaluation: semantic similarity + LLM-as-judge faithfulness scoring
- MLflow: logs latency, tokens, params, artifacts

### Prompt strategies
- `grounded`: concise answer strictly from retrieved context
- `cot`: step-by-step reasoning (still grounded)
- `self_consistency`: generates 3 candidates and uses an LLM judge to pick the most faithful answer

## Quickstart
1. Put public guideline PDFs in `data/docs/`  
2. Build index:
```bash
python ingest.py


## FastAPI
![Fast API](images/ollama_screenshot.png)

## Evaluation Results

- Semantic similarity average: 0.7137191295623779
- Average latency: 92820.9565639495 ms
- Inference setup: Ollama local model on CPU

## Design Decisions

- Used hybrid retrieval (dense + BM25) to improve recall
- Used citation-grounded prompting to reduce hallucinations
- Added safety guardrails for high-risk healthcare queries
- Supported both hosted and local LLM inference
- Added experiment tracking for reproducibility

## Experiment Tracking

MLflow is used to track experiments including:

- Prompt parameters
- Retrieval settings
- Latency metrics
- Token usage
- Generated responses

Example MLflow dashboard:

![MLflow Dashboard](images/mlflow_dashboard.png)
![MLflow Dashboard](images/mlflow_run.png)