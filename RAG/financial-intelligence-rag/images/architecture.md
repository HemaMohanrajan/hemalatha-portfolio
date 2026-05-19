```mermaid
flowchart TD

A[SEC Filings] --> C[Cleaning & Normalization]
B[News Data] --> C

C --> D[SEC Chunking]
D --> E[Unified Corpus]

E --> F[BM25 Retrieval]
E --> G[Dense Embeddings FAISS]

F --> H[Hybrid Retrieval]
G --> H

H --> I[Cross Encoder Reranking]
I --> J[Top Context Selection]

J --> K[LLM Ollama - llama3]
K --> L[Final Answer]