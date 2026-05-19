# 🚀 Financial RAG System  
### Hybrid Retrieval over SEC Filings + Financial News  

> 🚀 Production-grade RAG pipeline using Hybrid Retrieval + Reranking + LLM

![Python](https://img.shields.io/badge/Python-3.10-blue)
![RAG](https://img.shields.io/badge/AI-RAG-orange)
![FAISS](https://img.shields.io/badge/Vector-FAISS-green)
![BM25](https://img.shields.io/badge/Retrieval-BM25-yellow)

---

## 🔥 Overview

This project builds a **production-grade Retrieval-Augmented Generation (RAG)** system that combines:

- 📄 SEC 10-K filings (structured financial disclosures)  
- 📰 Financial news headlines (real-time market signals)  

to answer complex financial questions using **hybrid retrieval, reranking, and LLM-based generation**.

---

## 🏗️ Architecture

```mermaid
flowchart TD
A[SEC Filings] --> C[Cleaning & Normalization]
B[News Data] --> C
C --> D[SEC Chunking]
D --> E[Unified Corpus]
E --> F[BM25 Retrieval]
E --> G[Dense Embeddings (FAISS)]
F --> H[Hybrid Retrieval]
G --> H
H --> I[Cross-Encoder Reranking]
I --> J[Top Context Selection]
J --> K[LLM (Ollama - llama3)]
K --> L[Final Answer]
```

## 🧠 Tech Stack
- Python
- FAISS (Vector Search)
- BM25 (Sparse Retrieval)
- Sentence Transformers (Embeddings)
- Cross-Encoder (Reranking)
- Ollama (llama3) (LLM)
- NLTK (Text Processing)



## ⚡ Key Highlights
🔎 Hybrid Retrieval (BM25 + Dense Embeddings)
🎯 Cross-Encoder Reranking for improved relevance
✂️ Optimized SEC Chunking Strategy
🏗️ Production-Style Modular Architecture
🤖 End-to-End RAG Pipeline with LLM integration


## ⚙️ Key Features
✅ Hybrid search (BM25 + FAISS)
✅ Cross-encoder reranking
✅ Optimized SEC chunking
✅ Modular production design
✅ LLM integration using Ollama (llama3)


## 🎯 Example Query

What risks did tech companies report in 2021?


## 🚀 How It Works

Load SEC filings and financial news datasets
Clean and normalize text data
Apply chunking for long SEC documents
Build unified corpus
Perform hybrid retrieval (BM25 + embeddings)
Apply cross-encoder reranking
Select top relevant context
Generate answer using LLM

## 💼 Real-World Applications

📈 Financial risk analysis
💰 Investment research
⚖️ Regulatory and compliance insights
🤖 AI-powered financial assistants and decision support systems


## ⚙️ Setup & Run

1. Clone the repository
git clone https://github.com/HemaMohanrajan/hemalatha-portfolio/tree/main/RAG/financial-intelligence-rag.git
cd financial-intelligence-rag

2. Create virtual environment
python -m venv venv
venv\Scripts\activate

3. Install dependencies
pip install -r requirements.txt

4. Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"

5. Setup environment variables
Create a .env file:
OLLAMA_BASE_URL=http://localhost:11434

6. Run Ollama (in separate terminal)
ollama serve

7. Run the project
python -m src.main


## 🔮 Future Improvements
🔗 OpenAI / multi-LLM integration
📊 Evaluation metrics (Recall@K, MRR)
🌐 FastAPI deployment
🎨 Streamlit UI for interactive querying
⚡ Performance optimization
🎯 Why This Project Matters

This project demonstrates how modern AI systems:

Combine structured + unstructured data
Improve retrieval using hybrid search
Enhance accuracy with reranking
Generate context-aware responses using LLMs

👉 This reflects real-world production AI system design used in industry.

⭐ If you found this useful
⭐ Star the repository
🍴 Fork and experiment
💬 Share feedback

