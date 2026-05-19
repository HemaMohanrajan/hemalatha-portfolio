from src.retrieval.hybrid import HybridRetriever
from src.retrieval.reranker import Reranker
from src.rag.llm import OllamaLLM

class RAGPipeline:
    def __init__(self, corpus_df):
        self.df = corpus_df
        self.retriever = HybridRetriever(corpus_df)
        self.reranker = Reranker()
        self.llm = OllamaLLM()

    def answer(self, query):
        ids = self.retriever.search(query, 50)
        texts = self.df["text"].tolist()

        ids = self.reranker.rerank(query, texts, ids)

        context = "\n\n".join([texts[i] for i in ids[:5]])

        return self.llm.generate(query, context)