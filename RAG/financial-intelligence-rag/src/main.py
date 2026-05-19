from pathlib import Path
import pandas as pd

from src.ingestion.news_loader import load_news_data
from src.ingestion.sec_loader import load_sec_sections
from src.processing.cleaner import clean_news_data
from src.processing.chunker import build_chunks
from src.rag.pipeline import RAGPipeline

NEWS_DIR = Path("data/news")
SEC_DIR = Path("data/sec")

def build_corpus():
    news = load_news_data(NEWS_DIR)
    news = clean_news_data(news)

    sec = load_sec_sections(SEC_DIR)
    sec_chunks = build_chunks(sec)

    sec_df = pd.DataFrame(sec_chunks)
    news["source"] = "news"

    return pd.concat([
        sec_df[["text", "source"]],
        news[["text", "source"]]
    ], ignore_index=True)

def main():
    corpus = build_corpus()
    rag = RAGPipeline(corpus)

    query = "What risks did tech companies report in 2021?"
    answer = rag.answer(query)

    print("\nANSWER:\n", answer)

if __name__ == "__main__":
    main()