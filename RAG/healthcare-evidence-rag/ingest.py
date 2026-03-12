import os
from pathlib import Path
from pypdf import PdfReader
from src.config import settings
from src.chunking import chunk_text
from src.embeddings import Embedder
from src.vectorstore import FaissStore

def read_pdf_text(pdf_path: str):
    reader = PdfReader(pdf_path)
    pages = []
    for i, page in enumerate(reader.pages):
        txt = page.extract_text() or ""
        pages.append((i + 1, txt))
    return pages

def main():
    docs_dir = Path(settings.docs_dir)
    index_dir = Path(settings.index_dir)
    docs_dir.mkdir(parents=True, exist_ok=True)
    index_dir.mkdir(parents=True, exist_ok=True)

    embedder = Embedder(settings.embedding_model)

    all_chunks = []
    metadatas = []

    pdfs = sorted([p for p in docs_dir.glob("*.pdf")])
    if not pdfs:
        raise SystemExit(f"No PDFs found in {docs_dir}. Add public medical guideline PDFs first.")

    for pdf in pdfs:
        doc_id = pdf.stem
        for page_num, text in read_pdf_text(str(pdf)):
            chunks = chunk_text(doc_id=doc_id, page=page_num, text=text)
            for c in chunks:
                metadatas.append({
                    "doc_id": c.doc_id,
                    "page": c.page,
                    "chunk_id": c.chunk_id,
                    "text": c.text
                })
                all_chunks.append(c.text)

    vecs = embedder.embed_texts(all_chunks)
    store = FaissStore(dim=vecs.shape[1])
    store.add(vecs, metadatas)
    store.save(str(index_dir))

    print(f"✅ Built index: {len(metadatas)} chunks saved to {index_dir}")

if __name__ == "__main__":
    main()