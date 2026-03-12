from dataclasses import dataclass

@dataclass(frozen=True)
class Chunk:
    doc_id: str
    page: int
    chunk_id: int
    text: str

def chunk_text(doc_id: str, page: int, text: str, chunk_size: int = 900, overlap: int = 150):
    text = " ".join(text.split())
    chunks = []
    start = 0
    cid = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        chunk = text[start:end]
        if len(chunk.strip()) > 50:
            chunks.append(Chunk(doc_id=doc_id, page=page, chunk_id=cid, text=chunk))
            cid += 1
        start = max(end - overlap, end)
    return chunks