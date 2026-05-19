from nltk.tokenize import sent_tokenize

def chunk_text(text, max_chars=1200, overlap=200):
    sentences = sent_tokenize(text)
    chunks, current = [], ""

    for sent in sentences:
        if len(current) + len(sent) < max_chars:
            current += " " + sent
        else:
            chunks.append(current.strip())
            current = current[-overlap:] + " " + sent

    if current:
        chunks.append(current.strip())

    return chunks


def build_chunks(df):
    records = []
    for _, row in df.iterrows():
        chunks = chunk_text(row["raw_text"])
        for i, c in enumerate(chunks):
            records.append({
                "text": c,
                "source": "sec"
            })
    return records