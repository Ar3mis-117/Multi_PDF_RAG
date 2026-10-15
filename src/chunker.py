def chunk_pages(pages, source, chunk_size=800, overlap=150):
    """Split each page into overlapping chunks, keeping source + page metadata."""
    chunks = []
    for page in pages:
        text = page["text"].strip()
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunks.append({
                "text": text[start:end],
                "source": source,
                "page": page["page"],
            })
            if end >= len(text):
                break
            start = end - overlap
    return chunks