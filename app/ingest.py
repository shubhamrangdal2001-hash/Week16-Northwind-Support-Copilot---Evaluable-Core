"""Ingestion: load corpus -> chunk -> embed -> store in the vector DB."""
from __future__ import annotations

from pathlib import Path

from .config import CORPUS_DIR, Settings, get_settings
from .corpus import write_corpus
from .providers import get_embedder
from .vectorstore import get_vectorstore


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Character-based sliding-window chunker with overlap.

    Small docs (most of the corpus) fit in a single chunk; longer ones split
    with overlap so a fact near a boundary is not lost.
    """
    text = text.strip()
    if len(text) <= chunk_size:
        return [text] if text else []
    chunks: list[str] = []
    start = 0
    step = max(1, chunk_size - overlap)
    while start < len(text):
        chunk = text[start : start + chunk_size].strip()
        if chunk:
            chunks.append(chunk)
        start += step
    return chunks


def load_corpus() -> dict[str, str]:
    """Read every markdown file in the corpus dir as {doc_id: text}."""
    if not any(CORPUS_DIR.glob("*.md")):
        write_corpus()
    docs: dict[str, str] = {}
    for path in sorted(Path(CORPUS_DIR).glob("*.md")):
        docs[path.stem] = path.read_text(encoding="utf-8")
    return docs


def ingest(settings: Settings | None = None, rebuild: bool = True) -> dict:
    settings = settings or get_settings()
    docs = load_corpus()
    embedder = get_embedder(settings)
    store = get_vectorstore(settings)
    if rebuild:
        store.reset()

    ids, texts, metadatas = [], [], []
    for doc_id, body in docs.items():
        for j, chunk in enumerate(chunk_text(body, settings.chunk_size, settings.chunk_overlap)):
            ids.append(f"{doc_id}::chunk{j}")
            texts.append(chunk)
            metadatas.append({"doc_id": doc_id, "chunk": j})

    # Embed in batches to stay under API limits.
    batch = 64
    embeddings = []
    for i in range(0, len(texts), batch):
        embeddings.extend(embedder.embed(texts[i : i + batch]).tolist())
    store.add(ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas)

    return {
        "documents": len(docs),
        "chunks": len(ids),
        "store": type(store).__name__,
        "provider": settings.provider,
    }


if __name__ == "__main__":
    print(ingest())
