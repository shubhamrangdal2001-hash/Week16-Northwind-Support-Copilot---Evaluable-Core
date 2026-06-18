"""Vector store abstraction.

Uses Chroma (the tool required by the brief) when ``chromadb`` is installed.
Falls back to a tiny numpy-backed JSON store so the pipeline still runs end to
end in an offline sandbox where chromadb is not available. Both expose the same
``add`` / ``query`` / ``count`` / ``reset`` interface.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .config import CHROMA_DIR, STORE_DIR, Settings


class SimpleVectorStore:
    """Cosine-similarity store persisted as a single JSON file (offline fallback)."""

    def __init__(self, collection: str):
        self.path = Path(STORE_DIR) / f"{collection}.json"
        self.ids: list[str] = []
        self.embeddings: list[list[float]] = []
        self.documents: list[str] = []
        self.metadatas: list[dict] = []
        if self.path.exists():
            self._load()

    def _load(self) -> None:
        data = json.loads(self.path.read_text())
        self.ids = data["ids"]
        self.embeddings = data["embeddings"]
        self.documents = data["documents"]
        self.metadatas = data["metadatas"]

    def _save(self) -> None:
        self.path.write_text(
            json.dumps(
                {
                    "ids": self.ids,
                    "embeddings": self.embeddings,
                    "documents": self.documents,
                    "metadatas": self.metadatas,
                }
            )
        )

    def reset(self) -> None:
        self.ids, self.embeddings, self.documents, self.metadatas = [], [], [], []
        if self.path.exists():
            self.path.unlink()

    def add(self, ids, embeddings, documents, metadatas) -> None:
        self.ids.extend(ids)
        self.embeddings.extend([list(map(float, e)) for e in embeddings])
        self.documents.extend(documents)
        self.metadatas.extend(metadatas)
        self._save()

    def count(self) -> int:
        return len(self.ids)

    def query(self, embedding, top_k: int) -> list[dict]:
        if not self.embeddings:
            return []
        mat = np.array(self.embeddings, dtype=np.float32)
        q = np.array(embedding, dtype=np.float32)
        mnorm = np.linalg.norm(mat, axis=1)
        qnorm = np.linalg.norm(q)
        denom = np.where(mnorm == 0, 1e-9, mnorm) * (qnorm if qnorm else 1e-9)
        sims = mat @ q / denom
        order = np.argsort(-sims)[:top_k]
        return [
            {
                "id": self.ids[i],
                "text": self.documents[i],
                "doc_id": self.metadatas[i].get("doc_id", self.ids[i]),
                "score": float(sims[i]),
            }
            for i in order
        ]

    def all_chunks(self) -> list[dict]:
        return [
            {"id": self.ids[i], "text": self.documents[i],
             "doc_id": self.metadatas[i].get("doc_id", self.ids[i])}
            for i in range(len(self.ids))
        ]


class ChromaVectorStore:
    def __init__(self, collection: str):
        import chromadb  # lazy import

        self.client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        self.name = collection
        self.collection = self.client.get_or_create_collection(
            name=collection, metadata={"hnsw:space": "cosine"}
        )

    def reset(self) -> None:
        try:
            self.client.delete_collection(self.name)
        except Exception:
            pass
        self.collection = self.client.get_or_create_collection(
            name=self.name, metadata={"hnsw:space": "cosine"}
        )

    def add(self, ids, embeddings, documents, metadatas) -> None:
        self.collection.add(
            ids=list(ids),
            embeddings=[list(map(float, e)) for e in embeddings],
            documents=list(documents),
            metadatas=list(metadatas),
        )

    def count(self) -> int:
        return self.collection.count()

    def query(self, embedding, top_k: int) -> list[dict]:
        res = self.collection.query(
            query_embeddings=[list(map(float, embedding))],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        out = []
        ids = res.get("ids", [[]])[0]
        docs = res.get("documents", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        dists = res.get("distances", [[]])[0]
        for i, doc_id in enumerate(ids):
            out.append(
                {
                    "id": doc_id,
                    "text": docs[i],
                    "doc_id": metas[i].get("doc_id", doc_id),
                    "score": 1.0 - float(dists[i]),  # cosine distance -> similarity
                }
            )
        return out

    def all_chunks(self) -> list[dict]:
        res = self.collection.get(include=["documents", "metadatas"])
        ids = res.get("ids", [])
        docs = res.get("documents", [])
        metas = res.get("metadatas", [])
        return [
            {"id": ids[i], "text": docs[i], "doc_id": metas[i].get("doc_id", ids[i])}
            for i in range(len(ids))
        ]


def get_vectorstore(settings: Settings):
    """Return Chroma if available, else the offline SimpleVectorStore."""
    try:
        import chromadb  # noqa: F401

        return ChromaVectorStore(settings.collection)
    except Exception:
        return SimpleVectorStore(settings.collection)
