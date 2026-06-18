"""Retrieval strategies: dense (vector) and hybrid (vector + keyword + RRF rerank).

The hybrid retriever runs a dense vector search and a lexical BM25-lite search,
then fuses the two ranked lists with Reciprocal Rank Fusion (RRF) - a simple,
robust reranker that needs no extra model and works offline. This is the Bonus
\"reranking / hybrid retrieval\" path; the experiment runner quantifies its lift.
"""
from __future__ import annotations

import math
from collections import Counter

from .config import Settings
from .providers import get_embedder, tokenize
from .vectorstore import get_vectorstore

RRF_K = 60  # standard RRF damping constant


class _KeywordIndex:
    """Tiny in-memory BM25 over the indexed chunks (corpus is small)."""

    def __init__(self, chunks: list[dict], k1: float = 1.5, b: float = 0.75):
        self.chunks = chunks
        self.k1, self.b = k1, b
        self.docs_tokens = [tokenize(c["text"]) for c in chunks]
        self.doc_len = [len(t) for t in self.docs_tokens]
        self.avgdl = (sum(self.doc_len) / len(self.doc_len)) if self.doc_len else 0.0
        self.df: Counter = Counter()
        for toks in self.docs_tokens:
            for term in set(toks):
                self.df[term] += 1
        self.n = len(chunks)

    def _idf(self, term: str) -> float:
        df = self.df.get(term, 0)
        return math.log(1 + (self.n - df + 0.5) / (df + 0.5))

    def search(self, query: str, top_k: int) -> list[dict]:
        q_terms = tokenize(query)
        scored = []
        for i, toks in enumerate(self.docs_tokens):
            if not toks:
                continue
            tf = Counter(toks)
            score = 0.0
            for term in q_terms:
                if term not in tf:
                    continue
                idf = self._idf(term)
                denom = tf[term] + self.k1 * (1 - self.b + self.b * self.doc_len[i] / (self.avgdl or 1))
                score += idf * (tf[term] * (self.k1 + 1)) / (denom or 1)
            if score > 0:
                scored.append((score, i))
        scored.sort(reverse=True)
        return [{**self.chunks[i], "score": float(s)} for s, i in scored[:top_k]]


def _rrf_fuse(ranked_lists: list[list[dict]], top_k: int) -> list[dict]:
    fused: dict[str, dict] = {}
    for ranking in ranked_lists:
        for rank, item in enumerate(ranking):
            cid = item["id"]
            entry = fused.setdefault(cid, {**item, "score": 0.0})
            entry["score"] += 1.0 / (RRF_K + rank + 1)
    return sorted(fused.values(), key=lambda x: -x["score"])[:top_k]


class Retriever:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.embedder = get_embedder(settings)
        self.store = get_vectorstore(settings)
        self._kw: _KeywordIndex | None = None

    def _keyword_index(self) -> _KeywordIndex:
        if self._kw is None:
            self._kw = _KeywordIndex(self.store.all_chunks())
        return self._kw

    def retrieve(self, question: str) -> list[dict]:
        top_k = self.settings.top_k
        if self.settings.retrieval_mode == "hybrid":
            cand = self.settings.candidate_k
            q_emb = self.embedder.embed([question])[0]
            dense = self.store.query(q_emb, cand)
            lexical = self._keyword_index().search(question, cand)
            return _rrf_fuse([dense, lexical], top_k)
        # default: dense only
        q_emb = self.embedder.embed([question])[0]
        return self.store.query(q_emb, top_k)
 