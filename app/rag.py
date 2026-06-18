"""The thin vertical RAG slice.

retrieve top-k -> assemble prompt with retrieved context -> LLM answer with
citations. Each step is wrapped with Langfuse @observe so a full trace
(retrieval + generation, with tokens / latency / cost) is logged per query.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .config import Settings, get_settings
from .observability import flush, observe
from .providers import get_llm
from .retrieval import Retriever


@dataclass
class RagResult:
    question: str
    answer: str
    contexts: list[dict]          # retrieved chunks [{id, doc_id, text, score}]
    retrieved_doc_ids: list[str]
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float
    cost_usd: float

    def context_texts(self) -> list[str]:
        return [c["text"] for c in self.contexts]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class RagPipeline:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self.retriever = Retriever(self.settings)
        self.llm = get_llm(self.settings)

    @observe(name="retrieve")
    def retrieve(self, question: str) -> list[dict]:
        return self.retriever.retrieve(question)

    @observe(name="generate")
    def generate(self, question: str, contexts: list[dict]):
        return self.llm.answer(question, contexts)

    @observe(name="rag_query")
    def query(self, question: str) -> RagResult:
        contexts = self.retrieve(question)
        ans = self.generate(question, contexts)
        seen: list[str] = []
        for c in contexts:
            if c["doc_id"] not in seen:
                seen.append(c["doc_id"])
        return RagResult(
            question=question,
            answer=ans.text,
            contexts=contexts,
            retrieved_doc_ids=seen,
            prompt_tokens=ans.prompt_tokens,
            completion_tokens=ans.completion_tokens,
            latency_ms=ans.latency_ms,
            cost_usd=ans.cost_usd,
        )


def ask(question: str, settings: Settings | None = None) -> RagResult:
    pipe = RagPipeline(settings)
    result = pipe.query(question)
    flush()
    return result


if __name__ == "__main__":
    import sys

    q = " ".join(sys.argv[1:]) or "How long does standard shipping take?"
    r = ask(q)
    print("Q:", r.question)
    print("A:", r.answer)
    print("Sources:", r.retrieved_doc_ids)
    print(f"latency={r.latency_ms:.0f}ms cost=${r.cost_usd:.6f}")
