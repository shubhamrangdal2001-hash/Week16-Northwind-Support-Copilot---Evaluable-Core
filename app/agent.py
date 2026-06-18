"""Minimal agent router (Tier B bounded action).

Decision is deliberately simple and auditable: if the question references a
specific order id (NW-####), call the bounded order-status tool; otherwise
answer with the grounded RAG pipeline. The whole route is traced.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .config import Settings, get_settings
from .observability import flush, observe
from .rag import RagPipeline, RagResult
from .tools import extract_order_id, format_order_answer, lookup_order_status


@dataclass
class AgentResult:
    question: str
    route: str            # "tool:order_status" | "rag"
    answer: str
    detail: Any = None


class Agent:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self.rag = RagPipeline(self.settings)

    @observe(name="agent_route")
    def handle(self, question: str) -> AgentResult:
        order_id = extract_order_id(question)
        if order_id:
            result = lookup_order_status(order_id)
            return AgentResult(
                question=question,
                route="tool:order_status",
                answer=format_order_answer(result),
                detail=result,
            )
        rag_result: RagResult = self.rag.query(question)
        return AgentResult(
            question=question,
            route="rag",
            answer=rag_result.answer,
            detail={"sources": rag_result.retrieved_doc_ids,
                    "latency_ms": rag_result.latency_ms,
                    "cost_usd": rag_result.cost_usd},
        )


def handle(question: str, settings: Settings | None = None) -> AgentResult:
    result = Agent(settings).handle(question)
    flush()
    return result


if __name__ == "__main__":
    import sys

    q = " ".join(sys.argv[1:]) or "What is the status of order NW-1001?"
    r = handle(q)
    print(f"[route={r.route}] {r.answer}")
 