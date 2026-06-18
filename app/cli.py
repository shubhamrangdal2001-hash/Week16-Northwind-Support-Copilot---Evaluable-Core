"""Command-line entry point.

    python -m app.cli build-corpus
    python -m app.cli ingest
    python -m app.cli ask "How long does standard shipping take?"

Run from the repository root.
"""
from __future__ import annotations

import argparse
import json

from .config import get_settings
from .corpus import write_corpus
from .ingest import ingest
from .rag import ask


def main() -> None:
    parser = argparse.ArgumentParser(description="Northwind Support Copilot - Evaluable Core")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("build-corpus", help="Write the synthetic Northwind corpus to disk")

    p_ing = sub.add_parser("ingest", help="Chunk, embed, and index the corpus")
    p_ing.add_argument("--top-k", type=int)
    p_ing.add_argument("--chunk-size", type=int)
    p_ing.add_argument("--chunk-overlap", type=int)

    p_ask = sub.add_parser("ask", help="Ask the copilot a question (RAG only)")
    p_ask.add_argument("question", nargs="+")
    p_ask.add_argument("--top-k", type=int)
    p_ask.add_argument("--retrieval-mode", choices=["vector", "hybrid"])

    p_agent = sub.add_parser("agent", help="Ask via the agent router (tool or RAG)")
    p_agent.add_argument("question", nargs="+")
    p_agent.add_argument("--retrieval-mode", choices=["vector", "hybrid"])

    args = parser.parse_args()

    if args.command == "build-corpus":
        paths = write_corpus()
        print(f"Wrote {len(paths)} documents.")
        return

    if args.command == "ingest":
        settings = get_settings(
            top_k=getattr(args, "top_k", None),
            chunk_size=getattr(args, "chunk_size", None),
            chunk_overlap=getattr(args, "chunk_overlap", None),
        )
        print(json.dumps(ingest(settings), indent=2))
        return

    if args.command == "ask":
        settings = get_settings(
            top_k=getattr(args, "top_k", None),
            retrieval_mode=getattr(args, "retrieval_mode", None),
        )
        result = ask(" ".join(args.question), settings)
        print(f"\nQ: {result.question}")
        print(f"A: {result.answer}\n")
        print(f"Sources : {result.retrieved_doc_ids}")
        print(f"Latency : {result.latency_ms:.0f} ms")
        print(f"Cost    : ${result.cost_usd:.6f}")
        return

    if args.command == "agent":
        from .agent import handle

        settings = get_settings(retrieval_mode=getattr(args, "retrieval_mode", None))
        result = handle(" ".join(args.question), settings)
        print(f"\nQ: {result.question}")
        print(f"[route = {result.route}]")
        print(f"A: {result.answer}")
        return


if __name__ == "__main__":
    main()
