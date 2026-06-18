"""Build the eval dataset, score it, and produce the baseline scorecard.

Flow:
  1. Load the golden set (eval/golden_set.jsonl).
  2. Run the RAG pipeline on every question to collect answer + retrieved contexts.
  3. Score with Ragas (faithfulness, answer_relevancy, context_precision,
     context_recall) when an LLM provider + ragas are available; otherwise fall
     back to transparent lexical HEURISTICS so the pipeline still runs offline.
  4. Compute abstention accuracy on adversarial rows.
  5. Write scorecard.json + scorecard.md (metric -> score -> Week 15 target -> pass/fail).

Usage (real):    python eval/run_eval.py
Usage (offline): LLM_PROVIDER=fake python eval/run_eval.py
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.config import RESULTS_DIR, get_settings  # noqa: E402
from app.providers import ABSTAIN, tokenize  # noqa: E402
from app.rag import RagPipeline  # noqa: E402

GOLDEN = ROOT / "eval" / "golden_set.jsonl"
TARGETS = ROOT / "eval" / "targets.json"
CORE_METRICS = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]


def load_golden() -> list[dict]:
    rows = []
    for line in GOLDEN.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def collect_predictions(rows: list[dict], settings) -> list[dict]:
    pipe = RagPipeline(settings)
    preds = []
    for row in rows:
        res = pipe.query(row["question"])
        preds.append(
            {
                **row,
                "answer": res.answer,
                "contexts": res.context_texts(),
                "retrieved_doc_ids": res.retrieved_doc_ids,
                "latency_ms": res.latency_ms,
                "cost_usd": res.cost_usd,
                "prompt_tokens": res.prompt_tokens,
                "completion_tokens": res.completion_tokens,
            }
        )
    return preds


def is_abstention(text: str) -> bool:
    t = text.lower()
    return ("i don't know" in t) or ("i do not know" in t) or (ABSTAIN.lower() in t)


# --------------------------------------------------------------------------- #
# Ragas path (real)
# --------------------------------------------------------------------------- #
def _get_ragas_llm(settings):
    """Return a LangChain LLM suitable for Ragas based on the active provider.

    For Groq: use ChatOpenAI pointed at Groq's OpenAI-compatible endpoint.
    This avoids import issues that can occur with ChatGroq inside Ragas.
    """
    if settings.provider == "groq":
        import os
        try:
            from langchain_openai import ChatOpenAI
            from ragas.llms import LangchainLLMWrapper
            groq_key = os.getenv("GROQ_API_KEY", "")
            return LangchainLLMWrapper(
                ChatOpenAI(
                    model=settings.chat_model,
                    temperature=0,
                    openai_api_key=groq_key,
                    openai_api_base="https://api.groq.com/openai/v1",
                ),
                bypass_n=True
            )
        except Exception as e:
            raise ImportError(
                f"Failed to create Groq-backed Ragas LLM: {e}. "
                "Ensure langchain-openai and a valid GROQ_API_KEY are set."
            )
    # Default: Ragas uses OPENAI_API_KEY from env automatically
    return None  # Ragas default (OpenAI)


def score_with_ragas(preds: list[dict], settings) -> dict:
    import sys
    from unittest.mock import MagicMock
    # Mock missing modules to satisfy ragas internal imports
    sys.modules["langchain_community.chat_models.vertexai"] = MagicMock()
    sys.modules["langchain_community.llms"] = MagicMock()
    sys.modules["langchain_community.llms.vertexai"] = MagicMock()

    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import (
        answer_relevancy,
        context_precision,
        context_recall,
        faithfulness,
    )

    # Adversarial rows have no ground-truth context; Ragas context_recall needs a
    # reference, so we score the four core metrics on answerable rows only and
    # report abstention separately.
    answerable = [p for p in preds if p.get("source_docs")]
    ds = Dataset.from_list(
        [
            {
                "question": p["question"],
                "answer": p["answer"],
                "contexts": p["contexts"],
                "ground_truth": p["ground_truth"],
            }
            for p in answerable
        ]
    )

    from ragas.run_config import RunConfig

    # Build evaluate() kwargs — only pass llm when using a non-default provider.
    eval_kwargs = dict(
        dataset=ds,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        run_config=RunConfig(max_workers=2),
    )
    ragas_llm = _get_ragas_llm(settings)
    if ragas_llm is not None:
        eval_kwargs["llm"] = ragas_llm

    from app.providers import get_embedder
    from langchain_core.embeddings import Embeddings

    class LangchainEmbeddingsWrapper(Embeddings):
        def __init__(self, embedder):
            self.embedder = embedder
        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            return [list(map(float, vec)) for vec in self.embedder.embed(texts)]
        def embed_query(self, text: str) -> list[float]:
            return list(map(float, self.embedder.embed([text])[0]))

    embedder = get_embedder(settings)
    eval_kwargs["embeddings"] = LangchainEmbeddingsWrapper(embedder)

    result = evaluate(**eval_kwargs)
    df = result.to_pandas()
    scores = {m: float(df[m].mean()) for m in CORE_METRICS if m in df.columns}
    scores["_engine"] = "ragas"
    scores["_per_row"] = df.to_dict(orient="records")
    return scores


# --------------------------------------------------------------------------- #
# Heuristic path (offline fallback - clearly labeled, NOT for submission)
# --------------------------------------------------------------------------- #
def _overlap(a: str, b: str) -> float:
    ta, tb = set(tokenize(a)), set(tokenize(b))
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def _coverage(answer: str, context: str) -> float:
    ta = set(tokenize(answer))
    tc = set(tokenize(context))
    if not ta:
        return 0.0
    return len(ta & tc) / len(ta)


def score_with_heuristics(preds: list[dict]) -> dict:
    answerable = [p for p in preds if p.get("source_docs")]
    faith, relev, prec, recall = [], [], [], []
    per_row = []
    for p in answerable:
        ctx = "\n".join(p["contexts"])
        f = _coverage(p["answer"], ctx)                 # claims grounded in context
        r = _overlap(p["answer"], p["question"] + " " + p["ground_truth"])
        gold = set(p["source_docs"])
        retrieved = p["retrieved_doc_ids"]
        hits = [d in gold for d in retrieved]
        cp = (sum(hits) / len(hits)) if hits else 0.0   # precision of retrieved set
        cr = (len(gold & set(retrieved)) / len(gold)) if gold else 0.0  # recall of gold docs
        faith.append(f); relev.append(r); prec.append(cp); recall.append(cr)
        per_row.append({"id": p["id"], "faithfulness": round(f, 3), "answer_relevancy": round(r, 3),
                        "context_precision": round(cp, 3), "context_recall": round(cr, 3)})
    mean = lambda xs: round(sum(xs) / len(xs), 4) if xs else 0.0
    return {
        "faithfulness": mean(faith),
        "answer_relevancy": mean(relev),
        "context_precision": mean(prec),
        "context_recall": mean(recall),
        "_engine": "heuristic-offline",
        "_per_row": per_row,
    }


def abstention_accuracy(preds: list[dict]) -> float:
    adv = [p for p in preds if not p.get("source_docs")]
    if not adv:
        return float("nan")
    correct = sum(1 for p in adv if is_abstention(p["answer"]))
    return round(correct / len(adv), 4)


def build_scorecard(scores: dict, abst: float, targets: dict, preds: list[dict]) -> dict:
    rows = []
    for m in CORE_METRICS:
        score = scores.get(m)
        target = targets.get(m)
        passed = (score is not None and target is not None and score >= target)
        rows.append({"metric": m, "score": score, "target": target, "pass": passed})
    if abst == abst:  # not NaN
        rows.append({"metric": "abstention_accuracy", "score": abst,
                     "target": targets.get("abstention_accuracy"),
                     "pass": abst >= targets.get("abstention_accuracy", 1.0)})
    lat = [p["latency_ms"] for p in preds]
    cost = [p["cost_usd"] for p in preds]
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "engine": scores.get("_engine"),
        "rows": rows,
        "operational": {
            "avg_latency_ms": round(sum(lat) / len(lat), 1) if lat else 0,
            "p95_latency_ms": round(sorted(lat)[int(0.95 * (len(lat) - 1))], 1) if lat else 0,
            "avg_cost_per_query_usd": round(sum(cost) / len(cost), 6) if cost else 0,
            "total_queries": len(preds),
        },
        "per_row": scores.get("_per_row", []),
    }


def scorecard_to_markdown(card: dict) -> str:
    lines = ["# Baseline Scorecard", ""]
    lines.append(f"Engine: **{card['engine']}**  |  Generated: {card['generated_at']}")
    if card["engine"] == "heuristic-offline":
        lines.append("")
        lines.append("> WARNING: offline heuristic scores - plumbing check only. "
                     "Run with real OpenAI keys + ragas for submittable numbers.")
    lines += ["", "| Metric | Score | Week 15 Target | Pass/Fail |", "|---|---|---|---|"]
    for r in card["rows"]:
        s = "-" if r["score"] is None else f"{r['score']:.3f}"
        t = "-" if r["target"] is None else f"{r['target']:.2f}"
        verdict = "PASS" if r["pass"] else "FAIL"
        lines.append(f"| {r['metric']} | {s} | {t} | {verdict} |")
    op = card["operational"]
    lines += [
        "", "## Operational (Tier A)", "",
        f"- Avg latency: {op['avg_latency_ms']} ms",
        f"- p95 latency: {op['p95_latency_ms']} ms",
        f"- Avg cost / query: ${op['avg_cost_per_query_usd']}",
        f"- Total queries: {op['total_queries']}",
    ]
    return "\n".join(lines) + "\n"


def run(settings=None, label: str = "baseline") -> dict:
    settings = settings or get_settings()
    rows = load_golden()
    targets = json.loads(TARGETS.read_text())
    preds = collect_predictions(rows, settings)

    try:
        if settings.is_offline:
            raise RuntimeError("offline provider -> heuristic scorer")
        scores = score_with_ragas(preds, settings)
    except Exception as exc:
        print(f"[eval] Using heuristic scorer ({exc})")
        scores = score_with_heuristics(preds)

    abst = abstention_accuracy(preds)
    card = build_scorecard(scores, abst, targets, preds)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / f"{label}_scorecard.json").write_text(json.dumps(card, indent=2))
    (RESULTS_DIR / f"{label}_scorecard.md").write_text(scorecard_to_markdown(card))
    (RESULTS_DIR / f"{label}_predictions.json").write_text(json.dumps(preds, indent=2))
    print(scorecard_to_markdown(card))
    return card


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default="baseline")
    args = ap.parse_args()
    run(label=args.label)
 