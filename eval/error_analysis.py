"""Error-analysis table of the 5 worst cases (Tier B).

Reads the latest baseline predictions + per-row scores, ranks the worst cases,
and tags each root cause as RETRIEVAL or GENERATION using a simple rule:

  * If none of the gold source docs were retrieved   -> RETRIEVAL
  * Else (right docs retrieved but answer still bad)  -> GENERATION
  * Adversarial row that failed to abstain            -> GENERATION (over-answering)

Writes eval/results/error_analysis.{json,md}. Run eval first so predictions exist.

Usage (offline): LLM_PROVIDER=fake LANGFUSE_ENABLED=false python eval/error_analysis.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.config import RESULTS_DIR  # noqa: E402
from eval.run_eval import is_abstention  # noqa: E402


def _quality(pred: dict, faith_lookup: dict) -> float:
    """Lower is worse. Combines retrieval recall and (if available) faithfulness."""
    gold = set(pred.get("source_docs") or [])
    retrieved = set(pred.get("retrieved_doc_ids") or [])
    if not gold:  # adversarial: bad iff it did NOT abstain
        return 1.0 if is_abstention(pred["answer"]) else 0.0
    recall = len(gold & retrieved) / len(gold)
    faith = faith_lookup.get(pred["id"], recall)
    return 0.5 * recall + 0.5 * faith


def _root_cause(pred: dict) -> str:
    gold = set(pred.get("source_docs") or [])
    retrieved = set(pred.get("retrieved_doc_ids") or [])
    if not gold:
        return "generation (over-answered adversarial)" if not is_abstention(pred["answer"]) else "none"
    if not (gold & retrieved):
        return "retrieval (gold doc not retrieved)"
    return "generation (right context, weak answer)"


def run(label: str = "baseline", n: int = 5) -> dict:
    preds_path = RESULTS_DIR / f"{label}_predictions.json"
    card_path = RESULTS_DIR / f"{label}_scorecard.json"
    if not preds_path.exists():
        raise SystemExit(f"Run eval first: missing {preds_path}")
    preds = json.loads(preds_path.read_text())
    card = json.loads(card_path.read_text()) if card_path.exists() else {}
    faith_lookup = {r["id"]: r.get("faithfulness", 0.0)
                    for r in card.get("per_row", []) if "id" in r}

    ranked = sorted(preds, key=lambda p: _quality(p, faith_lookup))
    worst = ranked[:n]
    table = [{
        "id": p["id"], "flavor": p.get("flavor"),
        "question": p["question"],
        "gold": p.get("source_docs"),
        "retrieved": p.get("retrieved_doc_ids"),
        "answer": p["answer"][:160],
        "root_cause": _root_cause(p),
    } for p in worst]

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "error_analysis.json").write_text(json.dumps(table, indent=2))
    md = ["# Error Analysis - 5 Worst Cases", "",
          "| id | flavor | root cause | gold | retrieved | answer (truncated) |",
          "|---|---|---|---|---|---|"]
    for r in table:
        md.append(
            f"| {r['id']} | {r['flavor']} | {r['root_cause']} | "
            f"{r['gold']} | {r['retrieved']} | {r['answer'].replace('|', '/')} |"
        )
    counts: dict[str, int] = {}
    for r in table:
        key = "retrieval" if r["root_cause"].startswith("retrieval") else (
            "generation" if r["root_cause"].startswith("generation") else "none")
        counts[key] = counts.get(key, 0) + 1
    md += ["", f"**Root-cause tally:** {counts}"]
    (RESULTS_DIR / "error_analysis.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    return {"worst": table, "counts": counts}


if __name__ == "__main__":
    run()
