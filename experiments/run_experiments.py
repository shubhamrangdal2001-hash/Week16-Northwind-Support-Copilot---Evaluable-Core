"""Run 2-3 improvement experiments and report metric deltas.

Each experiment changes exactly ONE variable vs the baseline, re-ingests if the
change affects indexing (chunk size), re-runs the same golden set, and records
the metric delta. Results are written to experiments/results/.

Usage (offline smoke):  LLM_PROVIDER=fake python experiments/run_experiments.py
Usage (real):           python experiments/run_experiments.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.config import EXPERIMENTS_DIR, get_settings  # noqa: E402
from app.ingest import ingest  # noqa: E402
from eval.run_eval import CORE_METRICS, run as run_eval  # noqa: E402

# Each experiment: name, settings overrides, whether it requires re-ingesting.
EXPERIMENTS = [
    {"name": "baseline", "overrides": {}, "reingest": True,
     "note": "top_k=4, chunk_size=700, prompt=grounded_v1"},
    {"name": "exp1_topk2", "overrides": {"top_k": 2}, "reingest": False,
     "note": "Fewer retrieved chunks (top_k 4 -> 2)"},
    {"name": "exp2_smallchunks", "overrides": {"chunk_size": 350, "chunk_overlap": 80}, "reingest": True,
     "note": "Smaller chunks (700 -> 350 chars)"},
    {"name": "exp3_prompt_v2", "overrides": {"prompt_variant": "grounded_v2"}, "reingest": False,
     "note": "Stricter, citation-forcing prompt (grounded_v2)"},
    {"name": "exp4_hybrid", "overrides": {"retrieval_mode": "hybrid"}, "reingest": False,
     "note": "BONUS: hybrid retrieval (vector + BM25 fused via RRF)"},
]


def _metrics(card: dict) -> dict:
    return {r["metric"]: r["score"] for r in card["rows"]}


def main() -> None:
    base_settings = get_settings()
    results = []
    baseline_metrics = None

    for exp in EXPERIMENTS:
        settings = base_settings.with_overrides(**exp["overrides"])
        if exp["reingest"]:
            ingest(settings, rebuild=True)
        card = run_eval(settings=settings, label=exp["name"])
        metrics = _metrics(card)
        if exp["name"] == "baseline":
            baseline_metrics = metrics
        delta = {
            m: (None if metrics.get(m) is None or baseline_metrics.get(m) is None
                else round(metrics[m] - baseline_metrics[m], 4))
            for m in CORE_METRICS
        }
        results.append({"name": exp["name"], "note": exp["note"],
                        "metrics": metrics, "delta_vs_baseline": delta,
                        "operational": card["operational"]})

    # Re-ingest back to baseline so the repo is left in a clean state.
    ingest(base_settings, rebuild=True)

    out = EXPERIMENTS_DIR / "experiments_summary.json"
    out.write_text(json.dumps(results, indent=2))
    _write_markdown(results)
    print(f"\nWrote {out}")


def _write_markdown(results: list[dict]) -> None:
    lines = ["# Experiment Results", "", "All experiments re-use the same golden set. "
             "Delta is vs the baseline run.", ""]
    header = "| Experiment | " + " | ".join(CORE_METRICS) + " | avg latency (ms) | note |"
    lines += [header, "|" + "---|" * (len(CORE_METRICS) + 3)]
    for r in results:
        cells = []
        for m in CORE_METRICS:
            v = r["metrics"].get(m)
            d = r["delta_vs_baseline"].get(m)
            if v is None:
                cells.append("-")
            elif r["name"] == "baseline" or d is None:
                cells.append(f"{v:.3f}")
            else:
                sign = "+" if d >= 0 else ""
                cells.append(f"{v:.3f} ({sign}{d:.3f})")
        lat = r["operational"]["avg_latency_ms"]
        lines.append(f"| {r['name']} | " + " | ".join(cells) + f" | {lat} | {r['note']} |")
    lines += ["", "## Decision", "",
              "_Fill in: which change you keep and why (the one with the best "
              "faithfulness / recall trade-off without a latency or cost regression)._"]
    (EXPERIMENTS_DIR / "experiments_summary.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
