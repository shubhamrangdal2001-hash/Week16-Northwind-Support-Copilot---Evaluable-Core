"""Citation-accuracy check (Tier B).

For every answerable golden question, the answer should cite at least one source
id like [returns-policy]. This script verifies:
  1. coverage   - did the answer cite any source at all?
  2. validity   - is each cited id a real corpus doc?
  3. support    - does the cited chunk actually appear in the retrieved context
                  (i.e. the citation is grounded, not invented)?
  4. correctness- is the cited id one of the gold source_docs?

Writes eval/results/citation_report.{json,md}.

Usage (offline): LLM_PROVIDER=fake LANGFUSE_ENABLED=false python eval/citation_check.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.config import RESULTS_DIR, get_settings  # noqa: E402
from app.corpus import DOCS  # noqa: E402
from app.rag import RagPipeline  # noqa: E402
from eval.run_eval import load_golden  # noqa: E402

CITE_RE = re.compile(r"\[([a-z0-9][a-z0-9-]*)\]")
VALID_IDS = set(DOCS.keys())


def run() -> dict:
    settings = get_settings()
    pipe = RagPipeline(settings)
    rows = [r for r in load_golden() if r.get("source_docs")]

    records, covered, valid_all, supported_all, correct_all = [], 0, 0, 0, 0
    for row in rows:
        res = pipe.query(row["question"])
        cited = CITE_RE.findall(res.answer)
        retrieved = set(res.retrieved_doc_ids)
        gold = set(row["source_docs"])

        has_cite = bool(cited)
        valid = [c for c in cited if c in VALID_IDS]
        supported = [c for c in cited if c in retrieved]
        correct = [c for c in cited if c in gold]

        covered += int(has_cite)
        valid_all += int(bool(cited) and len(valid) == len(cited))
        supported_all += int(bool(cited) and len(supported) == len(cited))
        correct_all += int(bool(correct))
        records.append({
            "id": row["id"], "cited": cited, "gold": sorted(gold),
            "all_valid": bool(cited) and len(valid) == len(cited),
            "all_supported": bool(cited) and len(supported) == len(cited),
            "hits_gold": bool(correct),
        })

    n = len(rows) or 1
    report = {
        "n": len(rows),
        "citation_coverage": round(covered / n, 3),
        "citation_validity": round(valid_all / n, 3),
        "citation_support": round(supported_all / n, 3),
        "citation_correctness": round(correct_all / n, 3),
        "records": records,
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "citation_report.json").write_text(json.dumps(report, indent=2))
    md = [
        "# Citation Accuracy", "",
        f"- Answerable questions: {report['n']}",
        f"- Coverage (answer cites something): {report['citation_coverage']}",
        f"- Validity (cited id is a real doc): {report['citation_validity']}",
        f"- Support (cited chunk was retrieved): {report['citation_support']}",
        f"- Correctness (cited id is a gold doc): {report['citation_correctness']}",
    ]
    (RESULTS_DIR / "citation_report.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    return report


if __name__ == "__main__":
    run()
 