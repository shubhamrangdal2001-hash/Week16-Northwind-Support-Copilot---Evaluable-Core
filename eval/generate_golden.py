"""Draft a synthetic golden set with the Ragas test-set generator.

The brief says: draft with the Ragas synthetic generator, then HAND-VERIFY
every row (raw synthetic data fails). This script produces a *draft* file
(`golden_set.generated.jsonl`) that you must review and merge into the
hand-verified `golden_set.jsonl`. It requires OpenAI keys + `pip install ragas`.

Usage:
    python eval/generate_golden.py --n 30
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.config import CORPUS_DIR  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30, help="number of Q&A pairs to draft")
    ap.add_argument("--out", default=str(ROOT / "eval" / "golden_set.generated.jsonl"))
    args = ap.parse_args()

    try:
        from langchain_community.document_loaders import DirectoryLoader
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings
        from ragas.testset import TestsetGenerator
    except Exception as exc:  # pragma: no cover
        print(
            "This script needs ragas + langchain + OpenAI keys.\n"
            "Install: pip install ragas langchain-openai langchain-community\n"
            f"Import error: {exc}"
        )
        sys.exit(1)

    docs = DirectoryLoader(str(CORPUS_DIR), glob="*.md").load()
    generator = TestsetGenerator.from_langchain(
        llm=ChatOpenAI(model="gpt-4o-mini"),
        embedding_model=OpenAIEmbeddings(model="text-embedding-3-small"),
    )
    testset = generator.generate_with_langchain_docs(docs, testset_size=args.n)
    df = testset.to_pandas()

    out = Path(args.out)
    with out.open("w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            f.write(
                json.dumps(
                    {
                        "id": f"gen-{_:03d}",
                        "flavor": "unverified",
                        "question": row.get("user_input") or row.get("question"),
                        "ground_truth": row.get("reference") or row.get("ground_truth"),
                        "source_docs": [],
                    }
                )
                + "\n"
            )
    print(f"Wrote {len(df)} DRAFT rows to {out}. Hand-verify before using!")


if __name__ == "__main__":
    main()
 