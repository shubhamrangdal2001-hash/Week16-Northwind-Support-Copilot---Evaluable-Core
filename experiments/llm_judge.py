"""Bonus: LLM-as-judge comparison of two prompt variants.

Runs the golden set under prompt grounded_v1 and grounded_v2, then asks a judge
LLM to pick the better answer per question (grounded in the retrieved context).
Supports LLM_PROVIDER=groq (GROQ_API_KEY) or LLM_PROVIDER=openai (OPENAI_API_KEY).

In offline (fake) mode it falls back to a transparent lexical judge so the
pipeline runs, clearly labelling that the verdicts are heuristic.

Usage (groq):    python experiments/llm_judge.py
Usage (offline): LLM_PROVIDER=fake LANGFUSE_ENABLED=false python experiments/llm_judge.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.config import EXPERIMENTS_DIR, get_settings  # noqa: E402
from app.providers import tokenize  # noqa: E402
from app.rag import RagPipeline  # noqa: E402
from eval.run_eval import load_golden  # noqa: E402

JUDGE_SYSTEM = (
    "You are an impartial judge. Given a question, the retrieved context, and two "
    "candidate answers (A and B), choose which answer is better grounded in the "
    "context and more helpful. Reply with exactly one token: A, B, or TIE."
)


def _heuristic_judge(question: str, ctx: str, a: str, b: str) -> str:
    ctx_tokens = set(tokenize(ctx))
    def grounded(ans: str) -> float:
        at = set(tokenize(ans))
        return len(at & ctx_tokens) / (len(at) or 1)
    ga, gb = grounded(a), grounded(b)
    if abs(ga - gb) < 0.02:
        return "TIE"
    return "A" if ga > gb else "B"


def _judge(settings, question, ctx, a, b) -> str:
    if settings.is_offline:
        return _heuristic_judge(question, ctx, a, b)

    user = (f"Question: {question}\n\nContext:\n{ctx}\n\n"
            f"Answer A:\n{a}\n\nAnswer B:\n{b}\n\nWhich is better? A, B, or TIE.")

    if settings.provider == "groq":
        from groq import Groq
        client = Groq()  # reads GROQ_API_KEY from env
        resp = client.chat.completions.create(
            model=settings.chat_model, temperature=0,
            messages=[{"role": "system", "content": JUDGE_SYSTEM},
                      {"role": "user", "content": user}],
        )
    else:
        from openai import OpenAI
        client = OpenAI()
        resp = client.chat.completions.create(
            model=settings.chat_model, temperature=0,
            messages=[{"role": "system", "content": JUDGE_SYSTEM},
                      {"role": "user", "content": user}],
        )

    out = (resp.choices[0].message.content or "TIE").strip().upper()
    return out if out in {"A", "B", "TIE"} else "TIE"


def run() -> dict:
    base = get_settings()
    pipe_a = RagPipeline(base.with_overrides(prompt_variant="grounded_v1"))
    pipe_b = RagPipeline(base.with_overrides(prompt_variant="grounded_v2"))
    rows = [r for r in load_golden() if r.get("source_docs")]

    tally = {"A": 0, "B": 0, "TIE": 0}
    records = []
    for row in rows:
        ra = pipe_a.query(row["question"])
        rb = pipe_b.query(row["question"])
        ctx = "\n".join(ra.context_texts())
        verdict = _judge(base, row["question"], ctx, ra.answer, rb.answer)
        tally[verdict] += 1
        records.append({"id": row["id"], "verdict": verdict})

    result = {
        "engine": "heuristic-offline" if base.is_offline else "llm-judge",
        "A_grounded_v1": tally["A"], "B_grounded_v2": tally["B"], "TIE": tally["TIE"],
        "winner": max(tally, key=tally.get),
        "records": records,
    }
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    (EXPERIMENTS_DIR / "llm_judge.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != "records"}, indent=2))
    return result


if __name__ == "__main__":
    run()
