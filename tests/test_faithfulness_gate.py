"""Tier A: DeepEval pytest gate.

Fails the build when faithfulness on a sampled set of golden questions drops
below FLOOR. Pytest-native, so CI can run it directly.

Supports LLM_PROVIDER=groq (GROQ_API_KEY) or LLM_PROVIDER=openai (OPENAI_API_KEY).
Skips automatically when running with the offline fake provider.

    pytest tests/test_faithfulness_gate.py -v
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

FLOOR = float(os.getenv("FAITHFULNESS_FLOOR", "0.90"))
SAMPLE_SIZE = int(os.getenv("GATE_SAMPLE_SIZE", "8"))


def _load_answerable_sample():
    golden = ROOT / "eval" / "golden_set.jsonl"
    rows = [json.loads(l) for l in golden.read_text().splitlines() if l.strip()]
    answerable = [r for r in rows if r.get("source_docs")]
    return answerable[:SAMPLE_SIZE]


def _configure_deepeval_for_provider():
    """Point DeepEval at Groq's OpenAI-compatible endpoint when using groq provider."""
    provider = os.getenv("LLM_PROVIDER", "openai")
    if provider == "groq":
        groq_key = os.getenv("GROQ_API_KEY", "")
        chat_model = os.getenv("CHAT_MODEL", "llama-3.1-8b-instant")
        # DeepEval respects these env vars via its OpenAI client
        os.environ.setdefault("OPENAI_API_KEY", groq_key)
        os.environ.setdefault("OPENAI_BASE_URL", "https://api.groq.com/openai/v1")
        return chat_model
    return os.getenv("CHAT_MODEL", "gpt-4o-mini")


@pytest.mark.skipif(
    os.getenv("LLM_PROVIDER", "openai") == "fake",
    reason="DeepEval needs a real judge LLM; skipped in offline mode.",
)
def test_faithfulness_above_floor():
    deepeval = pytest.importorskip("deepeval")
    from deepeval.metrics import FaithfulnessMetric
    from deepeval.test_case import LLMTestCase

    from app.config import get_settings
    from app.rag import RagPipeline

    judge_model = _configure_deepeval_for_provider()

    pipe = RagPipeline(get_settings())
    metric = FaithfulnessMetric(threshold=FLOOR, model=judge_model)

    failures = []
    for row in _load_answerable_sample():
        res = pipe.query(row["question"])
        tc = LLMTestCase(
            input=row["question"],
            actual_output=res.answer,
            retrieval_context=res.context_texts(),
        )
        metric.measure(tc)
        if metric.score < FLOOR:
            failures.append((row["id"], round(metric.score, 3), metric.reason))

    assert not failures, (
        f"Faithfulness below floor {FLOOR} for: "
        + "; ".join(f"{i} ({s})" for i, s, _ in failures)
    )
