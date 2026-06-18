"""Central configuration for the Evaluable Core RAG slice.

Everything is overridable via environment variables (see .env.example) so that
the experiment runner can sweep parameters without editing code.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, replace
from pathlib import Path

try:  # optional, only present once `pip install -r requirements.txt` has run
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # pragma: no cover - dotenv is optional
    pass

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "app" / "data"
CORPUS_DIR = DATA_DIR / "corpus"
CHROMA_DIR = DATA_DIR / "chroma"
STORE_DIR = DATA_DIR / "store"  # used by the offline SimpleVectorStore fallback
EVAL_DIR = ROOT / "eval"
RESULTS_DIR = EVAL_DIR / "results"
EXPERIMENTS_DIR = ROOT / "experiments" / "results"

for _d in (DATA_DIR, CORPUS_DIR, CHROMA_DIR, STORE_DIR, RESULTS_DIR, EXPERIMENTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)


def _get_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


def _get_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    # "openai" => real models (needs OPENAI_API_KEY).
    # "groq"   => Groq-hosted LLMs (needs GROQ_API_KEY) + local sentence-transformers embeddings.
    # "fake"   => deterministic offline provider for smoke tests / CI without network.
    provider: str = os.getenv("LLM_PROVIDER", "openai")
    chat_model: str = os.getenv("CHAT_MODEL", "gpt-4o-mini")
    embed_model: str = os.getenv("EMBED_MODEL", "text-embedding-3-small")
    top_k: int = _get_int("TOP_K", 16)
    chunk_size: int = _get_int("CHUNK_SIZE", 700)        # characters
    chunk_overlap: int = _get_int("CHUNK_OVERLAP", 120)  # characters
    collection: str = os.getenv("CHROMA_COLLECTION", "northwind_support")
    prompt_variant: str = os.getenv("PROMPT_VARIANT", "cot_grounded")
    # "vector" (dense only) or "hybrid" (dense + keyword fused via RRF reranker).
    retrieval_mode: str = os.getenv("RETRIEVAL_MODE", "hybrid")
    # How many candidates each retriever pulls before fusion/rerank.
    candidate_k: int = _get_int("CANDIDATE_K", 40)
    langfuse_enabled: bool = _get_bool("LANGFUSE_ENABLED", True)

    def with_overrides(self, **kwargs) -> "Settings":
        clean = {k: v for k, v in kwargs.items() if v is not None}
        return replace(self, **clean)

    @property
    def is_offline(self) -> bool:
        return self.provider == "fake"


def get_settings(**overrides) -> Settings:
    return Settings().with_overrides(**overrides)


ABSTAIN = "I am sorry, but I do not have enough information to answer your question."

GROUNDED_SYSTEM = {
    "grounded_v1": (
        "You are the Northwind Support Copilot. Answer the user's question using "
        "ONLY the provided context. Cite the supporting source id in square "
        "brackets, e.g. [returns-policy]. If the answer is not contained in the "
        f"context, reply exactly: {ABSTAIN}"
    ),
    "grounded_v2": (
        "You are the Northwind Support Copilot, a careful, concise support agent. "
        "Use ONLY the context passages below to answer. Every factual sentence must "
        "cite its source id like [shipping-domestic]. Do not use outside knowledge. "
        f"If the context does not contain the answer, reply exactly: {ABSTAIN} "
        "Keep the answer under 4 sentences."
    ),
    "cot_grounded": (
        "You are the Northwind Support Copilot. Provide a brief answer based solely on the given context, then explain your reasoning step‑by‑step to ensure faithfulness. Cite sources using brackets like [policy-id]. If the context lacks the answer, respond exactly with the abstention phrase."
    ),
}

# Pricing table (USD per 1M tokens) for cost-per-query estimation (Tier A).
# Update if you switch models. Used only for reporting; not billed.
PRICING_PER_1M = {
    # OpenAI
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "text-embedding-3-small": {"input": 0.02, "output": 0.0},
    # Groq (free tier / on-demand pricing as of 2025)
    "llama-3.1-8b-instant": {"input": 0.05, "output": 0.08},
    "llama3-8b-8192": {"input": 0.05, "output": 0.08},
    "llama-3.3-70b-versatile": {"input": 0.59, "output": 0.79},
    "mixtral-8x7b-32768": {"input": 0.24, "output": 0.24},
    "gemma2-9b-it": {"input": 0.20, "output": 0.20},
}
