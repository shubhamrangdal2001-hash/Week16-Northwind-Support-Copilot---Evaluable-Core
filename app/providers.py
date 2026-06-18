"""LLM + embedding providers.

Three backends share one interface so the rest of the code never branches on the
provider:

* ``openai`` - real models (text-embedding-3-small + a chat model). Requires
  OPENAI_API_KEY and network access.
* ``groq``   - Groq-hosted LLMs (fast & free-tier friendly). Requires GROQ_API_KEY.
  Uses sentence-transformers for local embeddings (no embedding API key needed).
* ``fake``   - deterministic, offline, zero-dependency backend used to smoke-test
  the full pipeline (ingest -> retrieve -> answer -> eval) in CI or without keys.
  It uses a hashing embedder and an extractive "LLM". Numbers from the fake
  backend are for plumbing verification only and must NOT be reported as results.
"""
from __future__ import annotations

import hashlib
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .config import PRICING_PER_1M, Settings

_WORD = re.compile(r"[a-z0-9]+")
_SENT = re.compile(r"(?<=[.!?])\s+")
FAKE_DIM = 384
ABSTAIN = "I don't know based on the available documentation."


def tokenize(text: str) -> list[str]:
    return _WORD.findall(text.lower())


@dataclass
class Answer:
    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0
    cost_usd: float = 0.0
    raw: Any = field(default=None, repr=False)


# --------------------------------------------------------------------------- #
# Embedders
# --------------------------------------------------------------------------- #
class FakeEmbedder:
    """Deterministic hashing bag-of-words embedder (offline)."""

    dim = FAKE_DIM

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            for tok in tokenize(text):
                h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
                out[i, h % self.dim] += 1.0
            norm = np.linalg.norm(out[i])
            if norm > 0:
                out[i] /= norm
        return out


class OpenAIEmbedder:
    def __init__(self, model: str):
        # Import OpenAI only when needed.
        # In fake/offline mode this block is never executed.
        try:
            from openai import OpenAI
        except ModuleNotFoundError:
            raise ImportError(
                "The 'openai' package is required for LLM_PROVIDER='openai'. "
                "Install it with `pip install openai` or switch to LLM_PROVIDER='fake'."
            )
        self.model = model
        self.client = OpenAI()

    def embed(self, texts: list[str]) -> np.ndarray:
        resp = self.client.embeddings.create(model=self.model, input=texts)
        return np.array([d.embedding for d in resp.data], dtype=np.float32)


class SentenceTransformerEmbedder:
    """Local embedding model via sentence-transformers (no API key needed).

    Used by the ``groq`` provider since Groq does not offer embedding endpoints.
    Default model: all-MiniLM-L6-v2 (384-dim, ~90 MB, downloads once on first use).
    Override with EMBED_MODEL env var (must be a sentence-transformers model name).
    """

    DEFAULT_MODEL = "all-MiniLM-L6-v2"

    def __init__(self, model: str | None = None):
        try:
            from sentence_transformers import SentenceTransformer
        except ModuleNotFoundError:
            raise ImportError(
                "The 'sentence-transformers' package is required for LLM_PROVIDER='groq'. "
                "Install it with `pip install sentence-transformers`."
            )
        self.model_name = model or self.DEFAULT_MODEL
        self._model = SentenceTransformer(self.model_name)
        self.dim = (self._model.get_embedding_dimension()
                    if hasattr(self._model, "get_embedding_dimension")
                    else self._model.get_sentence_embedding_dimension())

    def embed(self, texts: list[str]) -> np.ndarray:
        vecs = self._model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return np.array(vecs, dtype=np.float32)


# --------------------------------------------------------------------------- #
# Chat LLMs
# --------------------------------------------------------------------------- #
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
}


def build_prompt(question: str, contexts: list[dict], variant: str) -> list[dict]:
    system = GROUNDED_SYSTEM.get(variant, GROUNDED_SYSTEM["grounded_v1"])
    ctx_block = "\n\n".join(f"[{c['doc_id']}] {c['text']}" for c in contexts)
    user = f"Question: {question}\n\nContext:\n{ctx_block}"
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


class FakeLLM:
    """Extractive offline answerer: returns the best-matching context sentence."""

    def __init__(self, variant: str = "grounded_v1"):
        self.variant = variant

    def answer(self, question: str, contexts: list[dict]) -> Answer:
        start = time.perf_counter()
        q_tokens = set(tokenize(question))
        best_sent, best_doc, best_score = "", None, 0.0
        for c in contexts:
            for sent in _SENT.split(c["text"].replace("\n", " ")):
                s_tokens = set(tokenize(sent))
                if not s_tokens:
                    continue
                overlap = len(q_tokens & s_tokens) / len(q_tokens | s_tokens)
                if overlap > best_score:
                    best_score, best_sent, best_doc = overlap, sent.strip(), c["doc_id"]
        if best_score < 0.16 or not best_sent:
            text = ABSTAIN
        else:
            text = f"{best_sent} [{best_doc}]"
        latency = (time.perf_counter() - start) * 1000
        return Answer(
            text=text,
            prompt_tokens=sum(len(tokenize(c["text"])) for c in contexts),
            completion_tokens=len(tokenize(text)),
            latency_ms=latency,
        )


class OpenAILLM:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.model = settings.chat_model
        self._observed = False
        # Prefer the Langfuse-wrapped OpenAI client so calls are auto-traced.
        if settings.langfuse_enabled:
            try:
                from langfuse.openai import OpenAI  # type: ignore

                self.client = OpenAI()
                self._observed = True
                return
            except Exception:
                pass
        from openai import OpenAI

        self.client = OpenAI()

    def answer(self, question: str, contexts: list[dict]) -> Answer:
        messages = build_prompt(question, contexts, self.settings.prompt_variant)
        start = time.perf_counter()
        resp = self.client.chat.completions.create(
            model=self.model, messages=messages, temperature=0
        )
        latency = (time.perf_counter() - start) * 1000
        usage = getattr(resp, "usage", None)
        pt = getattr(usage, "prompt_tokens", 0) or 0
        ct = getattr(usage, "completion_tokens", 0) or 0
        price = PRICING_PER_1M.get(self.model, {"input": 0.0, "output": 0.0})
        cost = (pt * price["input"] + ct * price["output"]) / 1_000_000
        return Answer(
            text=resp.choices[0].message.content or "",
            prompt_tokens=pt,
            completion_tokens=ct,
            latency_ms=latency,
            cost_usd=cost,
            raw=resp,
        )


class GroqLLM:
    """Groq-hosted LLM using the official groq Python SDK.

    Groq's API is OpenAI-compatible but uses GROQ_API_KEY. The SDK automatically
    reads GROQ_API_KEY from the environment.
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self.model = settings.chat_model
        try:
            from groq import Groq
        except ModuleNotFoundError:
            raise ImportError(
                "The 'groq' package is required for LLM_PROVIDER='groq'. "
                "Install it with `pip install groq`."
            )
        self.client = Groq()  # reads GROQ_API_KEY from env automatically

    def answer(self, question: str, contexts: list[dict]) -> Answer:
        messages = build_prompt(question, contexts, self.settings.prompt_variant)
        start = time.perf_counter()
        
        from groq import RateLimitError
        import time as pytime

        max_attempts = 6
        for attempt in range(max_attempts):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model, messages=messages, temperature=0
                )
                break
            except RateLimitError as e:
                if attempt == max_attempts - 1:
                    raise
                # Extract wait time from error if possible, or back off exponentially
                wait_time = (2 ** attempt) + 3
                print(f"[groq] Rate limit hit. Retrying in {wait_time}s...")
                pytime.sleep(wait_time)
        latency = (time.perf_counter() - start) * 1000
        usage = getattr(resp, "usage", None)
        pt = getattr(usage, "prompt_tokens", 0) or 0
        ct = getattr(usage, "completion_tokens", 0) or 0
        price = PRICING_PER_1M.get(self.model, {"input": 0.0, "output": 0.0})
        cost = (pt * price["input"] + ct * price["output"]) / 1_000_000
        return Answer(
            text=resp.choices[0].message.content or "",
            prompt_tokens=pt,
            completion_tokens=ct,
            latency_ms=latency,
            cost_usd=cost,
            raw=resp,
        )


def get_embedder(settings: Settings):
    if settings.is_offline:
        return FakeEmbedder()
    if settings.provider == "groq":
        # Groq has no embedding API; use a local sentence-transformers model.
        # EMBED_MODEL can override to any sentence-transformers model name.
        st_model = settings.embed_model if settings.embed_model != "text-embedding-3-small" else None
        return SentenceTransformerEmbedder(st_model)
    return OpenAIEmbedder(settings.embed_model)


def get_llm(settings: Settings):
    if settings.is_offline:
        return FakeLLM(settings.prompt_variant)
    if settings.provider == "groq":
        return GroqLLM(settings)
    return OpenAILLM(settings)
