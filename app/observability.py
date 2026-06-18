"""Langfuse helpers.

Provides an ``observe`` decorator that is a no-op when Langfuse is not installed
or disabled, so the pipeline imports and runs everywhere. When Langfuse IS
configured (env keys + ``pip install langfuse``), the real decorator traces each
step (retrieval, generation) with tokens, latency, and cost.
"""
from __future__ import annotations

import functools
import os

_LANGFUSE_ON = os.getenv("LANGFUSE_ENABLED", "true").lower() in {"1", "true", "yes", "on"}

try:
    if _LANGFUSE_ON:
        from langfuse.decorators import observe as _observe  # type: ignore
        from langfuse.decorators import langfuse_context  # type: ignore

        _AVAILABLE = True
    else:  # pragma: no cover
        raise ImportError
except Exception:  # pragma: no cover - offline / not installed
    _AVAILABLE = False
    langfuse_context = None  # type: ignore

    def _observe(*d_args, **d_kwargs):
        # Support both @observe and @observe(name=...) usage.
        if len(d_args) == 1 and callable(d_args[0]) and not d_kwargs:
            return d_args[0]

        def deco(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                return func(*args, **kwargs)

            return wrapper

        return deco


observe = _observe
LANGFUSE_AVAILABLE = _AVAILABLE


def score_current_trace(name: str, value: float, comment: str = "") -> None:
    """Attach a numeric score to the active Langfuse trace (no-op if disabled)."""
    if not _AVAILABLE or langfuse_context is None:
        return
    try:
        langfuse_context.score_current_trace(name=name, value=value, comment=comment)
    except Exception:
        pass


def flush() -> None:
    if not _AVAILABLE or langfuse_context is None:
        return
    try:
        langfuse_context.flush()
    except Exception:
        pass
 