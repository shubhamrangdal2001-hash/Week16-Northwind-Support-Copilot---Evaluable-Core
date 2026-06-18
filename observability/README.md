# Observability (Langfuse)

The pipeline is instrumented with Langfuse via the `@observe()` decorator in
`app/rag.py` (steps: `retrieve`, `generate`, `rag_query`) and the
`langfuse.openai` drop-in client in `app/providers.py`, which auto-captures
tokens, latency, and cost for every LLM call.

## Setup

1. Create a free Langfuse cloud project (or self-host) and grab the keys.
2. Add to `.env`:
   ```
   LANGFUSE_ENABLED=true
   LANGFUSE_PUBLIC_KEY=pk-lf-...
   LANGFUSE_SECRET_KEY=sk-lf-...
   LANGFUSE_HOST=https://cloud.langfuse.com
   ```
3. `pip install langfuse` (already in requirements.txt).
4. Run any query: `python -m app.cli ask "How long does standard shipping take?"`
5. Open the Langfuse dashboard - you will see a `rag_query` trace with nested
   `retrieve` and `generate` spans.

## Required screenshots (put them here)

- `screenshots/trace.png` - one full trace (retrieval + generation spans, tokens, latency).
- `screenshots/dashboard.png` - the dashboard view (traces over time, latency, cost).

These two screenshots are referenced from the top-level `README.md`.

## Notes

- When `LANGFUSE_ENABLED=false` or the package is missing, `app/observability.py`
  swaps in a no-op `observe` decorator so the pipeline still runs (e.g. offline CI).
- `score_current_trace()` is available to push eval scores back onto traces if you
  want Ragas numbers visible per-trace in Langfuse.
 