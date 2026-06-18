# Before Production (Tier A)

A short, honest read on what it would take to run this RAG slice in production.

## Monitoring & alerting

- **Online quality**: keep the Langfuse `@observe` traces in prod; sample live
  traffic into a daily Ragas job (faithfulness + context recall) and alert if the
  rolling mean drops below the Week 15 floor (faithfulness < 0.90).
- **Operational SLOs**: alert on p95 latency and cost-per-query regressions
  (both are already emitted in the scorecard's `operational` block).
- **Abstention rate**: track the share of "I don't know" answers; a sudden spike
  usually means retrieval broke (index empty / embedding drift).
- **The CI gate** (`tests/test_faithfulness_gate.py`) blocks regressions before
  deploy; pair it with the online monitor for after-deploy drift.

## Freshness & re-indexing

- Treat the corpus as versioned. On any source-doc change, re-run `ingest`
  (idempotent: `rebuild=True` resets the collection) behind a content hash so we
  only re-embed changed chunks.
- Schedule a nightly delta re-index; keep an `indexed_at` per chunk so stale
  content can be detected and expired.
- Pin the embedding model + version; a model change requires a full re-embed and
  a fresh baseline scorecard (embeddings are not comparable across models).

## Guardrails

- **Grounding/abstention**: the prompt forces "answer only from context, else say
  I don't know" - the primary hallucination guardrail, measured by faithfulness
  and abstention accuracy.
- **Bounded tool**: the order-status action validates its input (`NW-####`),
  is read-only, and returns no PII beyond status - no free-form tool calls.
- **Input/output filtering**: add prompt-injection and PII checks on the user
  query and on retrieved chunks before they hit the LLM.
- **Citations**: every answer must cite a retrieved source id; the citation
  check (`eval/citation_check.py`) verifies cited ids are real and retrieved.

## Known gaps

- Golden set is small (32 rows) and synthetic-seeded; grow and diversify it.
- Single judge model for Ragas/LLM-as-judge introduces judge bias.
- No multi-tenant access control on the corpus yet.
