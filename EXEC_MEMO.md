# Executive Memo - Northwind Support Copilot, Evaluable Core

_One page. Interview-ready. Fill the bracketed numbers from your real run
(`eval/results/baseline_scorecard.md` and `experiments/results/`)._

## What I built

A thin but complete RAG vertical slice for the Northwind Support Copilot:
ingest -> chunk -> embed (`text-embedding-3-small`) -> Chroma -> retrieve top-k ->
grounded, citation-forcing prompt -> answer. The slice is wrapped end to end in
Langfuse traces and scored on a 32-question golden set with Ragas. A DeepEval
pytest gate fails CI if faithfulness drops below 0.90.

## Baseline scorecard

| Metric | Score | Target | Verdict |
|---|---|---|---|
| Faithfulness | [ ] | 0.90 | [ ] |
| Answer relevancy | [ ] | 0.80 | [ ] |
| Context precision | [ ] | 0.70 | [ ] |
| Context recall | [ ] | 0.80 | [ ] |
| Abstention accuracy | [ ] | 0.90 | [ ] |

Operational: avg latency [ ] ms, p95 [ ] ms, cost/query $[ ].

## The experiment that moved the needle most

I ran three one-variable experiments on the same golden set: top_k (4->2),
chunk size (700->350), and a stricter prompt (`grounded_v2`). The biggest
movement was **[experiment]**, which changed **[metric]** by **[delta]**.
I [kept/rejected] it because **[reason - e.g. it lifted context recall without a
faithfulness or latency regression]**.

## Honest read: what is and isn't production-ready

**Ready:** the system abstains correctly on out-of-corpus questions
([abstention] accuracy), answers are grounded with citations, and every change
is measured against a fixed test set with traces for debugging.

**Not ready:** [e.g. small golden set; no freshness/re-index plan for changing
docs; single judge model for Ragas; latency at p95 is [ ] ms]. Before production
I would add monitoring/alerting on the faithfulness gate, a re-index schedule,
and input/output guardrails.

## The interview answer

"How do you know it works?" - I don't eyeball it. I have a golden set covering
easy, ambiguous, multi-hop, and adversarial cases; I measure faithfulness,
answer relevancy, and retrieval quality on every change; traces let me see
whether a failure is retrieval or generation; and a CI gate blocks regressions
below my faithfulness floor.
 