# Experiment Results

All experiments re-use the same golden set. Delta is vs the baseline run.

| Experiment | faithfulness | answer_relevancy | context_precision | context_recall | avg latency (ms) | note |
|---|---|---|---|---|---|---|
| baseline | 0.669 | 0.227 | 0.250 | 0.720 | 0.1 | top_k=4, chunk_size=700, prompt=grounded_v1 |
| exp1_topk2 | 0.644 (-0.025) | 0.221 (-0.006) | 0.380 (+0.130) | 0.600 (-0.120) | 0.1 | Fewer retrieved chunks (top_k 4 -> 2) |
| exp2_smallchunks | 0.669 (+0.000) | 0.227 (+0.000) | 0.267 (+0.017) | 0.760 (+0.040) | 0.1 | Smaller chunks (700 -> 350 chars) |
| exp3_prompt_v2 | 0.669 (+0.000) | 0.227 (+0.000) | 0.267 (+0.017) | 0.760 (+0.040) | 0.1 | Stricter, citation-forcing prompt (grounded_v2) |
| exp4_hybrid | 0.686 (+0.017) | 0.229 (+0.002) | 0.287 (+0.037) | 0.800 (+0.080) | 0.1 | BONUS: hybrid retrieval (vector + BM25 fused via RRF) |

## Decision

_Fill in: which change you keep and why (the one with the best faithfulness / recall trade-off without a latency or cost regression)._
 