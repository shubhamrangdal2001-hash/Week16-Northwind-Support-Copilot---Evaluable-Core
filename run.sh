#!/usr/bin/env bash
# One command to reproduce the baseline scorecard end to end.
# Usage:
#   ./run.sh           # real run (needs .env with OPENAI_API_KEY)
#   ./run.sh smoke     # offline plumbing check (no keys/network)
set -euo pipefail
cd "$(dirname "$0")"

if [[ "${1:-}" == "smoke" ]]; then
  export LLM_PROVIDER=fake
  export LANGFUSE_ENABLED=false
  echo ">> OFFLINE SMOKE MODE (fake provider, no network)"
fi

python -m app.cli build-corpus
python -m app.cli ingest
python eval/run_eval.py
python experiments/run_experiments.py

echo
echo ">> Done. See eval/results/ and experiments/results/."
