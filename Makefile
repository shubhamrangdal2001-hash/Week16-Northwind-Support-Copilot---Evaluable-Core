.PHONY: install corpus ingest eval experiments gate ask agent citations errors judge analysis smoke clean

install:
	pip install -r requirements.txt

corpus:
	python -m app.cli build-corpus

ingest:
	python -m app.cli ingest

eval:
	python eval/run_eval.py

experiments:
	python experiments/run_experiments.py

# Tier B: citation accuracy + 5-worst-case error analysis (run eval first).
citations:
	python eval/citation_check.py

errors:
	python eval/error_analysis.py

# Bonus: LLM-as-judge prompt comparison.
judge:
	python experiments/llm_judge.py

# Tier B bundle: eval -> citation check -> error analysis.
analysis: eval citations errors

gate:
	pytest tests/test_faithfulness_gate.py -v

ask:
	python -m app.cli ask "$(Q)"

# Agent router: bounded order-status tool, else RAG.
agent:
	python -m app.cli agent "$(Q)"

# One command: build corpus, index, and produce the baseline scorecard.
reproduce: corpus ingest eval

# Offline plumbing check (no API keys / network needed).
smoke:
	LLM_PROVIDER=fake LANGFUSE_ENABLED=false python -m app.cli build-corpus
	LLM_PROVIDER=fake LANGFUSE_ENABLED=false python -m app.cli ingest
	LLM_PROVIDER=fake LANGFUSE_ENABLED=false python eval/run_eval.py
	LLM_PROVIDER=fake LANGFUSE_ENABLED=false python experiments/run_experiments.py
	LLM_PROVIDER=fake LANGFUSE_ENABLED=false python eval/citation_check.py
	LLM_PROVIDER=fake LANGFUSE_ENABLED=false python eval/error_analysis.py

clean:
	rm -rf app/data/chroma app/data/store eval/results experiments/results
