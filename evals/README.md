# 🧪 Agent Evaluation (DeepEval)

This suite uses [DeepEval](https://docs.confident-ai.com/) to evaluate the
quality of DeepArticle's LLM-powered agents with an **LLM-as-judge** approach.

Unlike the unit tests in `tests/` (which are offline and deterministic), these
evaluations make **real LLM calls** and need an API key. They reuse the
project's own multi-provider LLM factory as the judge, so a single
`GROQ_API_KEY` (or OpenAI / Anthropic / Google key) is enough.

## What gets evaluated

| Agent          | Metric(s)                                   | What it checks |
|----------------|---------------------------------------------|----------------|
| Query Analyzer | `GEval` (relevance & specificity)           | Are generated search queries on-topic and specific, not vague? |
| Summarizer     | `FaithfulnessMetric`, `AnswerRelevancyMetric` | Do summaries stay faithful to the abstract (no hallucinations) and actually summarize the paper? |

The judge model is `evals/eval_model.py::DeepArticleEvalModel`, which wraps the
configured provider. Golden inputs live in `evals/datasets.py` — extend them to
broaden coverage.

## Running

```bash
# Install the eval extra
pip install -e ".[eval]"        # or: pip install deepeval

# Make sure an API key is set (see .env.example)
#   GROQ_API_KEY=...

# Run the evaluation suite
pytest evals/ -v -m eval
```

If no provider key is configured, every eval test is **skipped** (not failed),
so they never break a normal `pytest tests/` run or CI.

## Notes

- **Cost / speed:** each metric makes one or more LLM calls, so the suite is
  slower than the unit tests. Keep `datasets.py` small.
- **Telemetry:** DeepEval collects anonymous usage telemetry by default. Opt out
  with `export DEEPEVAL_TELEMETRY_OPT_OUT=YES`.
- **Thresholds:** pass/fail thresholds are set per metric in the test files
  (e.g. faithfulness ≥ 0.7). Tune them as the agents improve.
