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
pytest src/evals/ -v -m eval
```

## 🏁 Benchmark (100 topics)

For a product-level quality report across many topics, run the benchmark over the
100 bilingual research questions in `benchmark_questions.py`:

```bash
python -m evals.benchmark --limit 10            # quick shallow run (query metrics)
python -m evals.benchmark --lang tr --limit 20  # Turkish subset
python -m evals.benchmark --deep --limit 5      # adds search + summary metrics
python -m evals.benchmark --safety --limit 5    # adds safety metrics (implies --deep)
python -m evals.benchmark                        # all topics (use a paid judge)
```

With `--safety`, generated summaries are also scored on six DeepEval safety
metrics — `BiasMetric`, `ToxicityMetric`, `PIILeakageMetric`, `MisuseMetric`,
`NonAdviceMetric`, `RoleViolationMetric`. Their score directions differ, so
pass/fail uses each metric's own `.success` rather than a fixed threshold.

> Each topic in `--deep`/`--safety` mode makes ~9–11 judge calls; the Groq
> free tier rate-limits (HTTP 429) on large runs. Use `--limit`, a higher-tier
> key, or a different judge (`LLM_PROVIDER`/`LLM_MODEL`) for the full set.

**Agent-trace metrics** (`TaskCompletion`, `StepEfficiency`, `PlanQuality`,
`PlanAdherence`, `ToolCorrectness`, `ArgumentCorrectness`) and **MCP** evaluation
need DeepEval tracing (`@observe`) instrumentation of the agents; results can be
pushed to [Confident AI](https://app.confident-ai.com/) via `CONFIDENT_API_KEY`.
This is planned as a follow-up.

It writes a timestamped JSON + Markdown report to `benchmark_results/` and prints
a summary table with per-metric **pass rates**, **means**, **latency** percentiles
and an overall product-readiness **verdict**.

| Metric | Bar | Mode | Meaning |
|--------|-----|------|---------|
| `query_relevance` | ≥ 0.60 | always | GEval: queries are on-topic & specific |
| `bilingual_coverage` | = true | always | produced both EN and TR queries |
| `query_count` | ≥ 10 | always | number of expanded queries |
| `retrieval_count` | ≥ 10 | `--deep` | unique papers found across databases |
| `dedup_integrity` | = true | `--deep` | zero duplicate titles in results |
| `summary_faithfulness` | ≥ 0.70 | `--deep` | summary grounded in the abstract |
| `summary_relevancy` | ≥ 0.60 | `--deep` | summary addresses the topic |

> **Cost:** shallow mode is ~2 LLM calls/topic (fast); `--deep` adds live search +
> a summary per topic (minutes/topic). Use `--limit` for quick checks and run the
> full 100 periodically as a quality gate.

If no provider key is configured, every eval test is **skipped** (not failed),
so they never break a normal `pytest tests/` run or CI.

## Notes

- **Cost / speed:** each metric makes one or more LLM calls, so the suite is
  slower than the unit tests. Keep `datasets.py` small.
- **Telemetry:** DeepEval collects anonymous usage telemetry by default. Opt out
  with `export DEEPEVAL_TELEMETRY_OPT_OUT=YES`.
- **Thresholds:** pass/fail thresholds are set per metric in the test files
  (e.g. faithfulness ≥ 0.7). Tune them as the agents improve.
