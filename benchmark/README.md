# Benchmark runner

This folder contains a small convenience wrapper to run the project-level
benchmark harness located in `src/evals/benchmark.py`.

Quick start

1. Ensure your environment has the required API keys (see `.env`), or set up a
   local LLM/judge implementation if you want to run the deep benchmark.
2. Run a quick shallow benchmark (no live search/summaries):

```bash
python benchmark/run_benchmark.py --limit 10
```

3. Run a deeper, end-to-end benchmark (may be slow and require external APIs):

```bash
python benchmark/run_benchmark.py --deep --limit 5
```

CI / Offline testing

The repository includes `tests/test_benchmark_system.py` which runs an
offline/system-level smoke test of the benchmark harness by mocking external
dependencies (LLMs, search) so it can run in CI without network access.
