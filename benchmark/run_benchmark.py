#!/usr/bin/env python3
"""
Convenience runner for the DeepArticle benchmark.

This wrapper calls the packaged benchmark harness `evals.benchmark` with
user-friendly defaults and writes reports into `benchmark_results/`.

Usage:
    python benchmark/run_benchmark.py --limit 10         # quick shallow run
    python benchmark/run_benchmark.py --deep --limit 5   # full run (slower)
"""
import argparse
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description="Run DeepArticle benchmark (wrapper)")
    parser.add_argument("--limit", type=int, default=0, help="cap number of topics (0 = all)")
    parser.add_argument("--lang", default="all", choices=["all", "en", "tr"], help="language subset")
    parser.add_argument("--deep", action="store_true", help="also run search + summary metrics (slower)")
    args = parser.parse_args()

    cmd = [sys.executable, "-m", "evals.benchmark", "--lang", args.lang]
    if args.limit:
        cmd += ["--limit", str(args.limit)]
    if args.deep:
        cmd.append("--deep")

    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
