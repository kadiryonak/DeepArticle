"""
Product-level benchmark for the DeepArticle pipeline.

Runs the system over a set of research topics (see ``benchmark_questions.py``)
and scores it with DeepEval metrics plus operational metrics, then writes a
report (JSON + Markdown) and prints a summary table.

Metrics
-------
Shallow (always, 1 LLM call per topic — the query analyzer):
  * query_relevance     — GEval (LLM judge): are the generated search queries
                          relevant and specific to the topic?         [≥ 0.60]
  * bilingual_coverage  — did it produce BOTH English and Turkish queries? [bool]
  * query_count         — number of expanded search queries.           [≥ 10]

Deep (``--deep``, adds live search + a summary per topic):
  * retrieval_count     — unique papers found across databases.        [≥ 10]
  * dedup_integrity     — zero duplicate titles in the merged results. [bool]
  * summary_faithfulness— FaithfulnessMetric: summary grounded in the
                          paper's abstract (no hallucination).         [≥ 0.70]
  * summary_relevancy   — AnswerRelevancyMetric: summary addresses the
                          topic.                                       [≥ 0.60]

Safety (``--safety``, implies ``--deep``; scores the generated summary):
  * bias, toxicity, pii_leakage, misuse, non_advice, role_violation — DeepEval
    safety metrics. Score directions differ, so pass/fail uses each metric's
    own ``.success`` flag.

Usage
-----
    python -m evals.benchmark --limit 10              # quick shallow run
    python -m evals.benchmark --lang tr --limit 20    # Turkish subset
    python -m evals.benchmark --deep --limit 5        # + search + summary metrics
    python -m evals.benchmark --safety --limit 5      # + safety metrics
    python -m evals.benchmark                          # all topics (use a paid judge)
"""

import argparse
import json
import os
import statistics
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

import sys
sys.path.insert(0, '..')

from utils.llm_factory import create_llm
from evals.benchmark_questions import get_questions
from evals.eval_model import get_eval_model

# Quality metric thresholds — higher is better (pass when score >= threshold).
THRESHOLDS = {
    "query_relevance": 0.60,
    "query_count": 10,
    "retrieval_count": 10,
    "summary_faithfulness": 0.70,
    "summary_relevancy": 0.60,
}

# Safety metrics scored on the generated summaries. Pass/fail comes from each
# deepeval metric's own ``.success`` (their score directions differ), so we keep
# only the set of names here for grouping/reporting.
SAFETY_METRICS = {
    "bias", "toxicity", "pii_leakage", "misuse", "non_advice", "role_violation",
}

_AGENT_ROLE = "academic research assistant"

_REPORT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "benchmark_results",
)


def _norm_title(t: str) -> str:
    return "".join(c for c in (t or "").lower() if c.isalnum())


def _score_query_relevance(topic: str, queries: List[str], judge) -> Optional[float]:
    """GEval relevance/specificity score for the generated queries (0-1)."""
    from deepeval.test_case import LLMTestCase, LLMTestCaseParams
    from deepeval.metrics import GEval

    metric = GEval(
        name="Query Relevance & Specificity",
        criteria=(
            "Evaluate whether the generated search queries are (1) clearly "
            "relevant to the research topic given as input, and (2) specific "
            "rather than overly generic. Penalize vague one-word queries and "
            "queries unrelated to the topic."
        ),
        evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
        model=judge,
        threshold=THRESHOLDS["query_relevance"],
    )
    tc = LLMTestCase(input=topic, actual_output="\n".join(f"- {q}" for q in queries))
    metric.measure(tc)
    return float(metric.score)


def _score_summary(topic: str, summary: str, abstract: str, judge) -> Dict[str, float]:
    """Faithfulness + answer-relevancy scores for a generated summary."""
    from deepeval.test_case import LLMTestCase
    from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric

    faith = FaithfulnessMetric(threshold=THRESHOLDS["summary_faithfulness"], model=judge)
    relev = AnswerRelevancyMetric(threshold=THRESHOLDS["summary_relevancy"], model=judge)

    tc = LLMTestCase(
        input=topic,
        actual_output=summary,
        retrieval_context=[abstract] if abstract else [summary],
    )
    faith.measure(tc)
    relev.measure(tc)
    return {"summary_faithfulness": float(faith.score), "summary_relevancy": float(relev.score)}


def _score_safety(topic: str, summary: str, judge) -> Dict[str, Dict[str, Any]]:
    """
    Safety evaluation of a generated summary across six dimensions.

    These deepeval metrics use mixed score directions (e.g. Bias: lower is safer;
    PIILeakage: higher is safer), so we rely on each metric's own ``.success``
    flag for pass/fail rather than imposing a direction. Returns
    ``{name: {"score": float, "success": bool}}``. Run defensively per-metric.
    """
    from deepeval.test_case import LLMTestCase
    from deepeval.metrics import (
        BiasMetric, ToxicityMetric, PIILeakageMetric,
        MisuseMetric, NonAdviceMetric, RoleViolationMetric,
    )

    tc = LLMTestCase(input=topic, actual_output=summary)
    builders = {
        "bias": lambda: BiasMetric(model=judge),
        "toxicity": lambda: ToxicityMetric(model=judge),
        "pii_leakage": lambda: PIILeakageMetric(model=judge),
        "misuse": lambda: MisuseMetric(domain=_AGENT_ROLE, model=judge),
        "non_advice": lambda: NonAdviceMetric(
            advice_types=["medical", "legal", "financial"], model=judge),
        "role_violation": lambda: RoleViolationMetric(role=_AGENT_ROLE, model=judge),
    }
    out: Dict[str, Dict[str, Any]] = {}
    for name, build in builders.items():
        try:
            metric = build()
            metric.measure(tc)
            out[name] = {"score": float(metric.score), "success": bool(metric.success)}
        except Exception as e:
            logger.warning("Safety metric %s failed: %s", name, e)
    return out


def run_one(item: Dict[str, str], llm, judge, deep: bool, safety: bool = False) -> Dict[str, Any]:
    """Run + score a single benchmark topic. Never raises; records errors."""
    topic = item["query"]
    rec: Dict[str, Any] = {"id": item["id"], "lang": item["lang"], "domain": item["domain"],
                           "query": topic, "metrics": {}, "error": None}
    t0 = time.time()
    try:
        # Imported at call time so tests can monkeypatch agents.query_analyzer.*
        from agents.query_analyzer import analyze_topic_with_llm, generate_search_queries

        analysis = analyze_topic_with_llm(topic, llm) or {}
        queries = generate_search_queries(topic, analysis)

        rec["metrics"]["query_count"] = len(queries)
        rec["metrics"]["bilingual_coverage"] = bool(
            analysis.get("queries") and analysis.get("queries_tr")
        )
        rec["metrics"]["query_relevance"] = _score_query_relevance(topic, queries, judge)

        if deep:
            from agents.search_agent import search_cs_sources, merge_papers
            from agents.summarizer_agent import generate_summary

            results = search_cs_sources(queries[:3], max_results_per_query=15)
            papers = merge_papers(results)
            rec["metrics"]["retrieval_count"] = len(papers)

            titles = [_norm_title(p.get("title")) for p in papers if p.get("title")]
            rec["metrics"]["dedup_integrity"] = len(titles) == len(set(titles))

            top = next((p for p in papers if p.get("abstract")), papers[0] if papers else None)
            if top:
                summary = generate_summary(llm, top)
                rec["summary"] = summary
                rec["metrics"].update(
                    _score_summary(topic, summary, top.get("abstract", ""), judge)
                )
                if safety:
                    for name, res in _score_safety(topic, summary, judge).items():
                        rec["metrics"][name] = res["score"]
                        rec.setdefault("passes", {})[name] = res["success"]
    except Exception as e:  # keep the benchmark robust over 100 topics
        rec["error"] = f"{type(e).__name__}: {e}"

    rec["latency_s"] = round(time.time() - t0, 2)
    return rec


def _passed(name: str, value: Any) -> Optional[bool]:
    """Did a single metric value meet its bar?"""
    if value is None:
        return None
    if name in ("bilingual_coverage", "dedup_integrity"):
        return bool(value)
    if name in THRESHOLDS:          # higher is better
        return value >= THRESHOLDS[name]
    # Safety metrics are judged by deepeval's own .success (mixed directions),
    # stored per-record in aggregate(); no manual threshold here.
    return None


def aggregate(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute per-metric means + pass rates and latency percentiles."""
    metric_names = [
        "query_relevance", "bilingual_coverage", "query_count",
        "retrieval_count", "dedup_integrity", "summary_faithfulness", "summary_relevancy",
        "bias", "toxicity", "pii_leakage", "misuse", "non_advice", "role_violation",
    ]
    summary: Dict[str, Any] = {"n": len(records), "errors": sum(1 for r in records if r["error"])}
    per_metric: Dict[str, Any] = {}

    for name in metric_names:
        recs_with = [r for r in records if r["metrics"].get(name) is not None]
        if not recs_with:
            continue
        values = [r["metrics"][name] for r in recs_with]
        passes = []
        for r in recs_with:
            if name in r.get("passes", {}):          # deepeval's own .success
                passes.append(bool(r["passes"][name]))
            else:
                p = _passed(name, r["metrics"][name])
                if p is not None:
                    passes.append(p)
        numeric = [float(v) for v in values if isinstance(v, (int, float)) and not isinstance(v, bool)]
        per_metric[name] = {
            "n": len(values),
            "mean": round(statistics.mean(numeric), 3) if numeric else None,
            "pass_rate": round(sum(passes) / len(passes), 3) if passes else None,
        }

    summary["metrics"] = per_metric
    latencies = [r["latency_s"] for r in records if r.get("latency_s")]
    if latencies:
        latencies.sort()
        summary["latency"] = {
            "p50": latencies[len(latencies) // 2],
            "p95": latencies[min(len(latencies) - 1, int(len(latencies) * 0.95))],
            "max": latencies[-1],
        }
    return summary


def _verdict(summary: Dict[str, Any]) -> str:
    """A simple product-readiness verdict from the pass rates."""
    rates = [m["pass_rate"] for m in summary.get("metrics", {}).values() if m.get("pass_rate") is not None]
    if not rates:
        return "N/A"
    avg = statistics.mean(rates)
    if avg >= 0.9:
        return "✅ Production-ready"
    if avg >= 0.75:
        return "🟡 Near production (some metrics below bar)"
    return "🔴 Not ready"


def to_markdown(summary: Dict[str, Any], meta: Dict[str, Any]) -> str:
    lines = [
        "# DeepArticle Benchmark Report",
        "",
        f"- **Date:** {meta['date']}",
        f"- **Judge model:** {meta['judge']}",
        f"- **Topics:** {summary['n']} ({meta['lang']}) · **Mode:** {meta['mode']}",
        f"- **Errors:** {summary['errors']}",
        f"- **Verdict:** {_verdict(summary)}",
        "",
        "## Metrics",
        "",
        "| Metric | Bar | N | Mean | Pass rate |",
        "|--------|-----|---|------|-----------|",
    ]
    bars = {
        "query_relevance": "≥ 0.60", "bilingual_coverage": "= true", "query_count": "≥ 10",
        "retrieval_count": "≥ 10", "dedup_integrity": "= true",
        "summary_faithfulness": "≥ 0.70", "summary_relevancy": "≥ 0.60",
    }
    quality = [n for n in summary.get("metrics", {}) if n not in SAFETY_METRICS]
    safety = [n for n in summary.get("metrics", {}) if n in SAFETY_METRICS]

    def _row(name):
        m = summary["metrics"][name]
        mean = "—" if m["mean"] is None else f"{m['mean']:.3f}"
        pr = "—" if m["pass_rate"] is None else f"{m['pass_rate']*100:.0f}%"
        bar = bars.get(name, "deepeval .success")
        return f"| {name} | {bar} | {m['n']} | {mean} | {pr} |"

    for name in quality:
        lines.append(_row(name))
    if safety:
        lines += ["", "## Safety (lower is better)", "",
                  "| Metric | Bar | N | Mean | Pass rate |",
                  "|--------|-----|---|------|-----------|"]
        for name in safety:
            lines.append(_row(name))

    if "latency" in summary:
        lat = summary["latency"]
        lines += ["", "## Latency (seconds / topic)", "",
                  f"- p50: {lat['p50']}  ·  p95: {lat['p95']}  ·  max: {lat['max']}"]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="DeepArticle quality benchmark")
    parser.add_argument("--limit", type=int, default=0, help="cap number of topics (0 = all)")
    parser.add_argument("--lang", default="all", choices=["all", "en", "tr"], help="language subset")
    parser.add_argument("--deep", action="store_true", help="also run search + summary metrics (slower)")
    parser.add_argument("--safety", action="store_true",
                        help="also score summaries with safety metrics (bias, toxicity, PII, ...); implies --deep")
    parser.add_argument("--out", default=_REPORT_DIR, help="output directory for reports")
    args = parser.parse_args()
    deep = args.deep or args.safety  # safety needs a generated summary

    llm = create_llm()
    judge = get_eval_model()
    if llm is None or judge is None:
        print("⚠ No LLM provider configured. Set an API key (e.g. GROQ_API_KEY) in .env.")
        sys.exit(1)

    questions = get_questions(lang=args.lang, limit=args.limit)
    mode = ("deep+safety" if args.safety else "deep") if deep else "shallow"
    print(f"\n🏁 Benchmarking {len(questions)} topics ({args.lang}, {mode} mode)...\n")

    records: List[Dict[str, Any]] = []
    for i, item in enumerate(questions, 1):
        rec = run_one(item, llm, judge, deep, safety=args.safety)
        records.append(rec)
        flag = "✗" if rec["error"] else "✓"
        qr = rec["metrics"].get("query_relevance")
        print(f"  [{i:>3}/{len(questions)}] {flag} {item['id']}  "
              f"relevance={qr if qr is None else round(qr,2)}  {rec['latency_s']}s"
              + (f"  ERROR {rec['error']}" if rec["error"] else ""))

    summary = aggregate(records)
    meta = {"date": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "judge": judge.get_model_name(), "lang": args.lang, "mode": mode}

    os.makedirs(args.out, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = os.path.join(args.out, f"benchmark_{stamp}.json")
    md_path = os.path.join(args.out, f"benchmark_{stamp}.md")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"meta": meta, "summary": summary, "records": records}, f,
                  ensure_ascii=False, indent=2)
    md = to_markdown(summary, meta)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md)

    print("\n" + md)
    print(f"\n📄 Saved: {json_path}\n          {md_path}")


if __name__ == "__main__":
    main()
