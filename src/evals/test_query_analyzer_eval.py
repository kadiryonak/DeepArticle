"""
DeepEval evaluation for the Query Analyzer agent.

Judges whether the search queries generated for a research topic are specific
and relevant (a known weak spot — generic queries find the wrong papers).

These tests make real LLM calls (for both generation and judging) and are
skipped automatically when no provider API key is configured. Run them with:

    pytest evals/ -v -m eval
"""

import pytest

import sys
sys.path.insert(0, '..')

from utils.llm_factory import create_llm
from agents.query_analyzer import analyze_topic_with_llm, generate_search_queries
from evals.datasets import QUERY_ANALYZER_TOPICS
from evals.eval_model import get_eval_model

HAS_LLM = create_llm() is not None

pytestmark = [
    pytest.mark.eval,
    pytest.mark.skipif(not HAS_LLM, reason="No LLM provider configured (set an API key in .env)"),
]


@pytest.mark.parametrize("topic", QUERY_ANALYZER_TOPICS)
def test_generated_queries_are_relevant_and_specific(topic):
    from deepeval import assert_test
    from deepeval.test_case import LLMTestCase, LLMTestCaseParams
    from deepeval.metrics import GEval

    llm = create_llm()
    analysis = analyze_topic_with_llm(topic, llm)
    queries = generate_search_queries(topic, analysis)

    actual_output = "\n".join(f"- {q}" for q in queries)

    relevancy = GEval(
        name="Query Relevance & Specificity",
        criteria=(
            "Evaluate whether the generated search queries are (1) clearly "
            "relevant to the research topic given as input, and (2) specific "
            "rather than overly generic. Penalize vague one-word queries and "
            "queries unrelated to the topic."
        ),
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
        ],
        model=get_eval_model(),
        threshold=0.6,
    )

    test_case = LLMTestCase(input=topic, actual_output=actual_output)
    assert_test(test_case, [relevancy])
