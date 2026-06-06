"""
DeepEval evaluation for the Summarizer agent.

Checks that LLM-generated summaries are:
  * Faithful — no claims that aren't supported by the abstract (hallucination
    detection), and
  * Relevant — actually summarize the paper.

These tests make real LLM calls and are skipped when no provider API key is
configured. Run them with:

    pytest evals/ -v -m eval
"""

import pytest

import sys
sys.path.insert(0, '..')

from utils.llm_factory import create_llm
from agents.summarizer_agent import generate_summary
from evals.datasets import SUMMARIZER_PAPERS
from evals.eval_model import get_eval_model

HAS_LLM = create_llm() is not None

pytestmark = [
    pytest.mark.eval,
    pytest.mark.skipif(not HAS_LLM, reason="No LLM provider configured (set an API key in .env)"),
]


@pytest.mark.parametrize("paper", SUMMARIZER_PAPERS, ids=lambda p: p["title"][:30])
def test_summary_is_faithful_and_relevant(paper):
    from deepeval import assert_test
    from deepeval.test_case import LLMTestCase
    from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric

    llm = create_llm()
    summary = generate_summary(llm, paper)

    eval_model = get_eval_model()

    # Faithfulness: the abstract is the retrieval context the summary must not
    # contradict or exceed.
    faithfulness = FaithfulnessMetric(
        threshold=0.7,
        model=eval_model,
        include_reason=True,
    )

    relevancy = AnswerRelevancyMetric(
        threshold=0.6,
        model=eval_model,
        include_reason=True,
    )

    test_case = LLMTestCase(
        input=f"Summarize the paper titled '{paper['title']}'.",
        actual_output=summary,
        retrieval_context=[paper["abstract"]],
    )

    assert_test(test_case, [faithfulness, relevancy])
