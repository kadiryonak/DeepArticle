"""
Golden datasets for evaluating DeepArticle agents.

These are small, hand-picked examples used as inputs for the DeepEval test
cases. Keep them realistic but compact so evaluation runs stay cheap.
"""

# Research topics fed to the Query Analyzer agent.
QUERY_ANALYZER_TOPICS = [
    "unit test generation using large language models",
    "retrieval augmented generation for question answering",
    "graph neural networks for drug discovery",
]

# (title, abstract) pairs fed to the Summarizer agent. Abstracts are the
# ground-truth context that summaries must stay faithful to.
SUMMARIZER_PAPERS = [
    {
        "title": "TestPilot: LLM-based Test Generation",
        "abstract": (
            "We present TestPilot, a tool that automatically generates unit "
            "tests for JavaScript using large language models. TestPilot does "
            "not require additional training or few-shot examples. Across 25 "
            "npm packages, TestPilot achieves a median statement coverage of "
            "70.2% and a median function coverage of 92.8%, substantially "
            "outperforming the state-of-the-art feedback-directed test "
            "generation technique."
        ),
    },
    {
        "title": "A Survey of Retrieval-Augmented Generation",
        "abstract": (
            "Retrieval-Augmented Generation (RAG) combines parametric language "
            "models with non-parametric external knowledge retrieval. This "
            "survey reviews RAG architectures, retrievers, and training "
            "strategies, and discusses open challenges such as retrieval "
            "quality, latency, and hallucination mitigation in knowledge-"
            "intensive tasks."
        ),
    },
]
