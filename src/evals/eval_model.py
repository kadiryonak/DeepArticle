"""
DeepEval judge model that reuses DeepArticle's multi-provider LLM factory.

DeepEval defaults to OpenAI as the judge LLM. This wrapper lets the evaluation
suite use whichever provider is already configured for the project (Groq,
OpenAI, Anthropic or Google), so you only need one API key.
"""

from typing import Optional, Any

import sys
sys.path.insert(0, '..')

from deepeval.models import DeepEvalBaseLLM

from config import LLM_PROVIDER, LLM_MODEL
from utils.llm_factory import create_llm


class DeepArticleEvalModel(DeepEvalBaseLLM):
    """Wraps a LangChain chat model so DeepEval can use it as a judge."""

    def __init__(self, llm: Optional[Any] = None):
        # Use a low temperature for stable, repeatable judgements.
        self._llm = llm or create_llm(temperature=0.0)

    def load_model(self):
        return self._llm

    def generate(self, prompt: str, schema: Optional[type] = None) -> Any:
        """
        Generate a response. When DeepEval passes a Pydantic ``schema`` (used by
        metrics like GEval for structured scoring), we use the model's native
        structured-output support.
        """
        if self._llm is None:
            raise RuntimeError(
                "No LLM configured. Set an API key (e.g. GROQ_API_KEY) in .env."
            )

        if schema is not None:
            structured = self._llm.with_structured_output(schema)
            return structured.invoke(prompt)

        return self._llm.invoke(prompt).content

    async def a_generate(self, prompt: str, schema: Optional[type] = None) -> Any:
        # LangChain models expose async, but a sync call keeps the wrapper
        # simple and provider-agnostic for the small eval workloads here.
        return self.generate(prompt, schema)

    def get_model_name(self) -> str:
        return f"DeepArticle judge ({LLM_PROVIDER or 'none'}/{LLM_MODEL})"


def get_eval_model() -> Optional["DeepArticleEvalModel"]:
    """Return a judge model, or ``None`` if no LLM provider is configured."""
    llm = create_llm(temperature=0.0)
    if llm is None:
        return None
    return DeepArticleEvalModel(llm)
