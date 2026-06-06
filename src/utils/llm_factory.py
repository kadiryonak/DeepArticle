"""
LLM Factory - Creates LLM instances for different providers.
Supports Groq, OpenAI, Anthropic, and Google.
"""

from typing import Optional, Any

from config import (
    LLM_PROVIDER, LLM_MODEL, LLM_TEMPERATURE,
    GROQ_API_KEY, OPENAI_API_KEY, ANTHROPIC_API_KEY, GOOGLE_API_KEY
)


def create_llm(
    provider: Optional[str] = None,
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
) -> Optional[Any]:
    """
    Create an LLM instance based on the specified or configured provider.

    Args:
        provider: LLM provider (groq, openai, anthropic, google). Defaults to config.
        model: Model name. Defaults to config.
        temperature: Temperature for generation. Defaults to config.
        top_p: Nucleus-sampling parameter. Pass for deterministic generation
            (e.g. temperature=0, top_p=1.0). Applied to all providers.
        top_k: Top-k sampling. Pass top_k=1 for greedy/deterministic decoding.
            Only Anthropic and Google support it; ignored for Groq/OpenAI.

    Returns:
        LLM instance or None if no valid provider/key
    """
    provider = provider or LLM_PROVIDER
    model = model or LLM_MODEL
    temperature = temperature if temperature is not None else LLM_TEMPERATURE

    if not provider:
        print("⚠ No LLM provider configured. Add an API key to .env")
        return None

    try:
        if provider == "groq":
            return _create_groq_llm(model, temperature, top_p, top_k)
        elif provider == "openai":
            return _create_openai_llm(model, temperature, top_p, top_k)
        elif provider == "anthropic":
            return _create_anthropic_llm(model, temperature, top_p, top_k)
        elif provider == "google":
            return _create_google_llm(model, temperature, top_p, top_k)
        else:
            print(f"⚠ Unknown provider: {provider}")
            return None
    except ImportError as e:
        print(f"⚠ Missing package for {provider}: {e}")
        print(f"   Install with: pip install langchain-{provider}")
        return None
    except Exception as e:
        print(f"⚠ Error creating {provider} LLM: {e}")
        return None


def _create_groq_llm(model, temperature, top_p=None, top_k=None):
    """Create Groq LLM instance. (Groq supports top_p, not top_k.)"""
    if not GROQ_API_KEY:
        print("⚠ GROQ_API_KEY not set")
        return None

    from langchain_groq import ChatGroq
    kwargs = {"api_key": GROQ_API_KEY, "model_name": model, "temperature": temperature}
    if top_p is not None:
        kwargs["model_kwargs"] = {"top_p": top_p}
    return ChatGroq(**kwargs)


def _create_openai_llm(model, temperature, top_p=None, top_k=None):
    """Create OpenAI LLM instance. (OpenAI supports top_p, not top_k.)"""
    if not OPENAI_API_KEY:
        print("⚠ OPENAI_API_KEY not set")
        return None

    from langchain_openai import ChatOpenAI
    kwargs = {"api_key": OPENAI_API_KEY, "model": model, "temperature": temperature}
    if top_p is not None:
        kwargs["top_p"] = top_p
    return ChatOpenAI(**kwargs)


def _create_anthropic_llm(model, temperature, top_p=None, top_k=None):
    """Create Anthropic LLM instance. (Supports both top_p and top_k.)"""
    if not ANTHROPIC_API_KEY:
        print("⚠ ANTHROPIC_API_KEY not set")
        return None

    from langchain_anthropic import ChatAnthropic
    kwargs = {"api_key": ANTHROPIC_API_KEY, "model": model, "temperature": temperature}
    if top_p is not None:
        kwargs["top_p"] = top_p
    if top_k is not None:
        kwargs["top_k"] = top_k
    return ChatAnthropic(**kwargs)


def _create_google_llm(model, temperature, top_p=None, top_k=None):
    """Create Google LLM instance. (Supports both top_p and top_k.)"""
    if not GOOGLE_API_KEY:
        print("⚠ GOOGLE_API_KEY not set")
        return None

    from langchain_google_genai import ChatGoogleGenerativeAI
    kwargs = {"google_api_key": GOOGLE_API_KEY, "model": model, "temperature": temperature}
    if top_p is not None:
        kwargs["top_p"] = top_p
    if top_k is not None:
        kwargs["top_k"] = top_k
    return ChatGoogleGenerativeAI(**kwargs)


def get_available_providers() -> list:
    """Get list of providers with valid API keys."""
    providers = []
    if GROQ_API_KEY:
        providers.append("groq")
    if OPENAI_API_KEY:
        providers.append("openai")
    if ANTHROPIC_API_KEY:
        providers.append("anthropic")
    if GOOGLE_API_KEY:
        providers.append("google")
    return providers


def test_llm_connection() -> bool:
    """Test if LLM can be created and invoked."""
    llm = create_llm()
    if not llm:
        return False
    
    try:
        response = llm.invoke("Say 'OK' if you can hear me.")
        return bool(response.content)
    except Exception as e:
        print(f"⚠ LLM test failed: {e}")
        return False
