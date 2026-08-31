"""LLM factories — the ONLY module that imports ``langchain_anthropic``.

Everything downstream takes a :class:`BaseChatModel`, so CI (which has no API key by
construction) injects ``GenericFakeChatModel`` and exercises identical code paths.
"""

from langchain_core.language_models.chat_models import BaseChatModel

from hedis_copilot.config import ConfigError, Settings


def _require_key(settings: Settings, purpose: str) -> str:
    if settings.anthropic_api_key is None:
        raise ConfigError(
            f"ANTHROPIC key required for {purpose}: set HEDIS_ANTHROPIC_API_KEY in .env "
            "(never committed; CI runs keyless with a fake model)"
        )
    return settings.anthropic_api_key.get_secret_value()


def answer_model(settings: Settings) -> BaseChatModel:
    """The answering model (default claude-sonnet-5, temperature 0)."""
    from langchain_anthropic import ChatAnthropic

    return ChatAnthropic(
        model_name=settings.answer_model,
        temperature=0,
        api_key=_require_key(settings, "answer generation"),  # type: ignore[arg-type]
        timeout=60,
        stop=None,
    )


def judge_model(settings: Settings) -> BaseChatModel:
    """The eval judge — deliberately a different, stronger model than the answerer."""
    from langchain_anthropic import ChatAnthropic

    return ChatAnthropic(
        model_name=settings.judge_model,
        temperature=0,
        api_key=_require_key(settings, "eval judging"),  # type: ignore[arg-type]
        timeout=120,
        stop=None,
    )
