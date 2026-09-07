from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from pydantic import SecretStr

from app.config import Settings
from app.llm.fake import ScriptedChatModel
from app.observability.logging import get_logger

log = get_logger(__name__)

_DEFAULT_MODEL = {
    "groq": "openai/gpt-oss-120b",
    "openai": "gpt-4o-mini",
    "anthropic": "claude-sonnet-4-5",
}


class LLMUnavailableError(RuntimeError):
    pass


def build_chat_model(settings: Settings, scripted: dict[str, Any] | None = None) -> BaseChatModel:
    provider = settings.llm_provider
    if provider == "fake":
        return ScriptedChatModel(responses=scripted or {})

    model = settings.llm_model or _DEFAULT_MODEL[provider]
    key = settings.api_key_for_provider()
    if not key:
        raise LLMUnavailableError(
            f"LLM_PROVIDER={provider} but no API key is set. "
            f"Set the matching key in .env, or use LLM_PROVIDER=fake to run offline."
        )

    # Each provider is constructed explicitly rather than through a shared kwargs
    # dict: the constructors disagree on which arguments are required, and a splat
    # would hide that behind an untyped mapping.
    if provider == "groq":
        from langchain_groq import ChatGroq

        return ChatGroq(
            api_key=SecretStr(key),
            model=model,
            temperature=settings.llm_temperature,
            timeout=settings.llm_timeout_seconds,
            max_retries=settings.llm_max_retries,
        )

    if provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            api_key=SecretStr(key),
            model=model,
            temperature=settings.llm_temperature,
            timeout=settings.llm_timeout_seconds,
            max_retries=settings.llm_max_retries,
        )

    from langchain_anthropic import ChatAnthropic

    return ChatAnthropic(
        api_key=SecretStr(key),
        model_name=model,
        temperature=settings.llm_temperature,
        timeout=settings.llm_timeout_seconds,
        max_retries=settings.llm_max_retries,
        max_tokens_to_sample=4096,
        stop=None,
    )


def usage_from(message: Any) -> dict[str, int]:
    usage = getattr(message, "usage_metadata", None)
    if isinstance(usage, dict):
        return {
            "input_tokens": int(usage.get("input_tokens", 0)),
            "output_tokens": int(usage.get("output_tokens", 0)),
            "total_tokens": int(usage.get("total_tokens", 0)),
        }
    return {}
