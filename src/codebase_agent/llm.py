"""Chat-model factory.

The client is always an OpenAI-compatible endpoint: ``ChatOpenAI`` with an
explicit ``base_url``. DeepSeek is the default provider, but nothing in the
codebase is bound to it - ``LLM_PROVIDER`` / ``LLM_BASE_URL`` / ``LLM_MODEL``
select any compatible service.
"""

from __future__ import annotations

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI

from .config import Settings


def build_chat_model(settings: Settings, *, require_api_key: bool = True) -> BaseChatModel:
    """Create the tool-calling chat model described by ``settings``."""
    api_key = settings.require_api_key() if require_api_key else (settings.api_key or "not-needed")
    return ChatOpenAI(
        model=settings.model,
        api_key=api_key,
        base_url=settings.base_url,
        temperature=settings.temperature,
        timeout=settings.timeout,
        max_retries=settings.max_retries,
    )
