"""Tests for environment-driven configuration (no secrets required)."""

from __future__ import annotations

from pathlib import Path

import pytest

from codebase_agent.config import ConfigError, Settings


def test_deepseek_is_the_default_provider() -> None:
    settings = Settings.from_env(env_file=None)
    assert settings.provider == "deepseek"
    assert settings.base_url == "https://api.deepseek.com/v1"
    assert settings.model == "deepseek-chat"
    assert settings.api_key is None
    assert settings.embedding_provider == "local"


def test_provider_preset_can_be_selected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    settings = Settings.from_env(env_file=None)
    assert settings.base_url == "https://api.openai.com/v1"
    assert settings.model == "gpt-4o-mini"


def test_explicit_base_url_and_model_win(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "openai-compatible")
    monkeypatch.setenv("LLM_BASE_URL", "http://localhost:8000/v1")
    monkeypatch.setenv("LLM_MODEL", "local-model")
    settings = Settings.from_env(env_file=None)
    assert settings.base_url == "http://localhost:8000/v1"
    assert settings.model == "local-model"


def test_unknown_provider_without_explicit_settings_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "totally-unknown")
    with pytest.raises(ConfigError):
        Settings.from_env(env_file=None)


def test_api_key_fallback_order(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-deepseek")
    assert Settings.from_env(env_file=None).api_key == "sk-deepseek"

    monkeypatch.setenv("LLM_API_KEY", "sk-generic")
    assert Settings.from_env(env_file=None).api_key == "sk-generic"


def test_numeric_and_boolean_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_TIMEOUT", "12.5")
    monkeypatch.setenv("AGENT_MAX_TOOL_CALLS", "3")
    monkeypatch.setenv("AGENT_REQUIRE_EVIDENCE", "0")
    settings = Settings.from_env(env_file=None)
    assert settings.timeout == pytest.approx(12.5)
    assert settings.max_tool_calls == 3
    assert settings.require_evidence is False


def test_invalid_numeric_value_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAG_TOP_K", "many")
    with pytest.raises(ConfigError):
        Settings.from_env(env_file=None)


def test_env_file_is_loaded(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("LLM_MODEL=from-dotenv\nLLM_BASE_URL=https://example.test/v1\n", encoding="utf-8")
    settings = Settings.from_env(env_file=env_file)
    assert settings.model == "from-dotenv"
    assert settings.base_url == "https://example.test/v1"


def test_require_api_key_raises_with_helpful_message() -> None:
    settings = Settings.from_env(env_file=None)
    with pytest.raises(ConfigError) as excinfo:
        settings.require_api_key()
    assert "LLM_API_KEY" in str(excinfo.value)


def test_safe_dict_masks_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_API_KEY", "sk-super-secret-value")
    payload = Settings.from_env(env_file=None).safe_dict()
    assert payload["api_key"] == "***alue"
    assert "sk-super-secret-value" not in str(payload)
    assert payload["model"] == "deepseek-chat"


def test_replace_returns_new_instance() -> None:
    settings = Settings.from_env(env_file=None)
    updated = settings.replace(top_k=99)
    assert settings.top_k != 99
    assert updated.top_k == 99
