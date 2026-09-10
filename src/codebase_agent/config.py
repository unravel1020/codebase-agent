"""Environment-driven configuration.

Nothing here is hard-coded to a vendor: the chat model is always an
OpenAI-compatible endpoint whose ``base_url``, ``model`` and ``api_key`` come
from the environment (optionally seeded by a git-ignored ``.env`` file).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

#: Known OpenAI-compatible endpoints. ``LLM_BASE_URL`` / ``LLM_MODEL`` always win,
#: so any other vendor works too (just set them explicitly).
PROVIDER_PRESETS: dict[str, dict[str, str | None]] = {
    "deepseek": {
        "base_url": "https://api.deepseek.com/v1",
        "model": "deepseek-chat",
        "api_key_env": "DEEPSEEK_API_KEY",
    },
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "model": "gpt-4o-mini",
        "api_key_env": "OPENAI_API_KEY",
    },
    "openai-compatible": {
        "base_url": None,
        "model": None,
        "api_key_env": None,
    },
}

DEFAULT_PROVIDER = "deepseek"


class ConfigError(RuntimeError):
    """Raised when configuration is missing or malformed."""


def _env(name: str, default: str | None = None) -> str | None:
    raw = os.environ.get(name)
    if raw is None:
        return default
    raw = raw.strip()
    return raw or default


def _env_int(name: str, default: int) -> int:
    raw = _env(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError as exc:  # pragma: no cover - defensive
        raise ConfigError(f"{name} must be an integer, got {raw!r}") from exc


def _env_float(name: str, default: float) -> float:
    raw = _env(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError as exc:  # pragma: no cover - defensive
        raise ConfigError(f"{name} must be a number, got {raw!r}") from exc


@dataclass(frozen=True)
class Settings:
    """Immutable snapshot of the runtime configuration."""

    # --- chat model (OpenAI-compatible) ---
    provider: str = DEFAULT_PROVIDER
    api_key: str | None = None
    base_url: str = "https://api.deepseek.com/v1"
    model: str = "deepseek-chat"
    temperature: float = 0.0
    timeout: float = 60.0
    max_retries: int = 2
    structured_output_method: str = "function_calling"

    # --- embeddings ---
    # "local" = deterministic offline hashing embeddings (no API key needed).
    # "openai" = any OpenAI-compatible /embeddings endpoint.
    embedding_provider: str = "local"
    embedding_model: str = "text-embedding-3-small"
    embedding_base_url: str | None = None
    embedding_api_key: str | None = None
    embedding_dim: int = 512

    # --- RAG ---
    top_k: int = 6
    chunk_size: int = 1200
    chunk_overlap: int = 150
    max_index_files: int = 400

    # --- tools / agent ---
    max_file_bytes: int = 512_000
    max_read_lines: int = 400
    max_search_results: int = 20
    max_tool_calls: int = 8
    max_iterations: int = 6
    memory_turns: int = 6
    require_evidence: bool = True

    repo_root: Path | None = field(default=None)

    # ------------------------------------------------------------------ #
    @classmethod
    def from_env(
        cls,
        env_file: str | Path | None = ".env",
        repo_root: str | Path | None = None,
        **overrides: Any,
    ) -> Settings:
        """Build settings from ``.env`` + process environment.

        Precedence: explicit ``overrides`` > environment > ``.env`` > defaults.
        """
        if env_file is not None:
            path = Path(env_file)
            if path.is_file():
                load_dotenv(path, override=False)

        provider = (_env("LLM_PROVIDER", DEFAULT_PROVIDER) or DEFAULT_PROVIDER).lower()
        preset = PROVIDER_PRESETS.get(provider, {})

        api_key = _env("LLM_API_KEY")
        if api_key is None:
            preset_key_env = preset.get("api_key_env")
            if preset_key_env:
                api_key = _env(preset_key_env)
            if api_key is None:
                api_key = _env("OPENAI_API_KEY")

        base_url = _env("LLM_BASE_URL") or preset.get("base_url")
        model = _env("LLM_MODEL") or preset.get("model")
        if not base_url or not model:
            raise ConfigError(
                f"provider {provider!r} has no preset: set LLM_BASE_URL and LLM_MODEL explicitly"
            )

        resolved_repo = repo_root or _env("REPO_ROOT")

        settings = cls(
            provider=provider,
            api_key=api_key,
            base_url=base_url,
            model=model,
            temperature=_env_float("LLM_TEMPERATURE", 0.0),
            timeout=_env_float("LLM_TIMEOUT", 60.0),
            max_retries=_env_int("LLM_MAX_RETRIES", 2),
            structured_output_method=_env("STRUCTURED_OUTPUT_METHOD", "function_calling"),
            embedding_provider=(_env("EMBEDDING_PROVIDER", "local") or "local").lower(),
            embedding_model=_env("EMBEDDING_MODEL", "text-embedding-3-small") or "",
            embedding_base_url=_env("EMBEDDING_BASE_URL"),
            embedding_api_key=_env("EMBEDDING_API_KEY"),
            embedding_dim=_env_int("EMBEDDING_DIM", 512),
            top_k=_env_int("RAG_TOP_K", 6),
            chunk_size=_env_int("RAG_CHUNK_SIZE", 1200),
            chunk_overlap=_env_int("RAG_CHUNK_OVERLAP", 150),
            max_index_files=_env_int("RAG_MAX_INDEX_FILES", 400),
            max_file_bytes=_env_int("MAX_FILE_BYTES", 512_000),
            max_read_lines=_env_int("MAX_READ_LINES", 400),
            max_search_results=_env_int("MAX_SEARCH_RESULTS", 20),
            max_tool_calls=_env_int("AGENT_MAX_TOOL_CALLS", 8),
            max_iterations=_env_int("AGENT_MAX_ITERATIONS", 6),
            memory_turns=_env_int("AGENT_MEMORY_TURNS", 6),
            require_evidence=(_env("AGENT_REQUIRE_EVIDENCE", "1") or "1").lower()
            not in {"0", "false", "no"},
            repo_root=Path(resolved_repo).expanduser().resolve() if resolved_repo else None,
        )
        if overrides:
            settings = settings.replace(**overrides)
        return settings

    # ------------------------------------------------------------------ #
    def replace(self, **overrides: Any) -> Settings:
        """Return a copy with selected fields replaced (frozen dataclass helper)."""
        from dataclasses import replace as _replace

        return _replace(self, **overrides)

    def require_api_key(self) -> str:
        """Return the API key or fail with an actionable message."""
        if not self.api_key:
            raise ConfigError(
                "No API key found. Copy .env.example to .env and set LLM_API_KEY "
                "(e.g. DEEPSEEK_API_KEY=sk-...). Offline modes "
                "(pytest, `python -m codebase_agent grep/read/search`) need no key."
            )
        return self.api_key

    def safe_dict(self) -> dict[str, Any]:
        """Serializable view with credentials redacted (safe for eval artifacts)."""

        def mask(value: str | None) -> str | None:
            if not value:
                return None
            return f"***{value[-4:]}" if len(value) > 4 else "***"

        return {
            "provider": self.provider,
            "base_url": self.base_url,
            "model": self.model,
            "temperature": self.temperature,
            "timeout": self.timeout,
            "structured_output_method": self.structured_output_method,
            "api_key": mask(self.api_key),
            "embedding_provider": self.embedding_provider,
            "embedding_model": self.embedding_model
            if self.embedding_provider != "local"
            else f"local-hashing-{self.embedding_dim}",
            "embedding_api_key": mask(self.embedding_api_key),
            "top_k": self.top_k,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "max_file_bytes": self.max_file_bytes,
            "max_tool_calls": self.max_tool_calls,
            "max_iterations": self.max_iterations,
            "repo_root": str(self.repo_root) if self.repo_root else None,
        }
