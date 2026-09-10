"""Shared fixtures.

The autouse fixture scrubs LLM/embedding environment variables so the suite is
hermetic: it never reads a developer's real key and never touches the network.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from codebase_agent import HashingEmbeddings, Repository, Settings, build_retriever

FIXTURE_ROOT = Path(__file__).resolve().parent / "fixtures" / "sample_repo"

_ENV_TO_CLEAR = (
    "LLM_API_KEY",
    "DEEPSEEK_API_KEY",
    "OPENAI_API_KEY",
    "LLM_PROVIDER",
    "LLM_BASE_URL",
    "LLM_MODEL",
    "LLM_TIMEOUT",
    "LLM_TEMPERATURE",
    "STRUCTURED_OUTPUT_METHOD",
    "EMBEDDING_PROVIDER",
    "EMBEDDING_MODEL",
    "EMBEDDING_BASE_URL",
    "EMBEDDING_API_KEY",
    "EMBEDDING_DIM",
    "REPO_ROOT",
    "AGENT_REQUIRE_EVIDENCE",
    "AGENT_MEMORY_TURNS",
    "AGENT_MAX_TOOL_CALLS",
    "AGENT_MAX_ITERATIONS",
)


@pytest.fixture(autouse=True)
def _hermetic_env(monkeypatch: pytest.MonkeyPatch):
    for name in _ENV_TO_CLEAR:
        monkeypatch.delenv(name, raising=False)
    yield


@pytest.fixture(autouse=True)
def _restore_environ():
    """Guarantee that ``load_dotenv`` side effects never leak between tests."""
    snapshot = dict(os.environ)
    yield
    os.environ.clear()
    os.environ.update(snapshot)


@pytest.fixture(scope="session")
def sample_repo_path() -> Path:
    assert FIXTURE_ROOT.is_dir(), f"fixture repository missing: {FIXTURE_ROOT}"
    return FIXTURE_ROOT


@pytest.fixture
def settings() -> Settings:
    return Settings(embedding_provider="local", embedding_dim=256, top_k=4)


@pytest.fixture
def repo(sample_repo_path: Path) -> Repository:
    return Repository(sample_repo_path, max_file_bytes=200_000, max_read_lines=400)


@pytest.fixture
def embeddings() -> HashingEmbeddings:
    return HashingEmbeddings(dim=256)


@pytest.fixture
def retriever(repo: Repository, embeddings: HashingEmbeddings, settings: Settings):
    return build_retriever(repo, settings, embeddings)
