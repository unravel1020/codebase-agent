"""Wiring helpers: one call builds a ready-to-use agent from a path."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel

from .agent import CodebaseAgent
from .config import Settings
from .embeddings import build_embeddings
from .llm import build_chat_model
from .memory import ConversationMemory
from .rag import CodeRetriever, build_retriever
from .repository import Repository
from .tools import build_tools_from_settings


def build_repository(
    repo_path: str | Path,
    settings: Settings,
    *,
    extra_excluded_dirs: tuple[str, ...] = (),
) -> Repository:
    """Open a repository with the configured size/line limits."""
    return Repository(
        repo_path,
        max_file_bytes=settings.max_file_bytes,
        max_read_lines=settings.max_read_lines,
        extra_excluded_dirs=extra_excluded_dirs,
    )


def build_agent(
    repo_path: str | Path,
    settings: Settings | None = None,
    *,
    llm: BaseChatModel | None = None,
    with_retrieval: bool = True,
    memory: ConversationMemory | None = None,
    repo: Repository | None = None,
    retriever: CodeRetriever | None = None,
    **settings_overrides: Any,
) -> CodebaseAgent:
    """Build a :class:`CodebaseAgent` for ``repo_path``.

    Pass ``llm`` to inject a fake/scripted model (tests, evals, offline demos);
    pass ``repo``/``retriever`` to reuse an already-indexed repository.
    """
    settings = settings or Settings.from_env(**settings_overrides)
    repo = repo if repo is not None else build_repository(repo_path, settings)

    if retriever is None and with_retrieval:
        retriever = build_retriever(repo, settings, build_embeddings(settings))

    tools = build_tools_from_settings(repo, settings, retriever)
    model = llm if llm is not None else build_chat_model(settings)
    return CodebaseAgent(
        model,
        tools,
        memory=memory,
        settings=settings,
        repo_name=repo.root.name,
    )
