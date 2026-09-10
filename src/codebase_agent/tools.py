"""LangChain tools exposed to the agent.

Three tools, each returning plain text the model can read:

* ``search_code``     - symbol/keyword search with line numbers
* ``read_file``       - bounded, line-numbered file read
* ``retrieve_context``- semantic top-k retrieval (RAG)

All filesystem access goes through :class:`~codebase_agent.repository.Repository`,
so path traversal, oversized files and binaries are rejected before any read.
Tool errors are returned as short ``ERROR:`` strings instead of raising, so the
agent loop can recover and keep its tool-call budget under control.
"""

from __future__ import annotations

from collections.abc import Sequence

from langchain_core.tools import BaseTool, tool

from .config import Settings
from .rag import CodeRetriever
from .repository import Repository, RepositoryError


def _format_hits(hits: Sequence) -> str:
    lines = []
    for hit in hits:
        marker = " [definition]" if hit.is_definition else ""
        lines.append(f"{hit.rel_path}:{hit.line}: {hit.text}{marker}")
    return "\n".join(lines)


def build_tools(
    repo: Repository,
    retriever: CodeRetriever | None = None,
    *,
    max_search_results: int | None = None,
    max_read_lines: int | None = None,
    top_k: int | None = None,
) -> list[BaseTool]:
    """Create the tool set bound to one repository.

    ``max_search_results`` / ``max_read_lines`` / ``top_k`` default to the
    repository's own limits when omitted.
    """
    search_limit = max_search_results or 20
    read_limit = max_read_lines or repo.max_read_lines
    default_k = top_k or 6

    @tool
    def search_code(query: str, max_results: int = 0) -> str:
        """Search the repository source code for a symbol or keyword.

        Use this first for questions about a specific name, class, function or
        string. Returns ``path:line: text`` matches, with definition lines
        ranked first. Read the file with read_file to see full context.

        Args:
            query: Symbol or keyword, e.g. "CalculatorService" or "parse_answer".
            max_results: Maximum matches to return (0 = server default).
        """
        limit = max_results if max_results > 0 else search_limit
        try:
            hits = repo.search(query, max_results=limit)
        except (RepositoryError, ValueError) as exc:
            return f"ERROR: {exc}"
        if not hits:
            return f"No matches for {query!r} in repository {repo.root.name}."
        header = f"{len(hits)} match(es) for {query!r}:"
        return f"{header}\n{_format_hits(hits)}"

    @tool
    def read_file(path: str, start_line: int = 1, end_line: int = 0) -> str:
        """Read a text file from the repository with line numbers.

        Paths are repository-relative; absolute paths outside the repository are
        rejected. Large files, binaries and directories are rejected.

        Args:
            path: Repository-relative path, e.g. "src/app/service.py".
            start_line: First line to return (1-based, default 1).
            end_line: Last line to return (0 = until end or the line budget).
        """
        try:
            slice_ = repo.read_text(
                path,
                start_line=start_line or 1,
                end_line=end_line or None,
                max_lines=read_limit,
            )
        except (RepositoryError, ValueError) as exc:
            return f"ERROR: {exc}"
        header = (
            f"{slice_.rel_path} (lines {slice_.start_line}-{slice_.end_line} "
            f"of {slice_.total_lines})"
        )
        if slice_.truncated:
            header += " [truncated to line budget]"
        return f"{header}\n{slice_.numbered()}"

    @tool
    def retrieve_context(query: str, k: int = 0) -> str:
        """Semantically retrieve the most relevant code chunks (RAG).

        Use this when you do not know the exact symbol name, or when you need a
        broader view of how a feature is implemented across files.

        Args:
            query: Natural-language question or description.
            k: Number of chunks to return (0 = server default).
        """
        if retriever is None:
            return "ERROR: retrieval is not enabled for this agent."
        try:
            chunks = retriever.retrieve(query, k or default_k)
        except Exception as exc:  # noqa: BLE001 - surface provider errors to the model
            return f"ERROR: retrieval failed: {exc}"
        if not chunks:
            return f"No retrieval results for {query!r}."
        blocks = [
            f"[{index + 1}] {chunk.location} (score={chunk.score:.3f})\n{chunk.text}"
            for index, chunk in enumerate(chunks)
        ]
        return "\n\n---\n\n".join(blocks)

    return [search_code, read_file, retrieve_context]


def build_tools_from_settings(
    repo: Repository,
    settings: Settings,
    retriever: CodeRetriever | None = None,
) -> list[BaseTool]:
    """Settings-driven variant used by the CLI and the agent factory."""
    return build_tools(
        repo,
        retriever,
        max_search_results=settings.max_search_results,
        max_read_lines=settings.max_read_lines,
        top_k=settings.top_k,
    )
