"""Command-line entry point: ``python -m codebase_agent <command>``.

``grep`` / ``read`` / ``index`` / ``search`` need no API key at all.
``ask`` needs a key unless ``--offline`` is passed (heuristic policy).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import ConfigError, Settings
from .embeddings import build_embeddings
from .factory import build_agent, build_repository
from .offline import build_offline_llm
from .rag import build_retriever
from .tools import build_tools_from_settings


def _settings(args: argparse.Namespace) -> Settings:
    return Settings.from_env(env_file=args.env_file)


def cmd_index(args: argparse.Namespace) -> int:
    settings = _settings(args)
    repo = build_repository(args.repo, settings)
    stats = repo.stats(max_files=settings.max_index_files)
    retriever = build_retriever(repo, settings)
    print(f"repository : {repo.root}")
    print(f"files      : {stats['files']} indexable ({stats['bytes']} bytes)")
    print(f"chunks     : {retriever.chunk_count} across {retriever.files_indexed} files")
    embedding_label = (
        f"local-hashing-{settings.embedding_dim}"
        if settings.embedding_provider == "local"
        else f"{settings.embedding_provider} / {settings.embedding_model}"
    )
    print(f"embeddings : {embedding_label}")
    return 0


def cmd_search(args: argparse.Namespace) -> int:
    settings = _settings(args)
    repo = build_repository(args.repo, settings)
    retriever = build_retriever(repo, settings)
    chunks = retriever.retrieve(args.query, args.k or settings.top_k)
    if not chunks:
        print("no results")
        return 1
    for index, chunk in enumerate(chunks, start=1):
        print(f"[{index}] {chunk.location} score={chunk.score:.3f}")
        print(f"    {chunk.preview(200)}")
    return 0


def cmd_grep(args: argparse.Namespace) -> int:
    settings = _settings(args)
    repo = build_repository(args.repo, settings)
    tools = {tool.name: tool for tool in build_tools_from_settings(repo, settings)}
    print(tools["search_code"].invoke({"query": args.query, "max_results": args.max_results}))
    return 0


def cmd_read(args: argparse.Namespace) -> int:
    settings = _settings(args)
    repo = build_repository(args.repo, settings)
    tools = {tool.name: tool for tool in build_tools_from_settings(repo, settings)}
    output = tools["read_file"].invoke(
        {"path": args.path, "start_line": args.start, "end_line": args.end}
    )
    print(output)
    return 0 if not output.startswith("ERROR:") else 2


def cmd_ask(args: argparse.Namespace) -> int:
    settings = _settings(args)
    llm = build_offline_llm() if args.offline else None
    agent = build_agent(args.repo, settings, llm=llm)
    result = agent.run(args.question)

    if args.json:
        print(json.dumps(result.to_dict(include_transcript=args.show_trace), indent=2))
        return 0

    print(result.answer.to_markdown())
    print()
    print(
        f"-- {result.tool_call_count} tool call(s), {result.iterations} model iteration(s), "
        f"{result.latency_ms:.0f} ms"
    )
    if args.show_trace:
        for index, record in enumerate(result.tool_calls, start=1):
            status = "ok" if record.ok else "error"
            print(f"   [{index}] {record.name}({record.args}) -> {status} "
                  f"{record.duration_ms:.0f} ms")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="codebase-agent",
        description="Ask questions about a local repository with an LLM + tool calling + RAG.",
    )
    parser.add_argument("--env-file", default=".env", help="path to .env (default: ./.env)")
    sub = parser.add_subparsers(dest="command", required=True)

    index = sub.add_parser("index", help="index a repository and report chunk stats")
    index.add_argument("--repo", default=".", help="repository root (default: .)")
    index.set_defaults(func=cmd_index)

    search = sub.add_parser("search", help="semantic top-k retrieval (no LLM)")
    search.add_argument("--repo", default=".", help="repository root (default: .)")
    search.add_argument("query", help="natural-language or symbol query")
    search.add_argument("-k", type=int, default=0, help="number of chunks (default: RAG_TOP_K)")
    search.set_defaults(func=cmd_search)

    grep = sub.add_parser("grep", help="symbol/keyword search (no LLM)")
    grep.add_argument("--repo", default=".", help="repository root (default: .)")
    grep.add_argument("query", help="symbol or keyword")
    grep.add_argument("-n", "--max-results", type=int, default=20, dest="max_results")
    grep.set_defaults(func=cmd_grep)

    read = sub.add_parser("read", help="read a file with line numbers (no LLM)")
    read.add_argument("--repo", default=".", help="repository root (default: .)")
    read.add_argument("path", help="repository-relative path")
    read.add_argument("--start", type=int, default=1)
    read.add_argument("--end", type=int, default=0)
    read.set_defaults(func=cmd_read)

    ask = sub.add_parser("ask", help="ask the agent a question (needs an API key unless --offline)")
    ask.add_argument("--repo", default=".", help="repository root (default: .)")
    ask.add_argument("question", help="question about the repository")
    ask.add_argument("--offline", action="store_true", help="use the heuristic policy, no API key")
    ask.add_argument("--json", action="store_true", help="print the structured result as JSON")
    ask.add_argument("--show-trace", action="store_true", help="print tool calls / transcript")
    ask.set_defaults(func=cmd_ask)

    return parser


def friendly_llm_error(exc: BaseException) -> str | None:
    """Translate provider failures into actionable advice.

    Returns ``None`` for anything that is not a recognisable LLM/network error,
    so real bugs still surface as tracebacks.
    """
    name = type(exc).__name__
    text = str(exc)
    lowered = text.lower()

    if "Authentication" in name or "PermissionDenied" in name or "401" in text or "invalid_api_key" in lowered:
        return (
            "LLM API key was rejected (401).\n"
            "  Check LLM_API_KEY in .env (DeepSeek keys: https://platform.deepseek.com/api_keys).\n"
            "  To test the whole pipeline without a key, add --offline."
        )
    if "RateLimit" in name or "429" in text:
        return f"The provider rate-limited the request (429). Retry later.\n  {text[:200]}"
    if "NotFound" in name or "404" in text:
        return (
            "The endpoint or model was not found (404).\n"
            "  Check LLM_BASE_URL (must end in /v1 for DeepSeek) and LLM_MODEL.\n"
            f"  {text[:200]}"
        )
    if "Timeout" in name or "Connection" in name or "connection" in lowered or "timed out" in lowered:
        return (
            f"Could not reach the LLM endpoint ({name}).\n"
            "  Check your network / proxy / LLM_BASE_URL, or raise LLM_TIMEOUT.\n"
            f"  {text[:200]}"
        )
    if "BadRequest" in name or "400" in text:
        return f"The provider rejected the request (400).\n  {text[:300]}"
    return None


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except ConfigError as exc:
        print(f"configuration error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:  # pragma: no cover
        print("interrupted", file=sys.stderr)
        return 130
    except Exception as exc:  # noqa: BLE001 - only provider errors are translated
        message = friendly_llm_error(exc)
        if message is None:
            raise
        print(f"llm error: {message}", file=sys.stderr)
        return 3


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
