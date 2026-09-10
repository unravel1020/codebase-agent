"""CLI tests - all offline (no API key, no network)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from codebase_agent.cli import friendly_llm_error, main

NO_ENV = "tests/.no-such-env-file"


def run(capsys: pytest.CaptureFixture[str], *args: str) -> tuple[int, str]:
    code = main(["--env-file", NO_ENV, *args])
    return code, capsys.readouterr().out


def test_index_command(sample_repo_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code, out = run(capsys, "index", "--repo", str(sample_repo_path))
    assert code == 0
    assert "chunks" in out
    assert "files      :" in out


def test_grep_command(sample_repo_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code, out = run(capsys, "grep", "--repo", str(sample_repo_path), "average")
    assert code == 0
    assert "app/calculator.py:" in out


def test_read_command(sample_repo_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code, out = run(
        capsys, "read", "--repo", str(sample_repo_path), "app/storage.py", "--start", "1", "--end", "12"
    )
    assert code == 0
    assert "class JsonStorage" in out


def test_read_command_reports_error_exit_code(
    sample_repo_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code, out = run(capsys, "read", "--repo", str(sample_repo_path), "../escape.txt")
    assert code == 2
    assert out.startswith("ERROR:")


def test_search_command(sample_repo_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code, out = run(capsys, "search", "--repo", str(sample_repo_path), "json storage", "-k", "3")
    assert code == 0
    assert "score=" in out


def test_ask_offline_end_to_end(sample_repo_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code, out = run(
        capsys,
        "ask",
        "--repo",
        str(sample_repo_path),
        "Where is the average of results computed?",
        "--offline",
        "--json",
    )
    assert code == 0
    payload = json.loads(out)
    assert payload["answer"]["summary"]
    assert payload["tool_call_count"] >= 1
    assert payload["used_evidence"] is True
    assert payload["tool_calls"][0]["name"] == "search_code"


def test_ask_offline_markdown_output(
    sample_repo_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code, out = run(
        capsys,
        "ask",
        "--repo",
        str(sample_repo_path),
        "How are results persisted?",
        "--offline",
        "--show-trace",
    )
    assert code == 0
    assert "Summary:" in out
    assert "tool call(s)" in out
    assert "[1] search_code(" in out


# --------------------------------------------------------------------------- #
# provider error translation (no network involved)
# --------------------------------------------------------------------------- #
class AuthenticationError(Exception):
    """Mimics openai.AuthenticationError by class name."""


class APIConnectionError(Exception):
    """Mimics openai.APIConnectionError by class name."""


def test_auth_error_points_at_env_file() -> None:
    message = friendly_llm_error(AuthenticationError("Error code: 401 - invalid_api_key"))
    assert message is not None
    assert "LLM_API_KEY" in message
    assert "--offline" in message


def test_connection_error_points_at_endpoint() -> None:
    message = friendly_llm_error(APIConnectionError("Connection error."))
    assert message is not None
    assert "LLM_BASE_URL" in message


def test_unrecognised_error_is_not_swallowed() -> None:
    assert friendly_llm_error(ValueError("a real bug")) is None
    assert friendly_llm_error(KeyError("missing")) is None
