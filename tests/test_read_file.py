"""Tests for Repository.read_text / the read_file tool."""

from __future__ import annotations

from pathlib import Path

import pytest

from codebase_agent import BinaryFileError, FileTooLargeError, Repository, RepositoryError
from codebase_agent.tools import build_tools


def test_read_whole_file(repo: Repository) -> None:
    slice_ = repo.read_text("app/calculator.py")
    assert slice_.rel_path == "app/calculator.py"
    assert slice_.start_line == 1
    assert slice_.end_line == slice_.total_lines
    assert "class Calculator" in slice_.text
    assert not slice_.truncated


def test_read_file_numbered_output(repo: Repository) -> None:
    slice_ = repo.read_text("app/calculator.py", 1, 5)
    rendered = slice_.numbered()
    assert slice_.end_line == 5
    assert rendered.startswith("1 | ")
    assert "from __future__ import annotations" in rendered
    assert "class Calculator" not in rendered, "window must stop at line 5"


def test_read_line_window(repo: Repository) -> None:
    full = repo.read_text("app/calculator.py")
    window = repo.read_text("app/calculator.py", start_line=3, end_line=6)
    assert window.start_line == 3
    assert window.end_line == 6
    assert window.text == "\n".join(full.text.splitlines()[2:6])


def test_read_respects_line_budget(sample_repo_path: Path) -> None:
    repo = Repository(sample_repo_path, max_read_lines=4)
    slice_ = repo.read_text("app/calculator.py")
    assert slice_.truncated is True
    assert slice_.end_line == 4


def test_read_missing_file(repo: Repository) -> None:
    with pytest.raises(RepositoryError):
        repo.read_text("app/does_not_exist.py")


def test_read_directory_is_rejected(repo: Repository) -> None:
    with pytest.raises(RepositoryError):
        repo.read_text("app")


def test_read_binary_suffix_is_rejected(repo: Repository) -> None:
    with pytest.raises(BinaryFileError):
        repo.read_text("data/blob.bin")


def test_read_binary_content_is_rejected(tmp_path: Path) -> None:
    (tmp_path / "weird.md").write_bytes(b"text\x00\x01\x02more")
    repo = Repository(tmp_path)
    with pytest.raises(BinaryFileError):
        repo.read_text("weird.md")


def test_read_oversized_file_is_rejected(tmp_path: Path) -> None:
    (tmp_path / "big.py").write_text("x = 1\n" * 5000, encoding="utf-8")
    repo = Repository(tmp_path, max_file_bytes=1000)
    with pytest.raises(FileTooLargeError):
        repo.read_text("big.py")


def test_invalid_line_range(repo: Repository) -> None:
    with pytest.raises(ValueError):
        repo.read_text("app/calculator.py", start_line=10, end_line=3)


def test_read_file_tool_returns_error_string(repo: Repository) -> None:
    """Tool failures are strings, not exceptions, so the agent loop survives."""
    tools = {tool.name: tool for tool in build_tools(repo)}
    output = tools["read_file"].invoke({"path": "app/nope.py"})
    assert output.startswith("ERROR:")
    assert "nope.py" in output


def test_read_file_tool_happy_path(repo: Repository) -> None:
    tools = {tool.name: tool for tool in build_tools(repo)}
    output = tools["read_file"].invoke({"path": "app/storage.py", "start_line": 1, "end_line": 12})
    assert output.startswith("app/storage.py (lines 1-12")
    assert "class JsonStorage" in output
