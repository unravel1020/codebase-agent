"""Path-traversal and sandbox tests.

These are the security-critical tests: no tool call may ever read outside the
repository root, whatever the model asks for.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from codebase_agent import PathSecurityError, Repository, RepositoryError
from codebase_agent.tools import build_tools


@pytest.fixture
def sandbox(tmp_path: Path) -> Repository:
    root = tmp_path / "repo"
    root.mkdir()
    (root / "inside.txt").write_text("inside\n", encoding="utf-8")
    (tmp_path / "outside.txt").write_text("OUTSIDE_SECRET_CONTENT\n", encoding="utf-8")
    return Repository(root)


@pytest.mark.parametrize(
    "attempt",
    [
        "../outside.txt",
        "sub/../../outside.txt",
        "./../outside.txt",
        pytest.param(
            "..\\outside.txt",
            marks=pytest.mark.skipif(
                os.name != "nt",
                reason="a backslash is an ordinary filename character on POSIX, not a separator",
            ),
        ),
    ],
)
def test_relative_traversal_is_blocked(sandbox: Repository, attempt: str) -> None:
    with pytest.raises(PathSecurityError):
        sandbox.resolve(attempt)


def test_absolute_path_outside_root_is_blocked(sandbox: Repository, tmp_path: Path) -> None:
    with pytest.raises(PathSecurityError):
        sandbox.resolve(str(tmp_path / "outside.txt"))


@pytest.mark.skipif(os.name == "nt", reason="covered by the Windows case above")
def test_backslash_is_a_plain_filename_on_posix(sandbox: Repository) -> None:
    """A Windows-style traversal string cannot escape a POSIX sandbox.

    On POSIX a backslash is not a separator, so ``..\\outside.txt`` is a single
    filename *inside* the root. It resolves inside the sandbox (and then simply
    does not exist), which is safe - the sandbox never needs to reject it, and the
    tool must still never return the file that really lives outside the root.
    """
    resolved = sandbox.resolve("..\\outside.txt")
    assert resolved.parent == sandbox.root

    tools = {tool.name: tool for tool in build_tools(sandbox)}
    output = tools["read_file"].invoke({"path": "..\\outside.txt"})
    assert output.startswith("ERROR:")
    assert "OUTSIDE_SECRET_CONTENT" not in output


def test_nul_byte_is_blocked(sandbox: Repository) -> None:
    with pytest.raises(PathSecurityError):
        sandbox.resolve("inside.txt\x00.png")


def test_empty_path_is_blocked(sandbox: Repository) -> None:
    with pytest.raises(PathSecurityError):
        sandbox.resolve("   ")


def test_root_itself_resolves_but_is_not_a_file(sandbox: Repository) -> None:
    resolved = sandbox.resolve(".")
    assert resolved == sandbox.root
    with pytest.raises(RepositoryError):
        sandbox.read_text(".")


def test_inside_path_is_allowed(sandbox: Repository) -> None:
    resolved = sandbox.resolve("inside.txt")
    assert resolved.parent == sandbox.root
    assert sandbox.to_rel_path(resolved) == "inside.txt"


def test_absolute_path_inside_root_is_allowed(sandbox: Repository) -> None:
    resolved = sandbox.resolve(str(sandbox.root / "inside.txt"))
    assert resolved == sandbox.root / "inside.txt"


def test_symlink_escape_is_blocked(sandbox: Repository, tmp_path: Path) -> None:
    link = sandbox.root / "escape.txt"
    try:
        os.symlink(tmp_path / "outside.txt", link)
    except (OSError, NotImplementedError):  # pragma: no cover - needs privileges
        pytest.skip("symlink creation not permitted on this platform")
    with pytest.raises(PathSecurityError):
        sandbox.resolve("escape.txt")


def test_read_file_tool_blocks_traversal(sandbox: Repository) -> None:
    tools = {tool.name: tool for tool in build_tools(sandbox)}
    for attempt in ("../outside.txt", "..\\outside.txt", "/etc/passwd", "C:\\Windows\\win.ini"):
        output = tools["read_file"].invoke({"path": attempt})
        assert output.startswith("ERROR:"), attempt
        # The file outside the root must never be read.
        assert "OUTSIDE_SECRET_CONTENT" not in output


def test_repository_rejects_non_directory(tmp_path: Path) -> None:
    file_path = tmp_path / "not_a_dir.txt"
    file_path.write_text("x", encoding="utf-8")
    with pytest.raises(Exception):
        Repository(file_path)
