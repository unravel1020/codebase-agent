"""Tests for Repository.search / the search_code tool."""

from __future__ import annotations

from codebase_agent import Repository
from codebase_agent.tools import build_tools


def test_search_finds_class_definition_first(repo: Repository) -> None:
    hits = repo.search("Calculator")
    assert hits, "expected at least one match for 'Calculator'"
    top = hits[0]
    assert top.is_definition
    assert top.rel_path == "app/calculator.py"
    assert top.text.startswith("class Calculator")


def test_search_finds_function_definition(repo: Repository) -> None:
    hits = repo.search("average")
    definitions = [hit for hit in hits if hit.is_definition]
    assert definitions, "expected the average() definition to be found"
    assert definitions[0].rel_path == "app/calculator.py"
    assert "def average" in definitions[0].text


def test_search_reports_line_numbers(repo: Repository) -> None:
    hits = repo.search("JsonStorage")
    assert hits
    for hit in hits:
        assert hit.line >= 1
        assert hit.text


def test_search_multi_token_query(repo: Repository) -> None:
    hits = repo.search("mean_result history")
    assert hits
    assert any(hit.rel_path == "app/service.py" for hit in hits)


def test_search_respects_max_results(repo: Repository) -> None:
    hits = repo.search("the", max_results=3)
    assert len(hits) <= 3


def test_search_no_match(repo: Repository) -> None:
    assert repo.search("zzz_definitely_not_present_zzz") == []


def test_search_empty_query(repo: Repository) -> None:
    assert repo.search("   ") == []


def test_search_regex_mode(repo: Repository) -> None:
    hits = repo.search(r"def\s+divide", regex=True)
    assert hits
    # Regex mode is exact: no token fallback, so every hit really matches.
    assert all("divide" in hit.text for hit in hits)
    assert hits[0].is_definition


def test_search_skips_binary_files(repo: Repository) -> None:
    """data/blob.bin contains SECRET_BINARY_MARKER but must never be searched."""
    assert repo.search("SECRET_BINARY_MARKER") == []


def test_binary_suffix_files_are_not_indexed(repo: Repository) -> None:
    indexed = {source.rel_path for source in repo.iter_source_files()}
    assert "data/blob.bin" not in indexed
    assert "data/cache.pyc" not in indexed
    assert "app/calculator.py" in indexed


def test_search_skips_excluded_dirs(tmp_path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "keep.py").write_text("MARKER_KEEP = 1\n", encoding="utf-8")
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "dep.py").write_text("MARKER_KEEP = 2\n", encoding="utf-8")
    repo = Repository(tmp_path)
    hits = repo.search("MARKER_KEEP")
    assert [hit.rel_path for hit in hits] == ["src/keep.py"]


def test_search_code_tool_output_format(repo: Repository) -> None:
    tools = {tool.name: tool for tool in build_tools(repo)}
    output = tools["search_code"].invoke({"query": "CalculatorService", "max_results": 5})
    assert "match(es) for" in output
    assert "app/service.py:" in output
    assert "[definition]" in output


def test_search_code_tool_no_match(repo: Repository) -> None:
    tools = {tool.name: tool for tool in build_tools(repo)}
    output = tools["search_code"].invoke({"query": "zzz_definitely_not_present_zzz"})
    assert output.startswith("No matches")
