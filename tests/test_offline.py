"""Tests for the deterministic offline doubles used by tests, evals and `ask --offline`."""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from codebase_agent import CodebaseAgent, Repository, ScriptedChatModel, Settings, build_offline_llm
from codebase_agent.offline import HeuristicPolicy, keywords, tool_call
from codebase_agent.tools import build_tools


def test_keywords_drop_stopwords() -> None:
    assert keywords("Where is the average of results computed?") == [
        "average",
        "results",
        "computed",
    ]


def test_scripted_model_replays_script() -> None:
    model = ScriptedChatModel(script=[AIMessage(content="one"), AIMessage(content="two")])
    assert model.invoke([HumanMessage(content="q")]).content == "one"
    assert model.invoke([HumanMessage(content="q")]).content == "two"
    assert model.invocation_count == 2


def test_scripted_model_records_tool_messages() -> None:
    model = ScriptedChatModel(script=[AIMessage(content="done")])
    model.invoke([ToolMessage(content="output", tool_call_id="1")])
    assert isinstance(model.calls[0][0], ToolMessage)


def test_policy_calls_search_then_read() -> None:
    policy = HeuristicPolicy()
    first = policy([HumanMessage(content="where is average defined")])
    assert first.tool_calls[0]["name"] == "search_code"
    second = policy(
        [
            HumanMessage(content="where is average defined"),
            first,
            ToolMessage(content="app/calculator.py:28: def average(x):", tool_call_id="1"),
        ]
    )
    assert second.tool_calls[0]["name"] == "read_file"
    assert second.tool_calls[0]["args"]["path"] == "app/calculator.py"


def test_policy_resets_between_questions(repo: Repository, retriever) -> None:
    """A second question must search again instead of reusing stale state."""
    agent = CodebaseAgent(
        build_offline_llm(),
        build_tools(repo, retriever),
        settings=Settings(require_evidence=False),
    )
    first = agent.run("Where is average defined?")
    second = agent.run("Where is JsonStorage defined?")

    assert first.tool_call_count == 2
    assert second.tool_call_count == 2
    assert "storage.py" in second.tool_calls[0].output
    assert second.tool_calls[0].args["query"] != first.tool_calls[0].args["query"]


def test_offline_agent_end_to_end(repo: Repository, retriever) -> None:
    agent = CodebaseAgent(
        build_offline_llm(), build_tools(repo, retriever), settings=Settings()
    )
    result = agent.run("How are calculation results persisted to disk?")

    assert result.used_evidence is True
    assert result.answer.relevant_files
    assert result.answer.confidence > 0.2
    assert any(
        record.name == "search_code" and record.ok for record in result.tool_calls
    )


def test_tool_call_helper_builds_ai_message() -> None:
    message = tool_call("read_file", {"path": "a.py"}, "custom-id")
    call = message.tool_calls[0]
    assert call["name"] == "read_file"
    assert call["args"] == {"path": "a.py"}
    assert call["id"] == "custom-id"
    assert isinstance(message, AIMessage)
