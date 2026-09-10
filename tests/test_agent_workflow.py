"""End-to-end agent loop tests with a scripted model.

No network, no API key: :class:`ScriptedChatModel` replays tool-call decisions,
while the real tools, real repository, real RAG index and the real agent loop
run unchanged.
"""

from __future__ import annotations

import json

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from codebase_agent import (
    CodebaseAgent,
    ConversationMemory,
    Repository,
    ScriptedChatModel,
    Settings,
    tool_call,
)
from codebase_agent.agent import EVIDENCE_NUDGE
from codebase_agent.tools import build_tools

FINAL_JSON = {
    "summary": "The average is computed by average() in app/calculator.py and used by CalculatorService.mean_result().",
    "relevant_files": ["app/calculator.py", "app/service.py"],
    "evidence": [
        {"file": "app/calculator.py", "lines": "26-30", "note": "defines average()"},
        {"file": "app/service.py", "lines": "44-46", "note": "calls average()"},
    ],
    "confidence": 0.85,
}


def make_agent(
    repo: Repository,
    retriever,
    script: list,
    *,
    settings: Settings | None = None,
    memory: ConversationMemory | None = None,
    **model_kwargs,
) -> tuple[CodebaseAgent, ScriptedChatModel]:
    llm = ScriptedChatModel(script=script, **model_kwargs)
    agent = CodebaseAgent(
        llm,
        build_tools(repo, retriever),
        memory=memory,
        settings=settings or Settings(),
        repo_name=repo.root.name,
    )
    return agent, llm


def test_agent_search_then_read_then_answer(repo: Repository, retriever) -> None:
    agent, llm = make_agent(
        repo,
        retriever,
        [
            tool_call("search_code", {"query": "average"}),
            tool_call("read_file", {"path": "app/calculator.py", "start_line": 1, "end_line": 40}),
            AIMessage(content=json.dumps(FINAL_JSON)),
        ],
    )
    result = agent.run("How is the average of results computed?")

    assert [record.name for record in result.tool_calls] == ["search_code", "read_file"]
    assert all(record.ok for record in result.tool_calls)
    assert result.used_evidence is True
    assert result.iterations == 3
    assert result.answer.confidence == pytest.approx(0.85)
    assert result.answer.relevant_files == ["app/calculator.py", "app/service.py"]
    assert not any("No repository evidence" in warning for warning in result.answer.warnings)

    # The model really saw the tool output.
    tool_messages = [m for m in llm.calls[-1] if isinstance(m, ToolMessage)]
    assert tool_messages and "class Calculator" in str(tool_messages[-1].content)


def test_retrieve_context_tool_is_usable(repo: Repository, retriever) -> None:
    agent, _ = make_agent(
        repo,
        retriever,
        [
            tool_call("retrieve_context", {"query": "where are results persisted", "k": 3}),
            AIMessage(content=json.dumps(FINAL_JSON)),
        ],
    )
    result = agent.run("Where are results persisted?")
    assert result.tool_calls[0].name == "retrieve_context"
    assert result.tool_calls[0].ok is True
    assert "app/storage.py" in result.tool_calls[0].output or "app/service.py" in (
        result.tool_calls[0].output
    )


def test_agent_is_nudged_when_it_answers_without_evidence(
    repo: Repository, retriever
) -> None:
    agent, llm = make_agent(
        repo,
        retriever,
        [
            AIMessage(content=json.dumps(FINAL_JSON)),  # lazy: no tool call
            tool_call("search_code", {"query": "average"}),
            AIMessage(content=json.dumps(FINAL_JSON)),
        ],
    )
    result = agent.run("How is the average computed?")

    assert llm.invocation_count == 3
    second_call = llm.calls[1]
    assert any(isinstance(m, HumanMessage) and m.content == EVIDENCE_NUDGE for m in second_call)
    assert result.tool_call_count == 1
    assert result.used_evidence is True


def test_confidence_is_clamped_without_evidence(repo: Repository, retriever) -> None:
    agent, _ = make_agent(
        repo,
        retriever,
        [AIMessage(content=json.dumps({**FINAL_JSON, "confidence": 0.99}))],
        settings=Settings(require_evidence=False),
    )
    result = agent.run("What does the code do?")

    assert result.tool_call_count == 0
    assert result.answer.confidence == pytest.approx(0.2)
    assert any("No repository evidence" in warning for warning in result.answer.warnings)


def test_confidence_is_capped_when_evidence_list_is_empty(
    repo: Repository, retriever
) -> None:
    payload = {**FINAL_JSON, "evidence": [], "confidence": 0.95}
    agent, _ = make_agent(
        repo,
        retriever,
        [
            tool_call("search_code", {"query": "average"}),
            AIMessage(content=json.dumps(payload)),
        ],
    )
    result = agent.run("How is the average computed?")
    assert result.answer.confidence == pytest.approx(0.5)
    assert any("cites no specific evidence" in warning for warning in result.answer.warnings)


def test_tool_budget_is_enforced(repo: Repository, retriever) -> None:
    agent, _ = make_agent(
        repo,
        retriever,
        [
            tool_call("search_code", {"query": "average"}),
            tool_call("search_code", {"query": "Calculator"}),
            AIMessage(content=json.dumps(FINAL_JSON)),
        ],
        settings=Settings(max_tool_calls=1),
    )
    result = agent.run("How is the average computed?")

    assert result.tool_call_count == 1
    assert any("Tool budget reached" in str(m.content) for m in result.messages)


def test_unknown_tool_is_reported_not_fatal(repo: Repository, retriever) -> None:
    agent, _ = make_agent(
        repo,
        retriever,
        [
            tool_call("delete_everything", {"path": "/"}),
            AIMessage(content=json.dumps(FINAL_JSON)),
        ],
    )
    result = agent.run("Do something dangerous")

    assert result.tool_calls[0].ok is False
    assert "unknown tool" in result.tool_calls[0].output
    assert result.answer.summary  # the loop still produced an answer


def test_failing_tool_is_reported_and_loop_continues(repo: Repository, retriever) -> None:
    agent, _ = make_agent(
        repo,
        retriever,
        [
            tool_call("read_file", {"path": "../escape.txt"}),
            tool_call("search_code", {"query": "Calculator"}),
            AIMessage(content=json.dumps(FINAL_JSON)),
        ],
    )
    result = agent.run("Read a file outside the repository")

    assert result.tool_calls[0].ok is False
    assert result.tool_calls[0].output.startswith("ERROR:")
    assert result.tool_calls[1].ok is True
    assert result.used_evidence is True


def test_structured_output_falls_back_to_text_parsing(
    repo: Repository, retriever
) -> None:
    fenced = f"```json\n{json.dumps(FINAL_JSON)}\n```"
    agent, _ = make_agent(
        repo,
        retriever,
        [
            tool_call("search_code", {"query": "average"}),
            AIMessage(content="done"),
            AIMessage(content=fenced),
        ],
        supports_structured_output=False,
    )
    result = agent.run("How is the average computed?")

    assert result.answer.relevant_files == ["app/calculator.py", "app/service.py"]
    assert any("structured output unavailable" in warning for warning in result.answer.warnings)


def test_unparseable_answer_degrades_gracefully(repo: Repository, retriever) -> None:
    agent, _ = make_agent(
        repo,
        retriever,
        [AIMessage(content="no tool call"), AIMessage(content="still no json")],
        settings=Settings(require_evidence=False),
        supports_structured_output=False,
    )
    result = agent.run("Anything")

    assert result.answer.confidence == pytest.approx(0.0)
    assert "still no json" in result.answer.summary


def test_structured_output_accepts_plain_dict(repo: Repository, retriever) -> None:
    agent, _ = make_agent(
        repo,
        retriever,
        [
            tool_call("search_code", {"query": "average"}),
            AIMessage(content="done"),
        ],
        structured_response=FINAL_JSON,
    )
    result = agent.run("How is the average computed?")
    assert result.answer.confidence == pytest.approx(0.85)


def test_memory_carries_context_between_questions(repo: Repository, retriever) -> None:
    memory = ConversationMemory(max_turns=3)

    first_agent, _ = make_agent(
        repo,
        retriever,
        [
            tool_call("search_code", {"query": "Calculator"}),
            AIMessage(content=json.dumps(FINAL_JSON)),
        ],
        memory=memory,
    )
    first_agent.run("Where is Calculator defined?")

    second_agent, second_llm = make_agent(
        repo,
        retriever,
        [
            tool_call("read_file", {"path": "app/calculator.py"}),
            AIMessage(content=json.dumps(FINAL_JSON)),
        ],
        memory=memory,
    )
    second_agent.run("Show me that file.")

    history = second_llm.calls[0]
    assert isinstance(history[0], SystemMessage)
    assert any(
        isinstance(m, HumanMessage) and "Where is Calculator defined?" in m.content
        for m in history
    )
    assert any(isinstance(m, ToolMessage) is False and "Show me that file" in str(m.content) for m in history)
    assert len(memory) == 4


def test_empty_question_is_rejected(repo: Repository, retriever) -> None:
    agent, _ = make_agent(repo, retriever, [])
    with pytest.raises(ValueError):
        agent.run("   ")


def test_agent_requires_at_least_one_tool(repo: Repository) -> None:
    with pytest.raises(ValueError):
        CodebaseAgent(ScriptedChatModel(script=[]), [])
