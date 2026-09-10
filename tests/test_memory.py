"""Tests for the sliding-window conversation memory."""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from codebase_agent import ConversationMemory


def test_records_turns_in_order() -> None:
    memory = ConversationMemory(max_turns=3)
    memory.add_user("q1")
    memory.add_assistant("a1")
    messages = memory.messages
    assert [type(m) for m in messages] == [HumanMessage, AIMessage]
    assert messages[0].content == "q1"
    assert messages[1].content == "a1"


def test_sliding_window_drops_oldest_turns() -> None:
    memory = ConversationMemory(max_turns=2)
    for index in range(4):
        memory.add_turn(f"q{index}", f"a{index}")
    contents = [m.content for m in memory.messages]
    assert contents == ["q2", "a2", "q3", "a3"]
    assert len(memory) == 4


def test_messages_property_returns_a_copy() -> None:
    memory = ConversationMemory()
    memory.add_user("q")
    snapshot = memory.messages
    snapshot.append(HumanMessage(content="injected"))
    assert len(memory) == 1


def test_clear_and_transcript() -> None:
    memory = ConversationMemory()
    memory.add_turn("q", "a")
    assert memory.transcript() == "user: q\nassistant: a"
    memory.clear()
    assert memory.transcript() == ""
    assert len(memory) == 0


def test_invalid_window_size() -> None:
    with pytest.raises(ValueError):
        ConversationMemory(max_turns=0)
