"""Minimal conversation memory.

A sliding window of the last N turns. Deliberately not a vector store, not a
summary chain, not a graph: multi-turn continuity is all this project needs.
"""

from __future__ import annotations

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage


class ConversationMemory:
    """Sliding-window chat history."""

    def __init__(self, max_turns: int = 6) -> None:
        if max_turns < 1:
            raise ValueError("max_turns must be >= 1")
        self.max_turns = max_turns
        self._messages: list[BaseMessage] = []

    def __len__(self) -> int:
        return len(self._messages)

    @property
    def messages(self) -> list[BaseMessage]:
        """Copy of the current window (safe to hand to a model)."""
        return list(self._messages)

    def add_user(self, text: str) -> None:
        self._append(HumanMessage(content=text))

    def add_assistant(self, text: str) -> None:
        self._append(AIMessage(content=text))

    def add_turn(self, question: str, answer_summary: str) -> None:
        """Record a completed turn (what the next question needs to know)."""
        self._append(HumanMessage(content=question))
        self._append(AIMessage(content=answer_summary))

    def _append(self, message: BaseMessage) -> None:
        self._messages.append(message)
        # Keep at most ``max_turns`` user/assistant pairs.
        limit = self.max_turns * 2
        if len(self._messages) > limit:
            self._messages = self._messages[-limit:]

    def clear(self) -> None:
        self._messages.clear()

    def transcript(self) -> str:
        """Plain-text rendering (used in the CLI ``--show-memory`` view)."""
        lines: list[str] = []
        for message in self._messages:
            role = "user" if isinstance(message, HumanMessage) else "assistant"
            lines.append(f"{role}: {message.content}")
        return "\n".join(lines)
