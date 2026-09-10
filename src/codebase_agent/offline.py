"""Deterministic offline doubles.

These exist so the whole pipeline - agent loop, tool calling, RAG, structured
output, evals - can run end to end with **no API key and no network**. They are
used by ``pytest`` and by ``evals/run_evals.py --mode mock-agent``.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Sequence
from typing import Any

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import RunnableLambda
from pydantic import ConfigDict, Field, PrivateAttr


def tool_call(name: str, args: dict[str, Any], call_id: str | None = None) -> AIMessage:
    """Build an assistant message that requests one tool call."""
    return AIMessage(
        content="",
        tool_calls=[{"name": name, "args": args, "id": call_id or f"call_{name}"}],
    )


class ScriptedChatModel(BaseChatModel):
    """A chat model that replays a script instead of calling an API.

    ``script`` is a list of :class:`AIMessage` (or plain strings) returned in
    order; ``policy`` is a callable ``messages -> AIMessage`` for state-dependent
    behaviour. ``bind_tools`` returns the model itself, so the real tool-calling
    code path in :mod:`codebase_agent.agent` is exercised unchanged.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    script: list[Any] = Field(default_factory=list)
    policy: Callable[[list[BaseMessage]], Any] | None = None
    structured_response: Any = None
    supports_structured_output: bool = True

    _cursor: int = PrivateAttr(default=0)
    _calls: list[list[BaseMessage]] = PrivateAttr(default_factory=list)
    _last_message: AIMessage | None = PrivateAttr(default=None)

    @property
    def _llm_type(self) -> str:
        return "scripted-chat-model"

    @property
    def calls(self) -> list[list[BaseMessage]]:
        """Every message list this model was invoked with, in order."""
        return [list(messages) for messages in self._calls]

    @property
    def invocation_count(self) -> int:
        return len(self._calls)

    def _next_message(self, messages: Sequence[BaseMessage]) -> Any:
        if self.policy is not None:
            return self.policy(list(messages))
        if self._cursor < len(self.script):
            message = self.script[self._cursor]
            self._cursor += 1
            return message
        raise RuntimeError(
            "ScriptedChatModel script exhausted: add more responses to `script` "
            "or provide a `policy`."
        )

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        self._calls.append(list(messages))
        message = self._next_message(messages)
        if isinstance(message, str):
            message = AIMessage(content=message)
        if not isinstance(message, AIMessage):
            raise TypeError(f"script entries must be AIMessage or str, got {type(message)!r}")
        self._last_message = message
        return ChatResult(generations=[ChatGeneration(message=message)])

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> ScriptedChatModel:
        """Ignore the tool list; the script already encodes the decisions."""
        return self

    def with_structured_output(self, schema: Any = None, **kwargs: Any) -> RunnableLambda:
        if not self.supports_structured_output:
            raise NotImplementedError("this fake model does not support structured output")

        def _respond(_input: Any) -> Any:
            payload = self.structured_response
            if payload is None:
                last = self._last_message or (
                    self.script[-1] if self.script else AIMessage(content="")
                )
                text = last.content if isinstance(last, AIMessage) else str(last)
                from .schemas import parse_answer

                return parse_answer(text)
            if schema is not None and hasattr(schema, "model_validate"):
                return schema.model_validate(payload)
            return payload

        return RunnableLambda(_respond)


_STOPWORDS = frozenset(
    {
        "the",
        "and",
        "for",
        "with",
        "what",
        "where",
        "which",
        "how",
        "does",
        "do",
        "is",
        "are",
        "was",
        "were",
        "this",
        "that",
        "these",
        "those",
        "from",
        "into",
        "about",
        "code",
        "file",
        "files",
        "used",
        "use",
        "uses",
        "called",
        "call",
        "implement",
        "implemented",
        "implementation",
        "defined",
        "define",
        "function",
        "method",
        "class",
        "repository",
        "repo",
        "project",
        "please",
        "tell",
        "explain",
        "when",
        "why",
        "who",
        "its",
        "it",
        "they",
        "them",
        "there",
        "here",
        "any",
        "all",
        "can",
        "you",
        "your",
    }
)

_HIT_RE = re.compile(r"^(?P<file>[^\s:][^:]*):(?P<line>\d+):\s*(?P<text>.*)$")


def keywords(text: str, limit: int = 6) -> list[str]:
    """Cheap keyword extraction used by the offline policy."""
    found: list[str] = []
    for token in re.findall(r"[A-Za-z_][A-Za-z0-9_]{2,}", text or ""):
        if token.lower() in _STOPWORDS:
            continue
        if token not in found:
            found.append(token)
        if len(found) >= limit:
            break
    return found


class HeuristicPolicy:
    """Deterministic stand-in for a tool-calling model.

    State machine over the message list: ``search_code`` -> ``read_file`` ->
    final structured JSON built from whatever the tools actually returned. This
    lets the real agent loop, tools and RAG run with no API key.
    """

    def __init__(self, max_files: int = 3) -> None:
        self.max_files = max_files
        self._question: str | None = None
        self._searched = False
        self._read = False
        self._hits: list[tuple[str, int, str]] = []
        self._read_files: list[str] = []

    # ------------------------------------------------------------------ #
    def _reset(self, question: str) -> None:
        self._question = question
        self._searched = False
        self._read = False
        self._hits = []
        self._read_files = []

    @staticmethod
    def _last_human(messages: list[BaseMessage]) -> str | None:
        for message in reversed(messages):
            if isinstance(message, HumanMessage) and str(message.content).strip():
                return str(message.content)
        return None

    def __call__(self, messages: list[BaseMessage]) -> AIMessage:
        current = self._last_human(messages)
        has_tool_output = any(isinstance(message, ToolMessage) for message in messages)
        # A new question (no tool output yet in this window) starts a new run.
        if current is not None and current != self._question and not has_tool_output:
            self._reset(current)
        elif self._question is None:
            self._question = current or ""

        self._refresh_state(messages)
        query = self._question or ""

        if not self._searched:
            self._searched = True
            terms = keywords(query)
            return tool_call("search_code", {"query": " ".join(terms) or query})

        if not self._read and self._hits:
            self._read = True
            file_path, line, _ = self._hits[0]
            return tool_call(
                "read_file",
                {
                    "path": file_path,
                    "start_line": max(1, line - 5),
                    "end_line": line + 45,
                },
            )

        return AIMessage(content=json.dumps(self._final_payload(query), ensure_ascii=False))

    # ------------------------------------------------------------------ #
    def _refresh_state(self, messages: list[BaseMessage]) -> None:
        if self._question is None:
            self._question = self._last_human(messages) or ""
        for message in messages:
            if not isinstance(message, ToolMessage):
                continue
            content = str(message.content)
            if message.name == "read_file":
                first_line = content.splitlines()[0] if content else ""
                file_path = first_line.split(" (lines")[0].strip()
                if file_path and file_path not in self._read_files:
                    self._read_files.append(file_path)
                continue
            for raw in content.splitlines():
                match = _HIT_RE.match(raw.strip())
                if match:
                    entry = (
                        match.group("file"),
                        int(match.group("line")),
                        match.group("text").strip(),
                    )
                    if entry not in self._hits:
                        self._hits.append(entry)

    def _final_payload(self, query: str) -> dict[str, Any]:
        files: list[str] = []
        for file_path in [hit[0] for hit in self._hits] + self._read_files:
            if file_path not in files:
                files.append(file_path)
        files = files[: self.max_files]

        evidence = [
            {
                "file": file_path,
                "lines": str(line),
                "note": f"matched {query!r}: {text[:120]}",
                "source_tool": "search_code",
            }
            for file_path, line, text in self._hits[:5]
        ]
        if not evidence and self._read_files:
            evidence = [
                {
                    "file": self._read_files[0],
                    "lines": None,
                    "note": "file was read to inspect the implementation",
                    "source_tool": "read_file",
                }
            ]

        if self._hits:
            file_path, line, text = self._hits[0]
            summary = (
                f"The most relevant match for {query!r} is {file_path}:{line} "
                f"(`{text[:160]}`). {len(files)} file(s) were involved."
            )
            confidence = 0.55
        elif self._read_files:
            summary = (
                f"No symbol matched {query!r}, but {self._read_files[0]} was read for context."
            )
            confidence = 0.3
        else:
            summary = f"No repository evidence was found for {query!r}."
            confidence = 0.1

        return {
            "summary": summary,
            "relevant_files": files,
            "evidence": evidence,
            "confidence": confidence,
            "assumptions": ["Offline heuristic policy: no LLM was called."],
            "warnings": [],
        }


def build_offline_llm(**kwargs: Any) -> ScriptedChatModel:
    """A :class:`ScriptedChatModel` driven by :class:`HeuristicPolicy`."""
    return ScriptedChatModel(policy=HeuristicPolicy())
