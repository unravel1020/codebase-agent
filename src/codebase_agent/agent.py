"""The tool-calling agent loop.

Deliberately explicit instead of a framework black box: the loop is
~60 lines and shows exactly what an "agent" is - prompt, tool choice, tool
execution, evidence accumulation, structured synthesis.

LangChain provides the primitives (``BaseChatModel.bind_tools``, ``@tool``,
``ToolMessage``); the control flow is ours so it stays readable, testable and
mockable without network access.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import BaseTool

from .config import Settings
from .memory import ConversationMemory
from .schemas import AnswerParseError, CodeAnswer, parse_answer

SYSTEM_PROMPT = """You are codebase-agent, an engineer that answers questions about ONE local repository.

Rules:
1. Never invent code. For any question about implementation details you MUST first
   gather repository evidence with the tools: `search_code` to locate a symbol,
   `read_file` to read it, `retrieve_context` for semantic/broad questions.
2. Prefer the smallest set of tool calls that proves your answer. Cite real
   file paths and line numbers.
3. If the repository does not contain the answer, say so and lower confidence.
4. Do not dump long file contents back to the user; summarize.
"""

SYNTHESIS_SYSTEM = """You convert an engineering investigation into a structured answer.

Return ONLY a JSON object with exactly these keys:
{
  "summary": string,                // 2-6 sentences, direct answer
  "relevant_files": [string],       // repository-relative paths
  "evidence": [{"file": string, "lines": string|null, "note": string, "source_tool": string|null}],
  "confidence": number,             // 0.0-1.0, based on evidence strength
  "assumptions": [string],
  "warnings": [string]
}

Rules:
- Evidence must come from the tool outputs below. Do not cite files you did not read.
- No hidden chain-of-thought: one short factual sentence per evidence item.
- If evidence is missing or partial, lower `confidence` and say why in `warnings`.
"""

EVIDENCE_NUDGE = """Stop. You answered without reading the repository.
Call at least one tool (`search_code`, `read_file` or `retrieve_context`) to gather
evidence, then answer. If the repository truly cannot answer the question, say so
explicitly and set confidence below 0.3.
"""

_MAX_STORED_OUTPUT = 4000


@dataclass
class ToolCallRecord:
    """One executed tool call (the agent's audit trail)."""

    name: str
    args: dict[str, Any]
    output: str
    ok: bool
    duration_ms: float
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "args": self.args,
            "ok": self.ok,
            "duration_ms": round(self.duration_ms, 2),
            "error": self.error,
            "output_preview": self.output[:500],
        }


@dataclass
class AgentResult:
    """Everything the caller (CLI, eval harness) needs."""

    question: str
    answer: CodeAnswer
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    iterations: int = 0
    latency_ms: float = 0.0
    messages: list[BaseMessage] = field(default_factory=list)

    @property
    def tool_call_count(self) -> int:
        return len(self.tool_calls)

    @property
    def used_evidence(self) -> bool:
        return any(record.ok for record in self.tool_calls)

    def to_dict(self, *, include_transcript: bool = False) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "question": self.question,
            "answer": self.answer.model_dump(),
            "tool_calls": [record.to_dict() for record in self.tool_calls],
            "tool_call_count": self.tool_call_count,
            "iterations": self.iterations,
            "latency_ms": round(self.latency_ms, 2),
            "used_evidence": self.used_evidence,
        }
        if include_transcript:
            payload["transcript"] = [
                {"type": message.type, "content": str(message.content)[:2000]}
                for message in self.messages
            ]
        return payload


class CodebaseAgent:
    """Tool-calling agent over one repository."""

    def __init__(
        self,
        llm: BaseChatModel,
        tools: list[BaseTool],
        *,
        memory: ConversationMemory | None = None,
        settings: Settings | None = None,
        system_prompt: str = SYSTEM_PROMPT,
        repo_name: str | None = None,
    ) -> None:
        if not tools:
            raise ValueError("at least one tool is required")
        self.settings = settings or Settings()
        self.llm = llm
        self.tools: dict[str, BaseTool] = {tool.name: tool for tool in tools}
        self.llm_with_tools = llm.bind_tools(tools)
        self.memory = memory if memory is not None else ConversationMemory(self.settings.memory_turns)
        self.system_prompt = system_prompt
        self.repo_name = repo_name

    # ------------------------------------------------------------------ #
    def run(self, question: str) -> AgentResult:
        """Answer one question, gathering repository evidence first."""
        if not question or not question.strip():
            raise ValueError("question must not be empty")

        started = time.perf_counter()
        messages: list[BaseMessage] = [SystemMessage(content=self.system_prompt)]
        messages.extend(self.memory.messages)
        messages.append(HumanMessage(content=question))

        records: list[ToolCallRecord] = []
        iterations = 0
        final: AIMessage | None = None

        for attempt in range(2):
            final, iterations = self._loop(messages, records, iterations)
            if records or not self.settings.require_evidence or final is None:
                break
            # No evidence at all: nudge once, then accept a low-confidence answer.
            messages.append(HumanMessage(content=EVIDENCE_NUDGE))

        answer = self._synthesize(question, messages, records, final)
        answer = self._apply_guards(answer, records)

        self.memory.add_turn(question, answer.summary)
        return AgentResult(
            question=question,
            answer=answer,
            tool_calls=records,
            iterations=iterations,
            latency_ms=(time.perf_counter() - started) * 1000.0,
            messages=messages,
        )

    # ------------------------------------------------------------------ #
    def _loop(
        self,
        messages: list[BaseMessage],
        records: list[ToolCallRecord],
        iterations: int,
    ) -> tuple[AIMessage | None, int]:
        """Run model -> tool -> model until the model answers or budgets run out."""
        final: AIMessage | None = None
        while iterations < self.settings.max_iterations:
            iterations += 1
            ai = self.llm_with_tools.invoke(messages)
            if not isinstance(ai, AIMessage):  # pragma: no cover - defensive
                ai = AIMessage(content=str(ai))
            messages.append(ai)

            if not ai.tool_calls:
                final = ai
                break
            if len(records) >= self.settings.max_tool_calls:
                budget_nudge = HumanMessage(
                    content=(
                        f"Tool budget reached ({self.settings.max_tool_calls} calls). "
                        "Answer now using the evidence you already have, without calling tools."
                    )
                )
                messages.append(budget_nudge)
                try:
                    # Deliberately use the unbound model: any tool calls returned here
                    # are a draft mistake and must never be executed or recorded.
                    forced = self.llm.invoke(messages)
                except Exception:
                    # Keep the existing synthesis/fallback path usable if the forced
                    # completion fails; the collected records remain authoritative.
                    final = ai
                    break
                if not isinstance(forced, AIMessage):  # pragma: no cover - defensive
                    forced = AIMessage(content=str(forced))
                messages.append(forced)
                final = forced
                break

            for call in ai.tool_calls:
                record, tool_message = self._execute(call)
                records.append(record)
                messages.append(tool_message)
        return final, iterations

    def _execute(self, call: dict[str, Any]) -> tuple[ToolCallRecord, ToolMessage]:
        name = str(call.get("name", ""))
        args = dict(call.get("args") or {})
        call_id = str(call.get("id") or f"call_{name}_{len(args)}")
        tool = self.tools.get(name)

        if tool is None:
            output = f"ERROR: unknown tool {name!r}. Available: {sorted(self.tools)}"
            record = ToolCallRecord(name, args, output, False, 0.0, "unknown tool")
            return record, ToolMessage(content=output, tool_call_id=call_id, name=name or "unknown")

        started = time.perf_counter()
        error: str | None = None
        try:
            output = str(tool.invoke(args))
        except Exception as exc:  # noqa: BLE001 - a tool crash must not kill the loop
            output = f"ERROR: tool {name} failed: {exc}"
            error = str(exc)
        duration_ms = (time.perf_counter() - started) * 1000.0
        ok = not output.startswith("ERROR:")
        if not ok:
            error = error or output.removeprefix("ERROR:").strip()

        record = ToolCallRecord(name, args, output, ok, duration_ms, error)
        return record, ToolMessage(
            content=output[:_MAX_STORED_OUTPUT], tool_call_id=call_id, name=name
        )

    # ------------------------------------------------------------------ #
    def _evidence_block(self, records: list[ToolCallRecord]) -> str:
        if not records:
            return ""
        blocks: list[str] = []
        for index, record in enumerate(records, start=1):
            args = json.dumps(record.args, ensure_ascii=False)
            blocks.append(
                f"### Tool {index}: {record.name}({args}) -> "
                f"{'ok' if record.ok else 'error'}\n{record.output[:3000]}"
            )
        return "\n\n".join(blocks)

    def _synthesize(
        self,
        question: str,
        messages: list[BaseMessage],
        records: list[ToolCallRecord],
        final: AIMessage | None = None,
    ) -> CodeAnswer:
        """Turn the investigation into the Pydantic structured answer."""
        evidence = self._evidence_block(records)
        draft = ""
        if final is not None:
            draft = final.content if isinstance(final.content, str) else str(final.content)
        repo_line = f"Repository: {self.repo_name}\n" if self.repo_name else ""
        prompt: list[BaseMessage] = [
            SystemMessage(content=SYNTHESIS_SYSTEM),
            HumanMessage(
                content=(
                    f"{repo_line}Question:\n{question}\n\n"
                    f"Repository evidence gathered by tools:\n"
                    f"{evidence or '(no tool calls were made - there is no evidence)'}\n\n"
                    f"Model's final draft (use only as a draft, not as evidence):\n"
                    f"{draft or '(none)'}\n\n"
                    "Return the JSON object now."
                )
            ),
        ]
        try:
            structured = self.llm.with_structured_output(
                CodeAnswer, method=self.settings.structured_output_method
            )
            result = structured.invoke(prompt)
            if isinstance(result, CodeAnswer):
                return result
            return CodeAnswer.model_validate(result)
        except Exception as exc:  # noqa: BLE001 - fall back to text parsing
            fallback = self.llm.invoke(prompt)
            text = fallback.content if isinstance(fallback.content, str) else str(fallback.content)
            try:
                answer = parse_answer(text)
            except AnswerParseError:
                answer = CodeAnswer(
                    summary=text.strip()[:1000] or "The model returned no usable answer.",
                    relevant_files=[],
                    evidence=[],
                    confidence=0.0,
                )
            answer.warnings.append(f"structured output unavailable, parsed text instead: {exc}")
            return answer

    @staticmethod
    def _apply_guards(answer: CodeAnswer, records: list[ToolCallRecord]) -> CodeAnswer:
        """Enforce 'no evidence -> low confidence'. Never trust a bare claim."""
        ok_records = [record for record in records if record.ok]
        if not ok_records:
            answer.warnings.append(
                "No repository evidence was collected; the answer is not verified against source."
            )
            answer.confidence = min(answer.confidence, 0.2)
        elif not answer.evidence:
            answer.warnings.append(
                "Tool calls succeeded but the answer cites no specific evidence."
            )
            answer.confidence = min(answer.confidence, 0.5)
        answer.confidence = max(0.0, min(1.0, answer.confidence))
        return answer
