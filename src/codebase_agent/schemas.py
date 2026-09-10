"""Pydantic schemas used for structured output.

The agent never asks the model for hidden chain-of-thought. It asks for a short,
checkable answer: a summary, the files that matter, the evidence that supports
them, and a confidence score.
"""

from __future__ import annotations

import json
import re
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

_FENCE_RE = re.compile(r"```(?:json|JSON)?\s*(.*?)```", re.DOTALL)


class AnswerParseError(ValueError):
    """Raised when a model response cannot be parsed into :class:`CodeAnswer`."""


class EvidenceItem(BaseModel):
    """One piece of repository evidence backing the answer."""

    model_config = ConfigDict(extra="ignore")

    file: str = Field(description="Repository-relative path, POSIX separators.")
    lines: str | None = Field(
        default=None,
        description="Line range such as '42-67', or a single line '42'. Null if unknown.",
    )
    note: str = Field(description="One sentence: what this evidence shows.")
    source_tool: str | None = Field(
        default=None, description="Tool that produced the evidence, e.g. read_file."
    )

    @field_validator("file")
    @classmethod
    def _normalize_file(cls, value: str) -> str:
        cleaned = value.strip().replace("\\", "/")
        return cleaned[2:] if cleaned.startswith("./") else cleaned


class CodeAnswer(BaseModel):
    """Structured answer returned by the agent."""

    model_config = ConfigDict(extra="ignore")

    summary: str = Field(description="Concise answer to the question (2-6 sentences).")
    relevant_files: list[str] = Field(
        default_factory=list,
        description="Repository-relative paths that matter for the answer.",
    )
    evidence: list[EvidenceItem] = Field(
        default_factory=list,
        description="Short, checkable evidence. No hidden reasoning traces.",
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="0 = pure guess, 1 = fully verified from repository evidence.",
    )
    assumptions: list[str] = Field(
        default_factory=list, description="Anything assumed but not verified."
    )
    warnings: list[str] = Field(
        default_factory=list,
        description="Limits of the answer, e.g. evidence was missing or truncated.",
    )

    @field_validator("relevant_files")
    @classmethod
    def _normalize_files(cls, values: list[str]) -> list[str]:
        seen: list[str] = []
        for value in values:
            cleaned = value.strip().replace("\\", "/")
            if cleaned.startswith("./"):
                cleaned = cleaned[2:]
            if cleaned and cleaned not in seen:
                seen.append(cleaned)
        return seen

    @field_validator("summary")
    @classmethod
    def _strip_summary(cls, value: str) -> str:
        return value.strip()

    def to_markdown(self) -> str:
        """Human-readable rendering used by the CLI."""
        lines = [f"Summary: {self.summary}", "", f"Confidence: {self.confidence:.2f}"]
        if self.relevant_files:
            lines += ["", "Relevant files:"]
            lines += [f"  - {path}" for path in self.relevant_files]
        if self.evidence:
            lines += ["", "Evidence:"]
            for item in self.evidence:
                loc = f"{item.file}:{item.lines}" if item.lines else item.file
                lines.append(f"  - [{loc}] {item.note}")
        if self.assumptions:
            lines += ["", "Assumptions:"] + [f"  - {a}" for a in self.assumptions]
        if self.warnings:
            lines += ["", "Warnings:"] + [f"  - {w}" for w in self.warnings]
        return "\n".join(lines)


def extract_json_object(text: str) -> str:
    """Pull the first JSON object out of a model response.

    Handles ```json fences and leading/trailing prose, which OpenAI-compatible
    models produce routinely.
    """
    if not text or not text.strip():
        raise AnswerParseError("empty model response")

    candidates: list[str] = []
    for match in _FENCE_RE.finditer(text):
        candidates.append(match.group(1))
    candidates.append(text)

    for candidate in candidates:
        start = candidate.find("{")
        end = candidate.rfind("}")
        if start == -1 or end <= start:
            continue
        blob = candidate[start : end + 1]
        try:
            json.loads(blob)
        except json.JSONDecodeError:
            continue
        return blob
    raise AnswerParseError(f"no JSON object found in response: {text[:200]!r}")


def parse_answer(text: str) -> CodeAnswer:
    """Parse raw model text into :class:`CodeAnswer`."""
    blob = extract_json_object(text)
    try:
        payload: Any = json.loads(blob)
    except json.JSONDecodeError as exc:  # pragma: no cover - guarded above
        raise AnswerParseError(str(exc)) from exc
    if not isinstance(payload, dict):
        raise AnswerParseError("expected a JSON object at the top level")
    return CodeAnswer.model_validate(payload)


def answer_json_schema() -> dict[str, Any]:
    """JSON schema for the structured answer (handy for prompts and debugging)."""
    return CodeAnswer.model_json_schema()
