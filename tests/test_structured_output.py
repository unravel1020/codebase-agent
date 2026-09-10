"""Tests for the Pydantic structured-output layer."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from codebase_agent.schemas import (
    AnswerParseError,
    CodeAnswer,
    EvidenceItem,
    answer_json_schema,
    extract_json_object,
    parse_answer,
)

VALID = {
    "summary": "The average is computed by average() in app/calculator.py.",
    "relevant_files": ["app/calculator.py"],
    "evidence": [
        {"file": "app/calculator.py", "lines": "26-30", "note": "defines average()"}
    ],
    "confidence": 0.8,
}


def test_parse_raw_json() -> None:
    answer = parse_answer(json.dumps(VALID))
    assert isinstance(answer, CodeAnswer)
    assert answer.summary.startswith("The average")
    assert answer.relevant_files == ["app/calculator.py"]
    assert answer.evidence[0].lines == "26-30"
    assert answer.confidence == pytest.approx(0.8)


def test_parse_fenced_json() -> None:
    text = f"Here you go:\n```json\n{json.dumps(VALID)}\n```\nDone."
    assert parse_answer(text).relevant_files == ["app/calculator.py"]


def test_parse_json_with_surrounding_prose() -> None:
    text = f"I inspected the repository. {json.dumps(VALID)} That is all."
    assert parse_answer(text).confidence == pytest.approx(0.8)


def test_parse_rejects_garbage() -> None:
    with pytest.raises(AnswerParseError):
        parse_answer("I could not find anything useful.")
    with pytest.raises(AnswerParseError):
        parse_answer("")
    with pytest.raises(AnswerParseError):
        parse_answer("{not valid json at all}")


def test_missing_required_field_fails() -> None:
    payload = {key: value for key, value in VALID.items() if key != "summary"}
    with pytest.raises(ValidationError):
        CodeAnswer.model_validate(payload)


def test_confidence_bounds_enforced() -> None:
    for bad in (-0.1, 1.5):
        with pytest.raises(ValidationError):
            CodeAnswer.model_validate({**VALID, "confidence": bad})


def test_relevant_files_are_normalized_and_deduplicated() -> None:
    answer = CodeAnswer.model_validate(
        {
            **VALID,
            "relevant_files": [
                ".\\app\\calculator.py",
                "./app/calculator.py",
                " app/calculator.py ",
                "",
            ],
        }
    )
    assert answer.relevant_files == ["app/calculator.py"]


def test_evidence_defaults_and_normalization() -> None:
    answer = CodeAnswer.model_validate(
        {
            "summary": "s",
            "confidence": 0.5,
            "evidence": [{"file": ".\\src\\a.py", "note": "n"}],
        }
    )
    assert answer.evidence[0].file == "src/a.py"
    assert answer.evidence[0].lines is None
    assert answer.assumptions == [] and answer.warnings == []


def test_extra_keys_are_ignored() -> None:
    answer = CodeAnswer.model_validate({**VALID, "chain_of_thought": "hidden reasoning"})
    assert not hasattr(answer, "chain_of_thought")


def test_extract_json_object_handles_nested_braces() -> None:
    blob = extract_json_object('prefix {"a": {"b": 1}} suffix')
    assert json.loads(blob) == {"a": {"b": 1}}


def test_json_schema_exposes_required_fields() -> None:
    schema = answer_json_schema()
    # Only fields without defaults are strictly required; the rest are optional
    # but always present in the parsed model.
    assert set(schema["required"]) == {"summary", "confidence"}
    assert set(schema["properties"]) >= {
        "summary",
        "relevant_files",
        "evidence",
        "confidence",
        "assumptions",
        "warnings",
    }
    assert schema["properties"]["confidence"]["minimum"] == 0.0
    assert schema["properties"]["confidence"]["maximum"] == 1.0


def test_evidence_item_is_validated() -> None:
    with pytest.raises(ValidationError):
        EvidenceItem.model_validate({"note": "missing file"})


def test_to_markdown_rendering() -> None:
    markdown = CodeAnswer.model_validate(VALID).to_markdown()
    assert "Summary:" in markdown
    assert "Confidence: 0.80" in markdown
    assert "app/calculator.py:26-30" in markdown
