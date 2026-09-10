"""Formatting helpers shared by the sample application."""

from __future__ import annotations


def format_result(value: float, precision: int = 2) -> str:
    """Format a number with a fixed number of decimals."""
    return f"{value:.{precision}f}"


def slugify(text: str) -> str:
    """Lowercase the text and join words with dashes."""
    return "-".join(text.lower().split())
