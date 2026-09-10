"""Pure arithmetic helpers for the sample application."""

from __future__ import annotations


class Calculator:
    """Stateless arithmetic operations."""

    def add(self, a: float, b: float) -> float:
        """Return a + b."""
        return a + b

    def subtract(self, a: float, b: float) -> float:
        """Return a - b."""
        return a - b

    def multiply(self, a: float, b: float) -> float:
        """Return a * b."""
        return a * b

    def divide(self, a: float, b: float) -> float:
        """Return a / b, raising ValueError for a zero divisor."""
        if b == 0:
            raise ValueError("division by zero")
        return a / b


def average(values: list[float]) -> float:
    """Return the arithmetic mean of values."""
    if not values:
        raise ValueError("average() requires at least one value")
    return sum(values) / len(values)
