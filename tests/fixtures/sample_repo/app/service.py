"""Application service that ties Calculator and JsonStorage together."""

from __future__ import annotations

from .calculator import Calculator, average
from .storage import JsonStorage


class CalculatorService:
    """Runs calculations and keeps a persistent history of results."""

    def __init__(self, storage: JsonStorage | None = None) -> None:
        self.calculator = Calculator()
        self.storage = storage or JsonStorage("history.json")
        self.history: list[dict] = self.storage.load()

    def compute(self, operation: str, a: float, b: float) -> float:
        """Apply operation to a and b, persist the result and return it."""
        if operation == "add":
            result = self.calculator.add(a, b)
        elif operation == "subtract":
            result = self.calculator.subtract(a, b)
        elif operation == "multiply":
            result = self.calculator.multiply(a, b)
        elif operation == "divide":
            result = self.calculator.divide(a, b)
        else:
            raise ValueError(f"unknown operation: {operation}")

        self.history.append({"operation": operation, "a": a, "b": b, "result": result})
        self.storage.save(self.history)
        return result

    def mean_result(self) -> float:
        """Return the average of every result computed so far."""
        return average([row["result"] for row in self.history])
