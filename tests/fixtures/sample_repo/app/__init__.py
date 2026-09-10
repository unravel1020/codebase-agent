"""Sample application package used by tests and offline demos."""

from .calculator import Calculator, average
from .service import CalculatorService
from .storage import JsonStorage

__all__ = ["Calculator", "CalculatorService", "JsonStorage", "average"]
