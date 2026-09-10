"""JSON-file persistence for calculation history."""

from __future__ import annotations

import json
from pathlib import Path


class JsonStorage:
    """Stores records as a JSON array in a single file."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def save(self, records: list[dict]) -> None:
        """Write all records to disk, replacing the previous contents."""
        self.path.write_text(json.dumps(records, indent=2), encoding="utf-8")

    def load(self) -> list[dict]:
        """Return the stored records, or an empty list if nothing was saved yet."""
        if not self.path.exists():
            return []
        return json.loads(self.path.read_text(encoding="utf-8"))
