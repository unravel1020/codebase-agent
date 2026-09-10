"""A tiny FAISS-backed vector index.

FAISS was chosen over Chroma because it is a single compiled wheel with no
server, no persistence layer and no telemetry - the simplest thing that gives
real top-k retrieval.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

try:  # pragma: no cover - import guard
    import faiss
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "faiss is required: pip install faiss-cpu (see requirements.txt)"
    ) from exc


@dataclass(frozen=True)
class RetrievedChunk:
    """One retrieved code chunk with citation metadata."""

    file: str
    start_line: int
    end_line: int
    score: float
    text: str
    chunk_id: int

    @property
    def location(self) -> str:
        return f"{self.file}:{self.start_line}-{self.end_line}"

    def preview(self, limit: int = 240) -> str:
        flat = " ".join(self.text.split())
        return flat[:limit] + ("..." if len(flat) > limit else "")


def _as_matrix(vectors: Sequence[Sequence[float]]) -> np.ndarray:
    matrix = np.asarray(vectors, dtype="float32")
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    return matrix


def _normalize(matrix: np.ndarray) -> np.ndarray:
    """L2-normalize so inner-product search equals cosine similarity."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return matrix / norms


class VectorIndex:
    """Exact inner-product index over L2-normalized embeddings."""

    def __init__(self, dim: int) -> None:
        if dim <= 0:
            raise ValueError("dim must be positive")
        self.dim = dim
        self._index = faiss.IndexFlatIP(dim)
        self._chunks: list[RetrievedChunk] = []

    def __len__(self) -> int:
        return len(self._chunks)

    @property
    def chunks(self) -> list[RetrievedChunk]:
        return list(self._chunks)

    def add(self, vectors: Sequence[Sequence[float]], chunks: Sequence[RetrievedChunk]) -> None:
        """Add embeddings and their chunks. Order must match."""
        if len(vectors) != len(chunks):
            raise ValueError(
                f"vectors ({len(vectors)}) and chunks ({len(chunks)}) must have equal length"
            )
        if not vectors:
            return
        matrix = _as_matrix(vectors)
        if matrix.shape[1] != self.dim:
            raise ValueError(f"expected {self.dim}-dim vectors, got {matrix.shape[1]}")
        self._index.add(_normalize(matrix))
        self._chunks.extend(chunks)

    def search(self, query_vector: Sequence[float], k: int = 6) -> list[RetrievedChunk]:
        """Return up to ``k`` chunks ordered by descending cosine similarity."""
        if len(self._chunks) == 0 or k <= 0:
            return []
        query = _normalize(_as_matrix([query_vector]))
        if query.shape[1] != self.dim:
            raise ValueError(f"expected {self.dim}-dim query, got {query.shape[1]}")
        scores, indices = self._index.search(query, min(k, len(self._chunks)))
        results: list[RetrievedChunk] = []
        for score, index in zip(scores[0], indices[0], strict=True):
            if index < 0:
                continue
            chunk = self._chunks[int(index)]
            results.append(
                RetrievedChunk(
                    file=chunk.file,
                    start_line=chunk.start_line,
                    end_line=chunk.end_line,
                    score=float(score),
                    text=chunk.text,
                    chunk_id=chunk.chunk_id,
                )
            )
        return results
