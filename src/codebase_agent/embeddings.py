"""Embedding backends.

``local`` (default) is a deterministic offline embedding model: no API key, no
network, stable across processes. It is good enough for code retrieval because
identifiers and keywords dominate code similarity. DeepSeek has no public
embeddings endpoint, so the offline default keeps the project runnable with a
DeepSeek-only setup.

``openai`` uses any OpenAI-compatible ``/embeddings`` endpoint.
"""

from __future__ import annotations

import hashlib
import math
import re
from typing import TYPE_CHECKING

from langchain_core.embeddings import Embeddings

if TYPE_CHECKING:  # pragma: no cover
    from .config import Settings

_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+")
_MASK64 = (1 << 64) - 1


class HashingEmbeddings(Embeddings):
    """Feature hashing over word tokens and character n-grams.

    Deterministic (blake2b, not Python's randomized ``hash``), L2-normalized,
    and therefore usable with a cosine/inner-product index.
    """

    def __init__(self, dim: int = 512, char_ngram: int = 3, ngram_weight: float = 0.3) -> None:
        if dim <= 0:
            raise ValueError("dim must be positive")
        self.dim = dim
        self.char_ngram = char_ngram
        self.ngram_weight = ngram_weight

    # ------------------------------------------------------------------ #
    @staticmethod
    def _digest(token: str) -> int:
        return int.from_bytes(hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest(), "big")

    def _accumulate(self, text: str, vector: list[float]) -> None:
        lowered = text.lower()
        for token in _TOKEN_RE.findall(lowered):
            self._add(vector, token, 1.0)
        if self.char_ngram > 0 and lowered:
            padded = f" {lowered} "
            for i in range(len(padded) - self.char_ngram + 1):
                gram = padded[i : i + self.char_ngram]
                if gram.strip():
                    self._add(vector, f"#{gram}", self.ngram_weight)

    def _add(self, vector: list[float], token: str, weight: float) -> None:
        digest = self._digest(token)
        index = digest % self.dim
        sign = 1.0 if (digest >> 63) & 1 else -1.0
        vector[index] += sign * weight

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dim
        self._accumulate(text or "", vector)
        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0.0:
            return vector
        return [value / norm for value in vector]

    # ------------------------------------------------------------------ #
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


def build_embeddings(settings: Settings) -> Embeddings:
    """Return the embedding backend selected by ``EMBEDDING_PROVIDER``."""
    provider = (settings.embedding_provider or "local").lower()
    if provider == "local":
        return HashingEmbeddings(dim=settings.embedding_dim)

    if provider in {"openai", "openai-compatible"}:
        from langchain_openai import OpenAIEmbeddings

        if not settings.embedding_api_key:
            raise ValueError(
                "EMBEDDING_PROVIDER=openai requires EMBEDDING_API_KEY "
                "(or set EMBEDDING_PROVIDER=local to stay offline)."
            )
        return OpenAIEmbeddings(
            model=settings.embedding_model,
            api_key=settings.embedding_api_key,
            base_url=settings.embedding_base_url or settings.base_url,
            # Custom (non-OpenAI) embedding servers do not accept tiktoken
            # pre-tokenization; disabling it keeps any compatible endpoint working.
            check_embedding_ctx_length=False,
        )

    raise ValueError(f"unknown EMBEDDING_PROVIDER: {settings.embedding_provider!r}")
