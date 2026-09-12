"""RAG pipeline: load repository -> chunk -> embed -> index -> retrieve."""

from __future__ import annotations

from collections.abc import Iterable

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter

from .config import Settings
from .repository import Repository
from .vectorstore import RetrievedChunk, VectorIndex

_PYTHON_SUFFIXES = {".py", ".pyi"}
_GENERIC_SEPARATORS = ["\n\n", "\n", " ", ""]
_EMBED_BATCH = 64


def _splitter_for(suffix: str, chunk_size: int, chunk_overlap: int) -> RecursiveCharacterTextSplitter:
    """Code-aware splitting: Python gets language separators, the rest generic."""
    if suffix in _PYTHON_SUFFIXES:
        return RecursiveCharacterTextSplitter.from_language(
            Language.PYTHON,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            add_start_index=True,
        )
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=_GENERIC_SEPARATORS,
        add_start_index=True,
    )


def _line_span(text: str, start_index: int, chunk: str) -> tuple[int, int]:
    """Convert a character offset into a 1-based inclusive line range."""
    start_line = text[:start_index].count("\n") + 1
    end_line = start_line + max(chunk.count("\n"), 0)
    return start_line, end_line


def _start_index(piece: Document, text: str, chunk: str) -> int:
    """Return the exact offset of a chunk inside its source text.

    The splitter reports a running cursor as ``start_index`` (it advances past
    the previous chunk before searching), so the same text occurring twice in one
    file resolves to two different offsets. Searching with a bare ``str.find``
    would send both copies to the first occurrence, which is exactly the bug this
    function avoids.
    """
    value = piece.metadata.get("start_index")
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return value
    found = text.find(chunk)
    return found if found >= 0 else 0


def chunk_documents(
    documents: Iterable[Document],
    *,
    chunk_size: int = 1200,
    chunk_overlap: int = 150,
) -> list[RetrievedChunk]:
    """Split documents into chunks carrying file + line metadata."""
    chunks: list[RetrievedChunk] = []
    next_id = 0
    for document in documents:
        text = document.page_content
        if not text.strip():
            continue
        suffix = str(document.metadata.get("suffix", ""))
        file_path = str(document.metadata.get("file", "<unknown>"))
        splitter = _splitter_for(suffix, chunk_size, chunk_overlap)
        for piece in splitter.split_documents([document]):
            chunk = piece.page_content
            if not chunk.strip():
                continue
            start_line, end_line = _line_span(text, _start_index(piece, text, chunk), chunk)
            chunks.append(
                RetrievedChunk(
                    file=file_path,
                    start_line=start_line,
                    end_line=end_line,
                    score=0.0,
                    text=chunk,
                    chunk_id=next_id,
                )
            )
            next_id += 1
    return chunks


class CodeRetriever:
    """Top-k retrieval over a repository's chunks."""

    def __init__(self, embeddings: Embeddings, index: VectorIndex, chunks: list[RetrievedChunk]) -> None:
        self.embeddings = embeddings
        self.index = index
        self._chunks = chunks

    # ------------------------------------------------------------------ #
    @classmethod
    def build(
        cls,
        repo: Repository,
        embeddings: Embeddings,
        *,
        chunk_size: int = 1200,
        chunk_overlap: int = 150,
        max_files: int | None = 400,
    ) -> CodeRetriever:
        """Load, chunk, embed and index a repository."""
        documents = repo.load_documents(max_files=max_files)
        chunks = chunk_documents(
            documents, chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        index = VectorIndex(dim=_embedding_dim(embeddings))
        if chunks:
            vectors: list[list[float]] = []
            for start in range(0, len(chunks), _EMBED_BATCH):
                batch = chunks[start : start + _EMBED_BATCH]
                vectors.extend(embeddings.embed_documents([chunk.text for chunk in batch]))
            index.add(vectors, chunks)
        return cls(embeddings, index, chunks)

    # ------------------------------------------------------------------ #
    @property
    def chunk_count(self) -> int:
        return len(self._chunks)

    @property
    def files_indexed(self) -> int:
        return len({chunk.file for chunk in self._chunks})

    def retrieve(self, query: str, k: int | None = None) -> list[RetrievedChunk]:
        """Return the top-k chunks for a natural-language or symbol query."""
        if not query or not query.strip() or self.chunk_count == 0:
            return []
        top_k = k or 6
        vector = self.embeddings.embed_query(query)
        return self.index.search(vector, top_k)


def _embedding_dim(embeddings: Embeddings) -> int:
    """Probe the embedding dimension once (cheap: one tiny string)."""
    dim = getattr(embeddings, "dim", None)
    if isinstance(dim, int) and dim > 0:
        return dim
    return len(embeddings.embed_query("dimension probe"))


def build_retriever(repo: Repository, settings: Settings, embeddings: Embeddings | None = None) -> CodeRetriever:
    """Convenience factory used by the CLI, the agent and the evals."""
    from .embeddings import build_embeddings

    return CodeRetriever.build(
        repo,
        embeddings or build_embeddings(settings),
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        max_files=settings.max_index_files,
    )
