"""Tests for chunking, embedding and FAISS retrieval (no API key involved)."""

from __future__ import annotations

import pytest
from langchain_core.documents import Document

from codebase_agent import HashingEmbeddings, Repository, VectorIndex, chunk_documents
from codebase_agent.rag import CodeRetriever, build_retriever
from codebase_agent.vectorstore import RetrievedChunk


def test_index_covers_the_repository(retriever: CodeRetriever) -> None:
    assert retriever.chunk_count > 0
    assert retriever.files_indexed >= 5


def test_chunks_carry_file_and_line_metadata(retriever: CodeRetriever) -> None:
    for chunk in retriever.index.chunks:
        assert chunk.file.endswith(".py") or chunk.file.endswith(".md") or chunk.file.endswith(".txt")
        assert chunk.start_line >= 1
        assert chunk.end_line >= chunk.start_line
        assert chunk.text.strip()


def test_retrieval_finds_persistence_code(retriever: CodeRetriever) -> None:
    chunks = retriever.retrieve("how are calculation results saved to disk", k=4)
    assert chunks
    files = [chunk.file for chunk in chunks]
    assert any(path in {"app/storage.py", "app/service.py"} for path in files), files


def test_retrieval_finds_average_implementation(retriever: CodeRetriever) -> None:
    chunks = retriever.retrieve("average of a list of values", k=4)
    assert chunks
    assert any(chunk.file == "app/calculator.py" for chunk in chunks)
    assert any("average" in chunk.text for chunk in chunks)


def test_retrieval_respects_k(retriever: CodeRetriever) -> None:
    assert len(retriever.retrieve("calculator", k=2)) <= 2
    assert len(retriever.retrieve("calculator", k=1)) == 1


def test_retrieval_scores_are_sorted(retriever: CodeRetriever) -> None:
    chunks = retriever.retrieve("JsonStorage save load", k=5)
    scores = [chunk.score for chunk in chunks]
    assert scores == sorted(scores, reverse=True)
    assert scores[0] > 0.0


def test_empty_query_returns_nothing(retriever: CodeRetriever) -> None:
    assert retriever.retrieve("") == []
    assert retriever.retrieve("   ") == []


def test_chunk_documents_are_deterministic(repo: Repository) -> None:
    documents = repo.load_documents()
    first = chunk_documents(documents, chunk_size=200, chunk_overlap=20)
    second = chunk_documents(documents, chunk_size=200, chunk_overlap=20)
    assert [(c.file, c.start_line, c.text) for c in first] == [
        (c.file, c.start_line, c.text) for c in second
    ]
    assert len(first) > len(documents), "small chunk size should split files"


def test_duplicate_chunk_text_keeps_its_own_line_range() -> None:
    """Regression: the same block twice must not both report the first location.

    ``chunk_documents`` used to resolve a chunk with ``text.find(chunk)``, so the
    second copy of a repeated block was cited at the first copy's line numbers.
    """
    block = "def average(values):\n    return sum(values) / len(values)\n"
    text = block + "\n" + ("padding = 1  # filler\n" * 120) + "\n" + block
    document = Document(page_content=text, metadata={"file": "dup.py", "suffix": ".py"})

    chunks = chunk_documents([document], chunk_size=200, chunk_overlap=0)
    lines = [chunk.start_line for chunk in chunks if "def average" in chunk.text]
    source = text.splitlines()

    assert len(lines) == 2, "the duplicated block should be chunked twice"
    assert lines[0] != lines[1], "both copies were reported at the same offset"
    assert lines[0] == 1
    assert lines[1] == text[: text.rfind("def average")].count("\n") + 1
    # The reported line must actually contain the code that was chunked.
    for start_line in lines:
        assert source[start_line - 1].strip() == "def average(values):"


def test_every_chunk_starts_on_the_line_it_reports(repo: Repository, retriever: CodeRetriever) -> None:
    """Invariant: a chunk's start_line points at its own first line."""
    cache: dict[str, list[str]] = {}
    for chunk in retriever.index.chunks:
        if chunk.file not in cache:
            cache[chunk.file] = repo.read_text(chunk.file, max_lines=10**6).text.splitlines()
        lines = cache[chunk.file]
        assert 1 <= chunk.start_line <= len(lines), chunk.location
        assert lines[chunk.start_line - 1].strip() == chunk.text.splitlines()[0].strip(), (
            f"{chunk.location} does not start on the line it claims"
        )


def test_hashing_embeddings_are_deterministic_and_normalized() -> None:
    embeddings = HashingEmbeddings(dim=128)
    vector = embeddings.embed_query("Calculator average")
    assert len(vector) == 128
    assert vector == embeddings.embed_query("Calculator average")
    assert abs(sum(value * value for value in vector) - 1.0) < 1e-9


def test_hashing_embeddings_similarity_ordering() -> None:
    embeddings = HashingEmbeddings(dim=512)
    query = embeddings.embed_query("average of values")

    def cosine(a: list[float], b: list[float]) -> float:
        return sum(x * y for x, y in zip(a, b, strict=True))

    near = cosine(query, embeddings.embed_query("def average(values): return sum(values)/len(values)"))
    far = cosine(query, embeddings.embed_query("kubernetes deployment yaml manifest"))
    assert near > far


def test_vector_index_orders_by_similarity() -> None:
    index = VectorIndex(dim=3)
    chunks = [
        RetrievedChunk("a.py", 1, 2, 0.0, "a", 0),
        RetrievedChunk("b.py", 1, 2, 0.0, "b", 1),
    ]
    index.add([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], chunks)
    results = index.search([0.9, 0.1, 0.0], k=2)
    assert [chunk.file for chunk in results] == ["a.py", "b.py"]
    assert results[0].score > results[1].score


def test_vector_index_validation() -> None:
    index = VectorIndex(dim=2)
    with pytest.raises(ValueError):
        index.add([[1.0, 0.0]], [])
    with pytest.raises(ValueError):
        index.add([[1.0, 0.0, 0.0]], [RetrievedChunk("a.py", 1, 1, 0.0, "a", 0)])
    assert index.search([1.0, 0.0], k=3) == []
    assert len(index) == 0


def test_build_retriever_from_settings(repo: Repository, settings) -> None:
    retriever = build_retriever(repo, settings, HashingEmbeddings(dim=64))
    assert retriever.chunk_count > 0
    assert retriever.retrieve("slugify", k=1)[0].file.endswith(".py")
