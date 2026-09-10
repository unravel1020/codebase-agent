"""Repository question evaluation harness.

Three modes:

* ``retrieval``  - RAG only. Reports hit rate@k, MRR and latency. No API key.
* ``mock-agent`` - full agent loop driven by the deterministic heuristic policy.
  Reports hit rate on the agent's ``relevant_files``, tool-call count, iterations
  and latency. No API key.
* ``llm``        - the real thing: needs ``LLM_API_KEY``. Same metrics as
  ``mock-agent``, plus the model's own answer quality signals.

Usage::

    python evals/run_evals.py                       # retrieval on this repo
    python evals/run_evals.py --mode mock-agent     # agent loop, offline
    python evals/run_evals.py --mode llm --repo D:\\Project\\other

Results are written to ``evals/results/<mode>-<timestamp>.json``.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:  # allow `python evals/run_evals.py`
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from codebase_agent.config import Settings  # noqa: E402
from codebase_agent.embeddings import build_embeddings  # noqa: E402
from codebase_agent.factory import build_agent, build_repository  # noqa: E402
from codebase_agent.offline import build_offline_llm  # noqa: E402
from codebase_agent.rag import build_retriever  # noqa: E402

DEFAULT_QUESTIONS = Path(__file__).resolve().parent / "questions.json"
DEFAULT_RESULTS_DIR = Path(__file__).resolve().parent / "results"


@dataclass
class QuestionSpec:
    id: str
    question: str
    expected_files: list[str] = field(default_factory=list)
    kind: str = "general"


@dataclass
class QuestionResult:
    id: str
    question: str
    expected_files: list[str]
    candidates: list[str]
    hit: bool
    rank: int | None
    latency_ms: float
    tool_calls: int = 0
    iterations: int = 0
    confidence: float | None = None
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "question": self.question,
            "expected_files": self.expected_files,
            "candidates": self.candidates,
            "hit": self.hit,
            "rank": self.rank,
            "latency_ms": round(self.latency_ms, 2),
            "tool_calls": self.tool_calls,
            "iterations": self.iterations,
            "confidence": self.confidence,
            "notes": self.notes,
        }


def _norm(path: str) -> str:
    return path.replace("\\", "/").lstrip("./")


def first_rank(candidates: list[str], expected: list[str]) -> int | None:
    """1-based rank of the first candidate matching any expected path."""
    if not expected:
        return None
    wanted = [_norm(path) for path in expected]
    for index, candidate in enumerate(candidates, start=1):
        current = _norm(candidate)
        for target in wanted:
            if current == target or current.endswith("/" + target) or target.endswith("/" + current):
                return index
    return None


def load_questions(path: Path) -> list[QuestionSpec]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [
        QuestionSpec(
            id=item["id"],
            question=item["question"],
            expected_files=item.get("expected_files", []),
            kind=item.get("kind", "general"),
        )
        for item in payload["questions"]
    ]


# --------------------------------------------------------------------------- #
def run_retrieval(
    repo,
    retriever,
    questions: list[QuestionSpec],
    k: int,
) -> list[QuestionResult]:
    results: list[QuestionResult] = []
    for spec in questions:
        started = time.perf_counter()
        chunks = retriever.retrieve(spec.question, k)
        latency_ms = (time.perf_counter() - started) * 1000.0
        candidates = [chunk.file for chunk in chunks]
        rank = first_rank(candidates, spec.expected_files)
        results.append(
            QuestionResult(
                id=spec.id,
                question=spec.question,
                expected_files=spec.expected_files,
                candidates=candidates,
                hit=rank is not None,
                rank=rank,
                latency_ms=latency_ms,
                notes=[f"kind={spec.kind}", f"top_score={chunks[0].score:.3f}"] if chunks else [],
            )
        )
    return results


def run_agent(
    repo,
    retriever,
    questions: list[QuestionSpec],
    settings: Settings,
    *,
    use_llm: bool,
) -> list[QuestionResult]:
    """Run every question through a fresh agent (questions stay independent)."""
    results: list[QuestionResult] = []
    for spec in questions:
        llm = None if use_llm else build_offline_llm()
        agent = build_agent(
            repo.root,
            settings,
            llm=llm,
            repo=repo,
            retriever=retriever,
        )
        result = agent.run(spec.question)
        candidates = list(result.answer.relevant_files)
        rank = first_rank(candidates, spec.expected_files)
        results.append(
            QuestionResult(
                id=spec.id,
                question=spec.question,
                expected_files=spec.expected_files,
                candidates=candidates,
                hit=rank is not None,
                rank=rank,
                latency_ms=result.latency_ms,
                tool_calls=result.tool_call_count,
                iterations=result.iterations,
                confidence=result.answer.confidence,
                notes=list(result.answer.warnings),
            )
        )
    return results


# --------------------------------------------------------------------------- #
def aggregate(results: list[QuestionResult], *, mode: str) -> dict[str, Any]:
    total = len(results)
    hits = [result for result in results if result.hit]
    ranks = [result.rank for result in hits if result.rank]
    latencies = [result.latency_ms for result in results]
    payload: dict[str, Any] = {
        "questions": total,
        "hits": len(hits),
        "hit_rate": round(len(hits) / total, 4) if total else 0.0,
        "mrr": round(statistics.fmean(1.0 / rank for rank in ranks), 4) if ranks else 0.0,
        "latency_ms_avg": round(statistics.fmean(latencies), 2) if latencies else 0.0,
        "latency_ms_max": round(max(latencies), 2) if latencies else 0.0,
    }
    if mode in {"mock-agent", "llm"}:
        payload["tool_calls_avg"] = round(
            statistics.fmean([result.tool_calls for result in results]), 2
        )
        payload["tool_calls_total"] = sum(result.tool_calls for result in results)
        payload["iterations_avg"] = round(
            statistics.fmean([result.iterations for result in results]), 2
        )
        payload["evidence_rate"] = round(
            sum(1 for result in results if result.tool_calls > 0) / total, 4
        )
        confidences = [result.confidence for result in results if result.confidence is not None]
        if confidences:
            payload["confidence_avg"] = round(statistics.fmean(confidences), 4)
    return payload


def print_report(report: dict[str, Any]) -> None:
    aggregate_data = report["aggregate"]
    print(f"\nmode            : {report['mode']}")
    print(f"repository      : {report['repo']}")
    print(f"index           : {report['index']['chunks']} chunks / {report['index']['files']} files")
    print(f"model           : {report['settings']['model']} via {report['settings']['base_url']}")
    print(f"embeddings      : {report['settings']['embedding_provider']}")
    print(
        f"hit rate        : {aggregate_data['hits']}/{aggregate_data['questions']} "
        f"= {aggregate_data['hit_rate']:.1%}"
    )
    print(f"MRR             : {aggregate_data['mrr']:.3f}")
    print(
        f"latency         : avg {aggregate_data['latency_ms_avg']:.1f} ms "
        f"/ max {aggregate_data['latency_ms_max']:.1f} ms"
    )
    if "tool_calls_avg" in aggregate_data:
        print(
            f"tool calls      : avg {aggregate_data['tool_calls_avg']:.2f} "
            f"/ total {aggregate_data['tool_calls_total']}"
        )
        print(f"evidence rate   : {aggregate_data['evidence_rate']:.1%}")
        if "confidence_avg" in aggregate_data:
            print(f"confidence avg  : {aggregate_data['confidence_avg']:.3f}")
    print("\nper question:")
    for item in report["results"]:
        mark = "HIT " if item["hit"] else "MISS"
        rank = f"rank={item['rank']}" if item["rank"] else "rank=-"
        extra = (
            f"tools={item['tool_calls']} iters={item['iterations']}"
            if report["mode"] in {"mock-agent", "llm"}
            else f"top={item['candidates'][0] if item['candidates'] else '-'}"
        )
        print(f"  {mark} {item['id']} {rank} {item['latency_ms']:7.1f}ms {extra}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate retrieval and agent behaviour on a repository.")
    parser.add_argument("--repo", default=str(PROJECT_ROOT), help="repository root to evaluate")
    parser.add_argument("--questions", default=str(DEFAULT_QUESTIONS), help="questions JSON file")
    parser.add_argument(
        "--mode",
        choices=("retrieval", "mock-agent", "llm"),
        default="retrieval",
        help="retrieval = RAG metrics only; mock-agent = offline agent loop; llm = real API",
    )
    parser.add_argument("-k", type=int, default=0, help="top-k for retrieval (0 = RAG_TOP_K)")
    parser.add_argument("--out", default=str(DEFAULT_RESULTS_DIR), help="output directory")
    parser.add_argument(
        "--exclude",
        nargs="*",
        default=["evals"],
        help="directories excluded from the index (default: evals, to avoid the "
        "questions file matching itself)",
    )
    args = parser.parse_args(argv)

    settings = Settings.from_env()
    k = args.k or settings.top_k
    questions = load_questions(Path(args.questions))

    repo = build_repository(args.repo, settings, extra_excluded_dirs=tuple(args.exclude or ()))
    retriever = build_retriever(repo, settings, build_embeddings(settings))

    if args.mode == "retrieval":
        results = run_retrieval(repo, retriever, questions, k)
    else:
        results = run_agent(repo, retriever, questions, settings, use_llm=args.mode == "llm")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report: dict[str, Any] = {
        "mode": args.mode,
        "timestamp": timestamp,
        "repo": str(repo.root),
        "repo_stats": repo.stats(max_files=settings.max_index_files),
        "index": {"chunks": retriever.chunk_count, "files": retriever.files_indexed},
        "k": k,
        "settings": settings.safe_dict(),
        "aggregate": aggregate(results, mode=args.mode),
        "results": [result.to_dict() for result in results],
    }

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{args.mode}-{timestamp}.json"
    out_file.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print_report(report)
    print(f"\nwrote {out_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
