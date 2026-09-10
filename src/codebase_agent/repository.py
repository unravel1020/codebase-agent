"""Sandboxed access to a local repository.

Every path that reaches the filesystem goes through :meth:`Repository.resolve`,
which resolves symlinks and refuses anything outside the repository root.
"""

from __future__ import annotations

import os
import re
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from langchain_core.documents import Document


class RepositoryError(RuntimeError):
    """Base class for repository access failures."""


class PathSecurityError(RepositoryError):
    """Raised when a requested path escapes the repository root."""


class FileTooLargeError(RepositoryError):
    """Raised when a file exceeds the configured size budget."""


class BinaryFileError(RepositoryError):
    """Raised when a file looks binary and is therefore not useful context."""


class FileNotFoundInRepoError(RepositoryError):
    """Raised when the requested path does not exist inside the repository."""


EXCLUDED_DIRS: frozenset[str] = frozenset(
    {
        ".git",
        ".hg",
        ".svn",
        ".venv",
        "venv",
        "env",
        "__pycache__",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".tox",
        "node_modules",
        "bower_components",
        "dist",
        "build",
        "target",
        ".next",
        ".nuxt",
        ".idea",
        ".vscode",
        "site-packages",
        ".terraform",
        "vendor",
    }
)

BINARY_SUFFIXES: frozenset[str] = frozenset(
    {
        ".pyc",
        ".pyo",
        ".pyd",
        ".so",
        ".dll",
        ".dylib",
        ".exe",
        ".bin",
        ".o",
        ".a",
        ".obj",
        ".lib",
        ".class",
        ".jar",
        ".war",
        ".zip",
        ".gz",
        ".bz2",
        ".xz",
        ".7z",
        ".tar",
        ".rar",
        ".png",
        ".jpg",
        ".jpeg",
        ".gif",
        ".bmp",
        ".ico",
        ".webp",
        ".svgz",
        ".pdf",
        ".doc",
        ".docx",
        ".xls",
        ".xlsx",
        ".ppt",
        ".pptx",
        ".mp3",
        ".mp4",
        ".avi",
        ".mov",
        ".wav",
        ".flac",
        ".ttf",
        ".otf",
        ".woff",
        ".woff2",
        ".eot",
        ".db",
        ".sqlite",
        ".sqlite3",
        ".pkl",
        ".pickle",
        ".npy",
        ".npz",
        ".pt",
        ".pth",
        ".onnx",
        ".parquet",
        ".lock",
    }
)

TEXT_SUFFIXES: frozenset[str] = frozenset(
    {
        ".py",
        ".pyi",
        ".js",
        ".mjs",
        ".cjs",
        ".ts",
        ".tsx",
        ".jsx",
        ".java",
        ".kt",
        ".kts",
        ".scala",
        ".cs",
        ".fs",
        ".go",
        ".rs",
        ".rb",
        ".php",
        ".swift",
        ".m",
        ".mm",
        ".c",
        ".h",
        ".cc",
        ".cpp",
        ".cxx",
        ".hpp",
        ".hxx",
        ".lua",
        ".pl",
        ".r",
        ".jl",
        ".sh",
        ".bash",
        ".zsh",
        ".ps1",
        ".bat",
        ".sql",
        ".html",
        ".htm",
        ".css",
        ".scss",
        ".sass",
        ".less",
        ".vue",
        ".svelte",
        ".json",
        ".jsonc",
        ".yaml",
        ".yml",
        ".toml",
        ".ini",
        ".cfg",
        ".conf",
        ".properties",
        ".env.example",
        ".md",
        ".rst",
        ".txt",
        ".adoc",
        ".gradle",
        ".dockerfile",
        ".cmake",
        ".proto",
        ".graphql",
        ".tf",
    }
)

TEXT_FILENAMES: frozenset[str] = frozenset(
    {
        "Dockerfile",
        "Makefile",
        "makefile",
        "GNUmakefile",
        "CMakeLists.txt",
        "Jenkinsfile",
        "Procfile",
        "requirements.txt",
        "pyproject.toml",
        "setup.py",
        "setup.cfg",
        ".gitignore",
        ".dockerignore",
        ".editorconfig",
        "README",
        "LICENSE",
    }
)

_DEF_TEMPLATES: tuple[str, ...] = (
    r"^\s*(?:async\s+)?def\s+{sym}\b",
    r"^\s*class\s+{sym}\b",
    r"^\s*func\s+{sym}\b",
    r"^\s*type\s+{sym}\b",
    r"^\s*(?:export\s+)?(?:default\s+)?(?:class|interface|type|enum|struct|trait)\s+{sym}\b",
    r"^\s*(?:const|let|var|function|fn|static|public|private|protected|internal|async)\b[^=;()]*\b{sym}\s*[(<=]",
    r"^\s*{sym}\s*[:=]",
    r"^\s*{sym}\s*\(",
)


@dataclass(frozen=True)
class SourceFile:
    """A text file inside the repository."""

    rel_path: str
    abs_path: Path
    size: int


@dataclass(frozen=True)
class SearchHit:
    """A single matching line."""

    rel_path: str
    line: int
    text: str
    score: float
    is_definition: bool = False


@dataclass(frozen=True)
class FileSlice:
    """A bounded window of a text file."""

    rel_path: str
    start_line: int
    end_line: int
    total_lines: int
    text: str
    truncated: bool = False

    def numbered(self, width: int | None = None) -> str:
        """Render with 1-based line numbers so the model can cite lines."""
        lines = self.text.splitlines()
        pad = width or len(str(self.end_line))
        return "\n".join(
            f"{self.start_line + offset:>{pad}} | {line}" for offset, line in enumerate(lines)
        )


class Repository:
    """A read-only, sandboxed view of a local repository."""

    def __init__(
        self,
        root: str | Path,
        *,
        max_file_bytes: int = 512_000,
        max_read_lines: int = 400,
        extra_excluded_dirs: tuple[str, ...] = (),
    ) -> None:
        candidate = Path(root).expanduser()
        if not candidate.exists():
            raise RepositoryError(f"repository root does not exist: {candidate}")
        if not candidate.is_dir():
            raise RepositoryError(f"repository root is not a directory: {candidate}")
        self.root: Path = candidate.resolve()
        self.max_file_bytes = max_file_bytes
        self.max_read_lines = max_read_lines
        self.excluded_dirs = EXCLUDED_DIRS | frozenset(extra_excluded_dirs)

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return f"Repository(root={str(self.root)!r})"

    # ------------------------------------------------------------------ #
    # path safety
    # ------------------------------------------------------------------ #
    def resolve(self, relative_path: str) -> Path:
        """Resolve ``relative_path`` inside the root or raise.

        Accepts POSIX or Windows separators, relative or absolute input. The
        result is always a real path under :attr:`root`.
        """
        if relative_path is None:
            raise PathSecurityError("path must not be None")
        raw = str(relative_path).strip().strip('"').strip("'")
        if not raw:
            raise PathSecurityError("path must not be empty")
        if "\x00" in raw:
            raise PathSecurityError("path must not contain NUL bytes")

        candidate = Path(raw)
        if not candidate.is_absolute():
            candidate = self.root / candidate
        try:
            resolved = candidate.resolve()
        except (OSError, RuntimeError) as exc:  # pragma: no cover - OS specific
            raise PathSecurityError(f"cannot resolve path {relative_path!r}: {exc}") from exc

        if not self._is_within_root(resolved):
            raise PathSecurityError(
                f"path escapes repository root: {relative_path!r} -> {resolved}"
            )
        return resolved

    def _is_within_root(self, resolved: Path) -> bool:
        try:
            resolved.relative_to(self.root)
        except ValueError:
            return False
        return True

    def to_rel_path(self, path: Path) -> str:
        """Repository-relative POSIX path (stable across platforms)."""
        return path.resolve().relative_to(self.root).as_posix()

    # ------------------------------------------------------------------ #
    # enumeration
    # ------------------------------------------------------------------ #
    def is_indexable(self, path: Path) -> bool:
        """Cheap pre-filter: text-ish, not excluded, within the size budget."""
        if path.name in self.excluded_dirs:
            return False
        suffix = path.suffix.lower()
        if suffix in BINARY_SUFFIXES:
            return False
        if suffix not in TEXT_SUFFIXES and path.name not in TEXT_FILENAMES:
            return False
        try:
            if path.stat().st_size > self.max_file_bytes:
                return False
        except OSError:  # pragma: no cover - race with filesystem
            return False
        return True

    def iter_source_files(self, max_files: int | None = None) -> Iterator[SourceFile]:
        """Yield indexable files, skipping VCS/dependency/binary noise."""
        count = 0
        for dirpath, dirnames, filenames in os.walk(self.root):
            dirnames[:] = sorted(
                name
                for name in dirnames
                if name not in self.excluded_dirs and not name.startswith(".git")
            )
            for filename in sorted(filenames):
                abs_path = Path(dirpath) / filename
                if not self.is_indexable(abs_path):
                    continue
                try:
                    size = abs_path.stat().st_size
                except OSError:  # pragma: no cover
                    continue
                yield SourceFile(self.to_rel_path(abs_path), abs_path, size)
                count += 1
                if max_files is not None and count >= max_files:
                    return

    # ------------------------------------------------------------------ #
    # reading
    # ------------------------------------------------------------------ #
    def _check_readable(self, path: Path) -> None:
        if not path.exists():
            raise FileNotFoundInRepoError(f"file not found in repository: {self._display(path)}")
        if path.is_dir():
            raise FileNotFoundInRepoError(
                f"{self._display(path)} is a directory; use search_code to explore it"
            )
        if not path.is_file():
            raise FileNotFoundInRepoError(f"{self._display(path)} is not a regular file")
        suffix = path.suffix.lower()
        if suffix in BINARY_SUFFIXES:
            raise BinaryFileError(f"refusing to read binary file: {self._display(path)}")
        size = path.stat().st_size
        if size > self.max_file_bytes:
            raise FileTooLargeError(
                f"{self._display(path)} is {size} bytes, limit is {self.max_file_bytes}"
            )

    def _display(self, path: Path) -> str:
        try:
            return self.to_rel_path(path)
        except ValueError:  # pragma: no cover - defensive
            return str(path)

    @staticmethod
    def _looks_binary(blob: bytes) -> bool:
        if b"\x00" in blob:
            return True
        if not blob:
            return False
        text_chars = bytes(range(32, 127)) + b"\n\r\t\f\b\x1b"
        non_text = sum(1 for byte in blob if byte not in text_chars and byte < 0x80)
        return non_text / len(blob) > 0.30

    def read_text(
        self,
        relative_path: str,
        start_line: int | None = None,
        end_line: int | None = None,
        *,
        max_lines: int | None = None,
    ) -> FileSlice:
        """Read a bounded slice of a text file, with 1-based line numbers."""
        path = self.resolve(relative_path)
        self._check_readable(path)

        with path.open("rb") as handle:
            blob = handle.read(self.max_file_bytes + 1)
        if len(blob) > self.max_file_bytes:
            raise FileTooLargeError(
                f"{self._display(path)} exceeds the {self.max_file_bytes}-byte limit"
            )
        if self._looks_binary(blob):
            raise BinaryFileError(f"refusing to read binary file: {self._display(path)}")

        text = blob.decode("utf-8", errors="replace")
        all_lines = text.splitlines()
        total = len(all_lines)

        first = 1 if start_line is None else max(1, int(start_line))
        last = total if end_line is None else min(total, int(end_line))
        if last < first:
            raise ValueError(f"end_line ({last}) must be >= start_line ({first})")

        limit = max_lines or self.max_read_lines
        truncated = False
        if last - first + 1 > limit:
            last = first + limit - 1
            truncated = True

        window = all_lines[first - 1 : last]
        return FileSlice(
            rel_path=self._display(path),
            start_line=first,
            end_line=last,
            total_lines=total,
            text="\n".join(window),
            truncated=truncated,
        )

    # ------------------------------------------------------------------ #
    # search
    # ------------------------------------------------------------------ #
    def search(
        self,
        query: str,
        *,
        max_results: int = 20,
        regex: bool = False,
        case_sensitive: bool = False,
    ) -> list[SearchHit]:
        """Search source files for a symbol or keyword.

        Ranking: definition lines first, then whole-query matches, then token
        matches. This makes ``search_code("Calculator")`` point at the class
        definition before any usage site.
        """
        raw = (query or "").strip()
        if not raw:
            return []

        tokens = [token for token in re.split(r"[^A-Za-z0-9_]+", raw) if token]
        flags = 0 if case_sensitive else re.IGNORECASE
        try:
            pattern = re.compile(raw if regex else re.escape(raw), flags)
        except re.error as exc:
            raise ValueError(f"invalid regex {raw!r}: {exc}") from exc
        # Regex mode is exact: only the user's pattern decides what matches.
        # Definition markers are still derived from the query words (a flag, not a filter).
        token_patterns = (
            [] if regex else [re.compile(re.escape(token), re.IGNORECASE) for token in tokens]
        )
        def_patterns = [
            re.compile(template.format(sym=re.escape(token)), re.IGNORECASE)
            for token in tokens
            for template in _DEF_TEMPLATES
        ]

        hits: list[SearchHit] = []
        for source in self.iter_source_files():
            try:
                with source.abs_path.open("rb") as handle:
                    blob = handle.read(self.max_file_bytes + 1)
            except OSError:  # pragma: no cover
                continue
            if len(blob) > self.max_file_bytes or self._looks_binary(blob):
                continue
            text = blob.decode("utf-8", errors="replace")
            for number, line in enumerate(text.splitlines(), start=1):
                if not line.strip():
                    continue
                is_definition = any(def_pattern.search(line) for def_pattern in def_patterns)
                if is_definition:
                    score = 4.0
                elif pattern.search(line):
                    score = 3.0 if case_sensitive else 2.0
                elif any(token_pattern.search(line) for token_pattern in token_patterns):
                    score = 1.0
                else:
                    continue
                hits.append(
                    SearchHit(
                        rel_path=source.rel_path,
                        line=number,
                        text=line.strip()[:300],
                        score=score,
                        is_definition=is_definition,
                    )
                )

        hits.sort(key=lambda hit: (-hit.score, hit.rel_path, hit.line))
        return hits[: max(1, max_results)]

    # ------------------------------------------------------------------ #
    # RAG loading
    # ------------------------------------------------------------------ #
    def load_documents(self, max_files: int | None = None) -> list[Document]:
        """Load indexable files as LangChain documents (one per file)."""
        documents: list[Document] = []
        for source in self.iter_source_files(max_files=max_files):
            try:
                slice_ = self.read_text(source.rel_path, max_lines=10**6)
            except RepositoryError:
                continue
            documents.append(
                Document(
                    page_content=slice_.text,
                    metadata={
                        "file": source.rel_path,
                        "suffix": Path(source.rel_path).suffix.lower(),
                        "size": source.size,
                    },
                )
            )
        return documents

    def stats(self, max_files: int | None = None) -> dict[str, int]:
        """File/size counters used by the CLI and the eval report."""
        files = 0
        total_bytes = 0
        for source in self.iter_source_files(max_files=max_files):
            files += 1
            total_bytes += source.size
        return {"files": files, "bytes": total_bytes}
