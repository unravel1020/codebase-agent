"""codebase-agent: a small LLM agent that answers questions about a local repository."""

from .agent import AgentResult, CodebaseAgent, ToolCallRecord
from .config import ConfigError, Settings
from .embeddings import HashingEmbeddings, build_embeddings
from .memory import ConversationMemory
from .offline import HeuristicPolicy, ScriptedChatModel, build_offline_llm, tool_call
from .rag import CodeRetriever, build_retriever, chunk_documents
from .repository import (
    BinaryFileError,
    FileTooLargeError,
    PathSecurityError,
    Repository,
    RepositoryError,
)
from .schemas import AnswerParseError, CodeAnswer, EvidenceItem, parse_answer
from .tools import build_tools, build_tools_from_settings
from .vectorstore import RetrievedChunk, VectorIndex

__version__ = "0.1.0"

__all__ = [
    "AgentResult",
    "AnswerParseError",
    "BinaryFileError",
    "CodeAnswer",
    "CodeRetriever",
    "CodebaseAgent",
    "ConfigError",
    "ConversationMemory",
    "EvidenceItem",
    "FileTooLargeError",
    "HashingEmbeddings",
    "HeuristicPolicy",
    "PathSecurityError",
    "Repository",
    "RepositoryError",
    "RetrievedChunk",
    "ScriptedChatModel",
    "Settings",
    "ToolCallRecord",
    "VectorIndex",
    "build_embeddings",
    "build_offline_llm",
    "build_retriever",
    "build_tools",
    "build_tools_from_settings",
    "chunk_documents",
    "parse_answer",
    "tool_call",
]
