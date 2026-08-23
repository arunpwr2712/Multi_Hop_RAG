"""Project configuration utilities for the multi-hop causal RAG system."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _load_env_file(env_path: Path) -> None:
    """Load key-value pairs from a .env file without external dependencies."""

    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


_load_env_file(_PROJECT_ROOT / ".env")


@dataclass
class LLMConfig:
    """Configuration for local Ollama text generation."""

    model: str = field(default_factory=lambda: os.getenv("MHC_RAG_LLM_MODEL", "llama3:latest"))
    base_url: str = field(default_factory=lambda: os.getenv("MHC_RAG_LLM_BASE_URL", "http://localhost:11434"))
    query_planning_enabled: bool = field(
        default_factory=lambda: os.getenv("MHC_RAG_QUERY_PLANNING_ENABLED", "true").strip().lower() == "true"
    )
    require_ollama: bool = field(
        default_factory=lambda: os.getenv("MHC_RAG_REQUIRE_OLLAMA", "true").strip().lower() == "true"
    )
    max_output_tokens: int = field(default_factory=lambda: int(os.getenv("MHC_RAG_MAX_OUTPUT_TOKENS", "1200")))
    temperature: float = field(default_factory=lambda: float(os.getenv("MHC_RAG_TEMPERATURE", "0.1")))
    timeout_seconds: int = field(default_factory=lambda: int(os.getenv("MHC_RAG_TIMEOUT", "45")))


@dataclass
class RetrievalConfig:
    """Configuration for embeddings and dense retrieval."""

    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    top_k: int = 5
    hop_top_k: int = 4
    similarity_metric: str = "cosine"
    faiss_index_path: Optional[str] = field(default_factory=lambda: os.getenv("MHC_RAG_FAISS_INDEX_PATH"))
    faiss_docstore_path: Optional[str] = field(default_factory=lambda: os.getenv("MHC_RAG_FAISS_DOCSTORE_PATH"))
    persist_faiss: bool = field(default_factory=lambda: os.getenv("MHC_RAG_PERSIST_FAISS", "true").lower() == "true")
    sync_new_documents_on_startup: bool = field(
        default_factory=lambda: os.getenv("MHC_RAG_SYNC_NEW_DOCS_ON_STARTUP", "false").lower() == "true"
    )


@dataclass
class PipelineConfig:
    """Configuration for the iterative multi-hop causal pipeline."""

    max_hops: int = 6
    max_candidate_paths: int = 5
    max_path_depth: int = 5
    min_edge_confidence: float = 0.2
    missing_link_similarity_threshold: float = 0.55


@dataclass
class AppConfig:
    """Top-level configuration container."""

    llm: LLMConfig = field(default_factory=LLMConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)


def build_default_config() -> AppConfig:
    """Construct the default application configuration."""

    return AppConfig()
