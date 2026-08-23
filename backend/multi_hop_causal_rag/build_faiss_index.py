"""Build a FAISS index from dataset files for offline retrieval setup.

This script is intentionally separate from chatbot runtime so dataset loading
happens only during index construction.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import List

# Allow direct execution: python multi_hop_causal_rag/build_faiss_index.py
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from multi_hop_causal_rag.config import RetrievalConfig, build_default_config
from multi_hop_causal_rag.data.hotpot_loader import hotpotqa_to_documents
from multi_hop_causal_rag.data.preprocessing import Document, merge_datasets
from multi_hop_causal_rag.data.worldtree_loader import worldtree_to_documents
from multi_hop_causal_rag.retrieval.retriever import CausalRetriever


def _default_index_paths() -> tuple[str, str]:
    """Resolve default FAISS index/docstore paths from config or project defaults."""

    config = build_default_config().retrieval
    project_root = Path(__file__).resolve().parent.parent
    index_path = config.faiss_index_path or str(project_root / "faiss_index_512" / "index.faiss")
    docstore_path = config.faiss_docstore_path or str(project_root / "faiss_index_512" / "index.pkl")
    return index_path, docstore_path


def _build_documents(
    hotpot_path: str | None,
    worldtree_path: str | None,
    max_hotpot: int | None,
    max_worldtree: int | None,
) -> List[Document]:
    """Load and merge documents from selected datasets."""

    datasets: List[List[Document]] = []

    if hotpot_path:
        print(f"Loading HotpotQA: {hotpot_path}")
        datasets.append(hotpotqa_to_documents(path=hotpot_path, max_samples=max_hotpot))

    if worldtree_path:
        print(f"Loading WorldTree: {worldtree_path}")
        datasets.append(worldtree_to_documents(path=worldtree_path, max_samples=max_worldtree))

    documents = merge_datasets(datasets)
    if not documents:
        raise RuntimeError(
            "No documents were loaded. Provide at least one valid dataset path using "
            "--hotpot-path and/or --worldtree-path."
        )

    return documents


def build_faiss_index(
    hotpot_path: str | None,
    worldtree_path: str | None,
    index_path: str,
    docstore_path: str,
    max_hotpot: int | None,
    max_worldtree: int | None,
    embedding_model_name: str | None,
) -> None:
    """Build and persist a FAISS index from provided datasets."""

    documents = _build_documents(hotpot_path, worldtree_path, max_hotpot, max_worldtree)
    print(f"Loaded documents: {len(documents)}")

    retrieval_config = RetrievalConfig(
        embedding_model_name=embedding_model_name or build_default_config().retrieval.embedding_model_name,
        faiss_index_path=index_path,
        faiss_docstore_path=docstore_path,
        persist_faiss=True,
        sync_new_documents_on_startup=False,
    )

    retriever = CausalRetriever(config=retrieval_config)
    retriever.build_vector_index(documents)
    retriever.vector_index.save(index_path=index_path, docstore_path=docstore_path)

    print("FAISS index build completed.")
    print(f"Index path: {index_path}")
    print(f"Docstore path: {docstore_path}")


def _parse_args() -> argparse.Namespace:
    """Parse command-line options for offline FAISS index building."""

    default_index_path, default_docstore_path = _default_index_paths()

    parser = argparse.ArgumentParser(description="Build FAISS index from dataset files.")
    parser.add_argument("--hotpot-path", type=str, default=None, help="Path to HotpotQA source file")
    parser.add_argument("--worldtree-path", type=str, default=None, help="Path to WorldTree source file")
    parser.add_argument("--index-path", type=str, default=default_index_path, help="Output FAISS index path")
    parser.add_argument("--docstore-path", type=str, default=default_docstore_path, help="Output docstore path")
    parser.add_argument("--max-hotpot", type=int, default=None, help="Optional cap on HotpotQA examples")
    parser.add_argument("--max-worldtree", type=int, default=None, help="Optional cap on WorldTree rows")
    parser.add_argument("--embedding-model", type=str, default=None, help="Sentence transformer model name")
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint for FAISS index construction."""

    args = _parse_args()
    build_faiss_index(
        hotpot_path=args.hotpot_path,
        worldtree_path=args.worldtree_path,
        index_path=args.index_path,
        docstore_path=args.docstore_path,
        max_hotpot=args.max_hotpot,
        max_worldtree=args.max_worldtree,
        embedding_model_name=args.embedding_model,
    )


if __name__ == "__main__":
    main()
