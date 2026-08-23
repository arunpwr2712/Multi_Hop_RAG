"""FAISS-backed dense vector index for document retrieval."""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Sequence

import faiss
import numpy as np

from ..data.preprocessing import Document


@dataclass
class RetrievalResult:
    """A ranked retrieval hit with similarity score and metadata."""

    document: Document
    score: float


class FaissVectorIndex:
    """In-memory FAISS index storing dense document embeddings."""

    def __init__(self) -> None:
        """Initialize an empty vector index."""

        self.documents: List[Document] = []
        self.index: faiss.IndexFlatIP | None = None
        self.dimension: int = 0

    def build(self, documents: Sequence[Document], embeddings: np.ndarray) -> None:
        """Build the FAISS index from a document collection and matching embeddings."""

        if len(documents) == 0:
            raise ValueError("Cannot build a FAISS index from an empty document set")
        if len(documents) != len(embeddings):
            raise ValueError("Document and embedding counts must match")

        self.documents = list(documents)
        self.dimension = int(embeddings.shape[1])
        self.index = faiss.IndexFlatIP(self.dimension)
        self.index.add(embeddings.astype("float32"))

    def add_documents(self, documents: Sequence[Document], embeddings: np.ndarray) -> None:
        """Append new documents and embeddings to an existing FAISS index."""

        if self.index is None:
            self.build(documents=documents, embeddings=embeddings)
            return
        if len(documents) == 0:
            return
        if len(documents) != len(embeddings):
            raise ValueError("Document and embedding counts must match")
        if embeddings.shape[1] != self.dimension:
            raise ValueError("Embedding dimension mismatch for existing FAISS index")

        self.documents.extend(list(documents))
        self.index.add(embeddings.astype("float32"))

    def load(self, index_path: str, docstore_path: str | None = None) -> None:
        """Load a FAISS index and associated document store from disk."""

        index_file = Path(index_path)
        if not index_file.exists():
            raise FileNotFoundError(f"FAISS index file not found: {index_file}")

        self.index = faiss.read_index(str(index_file))
        self.dimension = int(self.index.d)
        self.documents = []

        if docstore_path:
            self.documents = self._load_documents_from_docstore(Path(docstore_path))
        if len(self.documents) != int(self.index.ntotal):
            raise ValueError(
                "Loaded FAISS vectors and docstore size do not match. "
                f"index.ntotal={self.index.ntotal}, docstore_documents={len(self.documents)}"
            )

    def save(self, index_path: str, docstore_path: str) -> None:
        """Persist FAISS index and document store to disk."""

        if self.index is None:
            raise RuntimeError("Cannot save an uninitialized FAISS index")

        index_file = Path(index_path)
        docstore_file = Path(docstore_path)
        index_file.parent.mkdir(parents=True, exist_ok=True)
        docstore_file.parent.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self.index, str(index_file))
        payload = {
            "documents": [
                {
                    "doc_id": document.doc_id,
                    "text": document.text,
                    "source": document.source,
                    "metadata": document.metadata,
                }
                for document in self.documents
            ]
        }
        with docstore_file.open("wb") as handle:
            pickle.dump(payload, handle)

    def is_ready(self) -> bool:
        """Return whether the index has been built and can serve queries."""

        return self.index is not None and bool(self.documents)

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[RetrievalResult]:
        """Search the index with a normalized query embedding."""

        if not self.is_ready():
            raise RuntimeError("FAISS index has not been built")

        top_k = max(1, min(top_k, len(self.documents)))
        scores, indices = self.index.search(np.asarray([query_embedding], dtype="float32"), top_k)

        results: List[RetrievalResult] = []
        for score, index in zip(scores[0], indices[0]):
            if index < 0:
                continue
            results.append(RetrievalResult(document=self.documents[int(index)], score=float(score)))
        return results

    def _load_documents_from_docstore(self, docstore_path: Path) -> List[Document]:
        """Load documents from project-native or LangChain FAISS docstore formats."""

        if not docstore_path.exists():
            raise FileNotFoundError(f"Docstore file not found: {docstore_path}")

        with docstore_path.open("rb") as handle:
            payload = pickle.load(handle)

        documents = self._parse_native_docstore(payload)
        if documents:
            return documents

        documents = self._parse_langchain_docstore(payload)
        if documents:
            return documents

        raise ValueError(f"Unsupported docstore format in: {docstore_path}")

    def _parse_native_docstore(self, payload: Any) -> List[Document]:
        """Parse the project-native docstore payload format."""

        if isinstance(payload, dict) and isinstance(payload.get("documents"), list):
            return [
                Document(
                    doc_id=str(item["doc_id"]),
                    text=str(item["text"]),
                    source=str(item.get("source", "unknown")),
                    metadata=dict(item.get("metadata", {})),
                )
                for item in payload["documents"]
            ]
        return []

    def _parse_langchain_docstore(self, payload: Any) -> List[Document]:
        """Parse LangChain FAISS store payload `(docstore, index_to_docstore_id)`."""

        if not isinstance(payload, tuple) or len(payload) != 2:
            return []
        docstore, index_to_docstore_id = payload
        if not isinstance(index_to_docstore_id, dict):
            return []

        documents: List[Document] = []
        for vector_id in range(len(index_to_docstore_id)):
            store_id = index_to_docstore_id.get(vector_id)
            if store_id is None:
                continue
            source_doc = docstore._dict.get(store_id) if hasattr(docstore, "_dict") else None
            if source_doc is None:
                continue
            page_content = getattr(source_doc, "page_content", "")
            metadata = dict(getattr(source_doc, "metadata", {}) or {})
            documents.append(
                Document(
                    doc_id=str(metadata.get("doc_id", store_id)),
                    text=str(page_content),
                    source=str(metadata.get("source", "external-faiss")),
                    metadata=metadata,
                )
            )
        return documents
