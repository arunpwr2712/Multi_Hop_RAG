"""High-level retrieval interface for building and querying the dense index."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np

from ..config import RetrievalConfig
from ..data.preprocessing import Document
from .embedding_model import EmbeddingModel
from .vector_index import FaissVectorIndex, RetrievalResult


@dataclass
class RetrieverBundle:
    """Stateful retrieval bundle holding the embedding model and FAISS index."""

    embedding_model: EmbeddingModel
    vector_index: FaissVectorIndex


class CausalRetriever:
    """Dense retriever for iterative evidence retrieval in the causal pipeline."""

    def __init__(self, config: Optional[RetrievalConfig] = None) -> None:
        """Initialize the retriever with embedding and vector index components."""

        self.config = config or RetrievalConfig()
        self.embedding_model: Optional[EmbeddingModel] = None
        self.vector_index = FaissVectorIndex()
        self._warned_hash_fallback = False

    def _get_embedding_model(self) -> EmbeddingModel:
        """Create the embedding model on first use to reduce startup overhead."""

        if self.embedding_model is None:
            self.embedding_model = EmbeddingModel(self.config.embedding_model_name)
        return self.embedding_model

    def build_vector_index(self, documents: Sequence[Document]) -> None:
        """Embed the corpus and build the FAISS vector index."""

        embedding_model = self._get_embedding_model()
        embeddings = embedding_model.encode_texts([document.text for document in documents])
        self.vector_index.build(documents=documents, embeddings=embeddings)

    def initialize_with_documents(self, documents: Sequence[Document]) -> dict:
        """Initialize retrieval state from local FAISS files or build from scratch."""

        index_path = Path(self.config.faiss_index_path) if self.config.faiss_index_path else None
        docstore_path = Path(self.config.faiss_docstore_path) if self.config.faiss_docstore_path else None

        if index_path and docstore_path and index_path.exists() and docstore_path.exists():
            self.vector_index.load(index_path=str(index_path), docstore_path=str(docstore_path))

            existing_vectors = len(self.vector_index.documents)
            new_documents: List[Document] = []
            if self.config.sync_new_documents_on_startup:
                existing_doc_ids = {document.doc_id for document in self.vector_index.documents}
                new_documents = [document for document in documents if document.doc_id not in existing_doc_ids]
                if new_documents:
                    embedding_model = self._get_embedding_model()
                    embeddings = embedding_model.encode_texts([document.text for document in new_documents])
                    self.vector_index.add_documents(documents=new_documents, embeddings=embeddings)
                    if self.config.persist_faiss:
                        self.vector_index.save(index_path=str(index_path), docstore_path=str(docstore_path))

            return {
                "index_mode": "loaded_existing",
                "existing_vectors": existing_vectors,
                "new_documents_added": len(new_documents),
                "total_vectors": len(self.vector_index.documents),
                "sync_new_documents_on_startup": self.config.sync_new_documents_on_startup,
                "index_path": str(index_path),
                "docstore_path": str(docstore_path),
            }

        self.build_vector_index(documents)
        if index_path and docstore_path and self.config.persist_faiss:
            self.vector_index.save(index_path=str(index_path), docstore_path=str(docstore_path))

        return {
            "index_mode": "rebuilt_from_documents",
            "existing_vectors": 0,
            "new_documents_added": len(documents),
            "total_vectors": len(self.vector_index.documents),
            "index_path": str(index_path) if index_path else None,
            "docstore_path": str(docstore_path) if docstore_path else None,
        }

    def retrieve_documents(self, query: str, top_k: Optional[int] = None) -> List[RetrievalResult]:
        """Retrieve the top ranked evidence documents for a query."""

        search_k = top_k or self.config.top_k
        query_embedding = self._encode_query_with_fallback(query)
        return self.vector_index.search(query_embedding=query_embedding, top_k=search_k)

    def _encode_query_with_fallback(self, query: str) -> np.ndarray:
        """Encode query with sentence-transformer, or use hash fallback if model init fails."""

        try:
            embedding_model = self._get_embedding_model()
            return embedding_model.encode_query(query)
        except Exception as exc:
            if not self._warned_hash_fallback:
                print(f"Warning: embedding model unavailable ({exc}). Using hash-based query embedding fallback.")
                self._warned_hash_fallback = True

            if self.vector_index.dimension <= 0:
                raise RuntimeError(
                    "Cannot create fallback query embedding because FAISS index dimension is unknown. "
                    "Ensure a valid FAISS index/docstore is loaded."
                ) from exc
            return self._hash_query_embedding(query, self.vector_index.dimension)

    @staticmethod
    def _hash_query_embedding(query: str, dimension: int) -> np.ndarray:
        """Create a deterministic normalized sparse vector for emergency retrieval fallback."""

        vector = np.zeros(dimension, dtype="float32")
        for token in query.lower().split():
            digest = hashlib.md5(token.encode("utf-8")).digest()
            bucket = int.from_bytes(digest[:4], byteorder="little", signed=False) % dimension
            sign = 1.0 if (digest[4] % 2 == 0) else -1.0
            vector[bucket] += sign

        norm = float(np.linalg.norm(vector))
        if norm == 0.0:
            vector[0] = 1.0
            return vector
        return vector / norm


def build_vector_index(documents: Sequence[Document], config: Optional[RetrievalConfig] = None) -> RetrieverBundle:
    """Build a retriever bundle with embeddings and a FAISS index."""

    retriever = CausalRetriever(config=config)
    retriever.build_vector_index(documents)
    return RetrieverBundle(embedding_model=retriever._get_embedding_model(), vector_index=retriever.vector_index)


def retrieve_documents(
    query: str,
    top_k: int,
    retriever: Optional[CausalRetriever] = None,
    bundle: Optional[RetrieverBundle] = None,
) -> List[RetrievalResult]:
    """Retrieve top-k documents using either a retriever instance or a prebuilt bundle."""

    if retriever is None and bundle is None:
        raise ValueError("Either a CausalRetriever or RetrieverBundle must be provided")

    if retriever is not None:
        return retriever.retrieve_documents(query=query, top_k=top_k)

    ad_hoc_retriever = CausalRetriever()
    ad_hoc_retriever.embedding_model = bundle.embedding_model
    ad_hoc_retriever.vector_index = bundle.vector_index
    return ad_hoc_retriever.retrieve_documents(query=query, top_k=top_k)
