"""Sentence-transformer embedding utilities."""

from __future__ import annotations

import io
import os
import warnings
from contextlib import redirect_stderr, redirect_stdout
from typing import Sequence

import numpy as np
from sentence_transformers import SentenceTransformer

os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
warnings.filterwarnings("ignore")

try:
    from transformers.utils import logging as transformers_logging

    transformers_logging.set_verbosity_error()
except ImportError:
    pass

try:
    from huggingface_hub.utils import logging as hf_logging

    hf_logging.set_verbosity_error()
except ImportError:
    pass


class EmbeddingModel:
    """Wrapper around a sentence-transformers encoder."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        """Initialize the embedding encoder."""

        self.model_name = model_name
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            self.model = SentenceTransformer(model_name)

    def encode_texts(self, texts: Sequence[str]) -> np.ndarray:
        """Encode a batch of texts into L2-normalized dense vectors."""

        embeddings = self.model.encode(list(texts), convert_to_numpy=True, show_progress_bar=False)
        return self._normalize(embeddings)

    def encode_query(self, query: str) -> np.ndarray:
        """Encode a single query into a normalized dense vector."""

        return self.encode_texts([query])[0]

    @staticmethod
    def _normalize(embeddings: np.ndarray) -> np.ndarray:
        """Normalize embeddings to unit length for cosine similarity via inner product."""

        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        return embeddings / norms
