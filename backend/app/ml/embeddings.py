"""Text embedding backends.

The app talks to a single ``Embedder`` interface; which backend serves it is a
config decision (``EMBEDDING_BACKEND``):

- ``sentence-transformers`` — real biomedical embeddings (PubMedBERT). Heavy
  dependency (torch), installed via requirements-ml.txt; used in Docker/prod.
- ``hash`` — deterministic, dependency-free token-hashing embeddings. Not
  semantically meaningful, but texts sharing words land near each other, which
  is enough for tests and for local dev without torch.

All backends must produce vectors of EMBEDDING_DIM so the pgvector column and
index are backend-agnostic.
"""

import hashlib
import math
import re
from functools import lru_cache
from typing import Protocol

from app.config import settings

EMBEDDING_DIM = 768

_TOKEN = re.compile(r"[a-z0-9]+")


class Embedder(Protocol):
    def embed_texts(self, texts: list[str]) -> list[list[float]]: ...


class HashingEmbedder:
    """Feature-hashing bag-of-words embedder (test/dev fallback)."""

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def _embed(self, text: str) -> list[float]:
        vec = [0.0] * EMBEDDING_DIM
        for token in _TOKEN.findall(text.lower()):
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % EMBEDDING_DIM
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vec[index] += sign
        norm = math.sqrt(sum(v * v for v in vec))
        if norm == 0.0:
            return vec
        return [v / norm for v in vec]


class SentenceTransformerEmbedder:
    """PubMedBERT (or any sentence-transformers model) embedder.

    The model is loaded lazily on first use so importing the app never pulls
    torch into memory.
    """

    def __init__(self, model_name: str):
        self._model_name = model_name
        self._model = None

    def _get_model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:  # pragma: no cover - env-dependent
                raise RuntimeError(
                    "EMBEDDING_BACKEND=sentence-transformers but the package is "
                    "not installed; pip install -r requirements-ml.txt or set "
                    "EMBEDDING_BACKEND=hash"
                ) from exc
            self._model = SentenceTransformer(self._model_name)
        return self._model

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        embeddings = self._get_model().encode(texts, normalize_embeddings=True)
        return [list(map(float, row)) for row in embeddings]


@lru_cache(maxsize=1)
def get_embedder() -> Embedder:
    if settings.embedding_backend == "hash":
        return HashingEmbedder()
    return SentenceTransformerEmbedder(settings.embedding_model)
