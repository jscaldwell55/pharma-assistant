# backend/core_logic/embeddings.py
from typing import List
from dataclasses import dataclass
import numpy as np
from config import settings

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

@dataclass
class EmbeddingModel:
    model_name: str = settings.models.EMBEDDING_MODEL
    _model: "SentenceTransformer" = None  # type: ignore

    def _ensure_loaded(self) -> None:
        if self._model is None:
            if SentenceTransformer is None:
                raise RuntimeError("sentence-transformers is not installed.")
            self._model = SentenceTransformer(self.model_name)

    def encode(self, texts: List[str]) -> np.ndarray:
        self._ensure_loaded()
        return np.asarray(self._model.encode(texts, convert_to_numpy=True, normalize_embeddings=True))

    def encode_one(self, text: str) -> np.ndarray:
        return self.encode([text])[0]