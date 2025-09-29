# backend/core_logic/embeddings.py
from typing import List
from dataclasses import dataclass, field
import numpy as np
from .config import settings
import logging

logger = logging.getLogger(__name__)

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

@dataclass
class EmbeddingModel:
    """Wrapper for sentence transformer embedding model"""
    model_name: str = field(default_factory=lambda: settings.models.EMBEDDING_MODEL)
    _model: "SentenceTransformer" = field(default=None, init=False)  # type: ignore

    def _ensure_loaded(self) -> None:
        """Lazy load the model on first use"""
        if self._model is None:
            # Try to get from model manager first (for caching)
            try:
                from .model_manager import model_manager
                self._model = model_manager.get_embedder()
                logger.info("Got embedder from model manager cache")
            except Exception as e:
                # Fallback to direct loading
                logger.warning(f"Could not get embedder from cache: {e}")
                if SentenceTransformer is None:
                    raise RuntimeError("sentence-transformers is not installed.")
                self._model = SentenceTransformer(self.model_name)
                logger.info(f"Loaded embedding model directly: {self.model_name}")

    def encode(self, texts: List[str], **kwargs) -> np.ndarray:
        """Encode list of texts to embeddings"""
        self._ensure_loaded()
        return np.asarray(self._model.encode(
            texts, 
            convert_to_numpy=True, 
            normalize_embeddings=True,
            show_progress_bar=False,
            **kwargs
        ))

    def encode_one(self, text: str, **kwargs) -> np.ndarray:
        """Encode single text to embedding"""
        return self.encode([text], **kwargs)[0]
    
    @property
    def embedding_dim(self) -> int:
        """Get embedding dimension"""
        self._ensure_loaded()
        return self._model.get_sentence_embedding_dimension()