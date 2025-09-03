# reranker.py – Cross-encoder reranker
from typing import List, Dict, Any
from dataclasses import dataclass
from .config import settings

try:
    from sentence_transformers import CrossEncoder
except Exception:
    CrossEncoder = None

@dataclass
class CrossEncoderReranker:
    model_name: str = settings.models.RERANKER_MODEL
    _model: "CrossEncoder" = None  # type: ignore

    def _ensure_loaded(self) -> None:
        if self._model is None:
            if CrossEncoder is None:
                raise RuntimeError("sentence-transformers is not installed (CrossEncoder).")
            self._model = CrossEncoder(self.model_name)

    def rerank(self, query: str, passages: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
        self._ensure_loaded()
        pairs = [(query, p["text"]) for p in passages]
        scores = self._model.predict(pairs)
        for p, s in zip(passages, scores):
            p["rerank_score"] = float(s)
        passages.sort(key=lambda x: x["rerank_score"], reverse=True)
        return passages[:top_k]