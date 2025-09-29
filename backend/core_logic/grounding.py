# backend/core_logic/grounding.py
import os
import re
import logging
from typing import List, Dict, Any
import numpy as np

from sentence_transformers import SentenceTransformer, util

logger = logging.getLogger("grounding")

_MODEL_NAME = os.getenv("GROUNDING_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
_model = None


def _get_model():
    """Get the grounding model via the shared ModelManager singleton."""
    from .model_manager import model_manager
    return model_manager.get_grounding_model()


_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def _split_sentences(text: str, max_sentences: int) -> List[str]:
    if not text:
        return []
    sents = [s.strip() for s in _SENT_SPLIT.split(text) if s.strip()]
    return sents[:max_sentences]


def compute_grounding(draft: str, context: str) -> Dict[str, Any]:
    draft_sents = _split_sentences(draft, max_sentences=20)
    ctx_sents = _split_sentences(context, max_sentences=60)
    if not draft_sents or not ctx_sents:
        return {
            "avg_max_sim": 0.0,
            "min_max_sim": 0.0,
            "covered_frac": 0.0,
            "per_sentence": [],
            "draft_sentences": draft_sents,
            "context_sentences": len(ctx_sents),
        }

    model = _get_model()
    d_emb = model.encode(draft_sents, convert_to_tensor=True, normalize_embeddings=True, show_progress_bar=False)
    c_emb = model.encode(ctx_sents, convert_to_tensor=True, normalize_embeddings=True, show_progress_bar=False)
    sims = util.cos_sim(d_emb, c_emb)
    per_sentence = sims.max(dim=1).values.detach().cpu().numpy().tolist()

    min_thresh = float(os.getenv("GROUNDING_MIN_THRESH", "0.25"))
    avg_max = float(np.mean(per_sentence))
    min_max = float(np.min(per_sentence))
    covered_frac = float(np.mean([1.0 if s >= min_thresh else 0.0 for s in per_sentence]))

    return {
        "avg_max_sim": avg_max,
        "min_max_sim": min_max,
        "covered_frac": covered_frac,
        "per_sentence": per_sentence,
        "draft_sentences": draft_sents,
        "context_sentences": len(ctx_sents),
    }


def passes_grounding(details: Dict[str, Any]) -> bool:
    avg_thresh = float(os.getenv("GROUNDING_AVG_THRESH", "0.42"))
    covered_thresh = float(os.getenv("GROUNDING_COVERED_FRAC", "0.65"))
    return (
        details.get("avg_max_sim", 0.0) >= avg_thresh
        and details.get("covered_frac", 0.0) >= covered_thresh
    )


def _warmup_grounding_model() -> None:
    _ = _get_model()
