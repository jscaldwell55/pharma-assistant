# backend/core_logic/rag.py
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
import logging
import os
import re
import math
import time

logger = logging.getLogger("rag")

# Optional reranker - check if it should be skipped
SKIP_RERANKER = os.getenv("SKIP_RERANKER", "false").lower() == "true"

if not SKIP_RERANKER:
    try:
        from sentence_transformers import CrossEncoder
    except Exception:
        CrossEncoder = None
        logger.warning("CrossEncoder not available, will skip reranking")
else:
    CrossEncoder = None
    logger.info("Reranker disabled by SKIP_RERANKER environment variable")

# Lexical scoring
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    import numpy as np
except Exception:
    TfidfVectorizer = None
    np = None


def _minmax(xs: List[float]) -> List[float]:
    if not xs:
        return []
    lo, hi = min(xs), max(xs)
    if math.isclose(hi, lo):
        return [0.0 for _ in xs]
    return [(x - lo) / (hi - lo) for x in xs]


def _norm_q(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())


class _TTLCache:
    def __init__(self, maxsize: int = 256, ttl_sec: int = 1800):
        self.maxsize = maxsize
        self.ttl = ttl_sec
        self._d: Dict[str, Tuple[float, Any]] = {}

    def get(self, key: str):
        item = self._d.get(key)
        if not item:
            return None
        ts, val = item
        if time.time() - ts > self.ttl:
            self._d.pop(key, None)
            return None
        return val

    def set(self, key: str, val: Any):
        self._d[key] = (time.time(), val)
        if len(self._d) > self.maxsize:
            # evict oldest
            oldest_key = min(self._d.items(), key=lambda kv: kv[1][0])[0]
            self._d.pop(oldest_key, None)

    def clear(self):
        self._d.clear()


@dataclass
class RAGRetriever:
    vector_client: Any
    # Reduce candidates for memory efficiency
    top_k: int = int(os.getenv("RAG_TOPK", "25"))  # Reduced from 60
    final_k: int = int(os.getenv("RAG_FINALK", "5"))
    namespace: Optional[str] = os.getenv("PINECONE_NAMESPACE") or None

    # Models / weights - adjust when reranker is disabled
    reranker_name: str = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    W_RERANK: float = float(os.getenv("RAG_W_RERANK", "0.0" if SKIP_RERANKER else "0.60"))
    W_LEX: float = float(os.getenv("RAG_W_LEX", "0.50" if SKIP_RERANKER else "0.25"))
    W_VEC: float = float(os.getenv("RAG_W_VEC", "0.50" if SKIP_RERANKER else "0.15"))

    # Term-rescue settings
    RESCUE_MIN_HITS: int = int(os.getenv("RAG_RESCUE_MIN_HITS", "2"))

    # Cache settings
    ENABLE_CACHE: bool = bool(int(os.getenv("RAG_CACHE", "1")))
    CACHE_MAX: int = int(os.getenv("RAG_CACHE_MAX", "128"))  # Reduced cache size
    CACHE_TTL: int = int(os.getenv("RAG_CACHE_TTL", "1800"))

    def __post_init__(self):
        self._ce = None
        if not SKIP_RERANKER:
            try:
                from .model_manager import model_manager
                self._ce = model_manager.get_reranker()
                if self._ce:
                    logger.info("Got cached reranker from ModelManager")
                else:
                    logger.info("Reranker disabled or not available")
                    self.W_RERANK = 0.0
                    self.W_LEX = 0.50
                    self.W_VEC = 0.50
            except Exception as e:
                logger.warning("Reranker unavailable (%s) – vector+lexical only.", str(e))
                self.W_RERANK = 0.0
                self.W_LEX = 0.50
                self.W_VEC = 0.50
        else:
            logger.info("Reranker disabled by SKIP_RERANKER - using vector+lexical scoring only")

        self._cache = _TTLCache(self.CACHE_MAX, self.CACHE_TTL)
        self._last_cache_hit: bool = False

    # --- intent & query processing ---
    _INTENT_HINTS = {
        "ae": {"side", "effect", "effects", "adverse", "reaction", "reactions", "safety", "risk", "warning",
                "precaution", "contraindication", "symptom", "symptoms"},
        "dose": {"dose", "dosage", "dosing", "how much", "mg", "tablet", "frequency"},
        "interact": {"interact", "interaction", "interactions", "cyp3a", "contraindicated", "grapefruit"},
    }

    _SYNONYMS = {
        "dizziness": {"dizzy"}, "dizzy": {"dizziness"},
        "vomiting": {"vomit"},
        "nausea": {"nauseated"}, "nauseated": {"nausea"},
        "rash": {"hives", "urticaria"},
        "contraindication": {"contraindicated", "do not use"},
        "contraindicated": {"contraindication"},
        "interaction": {"interactions", "interact"},
        "interactions": {"interaction", "interact"},
        "interact": {"interaction", "interactions"},
        "diarrhea": {"diarrhoea"},
        "hemorrhage": {"haemorrhage"},
    }

    _SECTION_TITLES = {
        "ae": {"adverse reactions", "side effects", "safety"},
        "dose": {"dosage", "administration"},
        "interact": {"drug interactions", "interactions", "contraindications"},
    }

    def _tokenize(self, s: str) -> List[str]:
        return re.findall(r"[a-zA-Z0-9]+", (s or "").lower())

    def _detect_intent(self, q: str) -> Optional[str]:
        toks = set(self._tokenize(q))
        scores = {k: len(v & toks) for k, v in self._INTENT_HINTS.items()}
        best = max(scores, key=scores.get)
        return best if scores[best] > 0 else None

    def _expand_query_terms(self, q: str) -> List[str]:
        toks = self._tokenize(q)
        expanded = set(toks)
        for t in list(toks):
            if t in self._SYNONYMS:
                expanded |= self._SYNONYMS[t]
        for t in list(toks):
            expanded.add(t[:-1] if t.endswith("s") else t + "s")
        return sorted(expanded)

    def _build_variants(self, q: str, intent: Optional[str]) -> Tuple[List[str], List[str]]:
        base_terms = self._expand_query_terms(q)
        variants = [" ".join(base_terms)]
        if intent == "ae":
            variants.append(" ".join(base_terms + ["adverse reactions", "side effects", "safety", "risk"]))
        elif intent == "dose":
            variants.append(" ".join(base_terms + ["dosage", "administration"]))
        elif intent == "interact":
            variants.append(" ".join(base_terms + ["drug interactions", "contraindications"]))
        variants.insert(0, q.strip())
        key_terms = sorted(set(base_terms))
        return variants[:3], key_terms  # Reduced variants to save memory

    def _dedup(self, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        seen = set()
        out = []
        for r in items:
            meta = r.get("meta") or {}
            key = (
                meta.get("doc_id"),
                meta.get("chunk_id"),
                meta.get("source"),
                r.get("id"),
                (r.get("text") or "")[:96],
            )
            if key in seen:
                continue
            seen.add(key)
            out.append(r)
        return out

    def _section_boost(self, item: Dict[str, Any], intent: Optional[str]) -> float:
        if not intent:
            return 0.0
        txt = (item.get("text") or "").lower()
        meta = item.get("meta") or {}
        heading = (meta.get("heading") or meta.get("section") or "").lower()
        candidates = self._SECTION_TITLES.get(intent, set())
        if any(h in heading for h in candidates):
            return 0.15
        if any(h in txt[:250] for h in candidates):
            return 0.10
        return 0.0

    def _lexical_scores(self, query: str, cands: List[Dict[str, Any]]) -> List[float]:
        if TfidfVectorizer is None or np is None or not cands:
            return [0.0] * len(cands)
        texts = [query] + [c.get("text", "") or "" for c in cands]

        vec = TfidfVectorizer(
            ngram_range=(1, 1),  # Reduced from (1,2) to save memory
            max_features=5000,   # Reduced from 20000
            stop_words="english",
            norm="l2"
        )
        try:
            X = vec.fit_transform(texts)
        except Exception as exc:
            logger.warning("TF-IDF failed: %s, returning zeros", exc)
            return [0.0] * len(cands)
        qv = X[0]
        D = X[1:]
        sims = (D @ qv.T).toarray().flatten()
        sims = np.clip(sims, 0.0, 1.0).tolist()
        return sims

    def _rerank(self, query: str, cands: List[Dict[str, Any]]) -> List[float]:
        if self._ce is None or not cands:
            return [0.0] * len(cands)

        batch_size = 10
        all_scores = []

        for i in range(0, len(cands), batch_size):
            batch = cands[i:i + batch_size]
            pairs = [(query, r.get("text", "") or "") for r in batch]
            try:
                scores = self._ce.predict(pairs, convert_to_numpy=True, show_progress_bar=False)
                all_scores.extend(scores.tolist())
            except Exception as exc:
                logger.warning("Reranking batch failed: %s", exc)
                all_scores.extend([0.0] * len(batch))

        return all_scores

    # --------- Cache API ---------
    def clear_cache(self):
        self._cache.clear()

    @property
    def last_cache_hit(self) -> bool:
        return self._last_cache_hit

    # --------- Main retrieval ---------
    def retrieve(self, user_query: str) -> List[Dict[str, Any]]:
        intent = self._detect_intent(user_query)
        cache_key = f"{self.namespace or ''}::{_norm_q(user_query.lower().strip('?!. '))}"

        if self.ENABLE_CACHE:
            cached = self._cache.get(cache_key)
            if cached is not None:
                self._last_cache_hit = True
                logger.info("RAG cache hit for query")
                return cached
        self._last_cache_hit = False

        variants, key_terms = self._build_variants(user_query, intent)
        logger.info("Intent=%s · variants=%d", intent, len(variants))

        # Vector fan-out with timeout and error handling
        all_cands: List[Dict[str, Any]] = []
        per_variant = max(1, self.top_k // max(1, len(variants)))
        query_timeout = float(os.getenv("PINECONE_QUERY_TIMEOUT", "25.0"))  # Slightly less than gunicorn timeout

        for i, vq in enumerate(variants):
            try:
                logger.debug("Querying variant %d/%d: %s", i + 1, len(variants), vq[:100] + "..." if len(vq) > 100 else vq)
                hits = self.vector_client.query(vq, top_k=per_variant, namespace=self.namespace, timeout=query_timeout) or []
                all_cands.extend(hits)
                logger.debug("Variant %d returned %d candidates", i + 1, len(hits))
            except Exception as exc:
                logger.error("Vector query failed for variant %d (%s): %s", i + 1, vq[:50], exc)
                # Continue with other variants - partial results better than total failure
                continue

        merged = self._dedup(all_cands)
        logger.info("Retrieved %d merged candidates", len(merged))

        if not merged:
            if self.ENABLE_CACHE:
                self._cache.set(cache_key, [])
            return []

        vec_scores_raw = [float(r.get("score", 0.0) or 0.0) for r in merged]
        vec_scores = _minmax(vec_scores_raw)

        lex_scores = self._lexical_scores(user_query, merged)

        rr_scores = self._rerank(user_query, merged)
        rr_scores = _minmax(rr_scores)

        sec_boosts = [self._section_boost(r, intent) for r in merged]

        fused = []
        for i, r in enumerate(merged):
            s = (
                self.W_RERANK * rr_scores[i]
                + self.W_LEX * lex_scores[i]
                + self.W_VEC * vec_scores[i]
                + sec_boosts[i]
            )
            rr = dict(r)
            rr["rerank_score"] = float(rr_scores[i])
            rr["lex_score"] = float(lex_scores[i])
            rr["vec_score_norm"] = float(vec_scores[i])
            rr["section_boost"] = float(sec_boosts[i])
            rr["combo_score"] = float(s)
            fused.append(rr)

        fused.sort(key=lambda x: x["combo_score"], reverse=True)
        top = fused[: self.final_k]

        def _contains_any(text: str, terms: List[str]) -> bool:
            t = (text or "").lower()
            return any(term.lower() in t for term in terms)

        have_hits = sum(1 for r in top if _contains_any(r.get("text", ""), key_terms))

        if key_terms and have_hits < self.RESCUE_MIN_HITS:
            pool = [r for r in fused[self.final_k:] if _contains_any(r.get("text", ""), key_terms)]
            need = self.RESCUE_MIN_HITS - have_hits
            if pool and need > 0:
                non_matching = [r for r in top if not _contains_any(r.get("text", ""), key_terms)]
                non_matching.sort(key=lambda x: x["combo_score"])
                replace = non_matching[:need]
                keep = [r for r in top if r not in replace]
                top = keep + pool[:need]
                top.sort(key=lambda x: x["combo_score"], reverse=True)
                logger.info("Term-rescue applied: inserted %d matching candidate(s)", len(pool[:need]))

        logger.info(
            "Final %d (combo/rr/lex/vec): %s",
            len(top),
            [
                (
                    round(r["combo_score"], 3),
                    round(r["rerank_score"], 3),
                    round(r["lex_score"], 3),
                    round(r["vec_score_norm"], 3),
                )
                for r in top
            ],
        )

        if self.ENABLE_CACHE:
            self._cache.set(cache_key, top)

        return top
