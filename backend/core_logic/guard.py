# backend/core_logic/guard.py
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import re
import logging
import os
from collections import OrderedDict

from .config import settings
from .grounding import compute_grounding, passes_grounding  # grounding gate

logger = logging.getLogger("guard")

# ---------- sentence-transformers (offline-first) ----------
try:
    from sentence_transformers import SentenceTransformer, util
except Exception:
    SentenceTransformer = None  # type: ignore
    util = None  # type: ignore

# Use a faster model for guard/grounding by default; can override by ENV
_EMBED_MODEL_NAME = os.getenv("GROUNDING_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# ---------- singletons / caches ----------
_embed_model: Optional["SentenceTransformer"] = None

# Tiny LRU cache for text -> embedding to avoid recomputing within a turn
_TEXT_EMB_CACHE: "OrderedDict[str, Any]" = OrderedDict()
_TEXT_EMB_CACHE_MAX = int(os.getenv("GUARD_TEXT_EMB_CACHE", "512"))

def _get_embed_model() -> "SentenceTransformer":
    if SentenceTransformer is None:
        raise RuntimeError("sentence-transformers not installed")
    global _embed_model
    if _embed_model is None:
        logger.info("guard: loading embedding model once: %s", _EMBED_MODEL_NAME)
        _embed_model = SentenceTransformer(_EMBED_MODEL_NAME)
    return _embed_model

def _get_text_embedding(text: str):
    key = (text or "").strip()
    if not key:
        return None
    if key in _TEXT_EMB_CACHE:
        _TEXT_EMB_CACHE.move_to_end(key)
        return _TEXT_EMB_CACHE[key]
    model = _get_embed_model()
    emb = model.encode([key], convert_to_tensor=True, normalize_embeddings=True, show_progress_bar=False)
    _TEXT_EMB_CACHE[key] = emb
    if len(_TEXT_EMB_CACHE) > _TEXT_EMB_CACHE_MAX:
        _TEXT_EMB_CACHE.popitem(last=False)
    return emb

def _emb_probs(text: str, labels: Dict[str, str]) -> Dict[str, float]:
    model = _get_embed_model()
    t = _get_text_embedding(text)
    lab_texts = list(labels.values())
    import numpy as np
    if util is None:
        raise RuntimeError("sentence-transformers 'util' missing; cannot compute cosine similarities.")
    L = model.encode(lab_texts, convert_to_tensor=True, normalize_embeddings=True, show_progress_bar=False)
    sims = util.cos_sim(t, L).cpu().numpy().flatten()
    exps = np.exp(sims - sims.max())
    probs = exps / (exps.sum() + 1e-9)
    return {k: float(p) for k, p in zip(labels.keys(), probs)}

# ==================
# Regex / heuristics
# ==================
SAFE_NUMBERS = {
    "1-800-FDA-1088",  # FDA MedWatch (alphanumeric)
    "1-800-332-1088",  # FDA MedWatch (numeric)
    "1-800-222-1222",  # Poison Control
    "988",             # Suicide & Crisis Lifeline
}

PII_PATTERNS = [
    re.compile(r"\b\d{3}[-.\s]?\d{2}[-.\s]?\d{4}\b"),  # SSN
    re.compile(r"\b(?:\+?1[-.\s])?\(?\d{3}\)?[-.\s]\d{3}[-.\s]\d{4}\b"),  # phone
    re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),  # email
    re.compile(r"\b\d{1,5}\s+[A-Za-z0-9.\s]+\b(?:St|Street|Rd|Road|Ave|Avenue|Blvd|Lane|Ln)\b", re.I),  # address-ish
]

AE_HINTS = re.compile(
    r"adverse event|had a reaction|I (?:felt|had|get) (?:rash|hives|swelling|trouble breathing|dizzy|faint)"
    r"|went to (?:ER|emergency)|hospitali[sz]ed|passed out|seizure|anaphylaxis|side effect.*(report|file)",
    re.I
)

SELF_HARM = re.compile(
    r"\b(kill myself|suicide|want to die|end my life|hurt myself|self[- ]?harm|take my life)\b", re.I
)

JAILBREAK = re.compile(
    r"(ignore (?:previous|prior) (?:instructions|rules)|disregard .*policy|override|bypass|workaround|loophole|"
    r"do anything now|DAN|developer mode|prompt[- ]?injection|jailbreak|"
    r"act as|role[- ]?play|pretend|in character|simulate|imagine you are|"
    r"fiction|short story|story|poem|song|lyrics|narrative|as a joke|make up)",
    re.I
)

CRIME_VIOLENCE = re.compile(
    r"(kill|murder|poison|harm|injure|maim|assault|terror|bomb|explosive|make\s+(?:a|an)\s+bomb|"
    r"build\s+(?:weapon|gun)|obtain\s+(?:illegal|unlawful)\s+drugs|how to (?:kill|poison)|"
    r"laced?\s+(?:drink|food)|silencer|untraceable\s+gun|fentanyl\s+(?:synthesis|recipe)|"
    r"overdose\s+(?:someone|them)|commit\s+(?:crime|fraud|murder)|break\s+into|bypass\s+lock)",
    re.I
)

CLASS_LABELS = ["medical_advice", "off_label_use", "general_info"]

@dataclass
class SafetyLabels:
    medical_advice: bool = False
    off_label_use: bool = False
    phi_pii: bool = False
    adverse_event: bool = False
    self_harm: bool = False
    jailbreak: bool = False
    illicit_or_violent: bool = False
    toxicity: Optional[str] = None

@dataclass
class GuardDecision:
    allow: bool
    reason: str = ""
    labels: SafetyLabels = field(default_factory=SafetyLabels)
    enforced_risk_snippet: Optional[str] = None
    force_factual: bool = False

class Guard:
    def __init__(self) -> None:
        pass

    def _mnli_or_embedding_probs(self, text: str) -> Dict[str, float]:
        # Pure embedding path (MNLI disabled/offline)
        return _emb_probs(text, {
            "medical_advice": "A request for individualized medical advice about dosing, timing, contraindications.",
            "off_label_use": "A request about unapproved uses of a medicine for conditions or populations.",
            "general_info": "A general factual question about approved labeling or patient information.",
        })

    def detect_pii(self, text: str) -> bool:
        if not text:
            return False
        for n in SAFE_NUMBERS:
            if n in text:
                return False
        return any(p.search(text) for p in PII_PATTERNS)

    def detect_ae(self, text: str) -> bool:
        if not text:
            return False
        return AE_HINTS.search(text) is not None

    def detect_self_harm(self, text: str) -> bool:
        if not text:
            return False
        return SELF_HARM.search(text) is not None

    def detect_jailbreak(self, text: str) -> bool:
        if not text:
            return False
        return JAILBREAK.search(text) is not None

    def detect_illicit_or_violent(self, text: str) -> bool:
        if not text:
            return False
        if CRIME_VIOLENCE.search(text):
            return True
        probs = _emb_probs(text, {
            "illicit": "A request for instructions to commit wrongdoing, violence, or to harm others.",
            "benign": "A harmless, informational, or medical question within policy.",
        })
        return probs.get("illicit", 0.0) > 0.60

    def fair_balance_snippet(self, retrieved_chunks: List[Dict[str, Any]]) -> Optional[str]:
        """
        Scan retrieved chunks for a concise, factual safety/risk sentence to use as a fair-balance note.
        Skips headings/questions; prefers complete declarative statements mentioning risks or adverse effects.
        """
        risk_words = re.compile(
            r"\b(risk|warning|side effect|adverse|precaution|contraindication)\b", re.I
        )
        skip_patterns = re.compile(
            r"^(table|figure|page|reference id|section\s+\d+|what are|list of|possible side effects)", re.I
        )

        candidates: list[str] = []
        for ch in retrieved_chunks:
            txt = (ch.get("text") or "")
            if not risk_words.search(txt):
                continue
            for s in re.split(r"(?<=[.!?])\s+", txt):
                s = s.strip()
                if not s:
                    continue
                if not risk_words.search(s):
                    continue
                if skip_patterns.match(s):
                    continue
                if 30 <= len(s) <= 240:  # tighter range; avoids headings & walls of text
                    candidates.append(s)

        # Prefer the shortest meaningful candidate (concise snippet)
        if candidates:
            candidates.sort(key=len)
            return candidates[0]

        return None


    def needs_fair_balance(self, response_text: str, retrieved_chunks: List[Dict[str, Any]]) -> bool:
        benefits = re.compile(r"effective|works|relieves|reduces|improves|benefit|helps|efficacy", re.I)
        if response_text and benefits.search(response_text):
            return True
        for ch in retrieved_chunks:
            if benefits.search((ch.get("text") or "")):
                return True
        return False

    def evaluate(
        self,
        user_query: str,
        draft_response: str,
        retrieved_chunks: List[Dict[str, Any]],
        *,
        context_text: Optional[str] = None,
    ) -> GuardDecision:
        labels = SafetyLabels()
        labels.self_harm = self.detect_self_harm(user_query)
        labels.jailbreak = self.detect_jailbreak(user_query)
        labels.illicit_or_violent = self.detect_illicit_or_violent(user_query)

        if labels.self_harm:
            return GuardDecision(allow=False, reason="self_harm", labels=labels)
        if labels.illicit_or_violent:
            return GuardDecision(allow=False, reason="illicit_or_violent", labels=labels)

        probs = self._mnli_or_embedding_probs(user_query)
        labels.medical_advice = probs.get("medical_advice", 0.0) >= settings.guard.ADVICE_THRESHOLD
        labels.off_label_use = probs.get("off_label_use", 0.0) >= settings.guard.OFF_LABEL_THRESHOLD
        labels.phi_pii = self.detect_pii(user_query)
        labels.adverse_event = self.detect_ae(user_query)

        if labels.off_label_use:
            return GuardDecision(allow=False, reason="off_label", labels=labels)
        if labels.medical_advice:
            return GuardDecision(allow=False, reason="medical_advice", labels=labels)

        # Grounding gate
        if not context_text:
            context_text = "\n".join([(c.get("text") or "") for c in (retrieved_chunks or [])])
        g = compute_grounding(draft_response or "", context_text or "")
        if not passes_grounding(g):
            return GuardDecision(allow=False, reason="ungrounded", labels=labels, force_factual=True)

        risk_snip = None
        if settings.guard.FAIR_BALANCE_ENFORCED and self.needs_fair_balance(draft_response, retrieved_chunks):
            risk_snip = self.fair_balance_snippet(retrieved_chunks)

        return GuardDecision(allow=True, labels=labels, enforced_risk_snippet=risk_snip)

# Public warmup hook (called by app.py)
def preload_guard_models() -> None:
    _ = _get_embed_model()