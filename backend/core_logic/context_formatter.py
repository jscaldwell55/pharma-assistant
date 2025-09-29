# backend/core_logic/context_formatter.py
from typing import List, Dict, Any, Optional
import re
import os

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")
_RISK_WORDS = [
    "side effect", "adverse", "warning", "risk", "contraindication",
    "dizziness", "somnolence", "rash", "itch", "nausea", "vomit",
    "pregnant", "breast", "hepatic", "grapefruit", "drug interaction"
]

def _sentences(txt: str) -> List[str]:
    if not txt:
        return []
    sents = _SENT_SPLIT.split(txt.strip())
    return [s.strip() for s in sents if s.strip()]

def _score_sentence(q_tokens: set, s: str) -> int:
    t = set(re.findall(r"[a-zA-Z0-9]+", s.lower()))
    return len(q_tokens & t)

def _tokenize(s: str) -> set:
    return set(re.findall(r"[a-zA-Z0-9]+", (s or "").lower()))

def _contains_risk(s: str) -> bool:
    s_lower = s.lower()
    return any(w in s_lower for w in _RISK_WORDS)

def select_relevant_sentences(query: str, text: str, limit: int = 6) -> List[str]:
    qtok = _tokenize(query)
    sents = _sentences(text or "")
    scored = []
    for s in sents:
        score = _score_sentence(qtok, s)
        if _contains_risk(s):
            score += 2
        if score > 0:
            scored.append((score, s))
    scored.sort(key=lambda x: x[0], reverse=True)
    seen = set()
    out: List[str] = []
    for _, s in scored:
        if s.lower() in seen:
            continue
        out.append(s)
        seen.add(s.lower())
        if len(out) >= limit:
            break
    if not out:
        out = sents[:max(3, limit // 2)]
    return out

def build_context(retrieved: List[Dict[str, Any]], query: str, char_budget: int = 2000) -> Dict[str, Any]:
    pieces: List[str] = []
    sources: List[Dict[str, Any]] = []
    used = 0
    for idx, ch in enumerate(retrieved, start=1):
        text = ch.get("text") or ""
        if not text:
            continue
        sents = select_relevant_sentences(query, text, limit=6)
        snippet = " ".join(sents)
        if not snippet:
            continue
        block = f"[{idx}] {snippet}"
        if used + len(block) + 2 > char_budget:
            break
        pieces.append(block)
        sources.append({
            "id": idx,
            "meta": ch.get("meta", {}),
            "score": ch.get("rerank_score"),
            "vector_score": ch.get("score")
        })
        used += len(block) + 2
    ctx = "\n\n".join(pieces) if pieces else ""
    return {"context": ctx, "sources": sources}

def append_fair_balance(response_text: str, risk_snippet: Optional[str], position: Optional[str] = None) -> str:
    """
    position: "top" or "end" (default from FAIR_BALANCE_POSITION env, else "end")
    """
    if not risk_snippet:
        return response_text
    pos = (position or os.getenv("FAIR_BALANCE_POSITION", "end")).lower()
    note = f"**Safety note:** {risk_snippet}"
    if note.lower() in (response_text or "").lower():
        return response_text
    if pos == "top":
        return f"{note}\n\n{response_text}".strip()
    return (response_text or "").rstrip() + f"\n\n{note}"

def extractive_bullets(query: str, retrieved: List[Dict[str, Any]], max_items: int = 8) -> str:
    """
    Non-LLM fallback: pick high-signal sentences across retrieved chunks and return bullet points with citations.
    """
    picked: List[tuple[str, int]] = []
    for idx, ch in enumerate(retrieved, start=1):
        text = ch.get("text", "") or ""
        for s in select_relevant_sentences(query, text, limit=3):
            picked.append((s, idx))
    seen = set()
    bullets: List[str] = []
    for s, i in picked:
        key = s.lower()
        if key in seen:
            continue
        seen.add(key)
        bullets.append(f"- {s} [{i}]")
        if len(bullets) >= max_items:
            break
    if not bullets:
        return "I don’t have enough context to answer."
    return "Here’s what the approved documents say:\n\n" + "\n".join(bullets)
