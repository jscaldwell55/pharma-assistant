# regulatory_defaults.py
"""
Regulatory defaults (whitelisted, non-product-specific definitions).

These are SAFE, general-purpose definitions you may surface when the user asks
for a definition (e.g., "what is an adult?"). They are NOT product-specific;
always include the disclaimer returned with each definition.

Usage:
    from regulatory_defaults import try_answer_regulatory_default
    text = try_answer_regulatory_default(user_query)
    if text:
        return text  # Bypass RAG and grounding validator
"""

import re
from typing import Optional, Dict, List

# Canonical definitions (neutral, non-product-specific)
_DEFS: Dict[str, str] = {
    "adult": "Adult — individuals 18 years and older.",
    "pediatric": "Pediatric (children) — individuals under 18 years; commonly stratified into neonate, infant, child, and adolescent.",
    "child": "Child — 2–11 years.",
    "children": "Children — 2–11 years.",
    "adolescent": "Adolescent — 12–17 years.",
    "infant": "Infant — 1–23 months.",
    "neonate": "Neonate (newborn) — birth to 28 days.",
    "newborn": "Newborn (neonate) — birth to 28 days.",
    "geriatric": "Geriatric (older adult, elderly) — 65 years and older.",
    "older adult": "Older adult (geriatric, elderly) — 65 years and older.",
    "elderly": "Elderly (older adult, geriatric) — 65 years and older.",
}

# Phrases indicating a definition request
# Expanded to catch “whats definition of adult”, “what’s the definition of…”, “definition of…”
_DEF_PATTERNS: List[re.Pattern] = [
    re.compile(r"\bdefine\b\s+([a-z][a-z\s-]+)\??", re.I),
    re.compile(r"\bwhat\s+(?:is|are)\s+(an?\s+)?([a-z][a-z\s-]+)\??", re.I),
    re.compile(r"\bwhat'?s\s+(?:the\s+)?definition\s+of\s+([a-z][a-z\s-]+)\??", re.I),
    re.compile(r"\bdefinition\s+of\s+([a-z][a-z\s-]+)\??", re.I),
    re.compile(r"\bwhat\s+constitutes\s+([a-z][a-z\s-]+)\??", re.I),
    re.compile(r"\bhow\s+do\s+you\s+define\s+([a-z][a-z\s-]+)\??", re.I),
    re.compile(r"\bwhat\s+counts\s+as\s+([a-z][a-z\s-]+)\??", re.I),
]

_DISCLAIMER = (
    "This is a general regulatory definition and is not specific to Journvax. "
    "It is not medical advice. For whether this medicine is appropriate for you, "
    "please consult your healthcare provider."
)

def _normalize(term: str) -> str:
    t = term.strip().lower()
    # basic singularization for trailing 's'
    if t.endswith("s") and t[:-1] in _DEFS:
        return t[:-1]
    return t

def _lookup(term: str) -> Optional[str]:
    t = _normalize(term)
    if t in _DEFS:
        return _DEFS[t]
    # simple alias pass (already covered by keys)
    for k in _DEFS:
        if t == k:
            return _DEFS[k]
    return None

def _contains_definition_language(q: str) -> bool:
    ql = q.lower()
    triggers = [
        "define", "definition", "constitutes", "counts as", "how do you define",
        "what is", "what are", "what's", "whats"
    ]
    return any(t in ql for t in triggers)

def try_answer_regulatory_default(user_query: str) -> Optional[str]:
    """
    If the user is asking for a general definition that we allow, return a
    fully formatted answer including the disclaimer. Otherwise return None.
    """
    q = (user_query or "").strip()
    if not q:
        return None

    # Fast path: if a known term is present and the phrasing looks definitional
    for key in _DEFS.keys():
        if re.search(rf"\b{re.escape(key)}\b", q, re.I) and _contains_definition_language(q):
            text = _lookup(key)
            if text:
                return f"{text}\n\n{_DISCLAIMER}"

    # Pattern-driven path (extract the term and look it up)
    for pat in _DEF_PATTERNS:
        m = pat.search(q)
        if m:
            # Some patterns capture in group 2 (e.g., "what is an adult")
            term = m.group(2) if (m.lastindex and m.lastindex >= 2 and m.group(2)) else m.group(1)
            if term:
                text = _lookup(term)
                if text:
                    return f"{text}\n\n{_DISCLAIMER}"

    return None
