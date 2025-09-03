# backend/core_logic/conversational_agent.py
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import uuid
import time
import inspect
import logging
import hashlib
import os

from .guard import Guard
from .rag import RAGRetriever
from .context_formatter import build_context, append_fair_balance, extractive_bullets
from .prompts import AE_ROUTE_TEMPLATE, PHI_SANITIZE_NOTE

# Grounding metrics in trace (observability)
from .grounding import compute_grounding, passes_grounding

# For fallback message reuse
try:
    from .llm_client import NO_CONTEXT_FALLBACK_MESSAGE
except Exception:
    NO_CONTEXT_FALLBACK_MESSAGE = "I apologize, I don't seem to have that information. Can I assist you with something else?"

logger = logging.getLogger("agent")

# Expect your existing llm_client.py to export LLMClient with .generate(...)
try:
    from .llm_client import LLMClient  # must return grounded text from provided context
except Exception:
    class LLMClient:
        async def generate(self, system_prompt: str, context: str, user_query: str, conversation_history: List[Dict[str, str]]) -> str:
            return f"Here is what the approved docs say:\n\n{context}\n\nAnswer to your question: [DRAFT RESPONSE]"

@dataclass
class AgentDecision:
    response_text: str
    retrieved: List[Dict[str, Any]]
    safety_labels: Dict[str, Any]
    trace: Dict[str, Any]


class _TTLCache:
    def __init__(self, maxsize: int = 256, ttl_sec: int = 1800):
        self.maxsize = maxsize
        self.ttl = ttl_sec
        self._d: Dict[str, Any] = {}

    def _now(self) -> float:
        return time.time()

    def get(self, key: str):
        item = self._d.get(key)
        if not item:
            return None
        ts, val = item
        if self._now() - ts > self.ttl:
            self._d.pop(key, None)
            return None
        return val

    def set(self, key: str, val: Any):
        self._d[key] = (self._now(), val)
        if len(self._d) > self.maxsize:
            # evict oldest
            oldest_key = min(self._d.items(), key=lambda kv: kv[1][0])[0]
            self._d.pop(oldest_key, None)

    def clear(self):
        self._d.clear()


class ConversationalAgent:
    def __init__(self, retriever: RAGRetriever, llm_client: Optional[LLMClient] = None):
        self.guard = Guard()
        self.retriever = retriever
        self.llm = llm_client or LLMClient()
        # Answer cache (post-context, pre-guard).
        self._ans_cache = _TTLCache(
            maxsize=int(os.getenv("ANS_CACHE_MAX", "256")),
            ttl_sec=int(os.getenv("ANS_CACHE_TTL", "1800"))
        )

    def clear_answer_cache(self):
        self._ans_cache.clear()

    async def _maybe_await(self, fn, *args, **kwargs):
        res = fn(*args, **kwargs)
        if inspect.isawaitable(res):
            return await res
        return res

    def _answer_cache_key(self, system_prompt: str, user_query: str, top_chunks: List[Dict[str, Any]]) -> str:
        # Build a stable key from prompt + normalized query + identifiers of top chunks
        norm_q = " ".join(user_query.lower().split())
        ids = []
        for r in top_chunks:
            meta = r.get("meta") or {}
            piece = "|".join([
                str(meta.get("source", "")),
                str(meta.get("doc", "")),
                str(meta.get("chunk_id", "")),
                (r.get("id") or "")[:24],
            ])
            ids.append(piece)
        basis = f"{system_prompt}##{norm_q}##{';;'.join(ids)}"
        return hashlib.sha256(basis.encode("utf-8")).hexdigest()

    async def handle(self, user_query: str, conversation_history: List[Dict[str, str]], system_prompt: str = "") -> AgentDecision:
        t0 = time.time()
        trace_id = f"trace_{uuid.uuid4().hex}"
        stages: Dict[str, int] = {}
        cache_flags = {"retrieval_hit": False, "answer_hit": False}

        # 1) Retrieve (with internal cache)
        r0 = time.time()
        retrieved = self.retriever.retrieve(user_query)
        stages["retrieve_ms"] = int((time.time() - r0) * 1000)
        cache_flags["retrieval_hit"] = getattr(self.retriever, "last_cache_hit", False)
        logger.info("Retrieved %d chunks", len(retrieved))

        # 2) Build concise, sentence-level context with citations
        ctx_build0 = time.time()
        ctx = build_context(retrieved, user_query, char_budget=3000)
        context_str = ctx["context"]
        stages["context_build_ms"] = int((time.time() - ctx_build0) * 1000)
        logger.info("Context built: %d sources, %d chars", len(ctx["sources"]), len(context_str))

        # 3) Answer cache check (keyed on prompt + query + top chunk IDs)
        ans_key = self._answer_cache_key(system_prompt, user_query, retrieved)
        cached_answer = self._ans_cache.get(ans_key)
        if cached_answer:
            cache_flags["answer_hit"] = True
            draft = cached_answer
            stages["llm_ms"] = 0
            logger.info("Answer cache hit")
        else:
            gen0 = time.time()
            draft = await self._maybe_await(
                self.llm.generate,
                system_prompt=system_prompt,
                context=context_str,
                user_query=user_query,
                conversation_history=conversation_history  # Pass it through
            )
            stages["llm_ms"] = int((time.time() - gen0) * 1000)
            # Extractive fallback if placeholder/empty
            if (not draft) or ("[DRAFT RESPONSE]" in draft):
                logger.info("Using extractive fallback summarizer")
                draft = extractive_bullets(user_query, retrieved)
            # Put into answer cache
            self._ans_cache.set(ans_key, draft)

        # 4) Guard evaluation (enforces grounding via the same context)
        guard0 = time.time()
        decision = self.guard.evaluate(
            user_query,
            draft,
            retrieved,
            context_text=context_str,  # exact LLM context for grounding gate
        )
        stages["guard_ms"] = int((time.time() - guard0) * 1000)

        # Compute grounding details for trace (observability only)
        g_details = compute_grounding(draft or "", context_str or "")
        g_pass = passes_grounding(g_details)

        # Optional preface notes (sanitization/AE routing)
        prefix_notes = []
        if decision.labels.phi_pii:
            prefix_notes.append(PHI_SANITIZE_NOTE)
        if decision.labels.adverse_event:
            prefix_notes.append(AE_ROUTE_TEMPLATE)

        # -------- Refusal mapping for all policy blocks --------
        def _refusal_for_reason(reason: str) -> str:
            reason = (reason or "").lower()
            if reason == "medical_advice":
                return ("I can't provide individualized medical advice. "
                        "For guidance specific to you, please contact your healthcare provider.")
            if reason == "off_label":
                return ("I can only discuss approved/on-label information. "
                        "I can't provide guidance about unapproved or off-label uses.")
            if reason == "self_harm":
                return ("I'm really sorry you're going through this, but I can't help with that here. "
                        "If you're in immediate danger, call 911. You can also call or text 988 "
                        "to reach the Suicide & Crisis Lifeline for support.")
            if reason == "illicit_or_violent":
                return ("I can't assist with instructions or content that enables harm, violence, or illegal activity.")
            if reason == "jailbreak":
                return ("I can't ignore safety rules or pretend to act outside approved medical guidance.")
            if reason == "ungrounded":
                # Keep your existing safe fallback for ungrounded outputs
                return NO_CONTEXT_FALLBACK_MESSAGE
            # Default fallback
            return NO_CONTEXT_FALLBACK_MESSAGE

        # --- Apply refusal or deliver allowed answer ---
        if not decision.allow:
            safe_resp = _refusal_for_reason(decision.reason)
            resp_text = "\n\n".join(prefix_notes + [safe_resp]) if prefix_notes else safe_resp
        else:
            resp_text = draft
            if decision.enforced_risk_snippet:
                resp_text = append_fair_balance(resp_text, decision.enforced_risk_snippet)
            if prefix_notes:
                resp_text = "\n\n".join(prefix_notes + [resp_text])

        t1 = time.time()

        safety_labels = {
            "medical_advice": decision.labels.medical_advice,
            "off_label_use": decision.labels.off_label_use,
            "phi_pii": decision.labels.phi_pii,
            "adverse_event": decision.labels.adverse_event,
            "toxicity": decision.labels.toxicity,
            "illicit_or_violent": getattr(decision.labels, "illicit_or_violent", False),
            # Trace-only: expose grounding pass/fail at a glance
            "grounded": bool(g_pass),
        }

        trace = {
            "trace_id": trace_id,
            "timestamps": {
                "start": t0, "end": t1, "latency_ms": int((t1 - t0) * 1000),
                **stages
            },
            "cache": cache_flags,
            "user_query": user_query,
            "system_prompt": system_prompt,
            "retrieved": [
                {"text": r.get("text"), "vector_score": r.get("score"), "rerank_score": r.get("rerank_score"),
                 "meta": r.get("meta")}
                for r in retrieved
            ],
            "context_preview": context_str[:500],
            "draft_response": draft,
            "grounding": g_details,  # grounding metrics in trace
            "guard": {
                "allow": decision.allow,
                "reason": decision.reason,
                "labels": safety_labels,
                "enforced_risk_snippet": decision.enforced_risk_snippet,
            },
            "env": {
                "pinecone_index": os.getenv("PINECONE_INDEX"),
                "pinecone_namespace": os.getenv("PINECONE_NAMESPACE", ""),
            },
        }

        return AgentDecision(resp_text, retrieved, safety_labels, trace)