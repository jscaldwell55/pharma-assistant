# backend/core_logic/pinecone_vector.py
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import os
import logging

logger = logging.getLogger("vector")

# Pinecone SDK
try:
    from pinecone import Pinecone
except Exception as e:
    Pinecone = None
    _PC_IMPORT_ERR = e

# Load sentence transformers normally
try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

# CRITICAL FIX: Use the environment variable, not the hardcoded large model
DEFAULT_EMBED_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

@dataclass
class PineconeVectorClient:
    index_name: str
    environment: Optional[str] = os.getenv("PINECONE_ENVIRONMENT", "us-east-1")
    api_key: Optional[str] = os.getenv("PINECONE_API_KEY")
    namespace: Optional[str] = os.getenv("PINECONE_NAMESPACE") or None
    # CRITICAL: Don't override with large model here!
    embed_model_name: str = field(default_factory=lambda: os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"))

    _pc: Optional[Any] = field(default=None, init=False, repr=False)
    _index: Optional[Any] = field(default=None, init=False, repr=False)
    _embedder: Optional[Any] = field(default=None, init=False, repr=False)

    def __post_init__(self):
        if Pinecone is None:
            raise RuntimeError(
                "Pinecone import failed. Ensure the 'pinecone' package is installed. "
                f"Original import error: {repr(_PC_IMPORT_ERR)}"
            )
        if not self.api_key:
            raise RuntimeError("PINECONE_API_KEY is not set.")
        if not self.index_name:
            raise RuntimeError("index_name is required.")

        logger.info(
            "Initializing vector client index=%s ns=%s region=%s model=%s",
            self.index_name, self.namespace or "", self.environment, self.embed_model_name,
        )
        self._pc = Pinecone(api_key=self.api_key)
        # Assume index exists (create via build_index.py). This is fastest in hot path.
        self._index = self._pc.Index(self.index_name)

        # Load the embedder - will use the model from environment variable
        if SentenceTransformer is None:
            raise RuntimeError("sentence-transformers not installed for embedding model.")
        
        # Use the model from environment variable
        model_to_load = self.embed_model_name or DEFAULT_EMBED_MODEL
        logger.info("Loading embedder model: %s", model_to_load)
        self._embedder = SentenceTransformer(model_to_load)
        logger.info("Loaded embedder: %s", model_to_load)

    def query(self, text: str, top_k: int = 10, namespace: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Run a vector similarity search and return a list of:
        { "id": str, "text": str, "meta": dict, "score": float }
        """
        if not text:
            return []
        ns = namespace if namespace is not None else self.namespace
        vec = self._embedder.encode(text, normalize_embeddings=True).tolist()

        res = self._index.query(
            vector=vec,
            top_k=int(top_k),
            include_metadata=True,
            namespace=ns or "",
        )

        matches = getattr(res, "matches", None) or res.get("matches", [])
        out: List[Dict[str, Any]] = []
        for m in matches or []:
            _id = getattr(m, "id", None) or m.get("id")
            _score = float(getattr(m, "score", 0.0) or m.get("score", 0.0) or 0.0)
            _meta = getattr(m, "metadata", None) or m.get("metadata") or {}
            _text = _meta.get("text") or _meta.get("chunk") or ""
            out.append({
                "id": _id,
                "text": _text,
                "meta": _meta,
                "score": _score,
            })
        return out