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

# Load sentence transformers - with fallback
try:
    from sentence_transformers import SentenceTransformer
    _ST_AVAILABLE = True
except Exception:
    logger.warning("sentence-transformers not available, will use fallback")
    SentenceTransformer = None
    _ST_AVAILABLE = False

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
        logger.info("PineconeVectorClient initialization starting...")
        
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
        
        # Initialize Pinecone
        logger.info("Step 1: Creating Pinecone client...")
        self._pc = Pinecone(api_key=self.api_key)
        logger.info("Step 2: Pinecone client created, connecting to index...")
        
        # Connect to index
        try:
            self._index = self._pc.Index(self.index_name)
            logger.info("Step 3: Connected to Pinecone index successfully")
        except Exception as e:
            logger.error(f"Failed to connect to Pinecone index: {e}")
            raise
        
        # Load the embedder
        logger.info("Step 4: Loading embedding model...")
        model_to_load = self.embed_model_name or DEFAULT_EMBED_MODEL
        
        if not _ST_AVAILABLE:
            logger.warning("SentenceTransformer not available, using mock embedder")
            from .emergency_mock_embedder import MockEmbedder
            self._embedder = MockEmbedder(embedding_dim=384)
        else:
            import time
            start_time = time.time()
            try:
                self._embedder = SentenceTransformer(model_to_load)
                load_time = time.time() - start_time
                logger.info("Step 5: Loaded embedder %s in %.2f seconds", model_to_load, load_time)
            except Exception as e:
                load_time = time.time() - start_time
                logger.error(f"Failed to load embedding model after {load_time:.2f} seconds: {e}")
                raise
        
        logger.info("PineconeVectorClient initialization complete!")

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