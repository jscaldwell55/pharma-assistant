# backend/core_logic/pinecone_vector.py
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import os
import time
import logging

logger = logging.getLogger("vector")

# Pinecone SDK
try:
    from pinecone import Pinecone
except Exception as exc:
    Pinecone = None
    _PC_IMPORT_ERR = exc

# Load sentence transformers - with fallback
try:
    from sentence_transformers import SentenceTransformer
    _ST_AVAILABLE = True
except Exception:
    logger.warning("sentence-transformers not available, will use fallback")
    SentenceTransformer = None
    _ST_AVAILABLE = False

DEFAULT_EMBED_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")


@dataclass
class PineconeVectorClient:
    index_name: str
    environment: Optional[str] = os.getenv("PINECONE_ENVIRONMENT", "us-east-1")
    api_key: Optional[str] = os.getenv("PINECONE_API_KEY")
    namespace: Optional[str] = os.getenv("PINECONE_NAMESPACE") or None
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
            self.index_name,
            self.namespace or "",
            self.environment,
            self.embed_model_name,
        )

        logger.info("Step 1: Creating Pinecone client...")
        # Configure Pinecone client with connection timeouts
        client_config = {
            "api_key": self.api_key,
            "pool_threads": int(os.getenv("PINECONE_POOL_THREADS", "1")),  # Connection pool size
        }

        # Add additional headers for timeout configuration
        additional_headers = {
            "User-Agent": "pharma-assistant/1.0",
        }
        if additional_headers:
            client_config["additional_headers"] = additional_headers

        self._pc = Pinecone(**client_config)
        logger.info("Step 2: Pinecone client created, connecting to index...")

        # Create index connection with timeout
        connection_timeout = float(os.getenv("PINECONE_CONNECTION_TIMEOUT", "30.0"))
        start_time = time.time()

        try:
            self._index = self._pc.Index(self.index_name)

            connection_time = time.time() - start_time
            logger.info("Step 3: Connected to Pinecone index successfully in %.3fs", connection_time)

            if connection_time > 10.0:
                logger.warning("Slow Pinecone index connection: %.3fs", connection_time)

        except Exception as exc:
            connection_time = time.time() - start_time
            logger.error("Failed to connect to Pinecone index after %.3fs: %s", connection_time, exc)
            raise

        logger.info("Step 4: Loading embedding model...")
        model_to_load = self.embed_model_name or DEFAULT_EMBED_MODEL

        if not _ST_AVAILABLE:
            logger.warning("SentenceTransformer not available, using mock embedder")
            from .emergency_mock_embedder import MockEmbedder
            self._embedder = MockEmbedder(embedding_dim=384)
        else:
            
            start_time = time.time()
            try:
                from .model_manager import model_manager
                self._embedder = model_manager.get_embedder()
                load_time = time.time() - start_time
                logger.info("Step 5: Got cached embedder from ModelManager in %.2f seconds", load_time)
            except Exception as exc:
                load_time = time.time() - start_time
                logger.error("Failed to get embedding model after %.2f seconds: %s", load_time, exc)
                raise

        logger.info("PineconeVectorClient initialization complete!")

    def query(self, text: str, top_k: int = 10, namespace: Optional[str] = None, timeout: float = 30.0) -> List[Dict[str, Any]]:
        """
        Run a vector similarity search and return a list of:
        { "id": str, "text": str, "meta": dict, "score": float }

        Args:
            text: Query text to embed and search with
            top_k: Number of results to return
            namespace: Pinecone namespace to search in
            timeout: Request timeout in seconds (default: 30.0)
        """
        if not text:
            return []
        ns = namespace if namespace is not None else self.namespace

        # Generate embedding with proper error handling
        embed_start = time.time()
        try:
            vec = self._embedder.encode(
                text,
                normalize_embeddings=True,
                show_progress_bar=False,  # Disable tqdm progress bars in production
                convert_to_tensor=False
            ).tolist()

            embed_time = time.time() - embed_start
            if embed_time > 5.0:
                logger.warning("Slow embedding generation: %.2fs for text length %d", embed_time, len(text))

        except Exception as exc:
            embed_time = time.time() - embed_start
            logger.error("Embedding generation failed after %.2fs: %s", embed_time, exc)
            return []

        # Execute Pinecone query with timeout and retry logic
        query_start = time.time()
        max_retries = 2
        base_delay = 1.0

        for attempt in range(max_retries + 1):
            try:
                logger.debug("Pinecone query attempt %d/%d (timeout=%.1fs)", attempt + 1, max_retries + 1, timeout)

                res = self._index.query(
                    vector=vec,
                    top_k=int(top_k),
                    include_metadata=True,
                    namespace=ns or "",
                    _request_timeout=timeout,  # Critical: Add timeout parameter
                )

                query_time = time.time() - query_start
                logger.debug("Pinecone query completed in %.3fs", query_time)
                break

            except Exception as exc:
                query_time = time.time() - query_start
                if attempt < max_retries:
                    delay = base_delay * (2 ** attempt)  # Exponential backoff
                    logger.warning("Pinecone query attempt %d failed after %.3fs: %s. Retrying in %.1fs...",
                                 attempt + 1, query_time, exc, delay)
                    time.sleep(delay)
                    query_start = time.time()  # Reset timer for next attempt
                else:
                    logger.error("Pinecone query failed after %d attempts and %.3fs: %s",
                               max_retries + 1, query_time, exc)
                    return []

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
