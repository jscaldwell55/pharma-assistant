"""
Centralized model management with singleton pattern and lazy loading.
Reduces cold start by ensuring models are loaded only once and shared.
"""

import os
import time
import logging
import threading
from typing import Optional, Any

logger = logging.getLogger("model_manager")


class ModelManager:
    """
    Singleton model manager that handles lazy loading and caching of ML models.
    Thread-safe implementation ensures models are loaded only once.
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self._embedder = None
        self._embedder_lock = threading.Lock()
        self._embedder_loading = False

        self._reranker = None
        self._reranker_lock = threading.Lock()
        self._reranker_loading = False

        self._grounding_model = None
        self._grounding_lock = threading.Lock()
        self._grounding_loading = False

        # Model configuration from environment
        self.embedding_model = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        self.reranker_model = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
        self.grounding_model = os.getenv("GROUNDING_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        self.skip_reranker = os.getenv("SKIP_RERANKER", "false").lower() in {"true", "1", "yes"}

        # Use consistent cache paths from environment with fallbacks
        self.cache_dir = os.getenv("MODEL_CACHE_DIR") or os.getenv("TRANSFORMERS_CACHE")

        # Smart cache directory selection
        if not self.cache_dir:
            if os.path.exists("/app/.cache") and os.access("/app/.cache", os.W_OK):
                self.cache_dir = "/app/.cache"  # Production container
            elif os.path.exists(os.path.expanduser("~/.cache")):
                self.cache_dir = os.path.expanduser("~/.cache/huggingface")  # Local development
            else:
                import tempfile
                self.cache_dir = tempfile.mkdtemp(prefix="model_cache_")  # Fallback
                logger.warning("Using temporary cache directory: %s", self.cache_dir)

        os.environ["TRANSFORMERS_CACHE"] = self.cache_dir
        os.environ["HF_HOME"] = self.cache_dir
        os.environ["TORCH_HOME"] = self.cache_dir
        os.environ["SENTENCE_TRANSFORMERS_HOME"] = self.cache_dir
        os.environ["NLTK_DATA"] = os.path.join(self.cache_dir, "nltk_data")

        self._verify_cache_directory()

        self._initialized = True
        logger.info("ModelManager initialized with cache dir: %s", self.cache_dir)

    def _verify_cache_directory(self):
        """Log details about the transformer cache to aid debugging."""
        if not os.path.exists(self.cache_dir):
            logger.warning("Cache directory does not exist: %s", self.cache_dir)
            return

        st_cache = os.path.join(self.cache_dir, "sentence_transformers")
        if os.path.exists(st_cache):
            models = os.listdir(st_cache)
            logger.info("Found %d cached sentence transformer model(s)", len(models))
        else:
            logger.warning("Sentence transformers cache directory not found")

        hf_cache = os.path.join(self.cache_dir, "hub")
        if os.path.exists(hf_cache):
            models = os.listdir(hf_cache)
            logger.info("Found %d cached HuggingFace model(s)", len(models))

        try:
            total_size = sum(
                os.path.getsize(os.path.join(dirpath, filename))
                for dirpath, _, filenames in os.walk(self.cache_dir)
                for filename in filenames
            ) / (1024 * 1024)
            logger.info("Total cache size: %.1f MB", total_size)
        except Exception as exc:  # pragma: no cover - diagnostic only
            logger.warning("Could not calculate cache size: %s", exc)

        if os.getenv("TRANSFORMERS_OFFLINE") == "1":
            logger.info("Transformers offline mode enabled")
        else:
            logger.info("Transformers online mode - downloads permitted if required")

    def preload_all(self):
        """Preload every model during build or startup."""
        logger.info("Preloading all models...")
        start_time = time.time()

        self.get_embedder()

        if not self.skip_reranker:
            self.get_reranker()

        if self.grounding_model != self.embedding_model:
            self.get_grounding_model()

        total_time = time.time() - start_time
        logger.info("All models preloaded in %.2f seconds", total_time)

    def get_embedder(self) -> Any:
        """Return the embedding model, loading if needed."""
        if self._embedder is not None:
            return self._embedder

        with self._embedder_lock:
            if self._embedder is not None:
                return self._embedder

            if self._embedder_loading:
                logger.info("Embedder already loading in another thread")
                while self._embedder_loading:
                    time.sleep(0.1)
                return self._embedder

            self._embedder_loading = True

            try:
                logger.info("Loading embedding model: %s", self.embedding_model)
                start_time = time.time()

                from sentence_transformers import SentenceTransformer

                self._embedder = SentenceTransformer(
                    self.embedding_model,
                    cache_folder=self.cache_dir
                )

                load_time = time.time() - start_time
                logger.info("Embedding model loaded in %.2f seconds", load_time)

                if load_time > 5.0:
                    logger.warning("Slow embedder load (%.2fs) indicates cache miss", load_time)

                warmup_start = time.time()
                _ = self._embedder.encode("warmup", convert_to_tensor=False)
                logger.info("Embedding model warmup completed in %.3fs", time.time() - warmup_start)

                return self._embedder

            except Exception as exc:
                logger.error("Failed to load embedding model: %s", exc)
                raise
            finally:
                self._embedder_loading = False

    def get_reranker(self) -> Optional[Any]:
        """Return the reranker model or None if disabled."""
        if self.skip_reranker:
            return None

        if self._reranker is not None:
            return self._reranker

        with self._reranker_lock:
            if self._reranker is not None:
                return self._reranker

            if self._reranker_loading:
                logger.info("Reranker already loading in another thread")
                while self._reranker_loading:
                    time.sleep(0.1)
                return self._reranker

            self._reranker_loading = True

            try:
                logger.info("Loading reranker model: %s", self.reranker_model)
                start_time = time.time()

                from sentence_transformers import CrossEncoder

                self._reranker = CrossEncoder(self.reranker_model, max_length=512)

                load_time = time.time() - start_time
                logger.info("Reranker model loaded in %.2f seconds", load_time)

                _ = self._reranker.predict([["warmup query", "warmup text"]])
                return self._reranker

            except Exception as exc:
                logger.warning("Failed to load reranker model: %s", exc)
                logger.info("Continuing without reranker")
                self.skip_reranker = True
                return None
            finally:
                self._reranker_loading = False

    def get_grounding_model(self) -> Any:
        """Return the grounding model, reusing the embedder when identical."""
        if self.grounding_model == self.embedding_model:
            return self.get_embedder()

        if self._grounding_model is not None:
            return self._grounding_model

        with self._grounding_lock:
            if self._grounding_model is not None:
                return self._grounding_model

            if self._grounding_loading:
                logger.info("Grounding model already loading in another thread")
                while self._grounding_loading:
                    time.sleep(0.1)
                return self._grounding_model

            self._grounding_loading = True

            try:
                logger.info("Loading grounding model: %s", self.grounding_model)
                start_time = time.time()

                from sentence_transformers import SentenceTransformer

                self._grounding_model = SentenceTransformer(
                    self.grounding_model,
                    cache_folder=self.cache_dir
                )

                load_time = time.time() - start_time
                logger.info("Grounding model loaded in %.2f seconds", load_time)

                _ = self._grounding_model.encode("warmup", convert_to_tensor=False)
                return self._grounding_model

            except Exception as exc:
                logger.error("Failed to load grounding model: %s", exc)
                raise
            finally:
                self._grounding_loading = False

    def clear_cache(self):
        """Clear all loaded models to free memory."""
        logger.warning("Clearing model cache - models will be reloaded on demand")
        with self._embedder_lock:
            self._embedder = None
        with self._reranker_lock:
            self._reranker = None
        with self._grounding_lock:
            self._grounding_model = None

        import gc
        gc.collect()

    def get_status(self) -> dict:
        """Return current model loading status."""
        return {
            "embedder_loaded": self._embedder is not None,
            "reranker_loaded": self._reranker is not None if not self.skip_reranker else "skipped",
            "grounding_loaded": self._grounding_model is not None or self.grounding_model == self.embedding_model,
            "cache_dir": self.cache_dir,
            "models": {
                "embedding": self.embedding_model,
                "reranker": self.reranker_model if not self.skip_reranker else "disabled",
                "grounding": self.grounding_model,
            },
        }


# Global singleton instance
model_manager = ModelManager()
