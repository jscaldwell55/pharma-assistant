# preload_models.py
#!/usr/bin/env python3
"""
Pre-download and cache models during build phase.
This ensures models are available immediately at runtime.
"""

import os
import sys
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("preload")

def preload_models():
    """Download and cache all required models"""
    
    logger.info("=" * 60)
    logger.info("PRE-DOWNLOADING MODELS FOR BUILD CACHE")
    logger.info("=" * 60)
    
    # 1. Sentence Transformers embedding model
    logger.info("\n1. Downloading embedding model...")
    try:
        from sentence_transformers import SentenceTransformer
        model_name = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        model = SentenceTransformer(model_name)
        dim = model.get_sentence_embedding_dimension()
        logger.info(f"✓ Embedding model cached: {model_name} ({dim} dimensions)")
        del model  # Free memory
    except Exception as e:
        logger.error(f"✗ Failed to cache embedding model: {e}")
        sys.exit(1)
    
    # 2. Cross-encoder reranker (optional)
    if os.getenv("SKIP_RERANKER", "false").lower() != "true":
        logger.info("\n2. Downloading reranker model...")
        try:
            from sentence_transformers import CrossEncoder
            reranker_name = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
            reranker = CrossEncoder(reranker_name)
            logger.info(f"✓ Reranker model cached: {reranker_name}")
            del reranker
        except Exception as e:
            logger.warning(f"⚠ Could not cache reranker (will skip): {e}")
    else:
        logger.info("\n2. Skipping reranker (SKIP_RERANKER=true)")
    
    # 3. NLTK data
    logger.info("\n3. Downloading NLTK data...")
    try:
        import nltk
        nltk.download('punkt', quiet=True)
        nltk.download('maxent_ne_chunker', quiet=True)
        nltk.download('words', quiet=True)
        nltk.download('averaged_perceptron_tagger', quiet=True)
        logger.info("✓ NLTK data cached")
    except Exception as e:
        logger.warning(f"⚠ Could not cache NLTK data: {e}")
    
    logger.info("\n" + "=" * 60)
    logger.info("MODEL CACHING COMPLETE")
    logger.info("=" * 60)

if __name__ == "__main__":
    preload_models()
    sys.exit(0)