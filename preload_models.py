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
        
        # Only load if it's a HuggingFace model (not a local path)
        if not os.path.exists(model_name):
            logger.info(f"Downloading: {model_name}")
            model = SentenceTransformer(model_name)
            dim = model.get_sentence_embedding_dimension()
            logger.info(f"✓ Embedding model cached: {model_name} ({dim} dimensions)")
            del model  # Free memory immediately
            
            # Force garbage collection
            import gc
            gc.collect()
        else:
            logger.info(f"Skipping local model: {model_name}")
            
    except Exception as e:
        logger.error(f"✗ Failed to cache embedding model: {e}")
        sys.exit(1)
    
    # 2. Cross-encoder reranker (optional)
    skip_reranker = os.getenv("SKIP_RERANKER", "false").lower() in ["true", "1", "yes"]
    
    if not skip_reranker:
        logger.info("\n2. Downloading reranker model...")
        try:
            from sentence_transformers import CrossEncoder
            reranker_name = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
            
            # Check if it's a valid HuggingFace model
            if not os.path.exists(reranker_name):
                logger.info(f"Downloading: {reranker_name}")
                reranker = CrossEncoder(reranker_name)
                logger.info(f"✓ Reranker model cached: {reranker_name}")
                del reranker
                
                # Force garbage collection
                import gc
                gc.collect()
            else:
                logger.info(f"Skipping local model: {reranker_name}")
                
        except Exception as e:
            logger.warning(f"⚠ Could not cache reranker (will run without it): {e}")
            logger.info("Consider setting SKIP_RERANKER=true to avoid this warning")
    else:
        logger.info("\n2. Skipping reranker (SKIP_RERANKER=true)")
    
    # 3. NLTK data
    logger.info("\n3. Downloading NLTK data...")
    try:
        import nltk
        # Set NLTK data path to Render's cache directory
        nltk_data_dir = os.getenv("NLTK_DATA", "/opt/render/project/.cache/nltk_data")
        os.makedirs(nltk_data_dir, exist_ok=True)
        nltk.data.path.append(nltk_data_dir)
        
        # Download required data
        for package in ['punkt', 'maxent_ne_chunker', 'words', 'averaged_perceptron_tagger']:
            try:
                nltk.download(package, quiet=True, download_dir=nltk_data_dir)
            except:
                pass  # Continue even if some packages fail
                
        logger.info("✓ NLTK data cached")
    except Exception as e:
        logger.warning(f"⚠ Could not cache NLTK data: {e}")
    
    logger.info("\n" + "=" * 60)
    logger.info("MODEL CACHING COMPLETE")
    logger.info("=" * 60)

if __name__ == "__main__":
    preload_models()
    sys.exit(0)