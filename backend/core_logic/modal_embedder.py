# modal_embedder.py
"""
Modal serverless embedding service for pharma-assistant.
This handles all embedding operations without consuming Render's memory.
"""

import modal
import json
from typing import List, Union
import numpy as np
from fastapi import HTTPException
from pydantic import BaseModel

# Create Modal app
app = modal.App("pharma-embedder")

# Request models for better API structure
class EmbedRequest(BaseModel):
    text: str
    normalize: bool = True

class BatchEmbedRequest(BaseModel):
    texts: List[str]
    normalize: bool = True

# Build image with pre-downloaded model
image = (
    modal.Image.debian_slim()
    .pip_install(
        "sentence-transformers",
        "numpy",
        "torch",
        "transformers",
        "fastapi",
        "pydantic"
    )
    # Pre-download the model during image build
    .run_commands(
        "python -c \"from sentence_transformers import SentenceTransformer; "
        "model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2'); "
        "print('Model downloaded and cached')\""
    )
)

# Store model in class to reuse across invocations
@app.cls(
    image=image,
    cpu=2,
    memory=2048,  # 2GB memory
    timeout=60,    # 60 second timeout
)
class EmbedderService:
    def __enter__(self):
        from sentence_transformers import SentenceTransformer
        print("Loading model into memory...")
        self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        self.dimension = self.model.get_sentence_embedding_dimension()
        print(f"Model loaded. Embedding dimension: {self.dimension}")
    
    @modal.fastapi_endpoint(method="POST")
    def embed_single(self, request: EmbedRequest) -> dict:
        """Embed a single text string"""
        try:
            embedding = self.model.encode(
                request.text,
                normalize_embeddings=request.normalize,
                convert_to_numpy=True
            )
            return {
                "embedding": embedding.tolist(),
                "dimension": len(embedding),
                "text_length": len(request.text)
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Embedding failed: {str(e)}")

    @modal.fastapi_endpoint(method="POST")
    def embed_batch(self, request: BatchEmbedRequest) -> dict:
        """Embed multiple texts at once (more efficient)"""
        try:
            embeddings = self.model.encode(
                request.texts,
                normalize_embeddings=request.normalize,
                convert_to_numpy=True,
                batch_size=32
            )
            return {
                "embeddings": embeddings.tolist(),
                "dimension": embeddings.shape[1] if len(embeddings) > 0 else 0,
                "count": len(embeddings)
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Batch embedding failed: {str(e)}")

    @modal.fastapi_endpoint(method="GET")
    def get_info(self) -> dict:
        """Get model information"""
        return {
            "model_name": "sentence-transformers/all-MiniLM-L6-v2",
            "dimension": self.dimension,
            "status": "ready",
            "service": "pharma-embedder"
        }

# Standalone function for simple embedding (alternative to class)
@app.function(
    image=image,
    cpu=1,
    memory=1024,
    timeout=30
)
def embed_simple(text: str) -> List[float]:
    """Simple embedding function for single texts"""
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    embedding = model.encode(text, normalize_embeddings=True)
    return embedding.tolist()