"""
Modal serverless embedding service for pharma-assistant.
This handles all embedding operations without consuming Render's memory.
"""

import modal
import json
from typing import List, Union
import numpy as np

# Create Modal app
app = modal.App("pharma-embedder")

# Build image with pre-downloaded model
image = (
    modal.Image.debian_slim()
    .pip_install(
        "sentence-transformers",
        "numpy",
        "torch",
        "transformers"
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
    keep_warm=1,   # Keep 1 instance warm to avoid cold starts
    timeout=60,    # 60 second timeout
)
class EmbedderService:
    def __init__(self):
        from sentence_transformers import SentenceTransformer
        print("Loading model into memory...")
        self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        self.dimension = self.model.get_sentence_embedding_dimension()
        print(f"Model loaded. Embedding dimension: {self.dimension}")
    
    @modal.method()
    def embed_single(self, text: str, normalize: bool = True) -> List[float]:
        """Embed a single text string"""
        embedding = self.model.encode(
            text,
            normalize_embeddings=normalize,
            convert_to_numpy=True
        )
        return embedding.tolist()
    
    @modal.method()
    def embed_batch(self, texts: List[str], normalize: bool = True) -> List[List[float]]:
        """Embed multiple texts at once (more efficient)"""
        embeddings = self.model.encode(
            texts,
            normalize_embeddings=normalize,
            convert_to_numpy=True,
            batch_size=32
        )
        return embeddings.tolist()
    
    @modal.method()
    def get_info(self) -> dict:
        """Get model information"""
        return {
            "model_name": "sentence-transformers/all-MiniLM-L6-v2",
            "dimension": self.dimension,
            "status": "ready"
        }

# Standalone function for simple embedding (alternative to class)
@app.function(
    image=image,
    cpu=1,
    memory=1024,
    timeout=30
)
def quick_embed(text: str) -> List[float]:
    """Quick embedding function for single texts"""
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    embedding = model.encode(text, normalize_embeddings=True)
    return embedding.tolist()

# Local testing endpoint
@app.local_entrypoint()
def test():
    """Test the embedding service locally"""
    # Test class-based service
    embedder = EmbedderService()
    
    # Single embedding
    test_text = "What are the side effects?"
    embedding = embedder.embed_single.remote(test_text)
    print(f"Single embedding shape: {len(embedding)} dimensions")
    
    # Batch embedding
    test_batch = [
        "What is the dosage?",
        "Are there any drug interactions?",
        "How should I store this medication?"
    ]
    batch_embeddings = embedder.embed_batch.remote(test_batch)
    print(f"Batch embedding: {len(batch_embeddings)} texts encoded")
    
    # Get info
    info = embedder.get_info.remote()
    print(f"Service info: {info}")
    
    return "Embedding service test complete!"