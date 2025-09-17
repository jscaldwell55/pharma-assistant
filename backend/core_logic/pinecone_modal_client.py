# backend/core_logic/pinecone_modal_client.py
"""
Pinecone client that uses Modal for embeddings instead of local models.
This replaces pinecone_vector.py for the Modal architecture.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import os
import logging
import requests
import json
from urllib.parse import urljoin

logger = logging.getLogger("vector_modal")

# Pinecone SDK
try:
    from pinecone import Pinecone
except Exception as e:
    Pinecone = None
    _PC_IMPORT_ERR = e

@dataclass
class ModalEmbedder:
    """Client for Modal embedding service"""
    
    modal_endpoint: str
    api_key: Optional[str] = None
    timeout: int = 30
    
    def encode(self, text: str, normalize_embeddings: bool = True) -> List[float]:
        """Embed a single text using Modal service"""
        try:
            # Construct the full URL properly
            base_url = self.modal_endpoint.rstrip('/')  # Remove trailing slash if present
            url = f"{base_url}/embed_single"
            
            headers = {"Content-Type": "application/json"}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            
            response = requests.post(
                url,
                json={
                    "text": text,
                    "normalize": normalize_embeddings
                },
                headers=headers,
                timeout=self.timeout
            )
            response.raise_for_status()
            
            # Modal returns the embedding directly
            result = response.json()
            return result if isinstance(result, list) else result.get("embedding", [])
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Modal embedding request failed: {e}")
            raise RuntimeError(f"Failed to get embedding from Modal: {e}")
    
    def encode_batch(self, texts: List[str], normalize_embeddings: bool = True) -> List[List[float]]:
        """Embed multiple texts using Modal service"""
        try:
            base_url = self.modal_endpoint.rstrip('/')
            url = f"{base_url}/embed_batch"
            
            headers = {"Content-Type": "application/json"}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            
            response = requests.post(
                url,
                json={
                    "texts": texts,
                    "normalize": normalize_embeddings
                },
                headers=headers,
                timeout=self.timeout * 2  # Longer timeout for batch
            )
            response.raise_for_status()
            
            result = response.json()
            return result if isinstance(result, list) else result.get("embeddings", [])
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Modal batch embedding request failed: {e}")
            raise RuntimeError(f"Failed to get batch embeddings from Modal: {e}")
    
    def get_sentence_embedding_dimension(self) -> int:
        """Get embedding dimension from Modal service"""
        try:
            base_url = self.modal_endpoint.rstrip('/')
            url = f"{base_url}/get_info"

            response = requests.get(url, timeout=10)
            response.raise_for_status()
            info = response.json()
            return info.get("dimension", 384)  # Default to 384 if not found
        except:
            return 384  # Fallback dimension for all-MiniLM-L6-v2

@dataclass
class PineconeModalClient:
    """Pinecone client that uses Modal for embeddings"""
    
    index_name: str
    environment: Optional[str] = os.getenv("PINECONE_ENVIRONMENT", "us-east-1")
    api_key: Optional[str] = os.getenv("PINECONE_API_KEY")
    namespace: Optional[str] = os.getenv("PINECONE_NAMESPACE") or None
    modal_endpoint: str = os.getenv("MODAL_EMBEDDER_ENDPOINT", "")
    modal_api_key: Optional[str] = os.getenv("MODAL_API_KEY")
    
    _pc: Optional[Any] = field(default=None, init=False, repr=False)
    _index: Optional[Any] = field(default=None, init=False, repr=False)
    _embedder: Optional[ModalEmbedder] = field(default=None, init=False, repr=False)
    
    def __post_init__(self):
        if Pinecone is None:
            raise RuntimeError(
                "Pinecone import failed. Ensure 'pinecone-client' is installed."
            )
        if not self.api_key:
            raise RuntimeError("PINECONE_API_KEY is not set.")
        if not self.index_name:
            raise RuntimeError("index_name is required.")
        if not self.modal_endpoint:
            raise RuntimeError("MODAL_EMBEDDER_ENDPOINT is not set.")
        
        logger.info(
            "Initializing Pinecone+Modal client: index=%s ns=%s modal=%s",
            self.index_name, self.namespace or "", self.modal_endpoint
        )
        
        # Initialize Pinecone
        self._pc = Pinecone(api_key=self.api_key)
        self._index = self._pc.Index(self.index_name)
        logger.info("Connected to Pinecone index")
        
        # Initialize Modal embedder client
        self._embedder = ModalEmbedder(
            modal_endpoint=self.modal_endpoint,
            api_key=self.modal_api_key
        )
        logger.info("Modal embedder client initialized")
    
    def query(self, text: str, top_k: int = 10, namespace: Optional[str] = None) -> List[Dict[str, Any]]:
        """Run a vector similarity search"""
        if not text:
            return []
        
        ns = namespace if namespace is not None else self.namespace
        
        # Get embedding from Modal
        logger.debug(f"Getting embedding for query: {text[:50]}...")
        vec = self._embedder.encode(text, normalize_embeddings=True)
        
        # Query Pinecone
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
        
        logger.debug(f"Found {len(out)} matches")
        return out

    def upsert(self, vectors: List[Dict[str, Any]], namespace: Optional[str] = None) -> Dict[str, Any]:
        """Upsert vectors to Pinecone"""
        ns = namespace if namespace is not None else self.namespace

        # Convert texts to embeddings if needed
        processed_vectors = []
        for vector in vectors:
            if "values" not in vector and "text" in vector:
                # Need to embed the text
                embedding = self._embedder.encode(vector["text"], normalize_embeddings=True)
                vector["values"] = embedding
            processed_vectors.append(vector)

        return self._index.upsert(
            vectors=processed_vectors,
            namespace=ns or ""
        )

    def delete(self, ids: List[str], namespace: Optional[str] = None) -> Dict[str, Any]:
        """Delete vectors from Pinecone"""
        ns = namespace if namespace is not None else self.namespace
        return self._index.delete(ids=ids, namespace=ns or "")

    def describe_index_stats(self) -> Dict[str, Any]:
        """Get index statistics"""
        return self._index.describe_index_stats()

    def test_connection(self) -> bool:
        """Test both Pinecone and Modal connections"""
        try:
            # Test Pinecone
            stats = self.describe_index_stats()
            logger.info(f"Pinecone connection OK. Total vectors: {stats.get('total_vector_count', 0)}")

            # Test Modal
            dimension = self._embedder.get_sentence_embedding_dimension()
            logger.info(f"Modal connection OK. Embedding dimension: {dimension}")

            return True
        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            return False