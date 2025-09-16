#!/usr/bin/env python3
"""
Build Pinecone index from generated chunks.
This script reads chunks from JSONL and uploads them to Pinecone with embeddings.
"""

import os
import sys
import json
import time
from typing import List, Dict, Any
from pathlib import Path
import logging
from dotenv import load_dotenv

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Find and load .env file
script_dir = Path(__file__).parent
possible_env_paths = [
    script_dir / '.env',
    script_dir / '../.env',
    script_dir / '../../.env',
    script_dir / '../../backend/.env',
    Path.cwd() / '.env',
    Path.cwd() / 'backend/.env',
]

env_loaded = False
for env_path in possible_env_paths:
    if env_path.exists():
        load_dotenv(env_path)
        logger.info(f"Loaded environment from: {env_path}")
        env_loaded = True
        break

if not env_loaded:
    logger.warning("No .env file found, using system environment variables only")

# Import dependencies
try:
    from pinecone import Pinecone, ServerlessSpec
except ImportError:
    logger.error("pinecone-client not installed. Run: pip install pinecone-client")
    sys.exit(1)

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    logger.error("sentence-transformers not installed. Run: pip install sentence-transformers")
    sys.exit(1)

class PineconeIndexBuilder:
    """Build and manage Pinecone indexes"""
    
    def __init__(
        self,
        api_key: str = None,
        index_name: str = "pharma-assistant",
        environment: str = "us-east-1",
        embedding_model_name: str = None  # Changed: Now defaults to None
    ):
        self.api_key = api_key or os.getenv("PINECONE_API_KEY")
        if not self.api_key:
            raise ValueError(
                "PINECONE_API_KEY not found. Please either:\n"
                "1. Set it in your .env file\n"
                "2. Export it: export PINECONE_API_KEY='your-key'\n"
                "3. Pass it as --api-key argument"
            )
        
        self.index_name = index_name
        self.environment = environment
        self.namespace = os.getenv("PINECONE_NAMESPACE", "lilly")  # Using your namespace
        
        # Initialize Pinecone
        logger.info(f"Connecting to Pinecone...")
        self.pc = Pinecone(api_key=self.api_key)
        
        # Use environment variable or default to small model for memory efficiency
        if embedding_model_name is None:
            embedding_model_name = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        
        # Initialize embedding model
        logger.info(f"Loading embedding model: {embedding_model_name}")
        self.embedder = SentenceTransformer(embedding_model_name)
        self.embedding_dim = self.embedder.get_sentence_embedding_dimension()
        logger.info(f"Embedding dimension: {self.embedding_dim}")
        
        # Warn if using large model
        if "mpnet" in embedding_model_name.lower():
            logger.warning("⚠️  Using large model (all-mpnet-base-v2). This requires more memory!")
            logger.warning("   Consider using --embedding-model 'sentence-transformers/all-MiniLM-L6-v2' for lower memory usage")
    
    def create_index(self, recreate: bool = False) -> None:
        """Create Pinecone index if it doesn't exist"""
        try:
            existing_indexes = [index.name for index in self.pc.list_indexes()]
        except Exception as e:
            logger.error(f"Failed to list indexes. Check your API key and network connection: {e}")
            raise
        
        if self.index_name in existing_indexes:
            if recreate:
                logger.warning(f"Deleting existing index: {self.index_name}")
                self.pc.delete_index(self.index_name)
                time.sleep(5)  # Wait for deletion
            else:
                logger.info(f"Index {self.index_name} already exists, using it")
                self.index = self.pc.Index(self.index_name)
                
                # Check dimension compatibility
                try:
                    stats = self.index.describe_index_stats()
                    if stats.get('dimension') and stats['dimension'] != self.embedding_dim:
                        logger.error(f"❌ Dimension mismatch!")
                        logger.error(f"   Index has {stats['dimension']} dimensions")
                        logger.error(f"   Model creates {self.embedding_dim} dimensions")
                        logger.error(f"   Please either:")
                        logger.error(f"   1. Use --recreate flag to rebuild index with new model")
                        logger.error(f"   2. Use the original embedding model that matches {stats['dimension']} dimensions")
                        sys.exit(1)
                except:
                    pass  # If we can't check, proceed anyway
                return
        
        logger.info(f"Creating new index: {self.index_name} with {self.embedding_dim} dimensions")
        try:
            self.pc.create_index(
                name=self.index_name,
                dimension=self.embedding_dim,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region=self.environment
                )
            )
        except Exception as e:
            logger.error(f"Failed to create index: {e}")
            raise
        
        # Wait for index to be ready
        logger.info("Waiting for index to be ready...")
        time.sleep(10)
        
        self.index = self.pc.Index(self.index_name)
        logger.info(f"Index {self.index_name} created successfully with {self.embedding_dim} dimensions")
    
    def load_chunks(self, chunks_file: str) -> List[Dict[str, Any]]:
        """Load chunks from JSONL file"""
        chunks = []
        chunks_path = Path(chunks_file)
        
        if not chunks_path.exists():
            # Try common locations
            alternative_paths = [
                Path("backend/data/chunks/chunks.jsonl"),
                Path("data/chunks/chunks.jsonl"),
                Path("backend/data/chunks.jsonl"),
                Path("data/chunks.jsonl"),
            ]
            for alt_path in alternative_paths:
                if alt_path.exists():
                    chunks_path = alt_path
                    logger.info(f"Found chunks at: {chunks_path}")
                    break
            else:
                raise FileNotFoundError(f"Chunks file not found: {chunks_file}")
        
        with open(chunks_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    chunk = json.loads(line)
                    # Ensure required fields
                    if 'id' not in chunk or 'text' not in chunk:
                        logger.warning(f"Line {line_num}: Missing 'id' or 'text' field, skipping")
                        continue
                    chunks.append(chunk)
                except json.JSONDecodeError as e:
                    logger.warning(f"Line {line_num}: Invalid JSON, skipping: {e}")
        
        logger.info(f"Loaded {len(chunks)} valid chunks from {chunks_path}")
        return chunks
    
    def prepare_vectors(self, chunks: List[Dict[str, Any]], batch_size: int = 32) -> List[Dict[str, Any]]:
        """Prepare vectors for Pinecone upload"""
        vectors = []
        total_batches = (len(chunks) + batch_size - 1) // batch_size
        
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            texts = [chunk["text"] for chunk in batch]
            
            # Generate embeddings
            batch_num = i // batch_size + 1
            logger.info(f"Encoding batch {batch_num}/{total_batches} ({len(texts)} texts)")
            embeddings = self.embedder.encode(
                texts, 
                normalize_embeddings=True, 
                show_progress_bar=False,
                convert_to_numpy=True
            )
            
            # Create vector records
            for chunk, embedding in zip(batch, embeddings):
                metadata = chunk.get("metadata", chunk.get("meta", {}))
                
                # Ensure text is in metadata for retrieval
                if "text" not in metadata:
                    metadata["text"] = chunk["text"][:1000]  # Truncate for metadata limit
                if "chunk" not in metadata:
                    metadata["chunk"] = chunk["text"]  # Full text in separate field
                
                vector = {
                    "id": chunk["id"],
                    "values": embedding.tolist(),
                    "metadata": metadata
                }
                vectors.append(vector)
        
        logger.info(f"Prepared {len(vectors)} vectors")
        return vectors
    
    def upload_vectors(self, vectors: List[Dict[str, Any]], batch_size: int = 100) -> None:
        """Upload vectors to Pinecone"""
        total_vectors = len(vectors)
        total_batches = (total_vectors + batch_size - 1) // batch_size
        
        for i in range(0, total_vectors, batch_size):
            batch = vectors[i:i + batch_size]
            batch_num = i // batch_size + 1
            
            logger.info(f"Uploading batch {batch_num}/{total_batches} ({len(batch)} vectors)")
            
            try:
                if self.namespace:
                    self.index.upsert(vectors=batch, namespace=self.namespace)
                else:
                    self.index.upsert(vectors=batch)
            except Exception as e:
                logger.error(f"Failed to upload batch {batch_num}: {e}")
                raise
            
            # Small delay to avoid rate limiting
            time.sleep(0.1)
        
        logger.info(f"✅ Successfully uploaded {total_vectors} vectors to Pinecone")
    
    def verify_index(self) -> Dict[str, Any]:
        """Verify index statistics"""
        try:
            stats = self.index.describe_index_stats()
            logger.info(f"Index statistics: {stats}")
            return stats
        except Exception as e:
            logger.error(f"Failed to get index stats: {e}")
            return {}
    
    def test_query(self, query_text: str = "What are the side effects?", top_k: int = 3) -> None:
        """Test the index with a sample query"""
        logger.info(f"Testing query: '{query_text}'")
        
        # Encode query
        query_vector = self.embedder.encode(query_text, normalize_embeddings=True).tolist()
        
        # Query index
        try:
            if self.namespace:
                results = self.index.query(
                    vector=query_vector,
                    top_k=top_k,
                    include_metadata=True,
                    namespace=self.namespace
                )
            else:
                results = self.index.query(
                    vector=query_vector,
                    top_k=top_k,
                    include_metadata=True
                )
            
            # Display results
            logger.info(f"Found {len(results.matches)} matches:")
            for i, match in enumerate(results.matches, 1):
                logger.info(f"  {i}. Score: {match.score:.3f}")
                text_preview = match.metadata.get("text", "")[:100]
                logger.info(f"     Text: {text_preview}...")
        except Exception as e:
            logger.error(f"Query failed: {e}")

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Build Pinecone index from chunks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Build with default small model (recommended for low memory)
  python build_index.py --chunks data/chunks.jsonl
  
  # Explicitly use small model and recreate
  python build_index.py --chunks data/chunks.jsonl --embedding-model "sentence-transformers/all-MiniLM-L6-v2" --recreate
  
  # Use large model (requires more memory)
  python build_index.py --chunks data/chunks.jsonl --embedding-model "sentence-transformers/all-mpnet-base-v2"
  
  # Test with a custom query
  python build_index.py --chunks data/chunks.jsonl --test-query "What is the dosage?"
        """
    )
    
    parser.add_argument(
        "--chunks", "--jsonl",
        default="data/chunks.jsonl",
        help="Path to chunks JSONL file (default: data/chunks.jsonl)"
    )
    parser.add_argument(
        "--index-name",
        default=os.getenv("PINECONE_INDEX", "pharma-assistant"),
        help="Pinecone index name"
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="Pinecone API key (or set PINECONE_API_KEY env var)"
    )
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Delete and recreate index if it exists"
    )
    parser.add_argument(
        "--test-query",
        help="Test query to run after indexing"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Batch size for uploading (default: 100)"
    )
    parser.add_argument(
        "--embedding-model",
        default=None,  # Will use environment variable or small model default
        help="Embedding model to use (default: from EMBEDDING_MODEL env or all-MiniLM-L6-v2)"
    )
    parser.add_argument(
        "--namespace",
        default=os.getenv("PINECONE_NAMESPACE", "lilly"),
        help="Namespace for vectors (default: lilly)"
    )
    
    args = parser.parse_args()
    
    # Check if chunks file exists
    chunks_path = Path(args.chunks)
    if not chunks_path.is_absolute():
        # Try relative to script location
        chunks_path = Path(__file__).parent / args.chunks
    
    if not chunks_path.exists():
        logger.error(f"Chunks file not found: {chunks_path}")
        logger.info("Please run generate_chunks_from_pdfs.py first to create chunks")
        logger.info("Example: python generate_chunks_from_pdfs.py")
        sys.exit(1)
    
    try:
        # Initialize builder
        builder = PineconeIndexBuilder(
            api_key=args.api_key,
            index_name=args.index_name,
            embedding_model_name=args.embedding_model
        )
        
        # Set namespace if provided
        if args.namespace:
            builder.namespace = args.namespace
            logger.info(f"Using namespace: {args.namespace}")
        
        # Create/connect to index
        builder.create_index(recreate=args.recreate)
        
        # Load chunks
        chunks = builder.load_chunks(str(chunks_path))
        
        if not chunks:
            logger.warning("No chunks to upload")
            return
        
        # Prepare vectors
        logger.info("Preparing vectors...")
        vectors = builder.prepare_vectors(chunks, batch_size=32)
        
        # Upload to Pinecone
        logger.info("Uploading to Pinecone...")
        builder.upload_vectors(vectors, batch_size=args.batch_size)
        
        # Verify
        stats = builder.verify_index()
        
        # Test query
        if args.test_query:
            builder.test_query(args.test_query)
        else:
            # Default test
            builder.test_query("What is the recommended dosage?")
        
        logger.info("=" * 60)
        logger.info("✅ Index building complete!")
        logger.info(f"Index: {args.index_name}")
        if builder.namespace:
            logger.info(f"Namespace: {builder.namespace}")
        logger.info(f"Vectors: {len(vectors)}")
        logger.info(f"Embedding model: {builder.embedder.get_sentence_embedding_dimension()} dimensions")
        logger.info("Your index is ready to use!")
        
    except Exception as e:
        logger.error(f"Error building index: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()