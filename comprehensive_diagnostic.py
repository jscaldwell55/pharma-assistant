#!/usr/bin/env python3
"""
Comprehensive diagnostic script for pharma-assistant deployment issues.
Run this to identify configuration mismatches, missing dependencies, and potential bottlenecks.
"""

import os
import sys
import json
import time
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

class SystemDiagnostic:
    def __init__(self):
        self.issues = []
        self.warnings = []
        self.info = {}
        self.root_path = Path.cwd()
        
    def print_header(self, title: str):
        """Print a formatted section header"""
        print("\n" + "=" * 70)
        print(f"  {title}")
        print("=" * 70)
        
    def check_environment_variables(self):
        """Check all critical environment variables"""
        self.print_header("ENVIRONMENT VARIABLES CHECK")
        
        required_vars = {
            "PINECONE_API_KEY": "Required for vector database",
            "ANTHROPIC_API_KEY": "Required for Claude LLM",
            "PINECONE_INDEX": "Index name (default: pharma-assistant)",
            "PINECONE_NAMESPACE": "Namespace for vectors",
            "EMBEDDING_MODEL": "Model for embeddings",
        }
        
        optional_vars = {
            "SKIP_RERANKER": "Whether to skip cross-encoder",
            "USE_MOCK_EMBEDDER": "Emergency fallback",
            "RAG_TOPK": "Number of candidates to retrieve",
            "RAG_W_RERANK": "Weight for reranker",
            "PORT": "Server port (Render sets this)",
        }
        
        # Load .env files
        env_files = [
            self.root_path / ".env",
            self.root_path / "backend" / ".env",
            self.root_path / "backend" / "core_logic" / ".env",
        ]
        
        loaded_env = None
        for env_file in env_files:
            if env_file.exists():
                print(f"✓ Found .env at: {env_file}")
                loaded_env = env_file
                # Load it
                from dotenv import load_dotenv
                load_dotenv(env_file)
                break
        
        if not loaded_env:
            print("⚠️  No .env file found")
            self.warnings.append("No .env file found - relying on system environment")
        
        print("\nRequired Variables:")
        for var, desc in required_vars.items():
            value = os.getenv(var)
            if value:
                if "KEY" in var:
                    print(f"  ✓ {var}: ***{value[-4:]} ({desc})")
                else:
                    print(f"  ✓ {var}: {value} ({desc})")
                    self.info[var] = value
            else:
                print(f"  ✗ {var}: NOT SET ({desc})")
                self.issues.append(f"{var} is not set")
        
        print("\nOptional Variables:")
        for var, desc in optional_vars.items():
            value = os.getenv(var)
            if value:
                print(f"  • {var}: {value} ({desc})")
                self.info[var] = value
            else:
                print(f"  • {var}: not set ({desc})")
                
    def check_model_consistency(self):
        """Check if embedding model is consistent across configs"""
        self.print_header("MODEL CONSISTENCY CHECK")
        
        embedding_model = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        print(f"Configured model: {embedding_model}")
        
        # Check model dimensions
        model_dims = {
            "sentence-transformers/all-MiniLM-L6-v2": 384,
            "sentence-transformers/all-mpnet-base-v2": 768,
            "sentence-transformers/paraphrase-MiniLM-L3-v2": 384,
        }
        
        expected_dim = model_dims.get(embedding_model)
        if expected_dim:
            print(f"Expected dimensions: {expected_dim}")
            self.info["expected_dimensions"] = expected_dim
        else:
            print(f"⚠️  Unknown model dimensions for: {embedding_model}")
            self.warnings.append(f"Unknown embedding model: {embedding_model}")
            
        # Check if model files exist locally (for caching check)
        cache_paths = [
            Path.home() / ".cache" / "torch" / "sentence_transformers",
            Path("/opt/render/project/.cache"),
            Path(".cache"),
        ]
        
        print("\nModel cache locations:")
        model_cached = False
        for cache_path in cache_paths:
            if cache_path.exists():
                model_name = embedding_model.split("/")[-1]
                model_path = cache_path / model_name
                if model_path.exists():
                    print(f"  ✓ Model cached at: {model_path}")
                    model_cached = True
                    break
        
        if not model_cached:
            print("  ⚠️  Model not cached - will download on first use")
            self.warnings.append("Embedding model not cached - expect slow first load")
            
    def check_pinecone_config(self):
        """Check Pinecone configuration and connectivity"""
        self.print_header("PINECONE CONFIGURATION CHECK")
        
        index_name = os.getenv("PINECONE_INDEX", "pharma-assistant")
        namespace = os.getenv("PINECONE_NAMESPACE", "")
        api_key = os.getenv("PINECONE_API_KEY")
        
        if not api_key:
            print("✗ PINECONE_API_KEY not set - cannot check connectivity")
            return
            
        print(f"Index: {index_name}")
        print(f"Namespace: {namespace or '(default)'}")
        
        try:
            print("\nTesting Pinecone connection...")
            from pinecone import Pinecone
            
            start = time.time()
            pc = Pinecone(api_key=api_key)
            elapsed = time.time() - start
            print(f"  ✓ Client created in {elapsed:.2f}s")
            
            # List indexes
            start = time.time()
            indexes = list(pc.list_indexes())
            elapsed = time.time() - start
            print(f"  ✓ Listed {len(indexes)} indexes in {elapsed:.2f}s")
            
            # Check if our index exists
            index_names = [idx.name for idx in indexes]
            if index_name in index_names:
                print(f"  ✓ Index '{index_name}' exists")
                
                # Get index stats
                index = pc.Index(index_name)
                stats = index.describe_index_stats()
                
                dim = stats.get('dimension', 0)
                total = stats.get('total_vector_count', 0)
                
                print(f"  ✓ Index dimensions: {dim}")
                print(f"  ✓ Total vectors: {total}")
                
                self.info["pinecone_dimensions"] = dim
                self.info["pinecone_vectors"] = total
                
                # Check namespace
                if namespace and 'namespaces' in stats:
                    if namespace in stats['namespaces']:
                        ns_count = stats['namespaces'][namespace].get('vector_count', 0)
                        print(f"  ✓ Namespace '{namespace}' has {ns_count} vectors")
                        if ns_count == 0:
                            self.warnings.append(f"Namespace '{namespace}' exists but has no vectors")
                    else:
                        print(f"  ⚠️  Namespace '{namespace}' not found")
                        self.issues.append(f"Namespace '{namespace}' not found in index")
                        
                # Check dimension compatibility
                expected_dim = self.info.get("expected_dimensions")
                if expected_dim and dim != expected_dim:
                    print(f"\n  ✗ DIMENSION MISMATCH!")
                    print(f"     Model expects: {expected_dim} dimensions")
                    print(f"     Index has: {dim} dimensions")
                    self.issues.append(f"Dimension mismatch: model={expected_dim}, index={dim}")
            else:
                print(f"  ✗ Index '{index_name}' not found")
                print(f"     Available indexes: {', '.join(index_names)}")
                self.issues.append(f"Pinecone index '{index_name}' does not exist")
                
        except Exception as e:
            print(f"  ✗ Pinecone error: {e}")
            self.issues.append(f"Pinecone connection failed: {str(e)}")
            
    def check_file_structure(self):
        """Check if all required files exist"""
        self.print_header("FILE STRUCTURE CHECK")
        
        required_files = {
            "backend/api/server.py": "Flask API server",
            "backend/core_logic/conversational_agent.py": "Main agent logic",
            "backend/core_logic/rag.py": "RAG retrieval",
            "backend/core_logic/guard.py": "Safety guardrails",
            "backend/core_logic/pinecone_vector.py": "Vector client",
            "requirements.txt": "Python dependencies",
            "Procfile": "Render deployment config",
        }
        
        optional_files = {
            "gunicorn_config.py": "Gunicorn configuration",
            "render.yaml": "Render service config",
            "backend/core_logic/data/chunks.jsonl": "Document chunks",
        }
        
        print("Required files:")
        for filepath, desc in required_files.items():
            path = self.root_path / filepath
            if path.exists():
                size = path.stat().st_size
                print(f"  ✓ {filepath} ({size:,} bytes) - {desc}")
            else:
                print(f"  ✗ {filepath} - {desc}")
                self.issues.append(f"Missing required file: {filepath}")
                
        print("\nOptional files:")
        for filepath, desc in optional_files.items():
            path = self.root_path / filepath
            if path.exists():
                size = path.stat().st_size
                print(f"  • {filepath} ({size:,} bytes) - {desc}")
            else:
                print(f"  • {filepath} (not found) - {desc}")
                
    def check_memory_optimizations(self):
        """Check if memory optimizations are properly configured"""
        self.print_header("MEMORY OPTIMIZATION CHECK")
        
        skip_reranker = os.getenv("SKIP_RERANKER", "false").lower() == "true"
        use_mock = os.getenv("USE_MOCK_EMBEDDER", "false").lower() == "true"
        cache_max = int(os.getenv("RAG_CACHE_MAX", "256"))
        
        print(f"Reranker disabled: {skip_reranker}")
        print(f"Mock embedder: {use_mock}")
        print(f"Cache size: {cache_max} entries")
        
        if not skip_reranker:
            print("\n⚠️  Reranker is enabled - uses ~90MB additional memory")
            print("   Consider setting SKIP_RERANKER=true if memory is tight")
            
        embedding_model = os.getenv("EMBEDDING_MODEL", "")
        if "mpnet" in embedding_model.lower():
            print("\n⚠️  Using large embedding model (all-mpnet-base-v2)")
            print("   This uses ~420MB vs ~90MB for all-MiniLM-L6-v2")
            print("   Consider switching to the smaller model if memory is tight")
            
    def check_dependencies(self):
        """Check if all required Python packages are installed"""
        self.print_header("DEPENDENCY CHECK")
        
        required_packages = [
            "flask",
            "pinecone-client",
            "sentence-transformers",
            "anthropic",
            "numpy",
            "scikit-learn",
        ]
        
        print("Checking installed packages...")
        for package in required_packages:
            try:
                __import__(package.replace("-", "_"))
                print(f"  ✓ {package}")
            except ImportError:
                print(f"  ✗ {package} (not installed)")
                self.warnings.append(f"Package '{package}' not installed")
                
    def generate_summary(self):
        """Generate a summary of findings"""
        self.print_header("DIAGNOSTIC SUMMARY")
        
        if self.issues:
            print("\n🚨 CRITICAL ISSUES (must fix):")
            for issue in self.issues:
                print(f"  • {issue}")
        else:
            print("\n✅ No critical issues found")
            
        if self.warnings:
            print("\n⚠️  WARNINGS (should review):")
            for warning in self.warnings:
                print(f"  • {warning}")
                
        print("\n📊 KEY METRICS:")
        if "pinecone_dimensions" in self.info:
            print(f"  • Pinecone index dimensions: {self.info['pinecone_dimensions']}")
        if "expected_dimensions" in self.info:
            print(f"  • Model expected dimensions: {self.info['expected_dimensions']}")
        if "pinecone_vectors" in self.info:
            print(f"  • Total vectors in index: {self.info['pinecone_vectors']}")
            
        # Recommendations
        print("\n💡 RECOMMENDATIONS:")
        
        if self.issues:
            if any("Dimension mismatch" in i for i in self.issues):
                print("  1. Rebuild Pinecone index with matching embedding model")
                print("     python backend/core_logic/build_index.py --recreate")
                
            if any("not set" in i for i in self.issues):
                print("  2. Set missing environment variables in Render dashboard")
                
            if any("not found" in i for i in self.issues):
                print("  3. Check file paths and namespace configuration")
        else:
            print("  • System appears properly configured")
            print("  • If still having issues, check network connectivity and timeouts")
            
    def run(self):
        """Run all diagnostic checks"""
        print("\n" + "🔍 PHARMA ASSISTANT DIAGNOSTIC TOOL 🔍".center(70))
        print("Running comprehensive system checks...")
        
        self.check_environment_variables()
        self.check_model_consistency()
        self.check_pinecone_config()
        self.check_file_structure()
        self.check_memory_optimizations()
        self.check_dependencies()
        self.generate_summary()
        
        return len(self.issues) == 0

if __name__ == "__main__":
    diagnostic = SystemDiagnostic()
    success = diagnostic.run()
    sys.exit(0 if success else 1)