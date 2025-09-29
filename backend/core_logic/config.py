# backend/core_logic/config.py
from dataclasses import dataclass
import os

def _getenv(name: str, default: str) -> str:
    return os.getenv(name, default)

@dataclass(frozen=True)
class Models:
    # Consolidated embedding model for both RAG and Guardrails to save memory
    EMBEDDING_MODEL: str = _getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    # Cross-encoder reranker for better result ranking
    RERANKER_MODEL: str = _getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    # Grounding model (can be same as embedding to save memory)
    GROUNDING_MODEL: str = _getenv("GROUNDING_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

@dataclass(frozen=True)
class Retrieval:
    CANDIDATES: int = int(_getenv("RAG_TOPK", "40"))
    TOP_K_FINAL: int = int(_getenv("RAG_FINALK", "5"))
    MIN_SCORE: float = float(_getenv("RETRIEVAL_MIN_SCORE", "0.0"))
    # RAG weights
    W_RERANK: float = float(_getenv("RAG_W_RERANK", "0.60"))
    W_LEX: float = float(_getenv("RAG_W_LEX", "0.25"))
    W_VEC: float = float(_getenv("RAG_W_VEC", "0.15"))
    # Cache settings
    CACHE_MAX: int = int(_getenv("RAG_CACHE_MAX", "128"))
    CACHE_TTL: int = int(_getenv("RAG_CACHE_TTL", "1800"))
    # Term rescue
    RESCUE_MIN_HITS: int = int(_getenv("RAG_RESCUE_MIN_HITS", "2"))

@dataclass(frozen=True)
class Guardrails:
    ADVICE_THRESHOLD: float = float(_getenv("ADVICE_THRESHOLD", "0.55"))
    OFF_LABEL_THRESHOLD: float = float(_getenv("OFF_LABEL_THRESHOLD", "0.55"))
    FAIR_BALANCE_ENFORCED: bool = _getenv("FAIR_BALANCE_ENFORCED", "true").lower() == "true"
    # Grounding thresholds
    GROUNDING_AVG_THRESH: float = float(_getenv("GROUNDING_AVG_THRESH", "0.42"))
    GROUNDING_COVERED_FRAC: float = float(_getenv("GROUNDING_COVERED_FRAC", "0.65"))
    GROUNDING_MIN_THRESH: float = float(_getenv("GROUNDING_MIN_THRESH", "0.25"))

@dataclass(frozen=True)
class PineconeCfg:
    API_KEY: str = _getenv("PINECONE_API_KEY", "")
    ENVIRONMENT: str = _getenv("PINECONE_ENVIRONMENT", "us-east-1")
    INDEX: str = _getenv("PINECONE_INDEX", "pharma-assistant")
    NAMESPACE: str = _getenv("PINECONE_NAMESPACE", "default")
    # Connection settings
    POOL_THREADS: int = int(_getenv("PINECONE_POOL_THREADS", "1"))
    CONNECTION_TIMEOUT: float = float(_getenv("PINECONE_CONNECTION_TIMEOUT", "30.0"))
    QUERY_TIMEOUT: float = float(_getenv("PINECONE_QUERY_TIMEOUT", "25.0"))

@dataclass(frozen=True)
class LLMConfig:
    ANTHROPIC_API_KEY: str = _getenv("ANTHROPIC_API_KEY", "")
    CLAUDE_MODEL: str = _getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022")
    MAX_TOKENS: int = int(_getenv("ANTHROPIC_MAX_TOKENS", "1000"))
    TEMPERATURE: float = float(_getenv("ANTHROPIC_TEMPERATURE", "0.2"))
    NO_CONTEXT_FALLBACK_MESSAGE: str = _getenv(
        "NO_CONTEXT_FALLBACK_MESSAGE",
        "I apologize, I don't seem to have that information in the approved documentation. "
        "Can I assist you with something else?"
    )

@dataclass(frozen=True)
class ChunkingConfig:
    MAX_CHUNK_TOKENS: int = int(_getenv("MAX_CHUNK_TOKENS", "500"))
    CHUNK_OVERLAP: int = int(_getenv("CHUNK_OVERLAP", "50"))
    CHUNK_STRATEGY: str = _getenv("CHUNK_STRATEGY", "hybrid")

@dataclass(frozen=True)
class CloudRunConfig:
    """Cloud Run specific configuration"""
    CACHE_DIR: str = _getenv("MODEL_CACHE_DIR", "/app/.cache")
    SKIP_RERANKER: bool = _getenv("SKIP_RERANKER", "false").lower() == "true"
    TRANSFORMERS_CACHE: str = _getenv("TRANSFORMERS_CACHE", "/app/.cache")
    HF_HOME: str = _getenv("HF_HOME", "/app/.cache")
    TORCH_HOME: str = _getenv("TORCH_HOME", "/app/.cache")
    SENTENCE_TRANSFORMERS_HOME: str = _getenv("SENTENCE_TRANSFORMERS_HOME", "/app/.cache")
    NLTK_DATA: str = _getenv("NLTK_DATA", "/app/.cache/nltk_data")

@dataclass(frozen=True)
class Settings:
    models: Models = Models()
    retrieval: Retrieval = Retrieval()
    guard: Guardrails = Guardrails()
    pinecone: PineconeCfg = PineconeCfg()
    llm: LLMConfig = LLMConfig()
    chunking: ChunkingConfig = ChunkingConfig()
    cloud_run: CloudRunConfig = CloudRunConfig()

settings = Settings()

# Legacy support for direct imports
ANTHROPIC_API_KEY = settings.llm.ANTHROPIC_API_KEY
CLAUDE_MODEL = settings.llm.CLAUDE_MODEL
MAX_TOKENS = settings.llm.MAX_TOKENS
TEMPERATURE = settings.llm.TEMPERATURE
NO_CONTEXT_FALLBACK_MESSAGE = settings.llm.NO_CONTEXT_FALLBACK_MESSAGE
MAX_CHUNK_TOKENS = settings.chunking.MAX_CHUNK_TOKENS
CHUNK_OVERLAP = settings.chunking.CHUNK_OVERLAP