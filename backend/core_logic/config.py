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
    # Zero-shot classifier for guard (keep disabled to save memory)
    CLASSIFIER_MODEL: str = _getenv("CLASSIFIER_MODEL", "roberta-base-mnli")

@dataclass(frozen=True)
class Retrieval:
    CANDIDATES: int = int(_getenv("RETRIEVAL_CANDIDATES", "15"))
    TOP_K_FINAL: int = int(_getenv("RETRIEVAL_TOP_K_FINAL", "5"))
    MIN_SCORE: float = float(_getenv("RETRIEVAL_MIN_SCORE", "0.0"))

@dataclass(frozen=True)
class Guardrails:
    ADVICE_THRESHOLD: float = float(_getenv("ADVICE_THRESHOLD", "0.55"))
    OFF_LABEL_THRESHOLD: float = float(_getenv("OFF_LABEL_THRESHOLD", "0.55"))
    FAIR_BALANCE_ENFORCED: bool = _getenv("FAIR_BALANCE_ENFORCED", "true").lower() == "true"

@dataclass(frozen=True)
class Tracing:
    ENABLED: bool = _getenv("TRACE_ENABLED", "true").lower() == "true"
    DIR: str = _getenv("TRACE_DIR", "traces")
    EXPORT_JSONL: bool = _getenv("TRACE_EXPORT_JSONL", "true").lower() == "true"

@dataclass(frozen=True)
class PineconeCfg:
    API_KEY: str = _getenv("PINECONE_API_KEY", "")
    ENVIRONMENT: str = _getenv("PINECONE_ENVIRONMENT", "us-east-1")
    INDEX: str = _getenv("PINECONE_INDEX", "pharma-assistant")

@dataclass(frozen=True)
class LLMConfig:
    ANTHROPIC_API_KEY: str = _getenv("ANTHROPIC_API_KEY", "")
    CLAUDE_MODEL: str = _getenv("ANTHROPIC_MODEL", "claude-3-7-sonnet-latest")
    MAX_TOKENS: int = int(_getenv("ANTHROPIC_MAX_TOKENS", "1000"))
    TEMPERATURE: float = float(_getenv("ANTHROPIC_TEMPERATURE", "0.2"))
    NO_CONTEXT_FALLBACK_MESSAGE: str = _getenv(
        "NO_CONTEXT_FALLBACK_MESSAGE",
        "I apologize, I don't seem to have that information. Can I assist you with something else?"
    )

@dataclass(frozen=True)
class ChunkingConfig:
    MAX_CHUNK_TOKENS: int = int(_getenv("MAX_CHUNK_TOKENS", "500"))
    CHUNK_OVERLAP: int = int(_getenv("CHUNK_OVERLAP", "50"))

@dataclass(frozen=True)
class Settings:
    models: Models = Models()
    retrieval: Retrieval = Retrieval()
    guard: Guardrails = Guardrails()
    trace: Tracing = Tracing()
    pinecone: PineconeCfg = PineconeCfg()
    llm: LLMConfig = LLMConfig()
    chunking: ChunkingConfig = ChunkingConfig()

settings = Settings()

# Legacy support for direct imports
ANTHROPIC_API_KEY = settings.llm.ANTHROPIC_API_KEY
CLAUDE_MODEL = settings.llm.CLAUDE_MODEL
MAX_TOKENS = settings.llm.MAX_TOKENS
TEMPERATURE = settings.llm.TEMPERATURE
NO_CONTEXT_FALLBACK_MESSAGE = settings.llm.NO_CONTEXT_FALLBACK_MESSAGE
MAX_CHUNK_TOKENS = settings.chunking.MAX_CHUNK_TOKENS
CHUNK_OVERLAP = settings.chunking.CHUNK_OVERLAP

# If True, guard will try to load a LOCAL MNLI model from NLI_MODEL_PATH.
# If False, it will skip MNLI entirely and use the embedding classifier (quietly).
# Keep this disabled (False) to save memory even with 2GB
USE_NLI_CLASSIFIER: bool = os.getenv("USE_NLI_CLASSIFIER", "0") == "1"

# If you enable USE_NLI_CLASSIFIER, this must point to a local folder containing the HF model files.
# Example: "/app/models/roberta-base-mnli"
NLI_MODEL_PATH: str = os.getenv("NLI_MODEL_PATH", "").strip()