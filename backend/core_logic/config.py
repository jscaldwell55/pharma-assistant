# config.py – central settings
from dataclasses import dataclass
import os

def _getenv(name: str, default: str) -> str:
    return os.getenv(name, default)

@dataclass(frozen=True)
class Models:
    # Larger embedding model (768d) for stronger retrieval
    EMBEDDING_MODEL: str = _getenv("EMBEDDING_MODEL", "sentence-transformers/all-mpnet-base-v2")
    # Cross-encoder reranker
    RERANKER_MODEL: str = _getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    # Zero-shot classifier for guard
    CLASSIFIER_MODEL: str = _getenv("CLASSIFIER_MODEL", "roberta-base-mnli")

@dataclass(frozen=True)
class Retrieval:
    CANDIDATES: int = int(_getenv("RETRIEVAL_CANDIDATES", "20"))
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
class Settings:
    models: Models = Models()
    retrieval: Retrieval = Retrieval()
    guard: Guardrails = Guardrails()
    trace: Tracing = Tracing()
    pinecone: PineconeCfg = PineconeCfg()

settings = Settings()


# If True, guard will try to load a LOCAL MNLI model from NLI_MODEL_PATH.
# If False, it will skip MNLI entirely and use the embedding classifier (quietly).
USE_NLI_CLASSIFIER: bool = os.getenv("USE_NLI_CLASSIFIER", "0") == "1"

# If you enable USE_NLI_CLASSIFIER, this must point to a local folder containing the HF model files.
# Example: "/app/models/roberta-base-mnli"
NLI_MODEL_PATH: str = os.getenv("NLI_MODEL_PATH", "").strip()