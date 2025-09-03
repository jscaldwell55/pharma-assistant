# Pharma Assistant — System Overview & Safety Design

A safety-first pharmaceutical information assistant that answers only from approved, uploaded documents (labeling, PI, patient leaflets). The system emphasizes traceability, guardrails, fair-balance, and grounding, approximating the orchestration patterns used in enterprise GenAI stacks.

## Recent Enhancements (v3.0 - API Migration)

### API-First Architecture
- **Streamlit-free backend** - Core logic now runs as Flask API service
- **Browser extension ready** - RESTful `/api/rewrite` endpoint for browser integration
- **Conversation history support** - Full context awareness across turns
- **Modular package structure** - Clean separation between API layer and core logic
- **Lazy loading** - Models load on first request to optimize memory usage

### Enhanced Conversational AI
- **Context-aware responses** - Follow-up question understanding with conversation history
- **Natural dialogue flow** - Professional responses that build on previous exchanges
- **Session management** - Conversation state handled by client (browser extension)

### Knowledge Bridge Integration (Maintained)
- **Automatic synonym expansion** for better retrieval (e.g., Zepbound ↔ tirzepatide, "side effects" ↔ "adverse reactions")
- **Concept mapping** for medical terminology variations
- **Zero configuration** - works transparently without prompt changes
- Maintains strict safety boundaries while improving query understanding

### Natural Response Generation (Enhanced)
- **Removed prescriptive language** - no more "based on the provided context" qualifiers
- **Conversation-aware system prompts** for contextual, professional responses
- **Maintained safety** through grounding gates rather than verbose instructions

## What the System Does

1. **Provides RESTful API** for pharmaceutical Q&A with conversation context
2. **Answers user questions only using the provided documents** (no external knowledge)
3. **Retrieves high-recall passages** with enhanced query understanding via Knowledge Bridge
4. **Generates natural, conversational answers** (Claude 3.7 Sonnet) while maintaining strict grounding
5. **Enforces layered safety controls**:
   - Grounding gate (semantic similarity to retrieved context; hard gate in Guard)
   - Medical guardrails (advice/off-label detection, PHI/PII, AE)
   - Harmful/illegal intent pre-gates + reason-specific policy refusals
   - Fair-balance coupling (risk snippet insertion when benefits are mentioned)
6. **Emits structured JSON traces** per request (prompt, chunks, scores, labels, timings)
7. **Browser extension integration** with conversation history management

## Architecture Overview

```
Browser Extension ──► Flask API ──► Core Logic Package
    (State)           (server.py)     (conversational_agent.py)
                          │
                          ├─► RAG Retrieval + Knowledge Bridge
                          ├─► Claude 3.7 Sonnet Generation  
                          ├─► Multi-Layer Safety Guards
                          └─► Trace Export + Logging
```

## RAG (Retrieval-Augmented Generation) System

### Multi-Stage Retrieval Pipeline

The RAG system implements a sophisticated multi-stage retrieval and reranking pipeline designed for high precision and recall in pharmaceutical content.

#### Stage 1: Query Understanding & Expansion
```python
# Intent Detection
- Classifies queries into categories: adverse_events, dosage, interactions, general
- Uses semantic similarity to detect user intent patterns
- Applies intent-specific search strategies

# Knowledge Bridge Expansion  
- Expands drug names: "Zepbound" → ["zepbound", "tirzepatide"]
- Concept mapping: "side effects" → ["adverse reactions", "adverse events", "safety"]
- Morphological variants: "nausea" → ["nausea", "nauseated"]
```

#### Stage 2: Vector Search Fan-Out
```python
# Multi-Query Vector Search
- Generates 3-8 search variants per user query
- Executes parallel searches against Pinecone index
- Uses all-mpnet-base-v2 embeddings (768 dimensions)
- Retrieves 60+ candidates per variant for high recall
```

#### Stage 3: Hybrid Reranking & Fusion
```python
# Three-Signal Scoring System
1. Vector Similarity (15% weight)
   - Cosine similarity from Pinecone search
   - Normalized to [0,1] range

2. Lexical TF-IDF (25% weight)  
   - Scikit-learn TF-IDF vectorization
   - Captures exact term matches missed by embeddings
   - Handles medical terminology and abbreviations

3. Cross-Encoder Reranking (60% weight)
   - ms-marco-MiniLM-L-6-v2 cross-encoder
   - Deep bidirectional attention between query and passage
   - Most sophisticated relevance signal

# Section Boost (+0.10-0.15)
- Prioritizes chunks from relevant document sections
- "Adverse Reactions" sections for side effect queries
- "Dosage" sections for administration questions
```

#### Stage 4: Term Rescue
```python
# Ensures Key Term Coverage
- Identifies salient terms from user query
- Requires minimum 2 chunks containing key terms
- Replaces low-scoring chunks if term coverage insufficient
- Prevents semantic drift while maintaining precision
```

### Knowledge Bridge Integration

The Knowledge Bridge enhances retrieval without compromising safety boundaries.

#### Safe Query Expansion
```python
# Drug Synonym Mapping
drug_synonyms = {
    "zepbound": {"tirzepatide", "zepbound"},
    "tirzepatide": {"zepbound", "tirzepatide"}
}

# Medical Concept Mapping
concept_mappings = {
    "side effects": {"adverse reactions", "adverse events", "reactions"},
    "dosage": {"dose", "dosing", "administration", "how much"}
}
```

#### Expansion Process
1. **Entity Detection** - Identifies drugs and medical concepts in query
2. **Synonym Injection** - Adds verified synonyms to search variants  
3. **Context Enrichment** - Provides LLM with drug relationship context
4. **Safety Preservation** - All mappings manually curated from approved docs

## Multi-Layer Safety Architecture

### Layer 1: Pre-Retrieval Gates (Fast Blocking)

High-speed regex and embedding-based filters that block harmful queries before expensive retrieval operations.

#### Self-Harm Detection
```python
SELF_HARM = re.compile(
    r"\b(kill myself|suicide|want to die|end my life|hurt myself)\b", re.I
)
# → Immediate crisis resource routing, blocks all downstream processing
```

#### Criminal/Violence Intent
```python  
CRIME_VIOLENCE = re.compile(
    r"(kill|murder|poison|harm|injure|maim|assault|terror|bomb|explosive)\b", re.I
)
# → Policy refusal, no retrieval attempted
```

#### Jailbreak Prevention
```python
JAILBREAK = re.compile(
    r"(ignore (?:previous|prior) (?:instructions|rules)|act as|role[- ]?play)\b", re.I
)
# → Refuses fiction/roleplay attempts that could bypass medical guidelines
```

### Layer 2: Medical Classification Guards

Embedding-based classifiers that detect medical advice requests and off-label usage queries.

#### Medical Advice Detection
```python
# Three-Class Embedding Classifier
labels = {
    "medical_advice": "A request for individualized medical advice about dosing, timing, contraindications",
    "off_label_use": "A request about unapproved uses of a medicine for conditions or populations", 
    "general_info": "A general factual question about approved labeling or patient information"
}

# Threshold-Based Decision
if medical_advice_score >= 0.55:
    return "I can't provide individualized medical advice. Please contact your healthcare provider."
```

#### Off-Label Usage Blocking
- Detects queries about unapproved uses, populations, or indications
- Blocks before retrieval to prevent system from finding off-label information
- Configurable threshold (default: 0.55) balances precision/recall

### Layer 3: Content Screening

Regex-based detection of personally identifiable information and adverse event reports.

#### PHI/PII Protection
```python
PII_PATTERNS = [
    re.compile(r"\b\d{3}[-.\s]?\d{2}[-.\s]?\d{4}\b"),  # SSN
    re.compile(r"\b(?:\+?1[-.\s])?\(?\d{3}\)?[-.\s]\d{3}[-.\s]\d{4}\b"),  # Phone
    re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")  # Email
]
# → Privacy reminder, continues processing
```

#### Adverse Event Detection
```python
AE_HINTS = re.compile(
    r"adverse event|had a reaction|went to (?:ER|emergency)|hospitali[sz]ed", re.I
)
# → Adds AE reporting guidance, continues with medical information
```

## Grounding System (Critical Safety Layer)

The grounding system ensures all generated responses are semantically anchored to retrieved context, preventing hallucinations.

### Embedding-Based Grounding Validation

#### Sentence-Level Similarity Computation
```python
def compute_grounding(draft_response: str, context: str) -> Dict[str, float]:
    # Split into sentences
    draft_sentences = split_sentences(draft_response, max_sentences=20)
    context_sentences = split_sentences(context, max_sentences=60)
    
    # Encode with fast embedding model (all-MiniLM-L6-v2)
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    draft_embeddings = model.encode(draft_sentences)
    context_embeddings = model.encode(context_sentences)
    
    # Compute similarity matrix [draft_sentences × context_sentences]
    similarity_matrix = util.cos_sim(draft_embeddings, context_embeddings)
    
    # For each draft sentence, find max similarity to any context sentence
    max_similarities = similarity_matrix.max(dim=1).values
    
    return {
        "avg_max_sim": float(max_similarities.mean()),
        "min_max_sim": float(max_similarities.min()),
        "covered_frac": float((max_similarities >= 0.25).float().mean())
    }
```

#### Grounding Gate Decision
```python
def passes_grounding(grounding_details: Dict[str, float]) -> bool:
    # Configurable thresholds
    avg_threshold = 0.42  # Average similarity must exceed this
    coverage_threshold = 0.65  # Fraction of sentences that must be grounded
    
    avg_sim = grounding_details["avg_max_sim"]
    coverage = grounding_details["covered_frac"]
    
    return (avg_sim >= avg_threshold) and (coverage >= coverage_threshold)
```

### Grounding Failure Handling
```python
# Hard Refusal for Ungrounded Content
if not passes_grounding(grounding_details):
    return "I apologize, I don't seem to have that information. Can I assist you with something else?"
    # No partial information, no hedging - complete refusal
```

### Grounding Optimization Strategies

#### Context Quality
- **Sentence-level extraction** from retrieved chunks
- **Query-relevance ranking** prioritizes most relevant sentences
- **Character budget management** (2000 chars) ensures focused context

#### Embedding Model Selection  
- **Fast model** (all-MiniLM-L6-v2) for real-time grounding validation
- **Balanced** between speed and semantic understanding
- **Separate from retrieval embeddings** to avoid overfitting

#### Threshold Tuning
- **Conservative defaults** prioritize safety over coverage
- **Environment variable configuration** allows deployment-specific tuning
- **Trace logging** provides grounding scores for threshold optimization

## Fair-Balance Integration

Automatically detects benefit claims and couples them with risk information.

### Benefit Detection
```python
benefits_pattern = re.compile(
    r"effective|works|relieves|reduces|improves|benefit|helps|efficacy", re.I
)
```

### Risk Snippet Selection
```python
def fair_balance_snippet(retrieved_chunks) -> Optional[str]:
    # Scan for factual risk statements
    risk_words = re.compile(r"\b(risk|warning|side effect|adverse|precaution)\b", re.I)
    
    candidates = []
    for chunk in retrieved_chunks:
        for sentence in extract_sentences(chunk.text):
            if risk_words.search(sentence) and 30 <= len(sentence) <= 240:
                candidates.append(sentence)
    
    # Return shortest (most concise) risk statement
    return min(candidates, key=len) if candidates else None
```

### Balanced Response Construction
```python
if mentions_benefits(response) and has_risk_snippet:
    response = f"{response}\n\n{risk_snippet}"
```

## Performance Profile

- **RAG Retrieval**: ~0.6–1.5s (including Knowledge Bridge expansion and reranking)
- **Grounding Validation**: ~0.2–0.5s (fast embedding similarity computation)  
- **LLM Generation**: ~2–4s (Claude 3.7 Sonnet with conversation context)
- **Guard Processing**: ~0.3–0.8s (regex + embedding classification)
- **End-to-end**: ~3–7s typical (warm cache), up to 12s (cold start with model loading)

## Configuration

### RAG Tuning
```bash
RAG_TOPK=60              # Initial retrieval breadth
RAG_FINALK=5             # Final chunk count
RAG_W_RERANK=0.60        # Cross-encoder weight
RAG_W_LEX=0.25           # TF-IDF weight  
RAG_W_VEC=0.15           # Vector similarity weight
RAG_RESCUE_MIN_HITS=2    # Minimum term coverage
```

### Safety Thresholds
```bash
ADVICE_THRESHOLD=0.55         # Medical advice detection
OFF_LABEL_THRESHOLD=0.55      # Off-label usage detection
GROUNDING_AVG_THRESH=0.42     # Average sentence similarity
GROUNDING_COVERED_FRAC=0.65   # Minimum grounded sentence fraction
FAIR_BALANCE_ENFORCED=true    # Risk snippet insertion
```

### Model Configuration
```bash
EMBEDDING_MODEL=sentence-transformers/all-mpnet-base-v2      # Retrieval embeddings
GROUNDING_EMBED_MODEL=sentence-transformers/all-MiniLM-L6-v2 # Grounding validation
RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2         # Cross-encoder reranking
ANTHROPIC_MODEL=claude-3-7-sonnet-20250109                  # Claude model version
```

## Architecture Highlights

### Safety-First Design
- **Multiple pre-retrieval gates** block harmful queries before expensive operations
- **Strict grounding enforcement** prevents all hallucinations through semantic similarity
- **Layered medical guardrails** with embedding-based classification and regex screening
- **Traceable decision making** with comprehensive logging and structured traces

### Enhanced Query Understanding
- **Knowledge Bridge synonym expansion** improves recall without compromising safety
- **Multi-signal fusion scoring** combines vector, lexical, and cross-encoder signals
- **Intent-aware search variants** optimize retrieval for different query types
- **Term rescue mechanism** ensures key query terms are represented in results

### Production-Ready Architecture
- **API-first design** with clean separation between API layer and business logic
- **Lazy loading** optimizes memory usage for cloud deployment
- **Conversation state externalization** enables stateless, scalable API design
- **Comprehensive observability** with structured traces and performance metrics

## Migration Notes

### Removed Components
- **Streamlit UI**: Replaced with Flask API for browser extension integration
- **Session state management**: Externalized to client for stateless scaling
- **conversation.py**: Conversation context now passed in API requests

### Enhanced Components  
- **conversational_agent.py**: Accepts conversation history for context-aware responses
- **llm_client.py**: Enhanced message building with follow-up detection
- **server.py**: New Flask API entry point with lazy loading and dual endpoints

### Maintained Components
- **Complete safety system**: All guards, grounding, and fair-balance mechanisms preserved
- **RAG pipeline**: Full retrieval, reranking, and Knowledge Bridge functionality
- **Trace export**: Comprehensive logging and debugging capabilities maintained
- **Performance optimizations**: Caching, model singletons, and efficiency measures

## Known Limitations

- **Memory constraints**: Large embedding models require sufficient RAM (>1GB recommended)
- **Cold start latency**: Model loading adds 5-10 seconds to first request
- **Strict grounding**: Conservative thresholds may increase refusal rate for edge cases
- **Manual knowledge curation**: Knowledge Bridge mappings require ongoing maintenance
- **Client-side state**: Conversation persistence depends on browser extension implementation