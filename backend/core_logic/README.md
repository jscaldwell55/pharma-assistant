# Pharma Assistant — System Overview & Safety Design

A safety-first pharmaceutical information assistant that answers only from approved, uploaded documents (labeling, PI, patient leaflets). The system emphasizes traceability, guardrails, fair-balance, and grounding, approximating the orchestration patterns used in enterprise GenAI stacks.

## Recent Enhancements (v3.0 - API Migration)

### API-First Architecture
- **Streamlit-free backend** - Core logic now runs as Flask API service
- **Browser extension ready** - RESTful `/api/rewrite` endpoint for browser integration
- **Conversation history support** - Full context awareness across turns
- **Modular package structure** - Clean separation between API layer and core logic

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
4. **Generates natural, conversational answers** (Claude 3.5 Sonnet) while maintaining strict grounding
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

## End-to-End Flow (Hot Path)

### 0) API Request Processing
- Receives JSON with `text` (query) and `history` (conversation context)
- Validates input and extracts conversation history
- Routes to singleton ConversationalAgent instance

### 1) Pre-Gates (fast regex/embedding)
- Self-harm → crisis resources; blocked before retrieval
- Criminal/dangerous intent → clear policy refusal, no retrieval
- If blocked, no retrieval or medical "safety notes" appended

### 2) Enhanced Retrieval (RAG + Knowledge Bridge)
- **Knowledge Bridge expansion**: Drug synonyms and concept mappings
- Multi-query expansion (synonyms, morphology) + intent hints
- Vector fan-out on Pinecone (MPNet 768-d embeddings)
- Lexical TF-IDF boost and optional cross-encoder rerank
- Term-rescue: ensures coverage of user's salient terms
- Output: top-K chunks with composite scoring

### 3) Context Building
- Extracts sentence-level snippets most relevant to query
- Produces compact, citation-ready context within character budget
- Risk terms receive small boost for fair-balance

### 4) Conversation-Aware Generation (Claude 3.5 Sonnet)
- **Context-aware message building** with conversation history
- Follow-up question detection and contextual understanding
- Clean system prompt for natural responses
- Fallback to extractive bullets if LLM fails

### 5) Grounding Gate (Safety-Critical)
- Computes embedding similarity between draft and context
- Hard refusal if below thresholds (no hallucinations)
- Configurable thresholds via environment variables

### 6) Medical Guardrails
- **Advice/off-label detection**: Embedding classifier (primary)
- **PHI/PII screening**: Regex patterns
- **Adverse Event detection**: Heuristics → AE routing note
- **Jailbreak prevention**: Refuses fiction/role-play attempts

### 7) Fair-Balance Coupler
- Inserts risk snippet when benefits/efficacy mentioned
- Configurable placement (top/end)
- Never applied to refusal messages

### 8) API Response
- Returns JSON with `rewritten_text` containing final answer
- Full JSON trace for audit/debugging (server-side logging)

## Key Components

### API Layer (`backend/api/`)
- `server.py`: Flask application with `/api/rewrite` endpoint
- `requirements.txt`: API-specific dependencies
- CORS-enabled for browser extension integration

### Core Logic Package (`backend/core_logic/`)
- **Modular package structure** with `__init__.py`
- **Relative imports** for clean dependency management
- All business logic isolated from presentation layer

### Knowledge Bridge (`knowledge_bridge.py`) - Enhanced
- `PharmaceuticalKnowledgeBridge`: Manages drug/concept synonyms
- `EnhancedRAGRetriever`: Wraps base retriever with query expansion
- Zero-config integration with existing safety systems

### Conversational Agent (`conversational_agent.py`) - Updated
- **Conversation history support** - accepts and processes dialogue context
- **Context-aware caching** with conversation-sensitive cache keys
- Maintains all existing safety controls

### LLM Client (`llm_client.py`) - Enhanced
- **Conversation-aware message building** with history integration
- **Follow-up detection** for contextual responses
- Clean, natural system prompts
- Robust error handling with conversation context

### Retrieval & Scoring (`rag.py`)
- Intent detection + multi-query expansion
- Pinecone vector search (all-mpnet-base-v2, 768-d)
- Lexical TF-IDF + optional CrossEncoder rerank
- Fusion scoring with term-rescue

### Grounding & Guards (`grounding.py`, `guard.py`)
- Embedding-based similarity checking
- Hard gate for ungrounded content
- Multi-layer safety controls
- Reason-specific refusal messages

## Configuration

### Core Settings
```bash
PINECONE_API_KEY=...
PINECONE_ENVIRONMENT=us-east-1
PINECONE_INDEX=pharma-assistant
ANTHROPIC_API_KEY=...
```

### Retrieval Tuning
- `RAG_TOPK`, `RAG_FINALK`: Control retrieval breadth
- `RAG_W_RERANK`, `RAG_W_LEX`, `RAG_W_VEC`: Scoring weights
- `RAG_RESCUE_MIN_HITS`: Term coverage threshold

### Safety Controls
- `ADVICE_THRESHOLD`, `OFF_LABEL_THRESHOLD`: Classification thresholds
- `GROUNDING_AVG_THRESH`: Default 0.45
- `GROUNDING_COVERED_FRAC`: Default 0.70
- `FAIR_BALANCE_ENFORCED`: Enable/disable risk snippets

### API Settings
- Flask runs on port 5000 by default
- CORS enabled for browser extension integration
- Environment variables loaded via python-dotenv

## Installation & Running

### Backend API Setup
```bash
# Navigate to API directory
cd backend/api

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# Install dependencies
pip install -r requirements.txt

# Set environment variables (create .env file)
echo "PINECONE_API_KEY=your_key_here" >> .env
echo "ANTHROPIC_API_KEY=your_key_here" >> .env
echo "PINECONE_INDEX=pharma-assistant" >> .env

# Run the API server
python server.py
```

### Project Structure
```
pharma-assistant-benchmark/
└── backend/
    ├── api/
    │   ├── server.py              # Flask API entry point
    │   └── requirements.txt       # API dependencies
    └── core_logic/
        ├── __init__.py            # Package initialization
        ├── conversational_agent.py
        ├── llm_client.py
        ├── rag.py
        ├── guard.py
        ├── knowledge_bridge.py
        └── ... (all other core files)
```

### Browser Extension Integration
The API accepts POST requests to `/api/rewrite` with:
```json
{
  "text": "What are the side effects of Zepbound?",
  "history": [
    {"role": "user", "content": "What is Zepbound used for?"},
    {"role": "assistant", "content": "Zepbound is indicated for..."}
  ]
}
```

Returns:
```json
{
  "rewritten_text": "The most common side effects of Zepbound include..."
}
```

## Performance Profile

- **API Startup**: ~5-10s (model loading, singleton initialization)
- **Retrieval**: ~0.6–1.5s (with Knowledge Bridge expansion)
- **LLM Generation**: ~2–4s (Claude 3.5 Sonnet with conversation context)
- **Guard Checks**: ~0.8–2.5s (including grounding)
- **End-to-end**: ~3–7s typical (warm cache)
- **Conversation-aware responses**: +0.2-0.5s for context processing

## Architecture Highlights

### API-First Design
- Clean separation between API layer and business logic
- RESTful endpoint design for browser extension integration
- Conversation state managed by client for scalability

### Safety-First Design (Maintained)
- Multiple pre-retrieval gates
- Strict grounding enforcement
- Layered medical guardrails
- Traceable decision making

### Enhanced Query Understanding (Maintained)
- Automatic synonym expansion
- Concept mapping for medical terms
- No prompt engineering required
- Maintains safety boundaries

### Natural Communication (Enhanced)
- Professional, conversational responses
- Context-aware follow-up understanding
- No unnecessary qualifiers
- Clean refusal messages

### Conversation Continuity (New)
- Full conversation history integration
- Follow-up question understanding
- Contextual reference resolution
- Session management via client

## Migration Notes

### Removed Components
- **Streamlit UI**: Replaced with Flask API
- **Session state management**: Now handled by browser extension
- **conversation.py**: Conversation state managed by client

### Enhanced Components  
- **conversational_agent.py**: Now accepts conversation history
- **llm_client.py**: Context-aware message building
- **server.py**: New Flask API entry point

### Maintained Components
- All core safety systems (guards, grounding, fair-balance)
- Knowledge Bridge query expansion
- RAG retrieval and scoring
- Trace export and logging

## Known Limitations

- Conversation persistence depends on client implementation
- Very strict grounding thresholds may increase refusals
- Knowledge Bridge mappings must be manually maintained
- API startup time includes model loading overhead
- No built-in rate limiting (should be handled at reverse proxy level)