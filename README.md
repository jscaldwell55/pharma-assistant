# Pharma Assistant - Production Deployment

A safety-first pharmaceutical information assistant with RAG, grounding, and multi-layer safety controls.

## Architecture

- **Backend**: Google Cloud Run (Flask + Gunicorn)
- **Frontend**: Firebase Hosting
- **Vector DB**: Pinecone
- **LLM**: Anthropic Claude 3.5 Sonnet
- **Embeddings**: Sentence Transformers





## Project Structure

```
pharma-assistant/
├── backend/
│   ├── api/              # Flask API server
│   └── core_logic/       # RAG, safety, LLM logic
├── frontend/             # Static web UI
├── scripts/              # Deployment scripts
├── Dockerfile            # Container configuration
├── requirements.txt      # Python dependencies
├── cloudbuild.yaml       # Cloud Build config
└── firebase.json         # Firebase config
```

## Key Features

### Safety Architecture
- **Pre-retrieval guards**: Violence, self-harm, jailbreak detection
- **Medical classification**: Medical advice & off-label blocking
- **Grounding enforcement**: Response must be grounded in retrieved docs
- **Fair balance**: Automatic risk disclosure with benefits

### RAG Pipeline
1. **Query expansion**: Knowledge bridge for synonyms
2. **Vector search**: Pinecone with multiple query variants
3. **Hybrid scoring**: Reranking + lexical + vector scores
4. **Term rescue**: Ensures key terms appear in results

### Performance
- **Cold start**: ~12-15 seconds (models cached in Docker)
- **Warm requests**: 2-4 seconds
- **Auto-scaling**: 0-10 instances

## Configuration Options

### Environment Variables

#### Required
- `PINECONE_API_KEY`: Pinecone API key
- `ANTHROPIC_API_KEY`: Anthropic API key
- `GCP_PROJECT_ID`: Google Cloud project ID

#### Optional
- `PINECONE_INDEX`: Index name (default: pharma-assistant)
- `PINECONE_NAMESPACE`: Namespace for vectors
- `EMBEDDING_MODEL`: Sentence transformer model
- `RERANKER_MODEL`: Cross-encoder model
- `ANTHROPIC_MODEL`: Claude model version
- `GROUNDING_AVG_THRESH`: Grounding threshold (0-1)
- `ADVICE_THRESHOLD`: Medical advice detection threshold


## Troubleshooting

### High latency
- Check if instance is cold (first request after idle)
- Verify models are cached in Docker image
- Check Pinecone query performance

### Out of memory
- Reduce batch sizes in RAG pipeline
- Use smaller embedding model
- Increase Cloud Run memory limit

### API errors
- Check environment variables are set
- Verify API keys are valid
- Check service logs for details

## Security Notes

- API keys stored in GCP Secret Manager
- HTTPS only for all endpoints
- CORS configured for specific origins
- Input sanitization and validation
- PHI/PII detection and blocking

