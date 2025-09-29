# Pharma Assistant - Production Deployment

A safety-first pharmaceutical information assistant with RAG, grounding, and multi-layer safety controls.

## Architecture

- **Backend**: Google Cloud Run (Flask + Gunicorn)
- **Frontend**: Firebase Hosting
- **Vector DB**: Pinecone
- **LLM**: Anthropic Claude 3.5 Sonnet
- **Embeddings**: Sentence Transformers

## Quick Start

### Prerequisites

1. Install required tools:
```bash
# Google Cloud CLI
curl https://sdk.cloud.google.com | bash

# Firebase CLI
npm install -g firebase-tools

# Python dependencies (for local development)
pip install -r requirements.txt
```

2. Set up accounts:
- Google Cloud Project
- Firebase Project
- Pinecone account
- Anthropic API key

### Configuration

1. Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```

2. Fill in your API keys in `.env`:
- `PINECONE_API_KEY`: Your Pinecone API key
- `ANTHROPIC_API_KEY`: Your Anthropic API key
- Update project IDs and other settings as needed

### Deployment

One-command deployment:
```bash
chmod +x scripts/deploy.sh
./scripts/deploy.sh
```

This will:
1. Enable required GCP APIs
2. Create/update secrets in Secret Manager
3. Build and push Docker container
4. Deploy to Cloud Run
5. Deploy frontend to Firebase Hosting

### Local Development

1. Run backend locally:
```bash
python backend/api/server.py
```

2. Run frontend locally:
```bash
cd frontend
python -m http.server 3000
```

3. Or use Docker:
```bash
./scripts/build_docker.sh
docker run -p 8080:8080 --env-file .env pharma-assistant:local
```

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
- **Cost**: $1-15/month depending on usage

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

## API Endpoints

### Health Check
```bash
curl https://your-service.run.app/api/health
```

### Chat
```bash
curl -X POST https://your-service.run.app/api/chat \
  -H "Content-Type: application/json" \
  -d '{"text": "What are the side effects?", "history": []}'
```

### Clear Cache
```bash
curl -X POST https://your-service.run.app/api/clear-cache
```

## Monitoring

View logs:
```bash
gcloud logs tail --project=$GCP_PROJECT_ID \
  --filter="resource.type=cloud_run_revision"
```

View metrics:
```bash
gcloud monitoring dashboards list --project=$GCP_PROJECT_ID
```

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

## License

Proprietary - All rights reserved