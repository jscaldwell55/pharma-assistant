# Multi-stage build for optimized container
FROM python:3.11-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install with specific compatible versions
RUN pip install --no-cache-dir --user \
    flask==3.0.0 \
    flask-cors==4.0.0 \
    gunicorn==21.2.0 \
    sentence-transformers==2.7.0 \
    huggingface-hub==0.24.5 \
    transformers==4.41.2 \
    torch==2.2.2 \
    scikit-learn==1.4.2 \
    numpy==1.26.4 \
    tiktoken==0.7.0 \
    nltk==3.8.1 \
    pinecone-client==3.0.0 \
    anthropic==0.34.2 \
    python-dotenv==1.0.0 \
    pandas==2.2.2 \
    langchain-text-splitters==0.2.2

# Download models during build (skip this step to avoid the error)
# Models will be downloaded on first use instead

# Production stage
FROM python:3.11-slim

WORKDIR /app

# Copy Python packages from builder
COPY --from=builder /root/.local /root/.local

# Make sure scripts in .local are usable
ENV PATH=/root/.local/bin:$PATH
ENV TRANSFORMERS_CACHE=/app/.cache
ENV HF_HOME=/app/.cache
ENV SENTENCE_TRANSFORMERS_HOME=/app/.cache

# Copy application code
COPY backend/ ./backend/

# Set environment variables
ENV PORT=8080
ENV PYTHONUNBUFFERED=1
ENV DEPLOYMENT_ENV=cloud_run

# Create cache directory
RUN mkdir -p /app/.cache

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8080/api/health || exit 1

# Run with gunicorn for production
CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 60 \
    --worker-class gthread --preload backend.api.server:app