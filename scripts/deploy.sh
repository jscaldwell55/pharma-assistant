#!/bin/bash

# Complete deployment script for Google Cloud Platform
set -e

# Load environment variables
if [ -f .env ]; then
    export $(cat .env | grep -v '^#' | xargs)
fi

# Check required environment variables
if [ -z "$GCP_PROJECT_ID" ]; then
    echo "Error: GCP_PROJECT_ID not set"
    exit 1
fi

if [ -z "$GCP_REGION" ]; then
    GCP_REGION="us-central1"
fi

if [ -z "$SERVICE_NAME" ]; then
    SERVICE_NAME="pharma-assistant-api"
fi

echo "🚀 Deploying Pharma Assistant to Google Cloud"
echo "Project: $GCP_PROJECT_ID"
echo "Region: $GCP_REGION"
echo "Service: $SERVICE_NAME"

# Set the active project
gcloud config set project $GCP_PROJECT_ID

# Enable required APIs
echo "📦 Enabling required APIs..."
gcloud services enable cloudbuild.googleapis.com \
    run.googleapis.com \
    containerregistry.googleapis.com \
    secretmanager.googleapis.com

# Create secrets if they don't exist
echo "🔐 Managing secrets..."
echo -n "$PINECONE_API_KEY" | gcloud secrets create PINECONE_API_KEY --data-file=- 2>/dev/null || \
    echo -n "$PINECONE_API_KEY" | gcloud secrets versions add PINECONE_API_KEY --data-file=-

echo -n "$ANTHROPIC_API_KEY" | gcloud secrets create ANTHROPIC_API_KEY --data-file=- 2>/dev/null || \
    echo -n "$ANTHROPIC_API_KEY" | gcloud secrets versions add ANTHROPIC_API_KEY --data-file=-

# Build and push container using Cloud Build
echo "🏗️ Building container image..."
gcloud builds submit \
    --tag gcr.io/$GCP_PROJECT_ID/$SERVICE_NAME:latest \
    --timeout=20m \
    .

# Deploy to Cloud Run
echo "☁️ Deploying to Cloud Run..."
gcloud run deploy $SERVICE_NAME \
    --image gcr.io/$GCP_PROJECT_ID/$SERVICE_NAME:latest \
    --region $GCP_REGION \
    --platform managed \
    --allow-unauthenticated \
    --min-instances 0 \
    --max-instances 10 \
    --cpu 2 \
    --memory 2Gi \
    --timeout 60 \
    --concurrency 80 \
    --set-env-vars "PINECONE_INDEX=$PINECONE_INDEX" \
    --set-env-vars "PINECONE_ENVIRONMENT=$PINECONE_ENVIRONMENT" \
    --set-env-vars "PINECONE_NAMESPACE=$PINECONE_NAMESPACE" \
    --set-env-vars "EMBEDDING_MODEL=$EMBEDDING_MODEL" \
    --set-env-vars "RERANKER_MODEL=$RERANKER_MODEL" \
    --set-env-vars "ANTHROPIC_MODEL=$ANTHROPIC_MODEL" \
    --set-secrets "PINECONE_API_KEY=PINECONE_API_KEY:latest" \
    --set-secrets "ANTHROPIC_API_KEY=ANTHROPIC_API_KEY:latest"

# Get the service URL
SERVICE_URL=$(gcloud run services describe $SERVICE_NAME --region $GCP_REGION --format 'value(status.url)')
echo "✅ Backend deployed to: $SERVICE_URL"

# Deploy frontend to Firebase
echo "🌐 Deploying frontend to Firebase..."

# Update API URL in frontend
sed -i.bak "s|https://pharma-assistant-api-[^.]*\.us-central1\.run\.app|$SERVICE_URL|g" frontend/app.js
rm frontend/app.js.bak

# Deploy to Firebase Hosting
firebase deploy --only hosting

echo "🎉 Deployment complete!"
echo "Backend API: $SERVICE_URL"
echo "Frontend: https://$GCP_PROJECT_ID.web.app"
echo ""
echo "Test endpoints:"
echo "  Health: $SERVICE_URL/api/health"
echo "  Warmup: $SERVICE_URL/api/warmup"