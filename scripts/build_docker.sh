#!/bin/bash

# Build Docker image locally for testing
set -e

echo "Building Docker image locally..."
docker build -t pharma-assistant:local .

echo "Image built successfully!"
echo "To run locally:"
echo "  docker run -p 8080:8080 --env-file .env pharma-assistant:local"