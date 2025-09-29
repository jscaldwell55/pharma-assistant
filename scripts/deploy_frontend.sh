#!/bin/bash

# Deploy only the frontend to Firebase
set -e

echo "Deploying frontend to Firebase Hosting..."
firebase deploy --only hosting

echo "Frontend deployed successfully!"