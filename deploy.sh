#!/bin/bash
# Enhanced CareAI Production Deployment Script

set -e

echo "🚀 Deploying CareAI Enhanced System to Cloud Run"
echo "================================================="

# Configuration
PROJECT_ID="${GOOGLE_CLOUD_PROJECT:-your-project-id}"
SERVICE_NAME="careai-enhanced"
REGION="${CLOUD_RUN_REGION:-us-central1}"
IMAGE_NAME="gcr.io/$PROJECT_ID/$SERVICE_NAME"

echo "📋 Configuration:"
echo "   Project: $PROJECT_ID"
echo "   Service: $SERVICE_NAME"
echo "   Region: $REGION"
echo "   Image: $IMAGE_NAME"
echo ""

# Check if gcloud is authenticated
if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" | grep -q .; then
    echo "❌ Not authenticated with gcloud. Please run: gcloud auth login"
    exit 1
fi

# Set project
gcloud config set project $PROJECT_ID

echo "🔨 Building Docker image..."
docker build -t $IMAGE_NAME:latest .

echo "📤 Pushing image to Container Registry..."
docker push $IMAGE_NAME:latest

echo "🚀 Deploying to Cloud Run..."

# Deploy with enhanced configuration
gcloud run deploy $SERVICE_NAME \
    --image $IMAGE_NAME:latest \
    --platform managed \
    --region $REGION \
    --allow-unauthenticated \
    --memory 4Gi \
    --cpu 2 \
    --concurrency 20 \
    --max-instances 10 \
    --min-instances 1 \
    --timeout 900s \
    --port 8080 \
    --set-env-vars "FLASK_ENV=production,LOG_LEVEL=INFO,MAX_CONCURRENT_UPLOADS=3,MAX_CONCURRENT_EMBEDDINGS=2" \
    --update-secrets "OPENAI_API_KEY=openai-api-key:latest,PINECONE_API_KEY=pinecone-api-key:latest,GOOGLE_APPLICATION_CREDENTIALS=firebase-credentials:latest"

# Get the service URL
SERVICE_URL=$(gcloud run services describe $SERVICE_NAME --region=$REGION --format="value(status.url)")

echo ""
echo "✅ Deployment Complete!"
echo "🌐 Service URL: $SERVICE_URL"
echo ""
echo "🧪 Test your enhanced deployment:"
echo "   Health: curl $SERVICE_URL/health"
echo "   Upload: curl -X POST $SERVICE_URL/upload -F 'file=@test.pdf' -F 'orgId=test'"
echo ""
echo "📊 Enhanced Features Active:"
echo "   ✅ 3x concurrent file processing"
echo "   ✅ Redis progress tracking (if Redis configured)"
echo "   ✅ Auto-fallback to memory progress"
echo "   ✅ Enhanced error handling"
echo "   ✅ Resource limits and monitoring"
echo ""
echo "🔧 Configure Redis for best performance:"
echo "   1. Create Memorystore instance"
echo "   2. Update REDIS_URL environment variable"
echo "   3. Redeploy for full enhanced performance"