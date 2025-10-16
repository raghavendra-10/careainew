#!/bin/bash
# Phase 1 Worker Startup Script

echo "🚀 Starting CareAI Production System - Phase 1"
echo "============================================="

# Check if Redis is running
if ! redis-cli ping > /dev/null 2>&1; then
    echo "❌ Redis is not running. Please start Redis first:"
    echo "   macOS: brew services start redis"
    echo "   Ubuntu: sudo systemctl start redis"
    echo "   Manual: redis-server"
    exit 1
fi

echo "✅ Redis is running"

# Check Python dependencies
python -c "import redis, celery" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "❌ Missing dependencies. Installing..."
    pip install -r requirements.txt
fi

echo "✅ Dependencies are installed"

# Set environment
export PYTHONPATH="$PWD:$PYTHONPATH"

# Start workers in background
echo "🔄 Starting Celery workers..."

# File upload worker
celery -A celery_config worker \
    --loglevel=info \
    --concurrency=5 \
    --queues=file_uploads \
    --hostname=file_worker@%h \
    --detach \
    --pidfile=./logs/file_worker.pid \
    --logfile=./logs/file_worker.log

# Embedding worker  
celery -A celery_config worker \
    --loglevel=info \
    --concurrency=3 \
    --queues=embeddings \
    --hostname=embedding_worker@%h \
    --detach \
    --pidfile=./logs/embedding_worker.pid \
    --logfile=./logs/embedding_worker.log

# Webhook worker
celery -A celery_config worker \
    --loglevel=info \
    --concurrency=10 \
    --queues=webhooks \
    --hostname=webhook_worker@%h \
    --detach \
    --pidfile=./logs/webhook_worker.pid \
    --logfile=./logs/webhook_worker.log

# Agent worker
celery -A celery_config worker \
    --loglevel=info \
    --concurrency=3 \
    --queues=agents \
    --hostname=agent_worker@%h \
    --detach \
    --pidfile=./logs/agent_worker.pid \
    --logfile=./logs/agent_worker.log

sleep 2

echo "✅ Workers started successfully"
echo ""
echo "📊 Monitor workers:"
echo "   celery -A celery_config inspect active"
echo "   tail -f logs/file_worker.log"
echo ""
echo "🌐 Start Flower monitoring (optional):"
echo "   flower -A celery_config --port=5555"
echo ""
echo "🚀 Start Flask application:"
echo "   python app.py"
echo ""
echo "🛑 Stop workers:"
echo "   ./stop_workers.sh"