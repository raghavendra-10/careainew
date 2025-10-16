#!/bin/bash
# Phase 1 Worker Shutdown Script

echo "🛑 Stopping CareAI Workers..."

# Stop workers using pidfiles
if [ -f "./logs/file_worker.pid" ]; then
    echo "Stopping file upload worker..."
    celery multi stop file_worker --pidfile=./logs/file_worker.pid
    rm -f ./logs/file_worker.pid
fi

if [ -f "./logs/embedding_worker.pid" ]; then
    echo "Stopping embedding worker..."
    celery multi stop embedding_worker --pidfile=./logs/embedding_worker.pid
    rm -f ./logs/embedding_worker.pid
fi

if [ -f "./logs/webhook_worker.pid" ]; then
    echo "Stopping webhook worker..."
    celery multi stop webhook_worker --pidfile=./logs/webhook_worker.pid
    rm -f ./logs/webhook_worker.pid
fi

if [ -f "./logs/agent_worker.pid" ]; then
    echo "Stopping agent worker..."
    celery multi stop agent_worker --pidfile=./logs/agent_worker.pid
    rm -f ./logs/agent_worker.pid
fi

# Fallback: kill any remaining celery processes
pkill -f "celery.*worker" 2>/dev/null

echo "✅ All workers stopped"
echo "📝 Check logs in ./logs/ directory"
echo "🚀 Flask application continues running normally"