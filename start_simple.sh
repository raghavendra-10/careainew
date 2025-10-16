#!/bin/bash
# Simple Phase 1 Startup - Redis Enhanced Threading

echo "🚀 Starting CareAI Enhanced System (Simple Mode)"
echo "================================================"

# Check if Redis is running
if redis-cli ping > /dev/null 2>&1; then
    echo "✅ Redis is running - Enhanced mode activated"
else
    echo "⚠️ Redis not running - Using memory-based progress"
    echo "   To start Redis: brew services start redis (macOS) or sudo systemctl start redis (Linux)"
fi

# Check dependencies
python -c "import redis" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Installing Redis dependency..."
    pip install redis
fi

echo "✅ Dependencies ready"

# Start Flask app with enhanced features
echo "🌟 Starting Flask app with enhanced threading..."
echo ""
echo "📋 Features Active:"
echo "   - Enhanced threading with resource limits"
echo "   - Redis-based progress tracking (if Redis available)"
echo "   - Automatic fallback to memory-based progress"
echo "   - Same API endpoints - no frontend changes!"
echo ""
echo "🔗 Test endpoints:"
echo "   Health: curl http://localhost:8002/health"
echo "   Upload: curl -X POST http://localhost:8002/upload -F 'file=@test.pdf'"
echo ""
echo "🛑 Stop: Ctrl+C"
echo ""

export PYTHONPATH="$PWD:$PYTHONPATH"
python app.py