#!/bin/bash

cd /home/rebellions/rebellions/kepco

echo "🚀 Starting Flask Web App..."
echo "================================"

# Check if already running
if ps aux | grep -q "[a]pp_web.py"; then
    echo "⚠️  Flask app is already running!"
    echo "Run ./stop_app.sh to stop it first"
    exit 1
fi

# Start the app in background
echo "Starting app_web.py..."
nohup python3 app_web.py > logs/app_web.log 2>&1 &
PID=$!

sleep 2

# Check if it started successfully
if ps -p $PID > /dev/null 2>&1; then
    echo "✅ Flask app started successfully!"
    echo "   PID: $PID"
    echo "   URL: http://localhost:5000"
    echo "   Log: logs/app_web.log"
    echo ""
    echo "To stop: ./stop_app.sh"
    exit 0
else
    echo "❌ Failed to start Flask app"
    echo "Check logs/app_web.log for errors"
    exit 1
fi

