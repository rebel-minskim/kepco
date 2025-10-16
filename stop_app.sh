#!/bin/bash

echo "🛑 Stopping Flask Web App..."
echo "================================"

# Find and kill app_web.py process
PID=$(ps aux | grep "[a]pp_web.py" | awk '{print $2}')

if [ -z "$PID" ]; then
    echo "❌ No app_web.py process found running"
    exit 0
fi

echo "Found process: PID $PID"
kill $PID

# Wait a bit and check if it's really stopped
sleep 1

# Check if still running
if ps -p $PID > /dev/null 2>&1; then
    echo "⚠️  Process still running, forcing kill..."
    kill -9 $PID
    sleep 1
fi

# Final check
if ps -p $PID > /dev/null 2>&1; then
    echo "❌ Failed to stop process"
    exit 1
else
    echo "✅ Flask app stopped successfully"
    exit 0
fi

