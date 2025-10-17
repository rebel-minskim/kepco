#!/bin/bash

echo "🛑 Stopping Flask Web App..."
echo "================================"

# Try graceful API shutdown first
echo "Attempting graceful shutdown via API..."
SHUTDOWN_RESPONSE=$(curl -s -X POST http://localhost:5000/shutdown 2>/dev/null)
if [ $? -eq 0 ]; then
    echo "✅ Shutdown API called successfully"
    sleep 2
    
    # Check if process stopped
    PIDS=$(ps aux | grep "[p]ython.*app_web.py" | awk '{print $2}')
    if [ -z "$PIDS" ]; then
        echo "✅ Flask app stopped gracefully"
        echo "Waiting for port 5000 to be released..."
        for i in {1..10}; do
            PORT_CHECK=$(ss -tlnp 2>/dev/null | grep :5000)
            if [ -z "$PORT_CHECK" ]; then
                echo "✅ Port 5000 is clear"
                exit 0
            fi
            sleep 0.5
        done
        exit 0
    fi
else
    echo "⚠️  Shutdown API not available, proceeding with signal-based shutdown..."
fi

# Find all app_web.py processes
PIDS=$(ps aux | grep "[p]ython.*app_web.py" | awk '{print $2}')

if [ -z "$PIDS" ]; then
    echo "❌ No app_web.py process found running"
    
    # Double-check port 5000
    PORT_PID=$(lsof -ti:5000 2>/dev/null || ss -lptn 2>/dev/null | grep :5000 | grep -oP 'pid=\K\d+' | head -1)
    if [ -n "$PORT_PID" ]; then
        echo "⚠️  Found process on port 5000: PID $PORT_PID"
        echo "Killing port-bound process..."
        # Check if port process is root
        PORT_USER=$(ps -p $PORT_PID -o user= 2>/dev/null)
        if [ "$PORT_USER" = "root" ]; then
            sudo kill -9 $PORT_PID 2>/dev/null
        else
            kill -9 $PORT_PID 2>/dev/null
        fi
        sleep 1
        echo "✅ Port 5000 cleared"
    fi
    
    exit 0
fi

echo "Found process(es): $PIDS"

# Check if any processes are running as root
ROOT_PIDS=$(ps -p $PIDS -o user= | grep -c root || true)
if [ "$ROOT_PIDS" -gt 0 ]; then
    echo "⚠️  Process is running as root, will need sudo for kill"
    NEED_SUDO=true
else
    NEED_SUDO=false
fi

# Try graceful shutdown first (SIGTERM)
for PID in $PIDS; do
    echo "Sending SIGTERM to PID $PID..."
    if [ "$NEED_SUDO" = true ]; then
        sudo kill $PID 2>/dev/null
    else
        kill $PID 2>/dev/null
    fi
done

# Wait up to 5 seconds for graceful shutdown
echo "Waiting for graceful shutdown..."
for i in {1..10}; do
    sleep 0.5
    STILL_RUNNING=$(ps aux | grep "[p]ython.*app_web.py" | awk '{print $2}')
    if [ -z "$STILL_RUNNING" ]; then
        echo "✅ Flask app stopped successfully"
        
        # Wait for port to be released (handle TIME_WAIT state)
        echo "Waiting for port 5000 to be released..."
        for j in {1..10}; do
            PORT_CHECK=$(ss -tlnp 2>/dev/null | grep :5000)
            if [ -z "$PORT_CHECK" ]; then
                echo "✅ Port 5000 is clear"
                exit 0
            fi
            sleep 0.5
        done
        echo "⚠️  Port 5000 may still be in TIME_WAIT state (will clear in ~30-60 seconds)"
        exit 0
    fi
done

# Force kill if still running
echo "⚠️  Process(es) still running after 5s, forcing kill..."
PIDS=$(ps aux | grep "[p]ython.*app_web.py" | awk '{print $2}')
for PID in $PIDS; do
    echo "Sending SIGKILL to PID $PID..."
    if [ "$NEED_SUDO" = true ]; then
        sudo kill -9 $PID 2>/dev/null
    else
        kill -9 $PID 2>/dev/null
    fi
done

# Final wait
sleep 1

# Final check
FINAL_CHECK=$(ps aux | grep "[p]ython.*app_web.py" | awk '{print $2}')
if [ -z "$FINAL_CHECK" ]; then
    echo "✅ Flask app stopped successfully (forced)"
    
    # Wait for port to be released (handle TIME_WAIT state)
    echo "Waiting for port 5000 to be released..."
    for i in {1..10}; do
        PORT_CHECK=$(ss -tlnp 2>/dev/null | grep :5000)
        if [ -z "$PORT_CHECK" ]; then
            echo "✅ Port 5000 is clear"
            exit 0
        fi
        sleep 0.5
    done
    echo "⚠️  Port 5000 may still be in TIME_WAIT state (will clear in ~30-60 seconds)"
    exit 0
else
    echo "❌ Failed to stop process"
    echo "Still running: $FINAL_CHECK"
    exit 1
fi

