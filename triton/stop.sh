#!/bin/bash

# Stop All Services Script for Kepco Triton Inference Server
# This script stops the Triton Inference Server

set -e

# Get the directory where the script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PID_FILE="$SCRIPT_DIR/.triton_server.pid"

# Function to stop server by PID
stop_by_pid() {
    local pid=$1
    if ps -p "$pid" > /dev/null 2>&1; then
        echo "Stopping Triton server (PID: $pid)..."
        kill "$pid"
        
        # Wait for process to stop
        local count=0
        while ps -p "$pid" > /dev/null 2>&1 && [ $count -lt 10 ]; do
            sleep 1
            count=$((count + 1))
        done
        
        # Force kill if still running
        if ps -p "$pid" > /dev/null 2>&1; then
            echo "Force killing server (PID: $pid)..."
            kill -9 "$pid"
            sleep 1
        fi
        
        if ! ps -p "$pid" > /dev/null 2>&1; then
            echo "Server stopped successfully"
            rm -f "$PID_FILE"
            return 0
        else
            echo "Warning: Failed to stop server"
            return 1
        fi
    else
        echo "No running server found with PID: $pid"
        rm -f "$PID_FILE"
        return 1
    fi
}

# Try to stop using PID file
if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")
    stop_by_pid "$PID"
    exit $?
fi

# If no PID file, try to find tritonserver process
echo "No PID file found. Searching for tritonserver processes..."

# Find tritonserver processes
PIDS=$(pgrep -f "tritonserver" || true)

if [ -z "$PIDS" ]; then
    echo "No running Triton server processes found"
    exit 0
fi

# Stop all found processes
for pid in $PIDS; do
    echo "Found Triton server process (PID: $pid)"
    stop_by_pid "$pid"
done

echo "All Triton server processes stopped"

