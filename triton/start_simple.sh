#!/bin/bash

# Start Services in Background Script for Kepco Triton Inference Server
# This script starts the Triton Inference Server in the background

set -e

# Get the directory where the script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Load environment variables from .env if it exists
if [ -f .env ]; then
    echo "Loading environment variables from .env..."
    export $(cat .env | grep -v '^#' | xargs)
fi

# Default configuration
TRITON_HTTP_PORT=${TRITON_HTTP_PORT:-8000}
TRITON_GRPC_PORT=${TRITON_GRPC_PORT:-8001}
TRITON_METRICS_PORT=${TRITON_METRICS_PORT:-8002}
MODEL_REPOSITORY=${MODEL_REPOSITORY:-./rbln_backend}
BACKEND_TYPE=${BACKEND_TYPE:-rbln}
LOG_VERBOSE=${LOG_VERBOSE:-1}

# PID file location
PID_FILE="$SCRIPT_DIR/.triton_server.pid"
LOG_FILE="$SCRIPT_DIR/triton_server.log"

# Check if server is already running
if [ -f "$PID_FILE" ]; then
    OLD_PID=$(cat "$PID_FILE")
    if ps -p "$OLD_PID" > /dev/null 2>&1; then
        echo "Triton server is already running (PID: $OLD_PID)"
        echo "Use ./stop.sh to stop it first"
        exit 1
    else
        echo "Removing stale PID file..."
        rm -f "$PID_FILE"
    fi
fi

# Determine model repository based on backend type
if [ "$BACKEND_TYPE" = "gpu" ]; then
    MODEL_REPOSITORY="./gpu_backend"
    echo "Using GPU backend (PyTorch)"
elif [ "$BACKEND_TYPE" = "rbln" ]; then
    MODEL_REPOSITORY="./rbln_backend"
    echo "Using NPU backend (RBLN)"
else
    echo "Warning: Unknown BACKEND_TYPE '$BACKEND_TYPE'. Using default: rbln"
    MODEL_REPOSITORY="./rbln_backend"
fi

# Check if model repository exists
if [ ! -d "$MODEL_REPOSITORY" ]; then
    echo "Error: Model repository not found: $MODEL_REPOSITORY"
    echo "Please ensure the model repository exists or set MODEL_REPOSITORY environment variable"
    exit 1
fi

# Find tritonserver executable
TRITON_SERVER_BIN=""
if command -v tritonserver &> /dev/null; then
    TRITON_SERVER_BIN="tritonserver"
elif [ -f "/opt/tritonserver/bin/tritonserver" ]; then
    TRITON_SERVER_BIN="/opt/tritonserver/bin/tritonserver"
elif [ -f "/usr/local/bin/tritonserver" ]; then
    TRITON_SERVER_BIN="/usr/local/bin/tritonserver"
else
    echo "Error: tritonserver command not found"
    echo "Please install NVIDIA Triton Inference Server"
    echo "See: https://github.com/triton-inference-server/server"
    echo ""
    echo "Common installation paths checked:"
    echo "  - PATH: $(command -v tritonserver 2>/dev/null || echo 'not found')"
    echo "  - /opt/tritonserver/bin/tritonserver"
    echo "  - /usr/local/bin/tritonserver"
    exit 1
fi

echo "=========================================="
echo "Starting Triton Inference Server (Background)"
echo "=========================================="
echo "Model Repository: $MODEL_REPOSITORY"
echo "HTTP Port: $TRITON_HTTP_PORT"
echo "gRPC Port: $TRITON_GRPC_PORT"
echo "Metrics Port: $TRITON_METRICS_PORT"
echo "Backend Type: $BACKEND_TYPE"
echo "Log File: $LOG_FILE"
echo "PID File: $PID_FILE"
echo "=========================================="
echo ""

# Start Triton server in background
# Note: --disable-auto-complete-config is a flag (no value)
# If you want to disable auto-complete, remove the line or set DISABLE_AUTO_COMPLETE=true
TRITON_ARGS=(
    --model-repository="$MODEL_REPOSITORY"
    --http-port="$TRITON_HTTP_PORT"
    --grpc-port="$TRITON_GRPC_PORT"
    --metrics-port="$TRITON_METRICS_PORT"
    --log-verbose="$LOG_VERBOSE"
    --exit-on-error=false
)

# Add --disable-auto-complete-config flag only if DISABLE_AUTO_COMPLETE is true
if [ "${DISABLE_AUTO_COMPLETE:-false}" = "true" ]; then
    TRITON_ARGS+=(--disable-auto-complete-config)
fi

nohup "$TRITON_SERVER_BIN" "${TRITON_ARGS[@]}" > "$LOG_FILE" 2>&1 &

# Save PID
SERVER_PID=$!
echo "$SERVER_PID" > "$PID_FILE"

echo "Triton server started in background"
echo "PID: $SERVER_PID"
echo "Logs: $LOG_FILE"
echo ""
echo "To stop the server, run: ./stop.sh"
echo "To view logs, run: tail -f $LOG_FILE"

# Wait a moment and check if process is still running
sleep 2
if ps -p "$SERVER_PID" > /dev/null 2>&1; then
    echo "Server is running successfully"
else
    echo "Error: Server failed to start. Check logs: $LOG_FILE"
    rm -f "$PID_FILE"
    exit 1
fi

