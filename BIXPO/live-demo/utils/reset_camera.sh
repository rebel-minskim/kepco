#!/bin/bash

echo "Resetting Camera Device..."
echo "================================"

# Stop any running app first
./stop_app.sh

# Kill any process using the camera
echo "Checking for processes using camera..."
CAMERA_PROCS=$(lsof /dev/video* 2>/dev/null | awk 'NR>1 {print $2}' | sort -u)

if [ -n "$CAMERA_PROCS" ]; then
    echo "Found processes using camera: $CAMERA_PROCS"
    for PID in $CAMERA_PROCS; do
        echo "Killing PID $PID..."
        kill -9 $PID 2>/dev/null
    done
    sleep 1
else
    echo "No processes using camera"
fi

# Unload and reload camera modules (requires sudo)
echo ""
echo "Attempting to reload camera modules (may require sudo)..."
if command -v sudo &> /dev/null; then
    sudo modprobe -r uvcvideo 2>/dev/null
    sleep 1
    sudo modprobe uvcvideo 2>/dev/null
    sleep 1
    echo "Camera modules reloaded"
else
    echo "sudo not available, skipping module reload"
fi

# Check if camera is available
echo ""
echo "Checking camera availability..."
if [ -e /dev/video0 ]; then
    echo "/dev/video0 is available"
    ls -l /dev/video*
else
    echo "/dev/video0 not found"
    echo "Available video devices:"
    ls -l /dev/video* 2>/dev/null || echo "No video devices found"
fi

echo ""
echo "================================"
echo "Camera reset complete!"
echo "You can now run ./start_app.sh"

