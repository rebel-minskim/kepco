#!/bin/bash
# Continuous bottleneck monitoring script

echo "🔍 Bottleneck Monitor - Press Ctrl+C to stop"
echo "======================================"
echo ""

while true; do
    clear
    echo "🔍 BOTTLENECK ANALYSIS - $(date '+%H:%M:%S')"
    echo "======================================"
    echo ""
    
    # Get bottleneck analysis
    curl -s http://localhost:5000/bottleneck | jq '
        .timings_ms | to_entries | 
        sort_by(.value) | 
        reverse | 
        .[] | 
        "\(.key): \(.value | floor)ms"
    ' 2>/dev/null || echo "⚠️  Waiting for server..."
    
    echo ""
    echo "======================================"
    curl -s http://localhost:5000/bottleneck | jq -r '.bottleneck, .recommendation' 2>/dev/null
    echo ""
    
    # Get FPS and NPU util
    echo "📊 Current Stats:"
    curl -s http://localhost:5000/stats | jq '{fps: .fps, npu_util: .npu_utilization, detections: .detections}' 2>/dev/null
    
    sleep 2
done

