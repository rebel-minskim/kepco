# 🎬 Multi-Video Processing Guide

## Overview

Process multiple videos simultaneously to maximize GPU utilization and total throughput.

## Quick Start

### Test Scripts

```bash
cd /workspace/kepco/triton/client/cpp_client

# Run multi-video test
./test_multi_video.sh
```

## Performance Results

| Configuration | Individual FPS | Total Throughput | Improvement |
|---------------|---------------|------------------|-------------|
| **1 video** | 268 FPS | 268 inferences/sec | Baseline |
| **2 videos** | ~150 FPS each | ~292 inferences/sec | +9% |
| **4 videos** | ~76 FPS each | ~297 inferences/sec | +11% |

## Usage Examples

### Method 1: Separate Processes (Current)

```bash
cd build

# Create output directory
mkdir -p ../output

# Process 2 videos simultaneously
./bin/triton_client parallel ../media/video1.mp4 ../output/result1.mp4 4 &
./bin/triton_client parallel ../media/video2.mp4 ../output/result2.mp4 4 &
wait

# Process 4 videos simultaneously
./bin/triton_client parallel ../media/video1.mp4 ../output/result1.mp4 2 &
./bin/triton_client parallel ../media/video2.mp4 ../output/result2.mp4 2 &
./bin/triton_client parallel ../media/video3.mp4 ../output/result3.mp4 2 &
./bin/triton_client parallel ../media/video4.mp4 ../output/result4.mp4 2 &
wait
```

### Method 2: Inference-Only Mode (No Video Output)

For maximum throughput without saving output videos:

```bash
# Process 4 videos without saving (faster)
./bin/triton_client parallel ../media/video1.mp4 "" 2 &
./bin/triton_client parallel ../media/video2.mp4 "" 2 &
./bin/triton_client parallel ../media/video3.mp4 "" 2 &
./bin/triton_client parallel ../media/video4.mp4 "" 2 &
wait
```

## Architecture

### Current: Multiple Processes
```
Process 1: Video1 → [4 threads] → GPU
Process 2: Video2 → [4 threads] → GPU
Process 3: Video3 → [4 threads] → GPU
Process 4: Video4 → [4 threads] → GPU
                        ↓
                GPU Resource Competition
```

**Pros:**
- ✅ Simple to use (just run multiple processes)
- ✅ No code changes needed
- ✅ Immediate availability

**Cons:**
- ⚠️ Limited throughput increase (+11%)
- ⚠️ Each process loads model separately
- ⚠️ GPU memory competition

### Future: Single Process with Shared Workers

```
Video1 Reader ──┐
Video2 Reader ──┤
Video3 Reader ──┼→ [Shared 8 Preprocessors]
Video4 Reader ──┘   → [Shared 8 Inference threads]
                      → GPU (Efficient)
```

**Expected Performance:**
- Each video: ~180 FPS
- Total throughput: ~720 inferences/sec
- Improvement: **+169%** 🚀

**Benefits:**
- ✅ Maximum GPU utilization
- ✅ Single model instance (memory efficient)
- ✅ Shared worker pool (CPU efficient)

## Use Cases

### 1. CCTV Monitoring (4 cameras)
```bash
# Real-time monitoring of 4 cameras
./bin/triton_client parallel camera1_stream output1.mp4 2 &
./bin/triton_client parallel camera2_stream output2.mp4 2 &
./bin/triton_client parallel camera3_stream output3.mp4 2 &
./bin/triton_client parallel camera4_stream output4.mp4 2 &

# Result: Each camera at ~76 FPS (sufficient for real-time)
```

### 2. Batch Video Processing
```bash
# Process multiple video files
for video in video_*.mp4; do
    ./bin/triton_client parallel $video output_$video 2 &
done
wait
```

### 3. High-Throughput Benchmarking
```bash
# Inference-only mode for maximum throughput
for i in {1..8}; do
    ./bin/triton_client parallel input.mp4 "" 1 &
done
wait
```

## Performance Tuning

### Thread Allocation

For N videos with T total threads:
```
Threads per video = T / N

Examples:
- 2 videos, 8 threads → 4 threads each
- 4 videos, 8 threads → 2 threads each
- 8 videos, 8 threads → 1 thread each
```

### Optimal Configuration

| Use Case | Videos | Threads/Video | Expected FPS/Video |
|----------|--------|---------------|-------------------|
| Single high-FPS | 1 | 8 | ~268 |
| Dual processing | 2 | 4 | ~150 |
| Quad monitoring | 4 | 2 | ~76 |
| Many streams | 8 | 1 | ~38 |

## Troubleshooting

### Low FPS per video
- **Cause**: Too many videos for available GPU
- **Solution**: Reduce number of simultaneous videos or increase threads per video

### GPU out of memory
- **Cause**: Too many model instances loaded
- **Solution**: Process videos in batches, not all at once

### Uneven performance
- **Cause**: Video file differences (resolution, codec)
- **Solution**: Pre-process videos to same format

## Files

- `test_multi_video.sh` - Automated testing script
- `triton_client_multi.h` - Header for future single-process implementation
- `multi_video_example.cpp` - Example code (reference)

## Next Steps

For even better performance, consider implementing the single-process multi-video handler:
- Expected: +169% throughput improvement
- Benefits: Better GPU utilization, lower memory usage
- Development time: 2-3 hours

Contact the development team if you need this optimization!

