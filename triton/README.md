# Triton Inference Server Integration

YOLO11 object detection with NVIDIA Triton Inference Server, supporting both GPU and NPU backends.

## Overview

This project provides Triton Inference Server integration for YOLO11 with:
- **GPU Backend**: PyTorch-based inference
- **NPU Backend**: Rebellions RBLN-optimized inference
- **C++ Client**: High-performance gRPC and HTTP client
- **Python Client**: Easy-to-use Python interface

## Directory Structure

```
triton/
├── client/
│   ├── cpp_client/              # C++ gRPC/HTTP client
│   │   ├── main.cpp            # Client entry point
│   │   ├── triton_client.cpp   # Triton client implementation
│   │   ├── grpc_client.cpp     # gRPC wrapper
│   │   ├── yolo_preprocess.cpp # YOLO preprocessing
│   │   ├── yolo_postprocess.cpp# YOLO postprocessing
│   │   ├── build.sh            # Build script
│   │   ├── test_multi_video.sh # Multi-video test
│   │   └── README.md           # C++ client docs
│   │
│   └── python_client/           # Python client
│       ├── main.py             # Basic client
│       ├── main_concurrent.py  # Concurrent inference
│       ├── main_concurrent_simple.py  # Simplified concurrent
│       ├── config.py           # Configuration
│       ├── utils/              # Utility modules
│       └── README.md           # Python client docs
│
├── gpu_backend/                 # GPU (PyTorch) backend
│   └── yolov11/
│       ├── config.pbtxt        # Triton config
│       └── 1/
│           ├── model.py        # PyTorch model wrapper
│           └── yolov11.pt      # PyTorch model
│
├── rbln_backend/                # NPU (RBLN) backend
│   └── yolov11/
│       ├── config.pbtxt        # Triton config
│       └── 1/
│           ├── model.py        # RBLN model wrapper
│           └── yolov11.rbln    # RBLN compiled model
│
├── docs/                        # Documentation
│   ├── ARCHITECTURE.md         # System architecture
│   ├── PERFORMANCE.md          # Performance benchmarks
│   └── README_MULTI_VIDEO.md   # Multi-video guide
│
├── models/                      # Model storage
│   ├── gpu/                    # GPU models
│   └── rbln/                   # RBLN models
│
├── .gitignore
└── README.md                    # This file
```

## Quick Start

### 1. Start Triton Server

**GPU Backend:**
```bash
tritonserver --model-repository=./gpu_backend
```

**NPU Backend:**
```bash
tritonserver --model-repository=./rbln_backend
```

### 2. Run Client

**C++ Client:**
```bash
cd client/cpp_client
./build.sh
./bin/triton_client parallel video.mp4 output.mp4 8
```

**Python Client:**
```bash
cd client/python_client
pip install -r requirements.txt
python main.py --video video.mp4 --output output.mp4
```

## Performance

### Single Video Processing
- **C++ Client**: ~30 FPS (GPU), ~35-40 FPS (NPU)
- **Python Client**: ~25-28 FPS (GPU), ~30-35 FPS (NPU)

### Multi-Video Concurrent Processing
- **2 Videos**: ~1.8x throughput improvement
- **4 Videos**: ~3.2x throughput improvement

See [docs/PERFORMANCE.md](docs/PERFORMANCE.md) for detailed benchmarks.

## Features

### C++ Client
- gRPC and HTTP support
- Multi-threaded processing
- Concurrent multi-video inference
- Zero-copy where possible
- Optimized preprocessing/postprocessing

### Python Client
- Simple and concurrent modes
- Queue-based frame management
- Async inference support
- Video/image/webcam inputs
- Easy integration

### Backend Support
- **GPU**: PyTorch with CUDA optimization
- **NPU**: Rebellions RBLN with async runtime
- Configurable batch sizes
- Dynamic batching support

## Performance Analysis

### Using Triton Performance Analyzer

Triton provides `perf_analyzer` tool for comprehensive performance benchmarking.

**Basic Usage:**
```bash
perf_analyzer -m yolov11 \
  -u localhost:8000 \
  --shape IMAGE:3,800,800 \
  --request-rate-range=100:200:10 \
  --max-threads 32 \
  --shared-memory system \
  -f results.csv
```

**Parameters Explained:**
- `-m yolov11`: Model name
- `-u localhost:8000`: Triton server URL (HTTP)
- `--shape IMAGE:3,800,800`: Input tensor shape (CHW format)
- `--request-rate-range=100:200:10`: Test request rates from 100 to 200, step by 10
- `--max-threads 32`: Maximum concurrent threads
- `--shared-memory system`: Use system shared memory for faster data transfer
- `-f results.csv`: Save results to CSV file

**Additional Useful Options:**
```bash
# Test with concurrency levels instead of request rate
perf_analyzer -m yolov11 \
  -u localhost:8000 \
  --shape IMAGE:3,800,800 \
  --concurrency-range 1:16:1

# Test with gRPC protocol
perf_analyzer -m yolov11 \
  -u localhost:8001 \
  -i grpc \
  --shape IMAGE:3,800,800 \
  --concurrency-range 8

# Measure latency percentiles
perf_analyzer -m yolov11 \
  -u localhost:8000 \
  --shape IMAGE:3,800,800 \
  --percentile=95 \
  --percentile=99

# Extended measurement period
perf_analyzer -m yolov11 \
  -u localhost:8000 \
  --shape IMAGE:3,800,800 \
  --measurement-interval 10000 \
  --concurrency-range 8
```

**Analyzing Results:**

The output provides:
- **Throughput**: Inferences per second
- **Latency**: Min, max, mean, and percentiles (p50, p90, p95, p99)
- **Client/Server Overhead**: Time breakdown
- **Concurrency**: Optimal concurrent request count

**Example Output:**
```
Concurrency: 8
  Request throughput: 152.4 infer/sec
  Avg latency: 52.1 ms (overhead 0.3 ms + queue 1.2 ms + compute 50.6 ms)
  p50 latency: 51.2 ms
  p90 latency: 54.8 ms
  p95 latency: 56.3 ms
  p99 latency: 59.1 ms
```

**Best Practices:**
1. Start with low concurrency and gradually increase
2. Test both HTTP and gRPC protocols
3. Use `--shared-memory` for better performance
4. Run long measurement periods (10s+) for stable results
5. Test different batch sizes if model supports batching
6. Monitor server-side metrics with `nvidia-smi` or `rbln-stat`

**Comparing GPU vs NPU:**
```bash
# Test GPU backend
perf_analyzer -m yolov11 -u localhost:8000 --shape IMAGE:3,800,800 \
  --concurrency-range 1:16:1 -f gpu_results.csv

# Test NPU backend (different port/server)
perf_analyzer -m yolov11 -u localhost:8100 --shape IMAGE:3,800,800 \
  --concurrency-range 1:16:1 -f npu_results.csv

# Compare results
diff gpu_results.csv npu_results.csv
```

## Requirements

### Server
- NVIDIA Triton Inference Server 2.x+
- CUDA 11.0+ (for GPU backend)
- Rebellions SDK (for NPU backend)

### C++ Client
- CMake 3.15+
- C++17 compiler
- Triton client libraries
- OpenCV 4.x
- gRPC

### Python Client
- Python 3.8+
- tritonclient
- opencv-python
- numpy

## Documentation

- **C++ Client**: [client/cpp_client/README.md](client/cpp_client/README.md)
- **Python Client**: [client/python_client/README.md](client/python_client/README.md)
- **Architecture**: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
- **Performance**: [docs/PERFORMANCE.md](docs/PERFORMANCE.md)
- **Multi-Video**: [docs/README_MULTI_VIDEO.md](docs/README_MULTI_VIDEO.md)

## License

Copyright © 2025 Rebellions Inc.

---

**Triton + YOLO11 Integration for KEPCO PoC**

