# Kepco Triton Inference Server

Production-ready YOLO11 object detection with NVIDIA Triton Inference Server, supporting both GPU and NPU (Rebellions RBLN) backends.

![Status](https://img.shields.io/badge/Status-Active-green) ![Platform](https://img.shields.io/badge/Platform-Triton%20Server-blue) ![Backend](https://img.shields.io/badge/Backend-GPU%20%7C%20NPU-orange)

## Overview

This project provides a complete Triton Inference Server integration for YOLO11 object detection, featuring:

- **GPU Backend**: PyTorch-based inference with CUDA optimization
- **NPU Backend**: Rebellions RBLN-optimized inference for high efficiency
- **Multi-video Processing**: Concurrent processing of multiple video streams
- **Performance Benchmarking**: Built-in performance analysis tools

> **Note**: C++ and Python clients (`client/cpp_client` and `client/python_client`) are only available in the `jpeg-byte` branch. Switch to the `jpeg-byte` branch to access these client implementations.

## Features

- **Dual Backend Support**: Switch between GPU (PyTorch) and NPU (RBLN) backends
- **High Performance**: Up to 89.8 FPS with C++ client, 30-35 FPS with Python client
- **Multiple Protocols**: Support for both gRPC and HTTP protocols
- **Concurrent Processing**: Multi-threaded pipeline for parallel video processing
- **Decoupled Mode**: Async streaming inference for improved throughput
- **Production Ready**: Comprehensive error handling, logging, and monitoring
- **Easy Integration**: Simple client APIs for quick deployment

## Tech Stack

### Server
- **NVIDIA Triton Inference Server** 2.x+
- **Python** 3.8+
- **PyTorch** 1.9+ (GPU backend)
- **Rebellions RBLN Runtime** (NPU backend)
  - **rebel-compiler SDK** >= 0.9.2.post1
  - Contact Rebellions for SDK access and installation

### C++ Client (jpeg-byte branch only)
- **CMake** 3.15+
- **C++17** compiler
- **Triton Client Libraries**
- **OpenCV** 4.x
- **gRPC** and **Protocol Buffers**

### Python Client (jpeg-byte branch only)
- **Python** 3.8+
- **tritonclient[grpc,http]** >= 2.30.0
- **opencv-python** >= 4.5.0
- **numpy** >= 1.21.0
- **PyYAML** >= 6.0

## Quick Start

### Prerequisites

```bash
# Install NVIDIA Triton Inference Server
# Follow instructions at: https://github.com/triton-inference-server/server

# For NPU backend, install Rebellions RBLN Runtime SDK
# Contact Rebellions for SDK access and installation instructions
# Required: rebel-compiler >= 0.8.3
```

### Installation

```bash
git clone [repo-url]
cd kepco/triton

# For C++ and Python clients, switch to jpeg-byte branch
git checkout jpeg-byte

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install Python dependencies
pip install -r requirements.txt

# For client dependencies (jpeg-byte branch only)
# pip install -r client/python_client/requirements.txt

# Build C++ client (jpeg-byte branch only, optional)
cd client/cpp_client
./build.sh
cd ../..
```

### Startup

**Option 1: Using startup scripts (recommended)**

```bash
# Start all services (foreground)
./start_all.sh

# Or start in background
./start_simple.sh
```

**Option 2: Manual startup**

```bash
# Start Triton Server with NPU backend
tritonserver --model-repository=./rbln_backend \
             --log-verbose=1 \
             --http-port=8000 \
             --grpc-port=8001 \
             --metrics-port=8002

# Or with GPU backend
tritonserver --model-repository=./gpu_backend \
             --log-verbose=1 \
             --http-port=8000 \
             --grpc-port=8001 \
             --metrics-port=8002
```

### Run Client

**Python Client:**

```bash
cd client/python_client

# Process video file
python main.py video ../media/test_video_HD.mp4 -o output.mp4

# Process image
python main.py image path/to/image.jpg -o output.jpg

# Health check
python main.py health_check --url localhost:8001
```

**C++ Client:**

```bash
cd client/cpp_client

# Build (if not already built)
./build.sh

# Process single video
./bin/triton_client video input.mp4 output.mp4

# Process multiple videos concurrently
./bin/triton_client parallel video1.mp4 output1.mp4 8
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Triton Inference Server                  │
│  ┌──────────────────┐         ┌──────────────────┐         │
│  │   GPU Backend    │         │   NPU Backend    │         │
│  │  (PyTorch/YOLO)  │         │  (RBLN/YOLO)     │         │
│  └──────────────────┘         └──────────────────┘         │
│           │                            │                    │
│           └────────────┬───────────────┘                    │
│                        │                                    │
│              ┌─────────▼─────────┐                         │
│              │  Model Repository │                         │
│              │   (yolov11)      │                         │
│              └─────────┬─────────┘                         │
└─────────────────────────┼───────────────────────────────────┘
                          │
         ┌────────────────┴────────────────┐
         │                                  │
    ┌────▼────┐                      ┌────▼────┐
    │  gRPC   │                      │  HTTP   │
    │  :8001  │                      │  :8000  │
    └────┬────┘                      └────┬────┘
         │                                  │
    ┌────┴──────────────────────────────────┴────┐
    │            Client Applications              │
    │  ┌──────────────┐    ┌──────────────┐      │
    │  │ C++ Client   │    │Python Client │      │
    │  │ (89.8 FPS)   │    │ (30-35 FPS)  │      │
    │  └──────────────┘    └──────────────┘      │
    └─────────────────────────────────────────────┘
```

### Directory Structure

```
triton/
├── client/
│   ├── cpp_client/              # C++ gRPC/HTTP client (jpeg-byte branch only)
│   │   ├── main.cpp            # Client entry point
│   │   ├── triton_client.cpp   # Triton client implementation
│   │   ├── grpc_client.cpp     # gRPC wrapper
│   │   ├── yolo_preprocess.cpp # YOLO preprocessing
│   │   ├── yolo_postprocess.cpp# YOLO postprocessing
│   │   ├── build.sh            # Build script
│   │   └── README.md           # C++ client docs
│   │
│   ├── python_client/           # Python client (jpeg-byte branch only)
│   │   ├── main.py             # Basic client
│   │   ├── main_concurrent.py  # Concurrent inference
│   │   ├── config.py           # Configuration
│   │   ├── utils/              # Utility modules
│   │   └── README.md           # Python client docs
│   │
│   └── client.py               # Legacy client
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
│           ├── yolov11.rbln    # RBLN compiled model
│           └── coco128.yaml   # Class names
│
├── start_all.sh                 # Start all services
├── start_simple.sh              # Start in background
├── stop.sh                      # Stop all services
├── .env.example                 # Environment variables template
├── .gitignore
└── README.md                    # This file
```

## Usage

> **Note**: The following client examples are only available in the `jpeg-byte` branch. Switch to that branch first: `git checkout jpeg-byte`

### Basic Video Processing

```bash
# Switch to jpeg-byte branch for client access
git checkout jpeg-byte

# Python client - single video
cd client/python_client
python main.py video ../media/test_video_HD.mp4 -o output.mp4

# Python client - async mode (faster)
python client.py infer_video \
    --video_path ../media/test_video_HD.mp4 \
    --output_path output.mp4 \
    --async_mode true \
    --max_async_requests 16

# C++ client - single video
cd client/cpp_client
./bin/triton_client video input.mp4 output.mp4

# C++ client - parallel processing
./bin/triton_client parallel video1.mp4 output1.mp4 8
```

### Image Processing

```bash
# Python client (jpeg-byte branch only)
python main.py image path/to/image.jpg -o output.jpg

# Or using client.py
python client.py infer_image --image_path image.jpg
```

### Health Check

```bash
# Check server status
python client.py health_check --url localhost:8001

# Or using Python client
python main.py health_check --url localhost:8001
```

### Performance Benchmarking

#### Using perf_analyzer

`perf_analyzer` is Triton's official performance testing tool. It provides detailed latency and throughput metrics.

**gRPC Streaming with Real Data:**
```bash
# gRPC endpoint with streaming, async mode, and custom input data
perf_analyzer -m yolov11 \
  -u 10.244.48.166:8001 \
  -i grpc \
  --streaming \
  --async \
  --input-data ../perf_data/perf_input.json \
  --measurement-interval 10000 \
  --concurrency-range 64
```

**Key Options:**
- `-m`: Model name
- `-u`: Server URL (format: `host:port`)
- `-i`: Protocol (`grpc` or `http`)
- `--streaming`: Enable streaming mode (for decoupled models)
- `--async`: Use async mode
- `--input-data`: Path to JSON file containing input data
- `--measurement-interval`: Measurement duration in milliseconds
- `--concurrency-range`: Concurrency range (format: `start:end:step` or single value)
- `-f`: Output CSV file path

**Input Data Format:**
The `perf_input.json` file should contain base64-encoded JPEG images:
```json
{
  "data": [{
    "INPUT__0": {
      "b64": "/9j/4AAQSkZJRg...",
      "shape": [74852]
    }
  }]
}
```

#### Using client.py

The `client.py` script provides a Python interface for video and image inference.

**Video Inference:**
```bash
# Synchronous mode (default, for non-decoupled models only)
python client.py infer_video \
    --video_path video.mp4 \
    --output_path output.mp4 \
    --save_output

# Asynchronous/streaming mode (faster, recommended, required for decoupled models)
python client.py infer_video \
    --video_path video.mp4 \
    --output_path output.mp4 \
    --async_mode true \
    --max_async_requests 16 \
    --save_output
```

> **Note**: If your model uses decoupled transaction policy (like the NPU backend), 
> `client.py` will automatically enable async mode. Synchronous mode is not supported 
> for decoupled models.

# Webcam input
python client.py infer_video \
    --video_path 0 \
    --async_mode true

# Custom server URL
python client.py infer_video \
    --video_path video.mp4 \
    --url 10.244.48.166:8001 \
    --async_mode true
```

**Image Inference:**
```bash
# Single image (non-decoupled models only)
python client.py infer_image \
    --image_path image.jpg

# Custom server URL
python client.py infer_image \
    --image_path image.jpg \
    --url localhost:8001
```

> **Note**: Image inference is not supported for decoupled models. 
> Use video inference with `--async_mode true` instead.

**Health Check:**
```bash
# Check server status
python client.py health_check --url localhost:8001
```

**Key Options:**
- `--video_path`: Video file path or `0` for webcam
- `--output_path`: Output video file path (optional)
- `--url`: Triton server URL (default: `localhost:8001`)
- `--async_mode`: Enable async/streaming mode (default: `false`)
- `--max_async_requests`: Maximum concurrent async requests (default: `16`)
- `--save_output`: Save output video (default: `false`)
- `--jpeg_quality`: JPEG compression quality 1-100 (default: `80`)
- `--verbose`: Enable verbose logging

**Performance Tips:**
- Use `--async_mode true` for better throughput (2-3x improvement)
- Adjust `--max_async_requests` based on server capacity (typically 8-32)
- Lower `--jpeg_quality` (e.g., 70) for faster encoding at slight quality loss
- Use `--save_output false` when only testing (saves disk I/O)

#### C++ Client Performance Test

```bash
# C++ client performance test
cd client/cpp_client
./bin/triton_client dummy --requests 900 --rate 90
```

## Configuration

### Python Dependencies

The project requires the following Python packages (see `requirements.txt`):

**Core Dependencies:**
- `numpy` >= 1.21.0
- `opencv-python` >= 4.5.0
- `torch` >= 1.9.0
- `ultralytics` >= 8.0.0
- `pyyaml` >= 6.0

**Triton Backend:**
- `triton-python-backend` >= 2.0.0

**Rebellions RBLN Runtime (NPU Backend):**
- `rebel-compiler` >= 0.8.3
  - Note: This SDK is provided by Rebellions. Contact Rebellions for access and installation instructions.

**Client Dependencies (jpeg-byte branch only):**
- `tritonclient[grpc,http]` >= 2.30.0

Install all dependencies:
```bash
pip install -r requirements.txt
```

### Environment Variables

Copy `.env.example` to `.env` and configure:

```bash
cp .env.example .env
```

Key environment variables:

- `TRITON_SERVER_URL`: Triton server URL (default: `localhost:8001`)
- `TRITON_HTTP_PORT`: HTTP port (default: `8000`)
- `TRITON_GRPC_PORT`: gRPC port (default: `8001`)
- `TRITON_METRICS_PORT`: Metrics port (default: `8002`)
- `MODEL_REPOSITORY`: Model repository path (default: `./rbln_backend`)
- `MODEL_NAME`: Model name (default: `yolov11`)
- `BACKEND_TYPE`: Backend type - `gpu` or `rbln` (default: `rbln`)
- `LOG_VERBOSE`: Enable verbose logging (default: `1`)
- `CONFIDENCE_THRESHOLD`: Detection confidence threshold (default: `0.20`)
- `IOU_THRESHOLD`: NMS IoU threshold (default: `0.50`)

See `.env.example` for all available options.

### Model Configuration

**GPU Backend** (`gpu_backend/yolov11/config.pbtxt`):
- Input: `[3, 800, 800]` FP32 tensor
- Output: Multiple FP32 tensors
- Instance group: 8 GPU instances

**NPU Backend** (`rbln_backend/yolov11/config.pbtxt`):
- Input: Variable-length UINT8 JPEG bytes
- Output: JSON string with detections
- Instance group: 2 model instances
- Decoupled mode: Enabled

## Performance

> **Note**: Client performance metrics below are for the `jpeg-byte` branch only.

### Single Video Processing

| Client | GPU Backend | NPU Backend |
|--------|------------|-------------|
| **C++ Client** (jpeg-byte) | ~30 FPS | ~35-40 FPS |
| **Python Client** (jpeg-byte) | ~25-28 FPS | ~30-35 FPS |

### Multi-Video Concurrent Processing

- **2 Videos**: ~1.8x throughput improvement
- **4 Videos**: ~3.2x throughput improvement
- **8 Videos**: ~4.5x throughput improvement

### C++ Client Performance

- **Peak Performance**: 89.8 FPS
- **Average Inference Time**: 28.6ms
- **End-to-End Latency**: 35.2ms
- **Throughput**: 90 req/s

## Documentation

- **C++ Client** (jpeg-byte branch): [client/cpp_client/README.md](client/cpp_client/README.md)
- **Python Client** (jpeg-byte branch): [client/python_client/README.md](client/python_client/README.md)

## Troubleshooting

### Server Not Starting

```bash
# Check if ports are available
netstat -tuln | grep -E '8000|8001|8002'

# Check Triton server logs
tail -f /var/log/triton/server.log

# Verify model repository
tritonserver --model-repository=./rbln_backend --exit-on-error
```

### Client Connection Issues

```bash
# Test server connectivity
python client.py health_check --url localhost:8001

# Check firewall settings
sudo ufw status

# Verify gRPC endpoint
grpc_health_probe -addr=localhost:8001
```

### Decoupled Model Errors

If you see an error like:
```
[StatusCode.UNIMPLEMENTED] ModelInfer RPC doesn't support models with decoupled transaction policy
```

This means you're trying to use synchronous mode with a decoupled model. Solutions:

1. **Use async mode** (recommended):
   ```bash
   python client.py infer_video --video_path video.mp4 --async_mode true
   ```

2. **client.py will auto-detect** and enable async mode automatically for decoupled models.

3. **Note**: Image inference (`infer_image`) is not supported for decoupled models. Use video inference instead.

### Performance Issues

```bash
# Monitor GPU usage
nvidia-smi -l 1

# Monitor NPU usage (if available)
rbln-stat

# Check system resources
htop
```

## License

Copyright © 2025 Rebellions Inc.

---

**Triton + YOLO11 Integration for KEPCO PoC**
