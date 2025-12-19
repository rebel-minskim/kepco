# KEPCO AI Demo Suite

AI-powered demonstration applications showcasing Rebellions NPU performance for KEPCO Proof of Concept.

![Status](https://img.shields.io/badge/Status-Active-green) ![Platform](https://img.shields.io/badge/Platform-Rebellions%20NPU-blue)

## What's Included

This demo suite contains three main components:

### 1. Performance Dashboard (`BIXPO/web/`)
Visual comparison of NPU vs GPU performance with real-time metrics and video processing demos.

**Key Metrics:**
- NPU: 36 FPS @ 50W → 0.72 FPS/Watt
- GPU: 24 FPS @ 325W → 0.074 FPS/Watt
- **NPU Advantage: 9.7x better efficiency**

### 2. Fire & Safety Detection (`BIXPO/live-demo/`)
Real-time YOLO-based detection system for industrial safety monitoring with 9 detection classes (fire, smoke, people, helmets, etc.).

**Performance:**
- 20-25 FPS with NPU acceleration
- <20ms latency
- Optimized for 4K displays

### 3. Triton Inference Server (`triton/`)
Production-ready YOLO11 inference with NVIDIA Triton Server, supporting both GPU and NPU backends.

**Features:**
- Python client with async streaming support
- gRPC and HTTP protocols
- Multi-video concurrent processing
- Performance benchmarking tools

## Quick Start

### Performance Dashboard
```bash
cd BIXPO/web

# Standalone (offline mode)
python3 build_standalone.py
open index_standalone.html

# Or with web server
python3 -m http.server 8080
# Open http://localhost:8080
```

### Fire & Safety Detection
```bash
cd BIXPO/live-demo

# Setup
python3 -m venv venv
source venv/bin/activate
pip install -r config/requirements_web.txt

# Run NPU version
python3 app_web.py
# Opens automatically at http://localhost:5000
```

### Triton Inference Server
```bash
cd triton

# Start server using startup script (recommended)
./start_all.sh
# Or start in background: ./start_simple.sh

# Run client (in another terminal)
cd client
python client.py infer_video \
    --video_path media/test_video_HD.mp4 \
    --output_path output.mp4 \
    --async_mode true \
    --save_output
```

## System Requirements

- **NPU**: Rebellions ATOM (ATOM-Max recommended)
- **OS**: Ubuntu 20.04+ / Linux
- **Python**: 3.8+
- **Camera**: USB webcam (for detection system)
- **Triton**: NVIDIA Triton Inference Server 2.x+ (for triton deployment)
- **Rebellions SDK**: rebel-compiler >= 0.9.2.post1
  - Required for NPU backend in Triton and live-demo applications
  - Contact Rebellions for SDK access and installation instructions

## Project Structure

```
kepco/
├── .gitignore                  # Root gitignore
├── README.md                   # This file
│
├── BIXPO/                      # BIXPO exhibition demos
│   ├── README.md              # BIXPO overview
│   │
│   ├── web/                   # Performance dashboard
│   │   ├── assets/           # CSS, JS, images, videos
│   │   ├── data/             # Performance data (JSON)
│   │   ├── index.html        # Main dashboard
│   │   ├── build_standalone.py
│   │   ├── README.md         # English docs
│   │   └── README_ko.md      # Korean docs
│   │
│   └── live-demo/            # Fire & safety detection
│       ├── app.py            # Streamlit GPU app
│       ├── app_web.py        # Flask NPU app
│       ├── start_app.sh
│       ├── stop_app.sh
│       ├── config/           # Configuration files
│       ├── docs/             # Documentation
│       ├── models/           # AI models (.pt, .rbln)
│       ├── static/           # Web assets
│       ├── templates/        # HTML templates
│       └── utils/            # Diagnostic tools
│
└── triton/                   # Triton Inference Server
    ├── README.md             # Triton docs
    ├── requirements.txt      # Python dependencies
    ├── start_all.sh          # Start server (foreground)
    ├── start_simple.sh       # Start server (background)
    ├── stop.sh               # Stop server
    ├── .env.example          # Environment variables template
    ├── client/
    │   ├── client.py         # Python gRPC client
    │   └── media/            # Test media files
    ├── gpu_backend/          # GPU (PyTorch) backend
    │   └── yolov11/         # YOLOv11 model
    ├── rbln_backend/         # NPU (RBLN) backend
    │   └── yolov11/         # YOLOv11 model
    └── perf_data/            # Performance test data
```

## Documentation

Detailed documentation for each component:

- **Performance Dashboard**: [BIXPO/web/README.md](BIXPO/web/README.md) (English), [README_ko.md](BIXPO/web/README_ko.md) (Korean)
- **Detection System**: [BIXPO/live-demo/docs/README.md](BIXPO/live-demo/docs/README.md)
- **Triton Server**: [triton/README.md](triton/README.md)
- **Exhibition Setup**: [BIXPO/README.md](BIXPO/README.md)

## Performance Comparison

| Application | Accelerator | FPS | Power | Efficiency | Latency |
|-------------|-------------|-----|-------|------------|---------|
| **Dashboard (Web)** | NPU | 36 | 50W | 0.72 FPS/W | - |
| **Dashboard (Web)** | GPU | 24 | 325W | 0.074 FPS/W | - |
| **Live Demo** | NPU | 20-25 | ~50W | ~0.45 FPS/W | <20ms |
| **Live Demo** | GPU | 15-30 | ~250W | ~0.08 FPS/W | 50-100ms |
| **Triton (Async)** | NPU | 30-35 | ~50W | ~0.65 FPS/W | 10-15ms |
| **Triton (Sync)** | GPU | 25-28 | ~250W | ~0.11 FPS/W | 50-100ms |

## Use Cases

### KEPCO-Specific Applications

1. **Power Infrastructure Monitoring**
   - Real-time fire detection in substations
   - Safety equipment compliance
   - Hazard detection

2. **Energy Efficiency Analysis**
   - NPU power consumption advantages
   - Cost-benefit analysis for AI deployments
   - Total Cost of Ownership (TCO) comparisons

3. **Edge AI Deployment**
   - Low-power edge devices
   - Distributed monitoring systems
   - 24/7 operation with minimal power

4. **Technical Demonstrations**
   - Live performance comparisons
   - Real-time AI capabilities
   - Stakeholder presentations

## Troubleshooting

**Dashboard not loading?**
- Use a web server (not file://)
- Verify video/data files exist

**Camera not working?**
```bash
ls /dev/video*  # Check camera device
```

**Port already in use?**
```bash
lsof -ti:5000 | xargs kill -9  # Kill Flask
lsof -ti:8501 | xargs kill -9  # Kill Streamlit
lsof -ti:8000 | xargs kill -9  # Kill Triton HTTP
```

**Triton server issues?**
```bash
cd triton

# Check server status
curl localhost:8000/v2/health/ready

# Stop server
./stop.sh

# Start server (foreground to see logs)
./start_all.sh

# Or start in background
./start_simple.sh
```

**More help?** See detailed docs in each application folder.

## Development

### Adding New Models

**For Live Demo:**
1. Place model in `BIXPO/live-demo/models/`
2. Update `config/data.yaml` with classes
3. Test with both `app.py` and `app_web.py`

**For Triton:**
1. Add model to `triton/gpu_backend/yolov11/` or `triton/rbln_backend/yolov11/`
2. Configure `config.pbtxt` in the model directory
3. Restart Triton server (use `./stop.sh` then `./start_all.sh`)
4. Test with `perf_analyzer` or `client.py`

### Running Tests

```bash
# Test live-demo utilities
cd BIXPO/live-demo
python3 utils/test_npu_inference.py
python3 utils/diagnose_camera.py

# Test Triton performance
cd triton
# Using perf_analyzer (for GPU backend with shape input)
perf_analyzer -m yolov11 -u localhost:8001 -i grpc --streaming --async \
    --input-data perf_data/perf_input.json

# Or using client.py
cd client
python client.py infer_video \
    --video_path media/test_video_HD.mp4 \
    --async_mode true \
    --verbose
```

## License

Copyright © 2025 Rebellions Inc.

---

**Built with Rebellions NPU Technology**

**For KEPCO Proof of Concept - 2025**
