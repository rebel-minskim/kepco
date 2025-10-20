# KEPCO AI Demo Suite

AI-powered demonstration applications showcasing Rebellions NPU performance and capabilities for KEPCO Proof of Concept.

![Project Status](https://img.shields.io/badge/Status-Active-green)
![Platform](https://img.shields.io/badge/Platform-Rebellions%20NPU-blue)

## Overview

This repository contains two demonstration applications designed for the KEPCO (Korea Electric Power Corporation) Proof of Concept, showcasing Rebellions NPU technology in real-world scenarios:

1. **Performance Comparison Dashboard** - Visual comparison of NPU vs GPU performance metrics
2. **Fire & Safety Detection System** - Real-time YOLO-based detection for industrial safety monitoring

## Project Structure

```
kepco/
├── BIXPO/                          # BIXPO demonstration applications
│   ├── web/                        # Performance comparison dashboard
│   │   ├── index.html              # Interactive dashboard
│   │   ├── index_standalone.html   # Offline version
│   │   ├── script.js               # Dashboard logic
│   │   ├── style.css               # Dashboard styling
│   │   ├── build_standalone.py     # Standalone builder
│   │   ├── npu_data.json           # NPU performance data
│   │   ├── gpu_data.json           # GPU performance data
│   │   ├── output_npu.mp4          # NPU processing video
│   │   ├── output_gpu.mp4          # GPU processing video
│   │   ├── image.png               # Dashboard preview
│   │   ├── README.md               # Dashboard documentation (English)
│   │   └── README_ko.md            # Dashboard documentation (Korean)
│   │
│   └── live-demo/                  # Fire & safety detection system
│       ├── app.py                  # Streamlit app (GPU)
│       ├── app_web.py              # Flask app (NPU)
│       ├── data.yaml               # Detection classes
│       ├── yolov11.pt              # PyTorch model
│       ├── yolov11.rbln            # RBLN compiled model
│       ├── requirements.txt        # GPU dependencies
│       ├── requirements_web.txt    # NPU dependencies
│       ├── templates/              # Web templates
│       ├── static/                 # Static assets
│       ├── start_app.sh            # Start script
│       ├── stop_app.sh             # Stop script
│       └── README.md               # Detection system documentation
│
└── README.md                       # This file
```

## Applications

### 1. Performance Comparison Dashboard

**Location**: `BIXPO/web/`

A real-time web dashboard that visually compares ATOM™-Max NPU and NVIDIA L40S GPU performance metrics for AI video processing workloads.

**Key Features**:
- Side-by-side video comparison of NPU vs GPU processing
- Real-time power consumption monitoring with animated gauges
- Performance efficiency tracking (FPS per Watt)
- Power usage visualization over time
- Standalone offline mode available

**Use Cases**:
- Performance demonstrations
- Technical presentations
- Efficiency analysis
- Stakeholder meetings

**Quick Start**:
```bash
cd BIXPO/web

# Option 1: Standalone (offline)
python3 build_standalone.py
open index_standalone.html

# Option 2: Web server
python3 -m http.server 8080
open http://localhost:8080/index.html
```

**Documentation**: See `BIXPO/web/README.md` for detailed usage instructions.

### 2. Fire & Safety Detection System

**Location**: `BIXPO/live-demo/`

Real-time object detection system for fire and safety monitoring using YOLOv11 on Rebellions NPU.

**Detected Objects**:
- Fire (flames)
- Smoke (white, black, opacity)
- People (person, head detection)
- Safety equipment (helmets)
- Safety hazards (ladders, falling persons)

**Two Applications Available**:

1. **Streamlit App** (`app.py`) - GPU Version
   - Interactive UI with controls and settings
   - Multiple input modes (webcam, video, images)
   - Real-time statistics and charts
   - Development and testing focused

2. **Flask Web App** (`app_web.py`) - NPU Version
   - High-performance MJPEG streaming
   - NPU accelerated with AsyncRuntime
   - Optimized for 4K displays
   - Production-ready (20-25 FPS)

**Use Cases**:
- Industrial fire safety monitoring
- Safety equipment compliance checking
- Hazard detection and alerts
- 24/7 surveillance systems

**Quick Start**:
```bash
cd BIXPO/live-demo

# GPU Version (Streamlit)
source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py

# NPU Version (Flask)
source venv/bin/activate
pip install -r requirements_web.txt
python3 app_web.py
```

**Documentation**: See `BIXPO/live-demo/docs/README.md` for detailed usage instructions.

## System Requirements

### Hardware
- **NPU**: Rebellions ATOM NPU (ATOM-Max recommended)
- **GPU**: NVIDIA GPU with CUDA support (for GPU comparison)
- **Camera**: USB webcam or built-in camera (for detection system)
- **Display**: Any resolution (optimized for 4K: 3840x2160)

### Software
- **OS**: Ubuntu 20.04+ / Linux
- **Python**: 3.8 or higher
- **CUDA**: Toolkit 11.0+ (for GPU applications)
- **Rebellions SDK**: Latest version (for NPU applications)
- **Browser**: Modern web browser (Chrome, Firefox, Safari, Edge)

## Installation

### 1. Clone Repository
```bash
git clone <repository-url>
cd kepco
```

### 2. Set Up Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

**For Performance Dashboard**:
```bash
cd BIXPO/web
# No dependencies needed for standalone version
# For development, standard Python 3 is sufficient
```

**For Detection System (GPU)**:
```bash
cd BIXPO/live-demo
pip install -r requirements.txt
```

**For Detection System (NPU)**:
```bash
cd BIXPO/live-demo
pip install -r requirements_web.txt
```

## Quick Reference

### Performance Dashboard
```bash
# Standalone (Offline)
cd BIXPO/web
python3 build_standalone.py
open index_standalone.html

# With Server (Development)
cd BIXPO/web
python3 -m http.server 8080
# Open http://localhost:8080/index.html
```

### Detection System
```bash
# GPU Version (Development)
cd BIXPO/live-demo
streamlit run app.py
# Open http://localhost:8501

# NPU Version (Production)
cd BIXPO/live-demo
python3 app_web.py
# Opens automatically at http://localhost:5000
```

## Performance Metrics

### NPU vs GPU Comparison (from Dashboard Data)

| Metric | ATOM™-Max NPU | NVIDIA L40S GPU | NPU Advantage |
|--------|---------------|-----------------|---------------|
| **FPS** | 36 | 24 | 1.5x faster |
| **Avg Power** | 50W | 325W | 6.5x lower |
| **Efficiency** | 0.72 FPS/W | 0.074 FPS/W | 9.7x better |
| **Performance/Watt** | Superior | Baseline | 6.3x multiplier |

### Detection System Performance

| Version | Accelerator | FPS | Latency | Best For |
|---------|-------------|-----|---------|----------|
| **Streamlit** | NVIDIA GPU | 15-30 | 50-100ms | Development |
| **Flask** | Rebellions NPU | 20-25 | 10-20ms | Production |

## Use Cases

### KEPCO-Specific Applications

1. **Power Infrastructure Monitoring**
   - Real-time fire detection in substations
   - Safety equipment compliance monitoring
   - Hazard detection (falling, unsafe conditions)

2. **Energy Efficiency Analysis**
   - NPU power consumption advantages
   - Cost-benefit analysis for AI deployments
   - Total Cost of Ownership (TCO) comparisons

3. **Edge AI Deployment**
   - Low-power edge devices
   - Distributed monitoring systems
   - 24/7 operation with minimal power consumption

4. **Technical Demonstrations**
   - Live performance comparisons
   - Real-time AI capabilities
   - Stakeholder presentations

## Documentation

Each application includes comprehensive documentation:

- **Performance Dashboard**: `BIXPO/web/README.md` (English) and `README_ko.md` (Korean)
- **Detection System**: `BIXPO/live-demo/docs/README.md` (English)
- **BIXPO Overview**: `BIXPO/README.md`

## Troubleshooting

### Common Issues

**Dashboard Not Loading**:
- Ensure you're using a web server (not file://)
- Check that video files exist
- Verify JSON data files are valid

**Camera Not Working (Detection System)**:
```bash
# List cameras
ls /dev/video*

# Test camera
python3 -c "import cv2; cap = cv2.VideoCapture(0); print(cap.isOpened())"
```

**Low FPS (Detection System)**:
- Check NPU utilization: `rbln-stat`
- Verify camera resolution (640x480 recommended)
- Monitor system resources

**Port Already in Use**:
```bash
# Find and kill process
lsof -ti:5000 | xargs kill -9  # Flask
lsof -ti:8501 | xargs kill -9  # Streamlit
lsof -ti:8080 | xargs kill -9  # HTTP server
```

## Development

### Adding New Features to Dashboard

1. Update data JSON files (`npu_data.json`, `gpu_data.json`)
2. Modify `script.js` for new visualizations
3. Update `style.css` for styling
4. Rebuild standalone: `python3 build_standalone.py`

### Modifying Detection Classes

1. Edit `BIXPO/live-demo/data.yaml`
2. Update class definitions
3. Retrain or reconfigure model
4. Test with both applications

### Custom Configurations

**Dashboard**: Edit files directly (HTML/CSS/JS)
**Detection System**: Modify `CONFIG` dictionary in application files

## Monitoring & Logging

### NPU Monitoring
```bash
# Real-time NPU stats
watch -n 0.1 rbln-stat

# Check NPU utilization
rbln-stat | grep "Util"
```

### Application Logs
```bash
# Detection system (Flask)
tail -f BIXPO/live-demo/logs/app_web.log

# Streamlit (console output)
# Logs displayed in terminal
```

### System Resources
```bash
# CPU and memory
htop

# GPU (if available)
nvidia-smi

# Disk usage
df -h
```

## Best Practices

### For Demonstrations

1. **Prepare Ahead**:
   - Test all applications before presentation
   - Verify camera and display connections
   - Check network connectivity

2. **Performance Dashboard**:
   - Use standalone version for offline demos
   - Prepare backup data files
   - Test on target display resolution

3. **Detection System**:
   - Use Flask (NPU) version for live demos
   - Set appropriate confidence thresholds
   - Use fullscreen mode (F11)

4. **System Health**:
   - Monitor NPU temperature and utilization
   - Check available disk space
   - Ensure stable power supply

## Performance Optimization

### Dashboard
- Use standalone version for best performance
- Compress video files for faster loading
- Optimize JSON data (remove unnecessary samples)

### Detection System
- Use 640x480 camera resolution
- Set JPEG quality to 80-90
- Enable NPU AsyncRuntime with parallel=2
- Process every frame for smooth detection

## Support & Resources

### Documentation
- Rebellions NPU: https://docs.rbln.ai/
- Ultralytics YOLO: https://docs.ultralytics.com/
- Streamlit: https://docs.streamlit.io/
- Flask: https://flask.palletsprojects.com/

### Debugging Checklist
- [ ] All dependencies installed
- [ ] Models downloaded and accessible
- [ ] Camera/video sources working
- [ ] Ports not blocked by firewall
- [ ] Sufficient disk space available
- [ ] NPU drivers and SDK up to date

## License

Copyright © 2025 Rebellions Inc.

## Contact

For technical support or questions about this demonstration suite, please contact the Rebellions technical team.

---

**Built with Rebellions NPU Technology**

**For KEPCO Proof of Concept - 2025**

