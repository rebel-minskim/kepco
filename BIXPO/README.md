# BIXPO Demonstration Suite

AI demonstration applications for KEPCO showcasing Rebellions NPU capabilities at BIXPO exhibition.

![Status](https://img.shields.io/badge/Status-Demo%20Ready-green)

## Overview

This directory contains two demonstration applications designed for the BIXPO (Business & Industry Expo) exhibition, showcasing Rebellions NPU technology for KEPCO (Korea Electric Power Corporation):

1. **Performance Comparison Dashboard** (`web/`) - Visual NPU vs GPU metrics
2. **Fire & Safety Detection System** (`live-demo/`) - Real-time AI-powered monitoring

## Directory Structure

```
BIXPO/
├── web/                        # Performance comparison dashboard
│   ├── index.html              # Interactive web dashboard
│   ├── index_standalone.html   # Offline standalone version
│   ├── script.js               # Dashboard animations and logic
│   ├── style.css               # Dashboard styling
│   ├── build_standalone.py     # Build tool for standalone version
│   ├── npu_data.json           # NPU performance metrics
│   ├── gpu_data.json           # GPU performance metrics
│   ├── output_npu.mp4          # NPU processing video demo
│   ├── output_gpu.mp4          # GPU processing video demo
│   ├── logo_rebellions.svg     # Rebellions logo
│   ├── logo_nvidia.svg         # NVIDIA logo
│   ├── image.png               # Dashboard screenshot
│   ├── README.md               # English documentation
│   └── README_ko.md            # Korean documentation
│
└── live-demo/                  # Fire & safety detection system
    ├── app.py                  # Streamlit application (GPU)
    ├── app_web.py              # Flask application (NPU)
    ├── data.yaml               # YOLO detection classes
    ├── yolov11.pt              # PyTorch model (GPU)
    ├── yolov11.rbln            # Compiled RBLN model (NPU)
    ├── requirements.txt        # Dependencies for GPU version
    ├── requirements_web.txt    # Dependencies for NPU version
    ├── templates/              # Flask web templates
    │   └── index.html          # Detection interface
    ├── static/                 # Static assets
    │   └── logo_rebellions.png # Company logo
    ├── start_app.sh            # Application start script
    ├── stop_app.sh             # Application stop script
    └── README.md               # System documentation
```

## Applications

### 1. Performance Comparison Dashboard (`web/`)

**Purpose**: Visual demonstration of NPU efficiency advantages over GPU for AI workloads

**Features**:
- Dual video comparison (NPU vs GPU processing results)
- Real-time power consumption gauges
- Performance efficiency graphs (FPS per Watt)
- Power usage tracking over time
- Automatic efficiency multiplier calculations

**Key Metrics Displayed**:
- Processing speed (FPS/Images per second)
- Power consumption (Watts)
- Performance efficiency (FPS/Watt)
- Energy efficiency comparison

**Quick Start**:
```bash
cd web

# Offline mode (no server needed)
python3 build_standalone.py
open index_standalone.html

# Development mode (with server)
python3 -m http.server 8080
open http://localhost:8080/index.html
```

**When to Use**:
- **Standalone**: Offline presentations, exhibitions, quick demos
- **Server Mode**: Development, frequent data updates, testing

**Documentation**: See `web/README.md` (English) or `web/README_ko.md` (Korean)

### 2. Fire & Safety Detection System (`live-demo/`)

**Purpose**: Real-time object detection for industrial safety monitoring using YOLOv11

**Detection Capabilities**:
- Fire and flame detection
- Smoke detection (white, black, opacity)
- Person and head detection
- Safety equipment (helmet) detection
- Hazard detection (ladder, falling person)

**Two Versions Available**:

#### A. Streamlit Application (`app.py`) - GPU Version

**Best For**: Development, testing, demonstrations with UI controls

**Features**:
- Interactive sidebar with all controls
- Multiple input modes (webcam, video, image upload)
- Real-time FPS and detection statistics
- Adjustable confidence and IoU thresholds
- Visual feedback with annotated frames

**Performance**:
- FPS: 15-30
- Accelerator: NVIDIA GPU (CUDA)
- Latency: 50-100ms
- Resolution: Camera default

**Usage**:
```bash
cd live-demo
source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
# Opens at http://localhost:8501
```

#### B. Flask Application (`app_web.py`) - NPU Version

**Best For**: Production deployment, live monitoring, exhibition displays

**Features**:
- High-performance MJPEG streaming
- NPU acceleration with AsyncRuntime
- Optimized for 4K displays
- Minimal latency
- Auto-opens browser
- 16:9 aspect ratio support

**Performance**:
- FPS: 20-25
- Accelerator: Rebellions NPU (RBLN)
- Latency: 10-20ms
- Resolution: 854x480 (16:9)
- NPU Utilization: 40-60%

**Usage**:
```bash
cd live-demo
source venv/bin/activate
pip install -r requirements_web.txt
python3 app_web.py
# Opens automatically at http://localhost:5000
```

**Documentation**: See `live-demo/docs/README.md`

## Exhibition Setup Guide

### For BIXPO Display Stations

**Hardware Setup**:
1. Connect 4K display (3840x2160 recommended)
2. Connect USB camera (for live-demo)
3. Ensure Rebellions NPU is properly installed
4. Verify stable power supply
5. Test network connectivity (if needed)

**Software Preparation**:

**Station 1: Performance Dashboard**
```bash
cd web
python3 build_standalone.py
# Copy index_standalone.html to display station
# Double-click to open in fullscreen (F11)
```

**Station 2: Live Detection Demo**
```bash
cd live-demo
source venv/bin/activate
python3 app_web.py
# Browser opens automatically
# Press F11 for fullscreen
```

**Pre-Demo Checklist**:
- [ ] Both applications tested and working
- [ ] Camera connected and functional (live-demo)
- [ ] Display resolution verified (4K optimal)
- [ ] Browser set to fullscreen mode
- [ ] NPU drivers and SDK up to date
- [ ] Backup data files prepared (dashboard)
- [ ] Emergency restart scripts ready
- [ ] Contact information for tech support available

## Performance Comparison

### Dashboard Metrics (NPU vs GPU)

| Metric | ATOM™-Max NPU | NVIDIA L40S GPU | NPU Advantage |
|--------|---------------|-----------------|---------------|
| Processing Speed | 36 imgs/s | 24 imgs/s | 1.5x faster |
| Average Power | 50W | 325W | 6.5x lower |
| Performance Efficiency | 0.72 FPS/W | 0.074 FPS/W | 9.7x better |
| Total Efficiency | - | - | 6.3x multiplier |

### Detection System Performance

| Metric | Streamlit (GPU) | Flask (NPU) | Winner |
|--------|----------------|-------------|---------|
| FPS | 15-30 | 20-25 | GPU (max) |
| Inference Time | 10-20ms | 30-35ms | GPU |
| Display Latency | 50-100ms | 10-20ms | NPU |
| Power Efficiency | Lower | Higher | NPU |
| Production Ready | No | Yes | NPU |
| 4K Optimized | No | Yes | NPU |

## Use Cases for KEPCO

### Power Infrastructure Applications

1. **Substation Safety Monitoring**
   - Real-time fire and smoke detection
   - 24/7 monitoring with low power consumption
   - Immediate alert capabilities

2. **Worker Safety Compliance**
   - Helmet and safety equipment detection
   - Hazardous situation detection (falling, unsafe conditions)
   - Automated compliance reporting

3. **Energy Efficiency Analysis**
   - NPU power consumption advantages
   - Cost-benefit analysis for AI deployment
   - Total Cost of Ownership (TCO) comparisons

4. **Edge AI Deployment**
   - Distributed monitoring systems
   - Low-power edge devices
   - Scalable infrastructure

## Technical Specifications

### Performance Dashboard

**Requirements**:
- Modern web browser (Chrome, Firefox, Safari, Edge)
- No server needed (standalone mode)
- H.264 video codec support
- JavaScript enabled

**Data Format**:
- JSON files with power and FPS metrics
- MP4 videos (H.264 encoded)
- SVG logos for branding

**Customization**:
- Edit JSON files for new data
- Rebuild standalone version
- Modify colors in CSS
- Adjust animation speeds in JavaScript

### Detection System

**Requirements**:
- Python 3.8+
- Rebellions NPU (for NPU version)
- NVIDIA GPU with CUDA (for GPU version)
- USB camera or video input
- Ubuntu 20.04+ / Linux

**Model**:
- YOLOv11 architecture
- 9 detection classes
- Compiled for RBLN (NPU version)
- PyTorch format (GPU version)

**Classes Detected**:
```yaml
0: person         # Person detection
1: fire           # Fire/flame detection
2: Smoke_White    # White smoke
3: Opacity_Smoke  # Dense smoke
4: Smoke_Black    # Black smoke
5: helmet         # Safety helmet
6: ladder         # Ladder detection
7: head           # Head detection
8: falling        # Falling person
```

## Troubleshooting

### Dashboard Issues

**Videos Not Playing**:
- Check video codec is H.264
- Verify file paths are correct
- Use web server instead of opening file directly
- Check browser console for errors

**Data Not Loading**:
- Validate JSON files (use jsonlint.com)
- Check file permissions
- Verify file names match code
- Rebuild standalone version

**Charts Not Animating**:
- Click play on videos to start
- Check Chart.js loaded properly
- Verify data has multiple samples
- Check browser console for errors

### Detection System Issues

**Camera Not Found**:
```bash
# List cameras
ls /dev/video*

# Test camera
python3 diagnose_camera.py
```

**Low FPS**:
```bash
# Monitor NPU
watch -n 0.1 rbln-stat

# Check bottlenecks
./monitor_bottleneck.sh
```

**NPU Not Utilized**:
- Verify model compiled for current SDK
- Check rbln-stat shows activity
- Recompile model if needed

**Port Already in Use**:
```bash
# Flask
lsof -ti:5000 | xargs kill -9

# Streamlit
lsof -ti:8501 | xargs kill -9
```

## Maintenance

### Regular Checks

**Before Each Demo**:
1. Test both applications
2. Verify camera functionality
3. Check display connections
4. Confirm NPU is operational
5. Test network if needed

**Daily**:
1. Clear browser cache
2. Restart applications
3. Check system logs
4. Monitor disk space
5. Verify NPU temperature

**Weekly**:
1. Update data if needed
2. Review detection accuracy
3. Optimize thresholds
4. Clean up log files
5. Backup configurations

### Updates

**Dashboard Data Update**:
```bash
cd web
# Edit npu_data.json and gpu_data.json
python3 build_standalone.py
```

**Detection Model Update**:
```bash
cd live-demo
# Replace yolov11.rbln (NPU) or yolov11.pt (GPU)
# Update data.yaml if classes changed
# Test with both applications
```

## Best Practices

### Exhibition Environment

1. **Display Setup**:
   - Use 4K displays for best impact
   - Set to fullscreen mode (F11)
   - Adjust brightness for environment
   - Position for optimal viewing angles

2. **Camera Setup** (Detection System):
   - Mount at appropriate height
   - Ensure good lighting
   - Test field of view
   - Adjust focus if needed

3. **System Stability**:
   - Use UPS for power backup
   - Monitor NPU temperature
   - Keep system resources available
   - Have restart procedures ready

4. **Visitor Interaction**:
   - Prepare demo script
   - Have technical specs ready
   - Show performance metrics clearly
   - Demonstrate key features

## Quick Reference

### Starting Applications

```bash
# Performance Dashboard (Standalone)
cd web
open index_standalone.html

# Performance Dashboard (Server)
cd web
python3 -m http.server 8080

# Detection System (NPU - Production)
cd live-demo
python3 app_web.py

# Detection System (GPU - Development)
cd live-demo
streamlit run app.py
```

### Stopping Applications

```bash
# Dashboard: Close browser
# Detection Flask: ./stop_app.sh or Ctrl+C
# Detection Streamlit: Ctrl+C in terminal
```

### Monitoring Commands

```bash
# NPU utilization
rbln-stat

# Application logs
tail -f live-demo/logs/app_web.log

# System resources
htop

# Camera devices
ls /dev/video*
```

## Documentation

Detailed documentation is available for each application:

- **Performance Dashboard**: `web/README.md` (English), `web/README_ko.md` (Korean)
- **Detection System**: `live-demo/docs/README.md` (English)
- **Main Project**: `../README.md`

## Support

### Debug Information to Collect

When reporting issues:
1. Application logs
2. NPU utilization (rbln-stat output)
3. System resource usage (htop)
4. Camera status (ls /dev/video*)
5. Browser console errors (F12)
6. Screenshots of the issue

### Contact

For technical support during BIXPO exhibition, contact the Rebellions technical team.

## License

Copyright © 2025 Rebellions Inc.

---

**BIXPO Demonstration Suite - Powered by Rebellions NPU**

**For KEPCO Proof of Concept - 2025**

