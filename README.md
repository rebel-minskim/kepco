# YOLO11 Fire & Safety Detection System

Real-time object detection system for fire and safety monitoring using YOLOv11 on Rebellions NPU.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [System Requirements](#system-requirements)
- [Installation](#installation)
- [Applications](#applications)
  - [Streamlit App (app.py)](#streamlit-app-apppy)
  - [Flask Web App (app_web.py)](#flask-web-app-app_webpy)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Performance](#performance)
- [Troubleshooting](#troubleshooting)

---

## 🎯 Overview

This system provides real-time detection of:
- **Fire** - Flame detection
- **Smoke** - White, black, and opacity smoke
- **People** - Person and head detection
- **Safety Equipment** - Helmet detection
- **Safety Hazards** - Ladder and falling detection

**Two Applications Available:**
1. **Streamlit App** (`app.py`) - Interactive UI with controls and settings
2. **Flask Web App** (`app_web.py`) - High-performance real-time video streaming

---

## ✨ Features

### Common Features (Both Apps)
- ✅ Real-time object detection using YOLOv11
- ✅ Rebellions NPU acceleration
- ✅ Multiple input sources (webcam, video, images)
- ✅ Custom class detection (9 classes from data.yaml)
- ✅ Adjustable confidence and IoU thresholds
- ✅ FPS monitoring

### Streamlit App (`app.py`) - GPU Version
- 🎛️ Rich interactive UI with sidebar controls
- 📊 Real-time statistics and charts
- 🎥 Multiple input modes (live webcam, snapshot, video upload, image upload)
- 📸 Frame-by-frame control
- 💾 Easy to customize and extend
- 🖥️ **GPU accelerated** (CUDA)

### Flask Web App (`app_web.py`) - NPU Version
- ⚡ High-performance MJPEG streaming
- 🚀 NPU accelerated with AsyncRuntime
- 🎯 Optimized for 4K displays
- 📹 16:9 aspect ratio support
- 🌐 Auto-opens browser
- 🔥 **2-3x faster** than Streamlit (20+ FPS vs 8-10 FPS)

---

## 💻 System Requirements

### Hardware
- **GPU Version**: NVIDIA GPU with CUDA support
- **NPU Version**: Rebellions NPU (ATOM or similar)
- **Camera**: USB webcam or built-in camera
- **Display**: Any resolution (optimized for 4K: 3840x2160)

### Software
- Python 3.8+
- Ubuntu 20.04+ / Linux
- CUDA Toolkit (for GPU version)
- Rebellions SDK (for NPU version)

---

## 📦 Installation

### 1. Clone Repository
```bash
cd /home/rebellions/rebellions/kepco
```

### 2. Create Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

**For GPU Version (Streamlit):**
```bash
pip install -r requirements.txt
```

**For NPU Version (Flask):**
```bash
pip install -r requirements_web.txt
```

### 4. Download Models
- GPU: `yolov11.pt` (PyTorch model)
- NPU: `yolov11.rbln` (Compiled RBLN model)

### 5. Configure Classes
Edit `data.yaml` to define your detection classes:
```yaml
nc: 9
names:
  0: person
  1: fire
  2: Smoke_White
  3: Opacity_Smoke
  4: Smoke_Black
  5: helmet
  6: ladder
  7: head
  8: falling
```

---

## 🚀 Applications

### Streamlit App (`app.py`)

**GPU-accelerated interactive application with rich UI**

#### Features:
- 🎛️ Interactive sidebar with all controls
- 📊 Real-time FPS and detection statistics
- 🎥 Multiple input modes:
  - Live webcam stream
  - Webcam snapshot
  - Video file upload
  - Image file upload
- ⚙️ Adjustable settings:
  - Confidence threshold (0.0 - 1.0)
  - IoU threshold (0.0 - 1.0)
  - Max FPS (1 - 30)
  - Camera index selection
- 📈 Visual feedback with annotated frames
- 💾 Easy to customize and extend

#### Usage:
```bash
streamlit run app.py
```

#### Performance:
- **FPS**: 15-30 FPS (depending on GPU)
- **Resolution**: Auto (uses camera default)
- **Processing**: GPU (CUDA)
- **Best For**: Development, testing, demonstrations with UI

#### Screenshot:
```
┌─────────────────────────────────────────────────────┐
│ 📹 Live Feed          │  🎯 Detection Results      │
│                       │                             │
│   [Original Video]    │   [Annotated Video]        │
│                       │                             │
├─────────────────────────────────────────────────────┤
│ Sidebar:              │ Stats:                      │
│ ⚙️ Settings           │ Frame: 1234                 │
│ 🎯 Confidence: 0.25   │ Objects: 3                  │
│ 🔍 IoU: 0.45          │ - person: 2                 │
│ 🎥 Max FPS: 15        │ - fire: 1                   │
└─────────────────────────────────────────────────────┘
```

---

### Flask Web App (`app_web.py`)

**NPU-accelerated high-performance web application**

#### Features:
- ⚡ MJPEG streaming for maximum FPS
- 🚀 Rebellions NPU acceleration with AsyncRuntime
- 🎯 Optimized for 4K displays
- 📹 16:9 aspect ratio (software cropping from 4:3)
- 🌐 Auto-opens browser
- 🔥 Real-time detection with minimal latency
- 📊 On-screen FPS and detection count

#### Usage:

**Option 1: Direct Run**
```bash
python3 app_web.py
```
Browser opens automatically to: http://localhost:5000

**Option 2: Using Scripts**
```bash
# Start
./start_app.sh

# Stop
./stop_app.sh
```

#### Performance:
- **FPS**: 20-25 FPS (NPU + camera limit)
- **Resolution**: 854x480 (16:9) cropped from 640x480 camera
- **Processing**: NPU (Rebellions RBLN)
- **Latency**: ~40-50ms
- **NPU Utilization**: 40-60%
- **Best For**: Production, live monitoring, 4K displays

#### Display Modes:
```bash
# Normal browser
http://localhost:5000

# Fullscreen
Press F11 in browser
```

#### Configuration:
Edit `CONFIG` in `app_web.py`:
```python
CONFIG = {
    'model_path': 'yolov11.rbln',
    'confidence': 0.25,        # Detection threshold
    'iou_threshold': 0.45,     # NMS threshold
    'camera_index': 0,         # Camera device
    'camera_width': 854,       # 16:9 aspect ratio
    'camera_height': 480,
    'jpeg_quality': 90,        # Compression quality
}
```

---

## 🎮 Quick Start

### For Development/Testing (GPU):
```bash
# Activate environment
source venv/bin/activate

# Run Streamlit app
streamlit run app.py

# Open browser to: http://localhost:8501
```

### For Production/Demos (NPU):
```bash
# Activate environment
source venv/bin/activate

# Run Flask app
python3 app_web.py

# Browser opens automatically to: http://localhost:5000
# Press F11 for fullscreen
```

---

## ⚙️ Configuration

### Detected Classes (data.yaml)

```yaml
nc: 9  # Number of classes

names:
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

### Camera Settings

**Streamlit (app.py):**
- Uses camera default resolution
- Adjustable FPS limit in sidebar
- Camera index selectable in UI

**Flask (app_web.py):**
- 854x480 (16:9) for 4K displays
- Auto-cropped from camera feed
- ~26 FPS camera throughput
- Edit `CONFIG` to change

### Detection Thresholds

| Parameter | Range | Default | Description |
|-----------|-------|---------|-------------|
| Confidence | 0.0 - 1.0 | 0.25 | Minimum detection confidence |
| IoU Threshold | 0.0 - 1.0 | 0.45 | Non-Maximum Suppression threshold |

**Lower confidence** = More detections (may include false positives)
**Higher confidence** = Fewer, more accurate detections

---

## 📊 Performance

### Comparison Table

| Metric | Streamlit (GPU) | Flask (NPU) |
|--------|----------------|-------------|
| **FPS** | 15-30 | 20-25 |
| **Accelerator** | NVIDIA GPU | Rebellions NPU |
| **Inference Time** | 10-20ms | 30-35ms |
| **Display Latency** | 50-100ms | 10-20ms |
| **UI Complexity** | High | Minimal |
| **Best For** | Development | Production |
| **Resolution** | Auto | 854x480 (16:9) |
| **4K Display** | Good | Optimized |

### NPU Performance Tips

1. **Camera Resolution**: Use 640x480 for max FPS, crop to 16:9
2. **AsyncRuntime**: Use `parallel=2` for better throughput
3. **JPEG Quality**: 80-90 balance between quality and speed
4. **Skip Frames**: Process every frame for smooth detection
5. **Display**: Fullscreen (F11) for best experience

### GPU Performance Tips

1. **Batch Size**: Keep at 1 for real-time
2. **FPS Limit**: Set to 15-20 for stability
3. **Resolution**: Use camera default
4. **Model**: YOLOv11n for fastest, YOLOv11l for accuracy

---

## 🔧 Troubleshooting

### Common Issues

#### 1. Camera Not Found
```bash
# List available cameras
ls /dev/video*

# Test camera
python3 -c "import cv2; cap = cv2.VideoCapture(0); print('Camera:', cap.isOpened())"
```

**Solution**: Change `camera_index` in config

#### 2. Low FPS (Flask)
- Check camera resolution (should be 640x480)
- Monitor NPU: `watch -n 0.1 rbln-stat`
- Check NPU utilization (should be 40-60%)

#### 3. NPU Not Utilized
- Verify model compiled for current SDK version
- Check: `rbln-stat` shows NPU activity
- Recompile model if needed

#### 4. Port Already in Use
```bash
# Find process using port 5000
lsof -ti:5000

# Kill process
kill -9 $(lsof -ti:5000)
```

#### 5. Model Not Found
```bash
# Check models exist
ls -lh yolov11.pt yolov11.rbln

# GPU model
ls yolov11.pt

# NPU model
ls yolov11.rbln
```

#### 6. Class Names Not Loading
- Verify `data.yaml` exists
- Check YAML syntax
- Ensure class indices match model

---

## 🎯 Use Cases

### Fire Safety Monitoring (Recommended: Flask NPU)
- Real-time fire and smoke detection
- 4K display support
- Low latency alerts
- Production-ready

### Safety Equipment Compliance (Both)
- Helmet detection
- Safety equipment monitoring
- PPE compliance checking

### Hazard Detection (Recommended: Flask NPU)
- Falling person detection
- Ladder safety monitoring
- Real-time alerts

### Development & Testing (Recommended: Streamlit GPU)
- Model evaluation
- Threshold tuning
- UI customization

---

## 📝 File Structure

```
kepco/
├── app.py                      # Streamlit GPU application
├── app_web.py                  # Flask NPU application  
├── data.yaml                   # Class definitions
├── requirements.txt            # GPU dependencies
├── requirements_web.txt        # NPU dependencies
├── yolov11.pt                  # PyTorch model (GPU)
├── yolov11.rbln               # Compiled RBLN model (NPU)
├── templates/
│   └── index.html             # Flask template
├── logs/
│   └── app_web.log            # Application logs
├── start_app.sh               # Start script
├── stop_app.sh                # Stop script
└── README.md                   # This file
```

---

## 🔗 Scripts

### Start Flask App
```bash
./start_app.sh
```

### Stop Flask App
```bash
./stop_app.sh
```

### View Logs
```bash
tail -f logs/app_web.log
```

### Monitor NPU
```bash
watch -n 0.1 rbln-stat
```

---

## 🌐 Remote Access

### Access from Another Device

**Find your IP:**
```bash
ip addr show | grep "inet "
```

**Access from browser:**
```
http://YOUR_IP:5000    # Flask NPU
http://YOUR_IP:8501    # Streamlit GPU
```

---

## 📚 Additional Resources

- **Rebellions NPU Docs**: https://docs.rbln.ai/
- **Ultralytics YOLO**: https://docs.ultralytics.com/
- **Streamlit Docs**: https://docs.streamlit.io/
- **Flask Docs**: https://flask.palletsprojects.com/

---

## 🎓 Best Practices

### For Production Deployment (Flask NPU)
1. Use Flask app (`app_web.py`)
2. Set camera to 854x480 (16:9)
3. Use fullscreen mode (F11)
4. Monitor NPU utilization
5. Log all detections
6. Set appropriate confidence threshold

### For Development (Streamlit GPU)
1. Use Streamlit app (`app.py`)
2. Adjust thresholds in sidebar
3. Test with different input sources
4. Analyze detection statistics
5. Fine-tune model if needed

---

## 🚀 Quick Reference

| Task | Streamlit (GPU) | Flask (NPU) |
|------|----------------|-------------|
| **Start** | `streamlit run app.py` | `python3 app_web.py` |
| **Stop** | `Ctrl+C` | `./stop_app.sh` |
| **URL** | http://localhost:8501 | http://localhost:5000 |
| **Fullscreen** | Browser F11 | Browser F11 |
| **Logs** | Terminal | `logs/app_web.log` |
| **FPS** | 15-30 | 20-25 |

---

## ✨ Summary

**Choose Streamlit (app.py) if you need:**
- Interactive UI with controls
- Easy customization
- Development and testing
- Multiple input modes
- GPU acceleration

**Choose Flask (app_web.py) if you need:**
- Maximum FPS (20-25)
- Production deployment
- 4K display optimization
- Minimal latency
- NPU acceleration
- 24/7 monitoring

---

## 📧 Support

For issues or questions:
1. Check logs: `tail -f logs/app_web.log`
2. Monitor NPU: `rbln-stat`
3. Test camera: `ls /dev/video*`
4. Verify models exist: `ls *.pt *.rbln`

---

**🎉 Enjoy your fire & safety detection system!**

