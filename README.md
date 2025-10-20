# KEPCO AI Demo Suite

AI-powered demonstration applications showcasing Rebellions NPU performance for KEPCO Proof of Concept.

![Status](https://img.shields.io/badge/Status-Active-green) ![Platform](https://img.shields.io/badge/Platform-Rebellions%20NPU-blue)

## What's Included

This demo suite contains two applications:

### 1. Performance Dashboard (`BIXPO/web/`)
Visual comparison of NPU vs GPU performance with real-time metrics and video processing demos.

**Key Metrics:**
- NPU: 36 FPS @ 50W → 0.72 FPS/Watt
- GPU: 24 FPS @ 325W → 0.074 FPS/Watt
- **NPU Advantage: 9.7x better efficiency**

### 2. Fire & Safety Detection (`BIXPO/live-demo/`)
Real-time YOLO-based detection system for industrial safety monitoring with 9 detection classes (fire, smoke, people, helmets, etc.).

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

## System Requirements

- **NPU**: Rebellions ATOM (ATOM-Max recommended)
- **OS**: Ubuntu 20.04+ / Linux
- **Python**: 3.8+
- **Camera**: USB webcam (for detection system)

## Documentation

Detailed documentation for each application:

- **Performance Dashboard**: [BIXPO/web/README.md](BIXPO/web/README.md) (English), [README_ko.md](BIXPO/web/README_ko.md) (Korean)
- **Detection System**: [BIXPO/live-demo/docs/README.md](BIXPO/live-demo/docs/README.md)
- **Exhibition Setup**: [BIXPO/README.md](BIXPO/README.md)

## Project Structure

```
kepco/
├── BIXPO/
│   ├── web/                    # Performance dashboard
│   │   ├── assets/            # CSS, JS, images, videos
│   │   ├── data/              # Performance data (JSON)
│   │   ├── index.html         # Main dashboard
│   │   └── build_standalone.py
│   │
│   └── live-demo/             # Fire & safety detection
│       ├── app_web.py         # Flask NPU application
│       ├── config/            # Configuration files
│       ├── docs/              # Documentation
│       ├── models/            # AI models (.pt, .rbln)
│       ├── templates/         # HTML templates
│       └── utils/             # Diagnostic tools
│
└── README.md                  # This file
```

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
```

**More help?** See detailed docs in each application folder.

## License

Copyright © 2025 Rebellions Inc.

---

**Built with Rebellions NPU Technology**

**For KEPCO Proof of Concept - 2025**
