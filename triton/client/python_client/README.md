# Triton Inference Client

A clean, modular client for running object detection inference using Triton Inference Server.

## Features

- **Modular Design**: Clean separation of concerns with dedicated modules
- **Multiple Modes**: Support for dummy, image, and video processing
- **Performance Tracking**: Built-in performance monitoring and statistics
- **Object Tracking**: Frame history-based object tracking for video processing
- **Configurable**: Easy configuration through command-line arguments and config files

## Project Structure

```
client/
├── main.py                 # Main entry point
├── config.py               # Configuration management
├── requirements.txt        # Python dependencies
├── data.yaml              # Class names configuration
├── README.md              # This file
├── utils/                  # Utility modules
│   ├── __init__.py
│   ├── models.py           # Data models and classes
│   ├── processing.py       # Image processing utilities
│   └── visualization.py    # Drawing and visualization utilities
├── media/                  # Media files directory
│   ├── 1.mp4
│   ├── 30sec.mp4
│   └── output.mp4
└── output/                 # Output directory (auto-created)
```

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Ensure Triton Inference Server is running with your model loaded.

## Usage

### Basic Usage

```bash
# Process a video file
python main.py video media/1.mp4 -o output/result.mp4

# Process a single image
python main.py image path/to/image.jpg -o output/result.jpg

# Run dummy inference (test server connection)
python main.py dummy
```

### Protocol Comparison

Compare HTTP vs gRPC performance:

```bash
# Detailed comparison (100 requests)
python protocol_comparison.py

# Extended comparison (500 requests)
python protocol_comparison.py -n 500
```

### Advanced Options

```bash
# Custom model and server settings
python main.py video media/30sec.mp4 \
    --model yolov11 \
    --url localhost:8001 \
    --width 800 --height 800 \
    --conf 0.25 --iou 0.65 \
    --fps 30.0 \
    -o output/result.mp4

# Print model information
python main.py dummy --model-info

# Verbose output
python main.py video media/30sec.mp4 --verbose
```

### Command Line Arguments

- `mode`: Run mode (`dummy`, `image`, `video`)
- `input`: Input file path
- `-o, --output`: Output file path
- `-m, --model`: Model name (default: yolov11)
- `--width, --height`: Input dimensions (default: 800x800)
- `--conf`: Confidence threshold (default: 0.20)
- `--iou`: IoU threshold (default: 0.50)
- `-u, --url`: Server URL (default: localhost:8001)
- `-t, --timeout`: Client timeout
- `-v, --verbose`: Verbose output
- `-f, --fps`: Output video FPS (default: 24.0)
- `-i, --model-info`: Print model information

## Configuration

The application uses a modular configuration system in `config.py`. You can customize:

- Model parameters (input size, thresholds, etc.)
- Server connection settings
- Video processing options
- File paths and directories

## Performance Monitoring

The client includes built-in performance tracking that provides:

- End-to-end latency statistics
- Per-stage timing (preprocessing, inference, postprocessing)
- FPS and throughput metrics
- P95 latency percentiles

## Object Tracking

For video processing, the client implements frame history-based object tracking:

- Maintains detection history across frames
- Filters out transient detections
- Reduces false positives through temporal consistency

## Development

The codebase is organized for maintainability:

- **`main.py`**: Entry point and high-level orchestration
- **`config.py`**: Configuration management
- **`models.py`**: Data models and performance tracking
- **`utils/`**: Reusable utility functions
- **`utils/processing.py`**: Image preprocessing and postprocessing
- **`utils/visualization.py`**: Drawing and visualization utilities

## Requirements

- Python 3.7+
- Triton Inference Server
- CUDA (for GPU inference)
- See `requirements.txt` for Python dependencies
