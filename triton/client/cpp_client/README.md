# C++ Triton Inference Client for YOLO Object Detection

A high-performance C++ client for running YOLO object detection inference via NVIDIA Triton Inference Server with gRPC protocol.

## 🎯 Key Features

- **gRPC Communication**: Direct protocol buffers communication with Triton server
- **YOLO v8/v11 Support**: Full postprocessing with NMS (Non-Maximum Suppression)
- **LetterBox Preprocessing**: Ultralytics-compatible image preprocessing
- **Multi-threaded Pipeline**: Parallel processing for high-FPS video inference
- **Performance Tracking**: Real-time FPS and latency metrics

## 📊 Performance Benchmarks

Tested on `1.mp4` (440 frames, 1080p):

| Mode | FPS | Processing Time | Speedup |
|------|-----|-----------------|---------|
| **Single-threaded** | 35.06 | 12.66s | Baseline |
| **Parallel (4 threads)** | **88.92** | 4.95s | **2.5x** |
| **Parallel (6 threads)** | 88.58 | 4.94s | 2.5x |
| **Parallel (8 threads)** | 88.06 | 4.95s | 2.5x |

**Optimal Configuration**: 4-6 inference threads

## 🏗️ Architecture

### File Structure

```
cpp_client/
├── main.cpp                    # Entry point, argument parsing
├── triton_client.h/cpp         # High-level client interface
├── grpc_client.h/cpp           # Low-level gRPC communication
├── grpc_service.proto          # Protobuf service definitions
├── yolo_preprocess.h/cpp       # YOLO input preprocessing (LetterBox)
├── yolo_postprocess.h/cpp      # YOLO output parsing & NMS
├── utils.h/cpp                 # Visualization & performance tracking
├── config.h                    # Configuration structures
├── CMakeLists.txt              # Build configuration
└── build.sh                    # Build automation script
```

### Pipeline Architecture

#### Single-threaded Mode (video)
```
Read Frame → Preprocess → Inference → Postprocess → Draw → Write
     ↓           ↓            ↓            ↓          ↓       ↓
   OpenCV    LetterBox      gRPC        YOLO+NMS   OpenCV  OpenCV
```

#### Multi-threaded Mode (parallel)
```
Thread 1:  Frame Reader
              ↓ (queue)
Thread 2:  Preprocessor (LetterBox)
              ↓ (queue)
Threads 3-N: Inference Workers (gRPC to Triton)
              ↓ (queue)
Thread N+1: Drawer & Writer (sequential for frame order)
```

## 🚀 Usage

### Build

```bash
./build.sh
```

Requirements:
- CMake 3.10+
- OpenCV 4.x
- gRPC & Protobuf
- C++17 compiler

### Run

```bash
# Test with synthetic data
./build/bin/triton_client dummy

# Single image inference
./build/bin/triton_client image input.jpg output.jpg

# Single-threaded video (35 FPS)
./build/bin/triton_client video input.mp4 output.mp4

# Multi-threaded video (88 FPS, 4 threads)
./build/bin/triton_client parallel input.mp4 output.mp4 4

# Multi-threaded video (6 threads)
./build/bin/triton_client parallel input.mp4 output.mp4 6
```

### Configuration

Edit `config.h` to customize:

```cpp
struct ServerConfig {
    std::string url = "localhost:8001";  // Triton server address
    int timeout_ms = 5000;               // Request timeout
};

struct ModelConfig {
    std::string name = "yolo11n";        // Model name on Triton
    int input_width = 640;               // Model input width
    int input_height = 640;              // Model input height
};

struct DetectionConfig {
    float confidence_threshold = 0.20f;  // Min confidence score
    float iou_threshold = 0.65f;         // NMS IoU threshold
    int max_detections = 1024;           // Max detections per frame
};
```

## 🔧 Technical Details

### 1. LetterBox Preprocessing

Matches Ultralytics preprocessing exactly:

```cpp
1. Calculate scale ratio: r = min(target_h/img_h, target_w/img_w)
2. Resize with aspect ratio: new_size = (img_w*r, img_h*r)
3. Add gray padding (value=114) to reach exact target size (640x640)
4. Normalize to [0, 1]: pixel / 255.0
5. Transpose HWC → CHW
6. Reverse channels: RGB → BGR (OpenCV compatibility)
```

### 2. YOLO Postprocessing

Output format: `[1, 84, 8400]`
- 84 channels: `[x, y, w, h, conf_class0, conf_class1, ..., conf_class79]`
- 8400 detections: 3 detection heads at different scales

Processing steps:
```cpp
1. Parse raw output tensor (column-major format)
2. Decode bounding boxes: (cx, cy, w, h) → (x1, y1, x2, y2)
3. Filter by confidence threshold (0.20)
4. Apply NMS across all classes (IoU threshold: 0.65)
5. Scale coordinates back to original image size
```

### 3. Multi-threading Synchronization

Thread-safe queues with condition variables:

```cpp
// Thread-safe queue structure
std::queue<FrameData> queue;
std::mutex mutex;
std::condition_variable cv;
std::atomic<bool> done_flag;

// Producer thread
{
    std::lock_guard<std::mutex> lock(mutex);
    queue.push(data);
}
cv.notify_one();

// Consumer thread
{
    std::unique_lock<std::mutex> lock(mutex);
    cv.wait(lock, [&]{ return !queue.empty() || done; });
    if (!queue.empty()) {
        data = queue.front();
        queue.pop();
    }
}
```

### 4. Performance Optimization Techniques

✅ **Implemented:**
- Multi-threaded pipeline parallelism (2.5x speedup)
- Frame data passing (avoid disk re-reads)
- Efficient memory layout (cv::Mat shallow copies)
- gRPC connection reuse

⏳ **Potential Future Improvements:**
- Batch inference (process multiple frames per request)
- GPU preprocessing (CUDA/OpenCV GPU module)
- Object pooling for memory allocations
- TensorRT optimized models

## 📝 Code Annotations

All major files contain extensive documentation:

- **Doxygen-style comments** for classes and functions
- **Inline explanations** for complex algorithms (LetterBox, NMS)
- **Performance notes** for optimization decisions
- **Reference links** to Ultralytics source code

Example:
```cpp
/**
 * @brief Preprocess image using LetterBox (Ultralytics-compatible)
 * @param image Input image (BGR format)
 * @return Preprocessed tensor [C, H, W] in RGB, normalized to [0, 1]
 * 
 * LetterBox steps (matching Ultralytics exactly):
 * 1. Calculate scale ratio to fit into target size (640x640)
 * 2. Resize image with aspect ratio preservation
 * 3. Add gray padding (value=114) to reach exact target size
 * ...
 */
```

## 🐛 Debugging

Enable debug output in source files:

```cpp
// triton_client.cpp - LetterBox debug
static bool print_debug = true;  // Set to false to disable

// yolo_postprocess.cpp - NMS debug  
static bool debug_nms = true;    // Set to false to disable
```

## 📊 Performance Metrics

Real-time stats displayed during inference:

```
Frame 100: 15 objects | E2E: 22.5ms | Pre: 4.2ms | Inf: 18.1ms | Post: 0.2ms
```

Summary at completion:

```
============================================================
PARALLEL PROCESSING SUMMARY
============================================================
Total frames: 440
Total time: 4.95s
Average FPS: 88.92
Inference threads: 4
============================================================
```

Breakdown:
- **E2E (End-to-End)**: Total latency per frame
- **Pre (Preprocessing)**: LetterBox transform
- **Inf (Inference)**: gRPC call + GPU execution
- **Post (Postprocessing)**: YOLO decode + NMS

## 🔗 Related Files

- Python client: `../triton_client.py`
- Server setup: `../../server/`
- Class names: `../config/coco_classes.txt`
- Test videos: `../media/`

## 📚 References

- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
- [Triton Inference Server](https://github.com/triton-inference-server/server)
- [gRPC Protocol](https://grpc.io/)
- [OpenCV Documentation](https://docs.opencv.org/)

## 🤝 Contributing

When modifying the code:
1. Maintain Doxygen-style documentation
2. Add performance measurements for optimizations
3. Run build and test before committing
4. Update this README with significant changes

## 📄 License

Copyright 2024 - KEPCO Project

