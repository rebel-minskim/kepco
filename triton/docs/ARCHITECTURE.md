# C++ Triton Client Architecture Documentation

## 📐 System Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                        C++ Triton Client                            │
│                                                                     │
│  ┌──────────┐   ┌──────────────┐   ┌─────────────┐   ┌─────────┐ │
│  │  main.cpp│──▶│TritonClient  │──▶│ GrpcClient  │──▶│ Triton  │ │
│  │   Entry  │   │  (High-level)│   │ (Low-level) │   │ Server  │ │
│  └──────────┘   └──────────────┘   └─────────────┘   └─────────┘ │
│                         │                                           │
│                         ├──▶ YoloPostprocessor (NMS, decode)       │
│                         ├──▶ PerformanceStats (FPS tracking)       │
│                         └──▶ Visualization (draw boxes)            │
└─────────────────────────────────────────────────────────────────────┘
```

## 🔄 Data Flow: Single-threaded Mode

```
┌─────────┐     ┌────────────┐     ┌──────────┐     ┌─────────────┐
│ OpenCV  │────▶│ LetterBox  │────▶│   gRPC   │────▶│   Triton    │
│VideoCapt│     │ Preprocess │     │ Inference│     │   Server    │
└─────────┘     └────────────┘     └──────────┘     └─────────────┘
                                          │
     ▲                                    ▼
     │                            ┌───────────────┐
     │                            │ YOLO Output   │
     │                            │ [1, 84, 8400] │
     │                            └───────────────┘
     │                                    │
     │                                    ▼
┌────────┐      ┌──────────┐     ┌──────────────┐
│VideoWrt│◀─────│   Draw   │◀────│ Postprocess  │
│  Save  │      │  Boxes   │     │ NMS + Decode │
└────────┘      └──────────┘     └──────────────┘

Performance: ~35 FPS
Latency breakdown:
- Read:        ~0.5ms
- Preprocess:  ~4ms
- Inference:   ~19ms
- Postprocess: ~0.2ms
- Draw:        ~1ms
Total:         ~25ms/frame
```

## 🚀 Data Flow: Multi-threaded Mode (Parallel Pipeline)

```
┌──────────────────────────────────────────────────────────────────┐
│                     PARALLEL PIPELINE                            │
└──────────────────────────────────────────────────────────────────┘

Thread 1: Frame Reader
┌──────────────┐
│ VideoCapture │──┐
│  Read Frame  │  │
└──────────────┘  │
                  ▼
           ┌─────────────┐
           │ raw_queue   │ (Thread-safe queue)
           │ cv::Mat     │
           └─────────────┘
                  │
                  ▼
Thread 2: Preprocessor
┌──────────────┐
│  LetterBox   │──┐
│ 640x640 pad  │  │
└──────────────┘  │
                  ▼
         ┌──────────────────┐
         │ preprocessed_queue│ (Thread-safe queue)
         │ [frame, tensor]  │
         └──────────────────┘
                  │
                  ▼
Threads 3-6: Inference Workers (4 parallel threads)
┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│ gRPC Worker 1│  │ gRPC Worker 2│  │ gRPC Worker 3│  │ gRPC Worker 4│
│   Triton     │  │   Triton     │  │   Triton     │  │   Triton     │
│  Inference   │  │  Inference   │  │  Inference   │  │  Inference   │
└──────────────┘  └──────────────┘  └──────────────┘  └──────────────┘
       │                 │                 │                 │
       └─────────────────┴─────────────────┴─────────────────┘
                                 │
                                 ▼
                       ┌──────────────────┐
                       │  inference_queue │ (Thread-safe queue)
                       │ [frame, results] │
                       └──────────────────┘
                                 │
                                 ▼
Thread 7: Drawer & Writer (Sequential for frame order)
┌──────────────┐
│ Postprocess  │ (NMS, decode, scale coords)
│  YOLO+NMS    │
└──────────────┘
        │
        ▼
┌──────────────┐
│ Draw Boxes   │ (cv::rectangle, cv::putText)
│   Labels     │
└──────────────┘
        │
        ▼
┌──────────────┐
│ VideoWriter  │ (Save to disk)
│   Save       │
└──────────────┘

Performance: ~88 FPS (2.5x speedup)
```

## 🧵 Thread Synchronization

```cpp
// Queue protection pattern (used 3 times in pipeline)
struct ThreadSafeQueue {
    std::queue<Data> queue;           // Actual data storage
    std::mutex mutex;                 // Protects queue access
    std::condition_variable cv;       // Signals new data
    std::atomic<bool> done;          // Signals pipeline completion
};

// Producer pattern (Reader, Preprocessor, Inference workers)
{
    std::lock_guard<std::mutex> lock(queue_mutex);
    queue.push(data);
}
condition_var.notify_one();  // Wake up consumer

// Consumer pattern (Preprocessor, Inference workers, Drawer)
{
    std::unique_lock<std::mutex> lock(queue_mutex);
    condition_var.wait(lock, [&]{ 
        return !queue.empty() || done_flag; 
    });
    
    if (!queue.empty()) {
        data = queue.front();
        queue.pop();
    }
}
```

## 📦 Class Hierarchy

```
┌────────────────────────────────────────────────────────────┐
│                      TritonClient                          │
│  ┌──────────────────────────────────────────────────────┐ │
│  │ Public Interface:                                     │ │
│  │  • connect()                                          │ │
│  │  • run_dummy_inference()                              │ │
│  │  • run_image_inference()                              │ │
│  │  • run_video_inference()        [35 FPS]             │ │
│  │  • run_video_inference_parallel() [88 FPS]           │ │
│  └──────────────────────────────────────────────────────┘ │
│                                                            │
│  ┌──────────────────────────────────────────────────────┐ │
│  │ Private Members:                                      │ │
│  │  • config_: ClientConfig                              │ │
│  │  • grpc_client_: unique_ptr<GrpcClient>               │ │
│  │  • yolo_postprocessor_: unique_ptr<YoloPostprocessor> │ │
│  │  • class_names_: vector<string>                       │ │
│  └──────────────────────────────────────────────────────┘ │
│                                                            │
│  ┌──────────────────────────────────────────────────────┐ │
│  │ Private Methods:                                      │ │
│  │  • load_class_names()                                 │ │
│  │  • is_server_live/ready()                             │ │
│  │  • is_model_ready()                                   │ │
│  │  • prepare_input_tensor()  [LetterBox]               │ │
│  │  • run_inference()          [gRPC + Postprocess]     │ │
│  │  • process_video_frame()    [Single-thread helper]   │ │
│  └──────────────────────────────────────────────────────┘ │
└────────────────────────────────────────────────────────────┘
                           │
                           │ uses
                           ▼
         ┌─────────────────┴─────────────────┐
         │                                    │
         ▼                                    ▼
┌────────────────┐                 ┌───────────────────┐
│   GrpcClient   │                 │YoloPostprocessor  │
├────────────────┤                 ├───────────────────┤
│ • stub_        │                 │ • postprocess()   │
│ • channel_     │                 │ • apply_nms()     │
│ • infer()      │                 │ • calculate_iou() │
│ • server_live()│                 │ • scale_coords()  │
│ • model_ready()│                 └───────────────────┘
└────────────────┘
```

## 🔍 LetterBox Preprocessing (Detailed)

```
Input Image: 1920x1080 (arbitrary size)
Target Size: 640x640 (model input)

Step 1: Calculate scale ratio
┌─────────────────┐
│  1920 x 1080    │
│  Original image │
└─────────────────┘
         │
         ▼
  r = min(640/1080, 640/1920)
  r = min(0.593, 0.333) = 0.333

Step 2: Resize with aspect ratio
┌─────────┐
│ 640x360 │
│ Resized │
└─────────┘
         │
         ▼
  new_w = round(1920 * 0.333) = 640
  new_h = round(1080 * 0.333) = 360

Step 3: Add padding (center mode)
  dh = (640 - 360) / 2 = 140
  padding_top = 140, padding_bottom = 140
  padding_left = 0, padding_right = 0

┌─────────────┐
│   (Gray)    │ ← padding_top = 140
├─────────────┤
│  640 x 360  │ ← actual image
├─────────────┤
│   (Gray)    │ ← padding_bottom = 140
└─────────────┘
  640 x 640

Step 4: Normalize & Transpose
  • pixel / 255.0  → [0.0, 1.0]
  • HWC → CHW: [640, 640, 3] → [3, 640, 640]
  • BGR → RGB: Reverse channel order

Output: [3, 640, 640] float32 tensor
```

## 📊 YOLO Output Format

```
Raw Output Shape: [1, 84, 8400]
                   │   │    │
                   │   │    └─ 8400 candidate detections
                   │   └────── 84 = 4 bbox coords + 80 class scores
                   └────────── Batch size (always 1)

Per-detection layout (84 values):
┌────────────────────────────────────────────────────┐
│ [0-3]:  cx, cy, w, h     (center x, center y,     │
│                            width, height)          │
│ [4-83]: confidence scores for 80 COCO classes     │
│         (person, bicycle, car, ..., toothbrush)   │
└────────────────────────────────────────────────────┘

Postprocessing Steps:
1. For each of 8400 detections:
   a. Find max confidence across 80 classes
   b. If confidence > threshold (0.20):
      - Decode bbox: (cx,cy,w,h) → (x1,y1,x2,y2)
      - Store: class_id, confidence, coordinates

2. Apply NMS (Non-Maximum Suppression):
   a. Sort by confidence (descending)
   b. For each box:
      - Compare with higher-confidence boxes
      - If IoU > threshold (0.65): suppress (remove)
   c. Keep only non-suppressed boxes

3. Scale coordinates back to original image size:
   a. Remove padding added by LetterBox
   b. Divide by scale ratio
   c. Clip to image boundaries

Output: Vector<Detection> (typically 5-30 objects)
```

## ⚡ Performance Bottleneck Analysis

### Single-threaded Mode (35 FPS)

```
Bottleneck: Sequential processing

┌────────┬──────┬─────┬──────┬──────┐
│  Read  │ Pre  │ Inf │ Post │ Draw │
│  0.5ms │  4ms │ 19ms│ 0.2ms│  1ms │
└────────┴──────┴─────┴──────┴──────┘
         Total: ~25ms/frame = 40 FPS theoretical
         Actual: ~35 FPS (video I/O overhead)

Inference (19ms) dominates the pipeline.
CPU idle during GPU inference!
```

### Multi-threaded Mode (88 FPS)

```
Optimization: Parallel inference workers

Time ──▶
0ms    10ms   20ms   30ms   40ms   50ms
│      │      │      │      │      │
Worker 1: [====Inf1====]      [====Inf5====]
Worker 2:      [====Inf2====]      [====Inf6====]
Worker 3:           [====Inf3====]      [====Inf7====]
Worker 4:                [====Inf4====]      [====Inf8====]

4 frames processed in 20ms = 200 FPS per-batch
Real FPS = 88 (limited by read/write/preprocess)

Remaining bottlenecks:
1. Frame reading (VideoCapture): ~3-5ms
2. Preprocessing (LetterBox): ~4ms
3. Drawing/Writing (sequential): ~2-3ms
```

## 🔮 Future Optimization Opportunities

### 1. Batch Inference (Potential 2x speedup)
```
Current: Process 1 frame per inference
Optimized: Process 4 frames per inference

Input Shape: [1, 3, 640, 640] → [4, 3, 640, 640]
Output Shape: [1, 84, 8400] → [4, 84, 8400]

Benefits:
- Better GPU utilization
- Amortized gRPC overhead
- Fewer context switches

Challenges:
- Frame synchronization
- Memory management
- Output parsing complexity
```

### 2. GPU Preprocessing (Potential 1.5x speedup)
```
Current: CPU-based OpenCV preprocessing (~4ms)
Optimized: CUDA kernels for LetterBox (~0.5ms)

cv::resize() → cudaResize()
cv::copyMakeBorder() → cudaPadding()
Memory copy → Direct GPU memory

Requires: OpenCV with CUDA support
```

### 3. Memory Pool (Reduce allocations)
```
Current: Allocate cv::Mat and vectors on each frame
Optimized: Pre-allocate and reuse memory

Object pool for:
- cv::Mat buffers (640x640x3)
- Inference input tensors (3x640x640 floats)
- Detection vectors
```

## 📚 References

- **Ultralytics LetterBox**: `ultralytics/data/augment.py`
- **Ultralytics NMS**: `ultralytics/utils/ops.py`
- **Triton gRPC Protocol**: `triton-inference-server/server`
- **YOLO v8 Architecture**: `ultralytics/nn/tasks.py`

---

*Last updated: October 2024*
*Performance benchmarks: NVIDIA GPU on 1080p video*

