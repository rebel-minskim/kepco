/**
 * @file triton_client.h
 * @brief Main Triton Inference Client class for YOLO object detection
 * 
 * This client provides a high-level interface for running YOLO inference via
 * NVIDIA Triton Inference Server using gRPC protocol.
 * 
 * Key Features:
 * - gRPC-based communication with Triton server
 * - YOLO v8/v11 postprocessing with NMS (Non-Maximum Suppression)
 * - LetterBox preprocessing (aspect ratio preservation + gray padding)
 * - Multi-threaded parallel processing pipeline (88 FPS on 1080p video)
 * - Real-time performance metrics tracking
 * 
 * Architecture:
 * - GrpcClient: Low-level gRPC communication with Triton
 * - YoloPostprocessor: Bounding box decoding, NMS, coordinate scaling
 * - PerformanceStats: Latency tracking and FPS calculation
 */

#pragma once

#include "config.h"
#include "utils.h"
#include "grpc_client.h"
#include "yolo_preprocess.h"
#include "yolo_postprocess.h"
#include <memory>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>

namespace triton_client {

/**
 * @class TritonClient
 * @brief High-level client for YOLO inference via Triton Inference Server
 * 
 * Manages the complete inference pipeline:
 * 1. Connection & health checks (ServerLive, ServerReady, ModelReady)
 * 2. Image preprocessing (LetterBox: resize + pad with aspect ratio preservation)
 * 3. gRPC inference request to Triton server
 * 4. YOLO postprocessing (bbox decode, confidence filter, NMS)
 * 5. Visualization (bounding boxes, class labels, confidence scores)
 */
class TritonClient {
public:
    /**
     * @brief Construct a new TritonClient
     * @param config Client configuration (server URL, model name, thresholds, etc.)
     */
    explicit TritonClient(const ClientConfig& config);
    
    /**
     * @brief Destructor - ensures proper disconnection from server
     */
    ~TritonClient();
    
    // ============================================================
    // CONNECTION MANAGEMENT
    // ============================================================
    
    /**
     * @brief Connect to Triton server and verify readiness
     * @return true if connection successful and server/model are ready
     * 
     * Performs health checks in order:
     * 1. ServerLive - Is the server process running?
     * 2. ServerReady - Is the server ready to accept requests?
     * 3. ModelReady - Is the specified model loaded and ready?
     */
    bool connect();
    
    /**
     * @brief Disconnect from Triton server and cleanup resources
     */
    void disconnect();
    
    /**
     * @brief Query and print model metadata (inputs, outputs, version)
     */
    void get_model_info();
    
    // ============================================================
    // INFERENCE METHODS
    // ============================================================
    
    /**
     * @brief Run dummy inference with synthetic data (for testing)
     * 
     * Generates a random tensor matching model input shape and sends
     * inference request to verify server connectivity and model execution.
     */
    void run_dummy_inference();
    
    /**
     * @brief Run inference on a single image
     * @param image_path Path to input image file
     * @param output_path Path to save visualization (empty = display only)
     * 
     * Pipeline: Load image → LetterBox preprocess → Inference → 
     *           YOLO postprocess → Draw detections → Save/Display
     */
    void run_image_inference(const std::string& image_path, const std::string& output_path = "");
    
    /**
     * @brief Run inference on video (single-threaded sequential processing)
     * @param video_path Path to input video file
     * @param output_path Path to save output video (empty = no saving)
     * 
     * Performance: ~35 FPS on 1080p video
     * Pipeline: Read frame → Preprocess → Infer → Postprocess → Draw → Write
     */
    void run_video_inference(const std::string& video_path, const std::string& output_path = "");
    
    /**
     * @brief Run inference on video with multi-threaded parallel pipeline
     * @param video_path Path to input video file
     * @param output_path Path to save output video (empty = no saving)
     * @param num_threads Number of parallel inference worker threads (default: 4)
     * 
     * Performance: ~88 FPS on 1080p video with 4 threads (2.5x speedup)
     * 
     * Multi-threaded Pipeline:
     * - Thread 1: Frame reader (VideoCapture)
     * - Thread 2: Preprocessor (LetterBox transform)
     * - Threads 3-N: Inference workers (gRPC calls to Triton)
     * - Thread N+1: Drawer/Writer (sequential, preserves frame order)
     * 
     * Thread-safe queues with condition variables ensure proper synchronization
     * while maximizing parallelism. Frames are passed through the pipeline with
     * their original cv::Mat to avoid re-reading from disk.
     */
    void run_video_inference_parallel(const std::string& video_path, const std::string& output_path = "", int num_threads = 4);
    
private:
    // ============================================================
    // MEMBER VARIABLES
    // ============================================================
    
    ClientConfig config_;                                     ///< Client configuration
    std::unique_ptr<GrpcClient> grpc_client_;                ///< gRPC communication handler
    std::unique_ptr<YoloPreprocessor> yolo_preprocessor_;    ///< YOLO input preprocessor (LetterBox)
    std::unique_ptr<YoloPostprocessor> yolo_postprocessor_;  ///< YOLO output parser (NMS, decode)
    std::vector<std::string> class_names_;                   ///< Class labels (e.g., "person", "car")
    
    // ============================================================
    // HELPER METHODS
    // ============================================================
    
    /**
     * @brief Load class names from file specified in config
     */
    void load_class_names();
    
    /**
     * @brief Check if Triton server process is alive
     * @return true if server is live
     */
    bool is_server_live();
    
    /**
     * @brief Check if Triton server is ready to accept requests
     * @return true if server is ready
     */
    bool is_server_ready();
    
    /**
     * @brief Check if specified model is loaded and ready
     * @param model_name Name of the model to check
     * @return true if model is ready
     */
    bool is_model_ready(const std::string& model_name);
    
    // ============================================================
    // INFERENCE PIPELINE HELPERS
    // ============================================================
    
    /**
     * @brief Preprocess image using LetterBox (Ultralytics-compatible)
     * @param image Input image (BGR format)
     * @return Preprocessed tensor [C, H, W] in RGB, normalized to [0, 1]
     * 
     * LetterBox steps (matching Ultralytics exactly):
     * 1. Calculate scale ratio to fit into target size (640x640)
     * 2. Resize image with aspect ratio preservation
     * 3. Add gray padding (value=114) to reach exact target size
     * 4. Normalize to [0, 1] (divide by 255)
     * 5. Transpose HWC → CHW
     * 6. Reverse channels: RGB → BGR (OpenCV reads BGR natively)
     */
    std::vector<float> prepare_input_tensor(const cv::Mat& image);
    
    /**
     * @brief Run inference and postprocess results
     * @param input_tensor Preprocessed input tensor [1, 3, H, W]
     * @param orig_width Original image width (for coordinate scaling)
     * @param orig_height Original image height (for coordinate scaling)
     * @return Vector of detected objects after NMS
     * 
     * Steps:
     * 1. Send gRPC inference request to Triton
     * 2. Parse raw output [1, 84, 8400] (YOLO format)
     * 3. Decode bounding boxes (cx, cy, w, h) → (x1, y1, x2, y2)
     * 4. Filter by confidence threshold
     * 5. Apply NMS to remove duplicate detections
     * 6. Scale coordinates back to original image size
     */
    std::vector<Detection> run_inference(const std::vector<float>& input_tensor, 
                                        int orig_width, int orig_height);
    
    /**
     * @brief Process a single video frame (used in single-threaded mode)
     * @param frame Input/output frame (modified in-place with detections)
     * @param stats Performance statistics tracker
     * @param frame_history Detection history for tracking (not currently used)
     */
    void process_video_frame(cv::Mat& frame, PerformanceStats& stats, 
                           std::vector<std::vector<Detection>>& frame_history);
};

} // namespace triton_client
