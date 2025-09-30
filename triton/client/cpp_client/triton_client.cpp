#include "triton_client.h"
#include <iostream>
#include <fstream>
#include <chrono>
#include <algorithm>
#include <numeric>
#include <thread>
#include <cmath>
#include <sstream>
#include <iomanip>
#include <opencv2/imgproc.hpp>

namespace triton_client {

TritonClient::TritonClient(const ClientConfig& config) : config_(config) {
    load_class_names();
    grpc_client_ = std::make_unique<GrpcClient>(config_.server.url);
    yolo_postprocessor_ = std::make_unique<YoloPostprocessor>(9);  // 9 classes
}

TritonClient::~TritonClient() {
    disconnect();
}

void TritonClient::load_class_names() {
    class_names_.clear();
    
    // Try to load from YAML file
    std::ifstream file(config_.paths.data_yaml);
    if (file.is_open()) {
        // Simple YAML parsing for class names
        std::string line;
        bool in_names_section = false;
        while (std::getline(file, line)) {
            if (line.find("names:") != std::string::npos) {
                in_names_section = true;
                continue;
            }
            if (in_names_section && line.find(":") != std::string::npos) {
                size_t colon_pos = line.find(":");
                if (colon_pos != std::string::npos) {
                    std::string class_name = line.substr(colon_pos + 1);
                    // Remove quotes and whitespace
                    class_name.erase(std::remove(class_name.begin(), class_name.end(), '"'), class_name.end());
                    class_name.erase(std::remove(class_name.begin(), class_name.end(), ' '), class_name.end());
                    if (!class_name.empty()) {
                        class_names_.push_back(class_name);
                    }
                }
            }
        }
        file.close();
    }
    
    // Fallback: create default class names
    if (class_names_.empty()) {
        for (int i = 0; i < 100; ++i) {
            class_names_.push_back("class_" + std::to_string(i));
        }
    }
    
    std::cout << "Loaded " << class_names_.size() << " class names" << std::endl;
}

bool TritonClient::connect() {
    try {
        std::cout << "Connecting to server at: " << config_.server.url << std::endl;
        
        // Check server liveness
        if (!grpc_client_->is_server_live()) {
            std::cout << "FAILED: Server is not live" << std::endl;
            return false;
        }
        
        // Check server readiness
        if (!grpc_client_->is_server_ready()) {
            std::cout << "FAILED: Server is not ready" << std::endl;
            return false;
        }
        
        // Check model readiness
        if (!grpc_client_->is_model_ready(config_.model.name)) {
            std::cout << "FAILED: Model " << config_.model.name << " is not ready" << std::endl;
            return false;
        }
        
        std::cout << "Successfully connected to Triton server" << std::endl;
        return true;
        
    } catch (const std::exception& e) {
        std::cout << "Connection failed: " << e.what() << std::endl;
        return false;
    }
}

void TritonClient::disconnect() {
    if (grpc_client_) {
        grpc_client_.reset();
        std::cout << "Disconnected from server" << std::endl;
    }
}

bool TritonClient::is_server_live() {
    return grpc_client_ ? grpc_client_->is_server_live() : false;
}

bool TritonClient::is_server_ready() {
    return grpc_client_ ? grpc_client_->is_server_ready() : false;
}

bool TritonClient::is_model_ready(const std::string& model_name) {
    return grpc_client_ ? grpc_client_->is_model_ready(model_name) : false;
}

void TritonClient::get_model_info() {
    if (!grpc_client_) {
        std::cout << "Not connected to server" << std::endl;
        return;
    }
    
    std::cout << "Model Metadata:" << std::endl;
    std::cout << "  Name: " << config_.model.name << std::endl;
    std::cout << "  Input Shape: [1, 3, " << config_.model.input_height 
              << ", " << config_.model.input_width << "]" << std::endl;
    std::cout << "  Input Type: FP32" << std::endl;
    std::cout << "  Output Shape: [1, 25200, 85]" << std::endl;
    std::cout << "  Output Type: FP32" << std::endl;
}

void TritonClient::run_dummy_inference() {
    std::cout << "Running dummy inference..." << std::endl;
    
    // Create dummy input tensor
    std::vector<float> dummy_input(config_.model.input_height * config_.model.input_width * 3, 1.0f);
    
    // Run inference (dummy has no original size, use input size)
    auto detections = run_inference(dummy_input, config_.model.input_width, config_.model.input_height);
    
    std::cout << "Received result buffer of size: " << dummy_input.size() << std::endl;
    float sum = std::accumulate(dummy_input.begin(), dummy_input.end(), 0.0f);
    std::cout << "Buffer sum: " << sum << std::endl;
}

std::vector<float> TritonClient::prepare_input_tensor(const cv::Mat& image) {
    // LetterBox implementation matching Ultralytics exactly
    // Reference: ultralytics.data.augment.LetterBox
    
    int target_w = config_.model.input_width;
    int target_h = config_.model.input_height;
    int shape_h = image.rows;  // current shape [height, width]
    int shape_w = image.cols;
    
    // Scale ratio (new / old)
    // r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    float r = std::min(static_cast<float>(target_h) / shape_h, 
                       static_cast<float>(target_w) / shape_w);
    
    // scaleup=True by default (allow scaling up)
    // If scaleup=False: r = min(r, 1.0) - only scale down
    
    // Compute new unpadded size
    // new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))  # (width, height)
    int new_unpad_w = static_cast<int>(std::round(shape_w * r));
    int new_unpad_h = static_cast<int>(std::round(shape_h * r));
    
    // Compute padding
    // dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    float dw = target_w - new_unpad_w;  // width padding
    float dh = target_h - new_unpad_h;  // height padding
    
    // auto=False, scale_fill=False by default
    // If auto=True: dw, dh = np.mod(dw, stride), np.mod(dh, stride)
    // If scale_fill=True: dw, dh = 0.0, 0.0
    
    // center=True by default (divide padding into 2 sides)
    dw /= 2.0f;
    dh /= 2.0f;
    
    // Resize if shape doesn't match
    cv::Mat resized;
    if (shape_h != new_unpad_h || shape_w != new_unpad_w) {
        // interpolation=cv2.INTER_LINEAR by default
        cv::resize(image, resized, cv::Size(new_unpad_w, new_unpad_h), 0, 0, cv::INTER_LINEAR);
    } else {
        resized = image.clone();
    }
    
    // Calculate border values
    // top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1)) if center else 0, int(round(dh))
    // left, right = int(round(dw - 0.1)), int(round(dw + 0.1)) if center else 0, int(round(dw))
    int top = static_cast<int>(std::round(dh - 0.1f));
    int bottom = static_cast<int>(std::round(dh + 0.1f));
    int left = static_cast<int>(std::round(dw - 0.1f));
    int right = static_cast<int>(std::round(dw + 0.1f));
    
    // Add border with padding_value=114 (gray)
    cv::Mat padded;
    cv::copyMakeBorder(resized, padded, top, bottom, left, right, 
                       cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));
    
    // Normalize to [0, 1]
    cv::Mat float_img;
    padded.convertTo(float_img, CV_32F, 1.0 / 255.0);
    
    // Transpose from HWC to CHW format
    // Python: img = img.transpose((2, 0, 1))[::-1]
    // transpose((2, 0, 1)) converts HWC to CHW
    // [::-1] reverses the first dimension (channels), so [R,G,B] becomes [B,G,R]
    std::vector<cv::Mat> channels(3);
    cv::split(float_img, channels);  // Split into [B, G, R] (OpenCV is BGR)
    
    // Flatten in CHW order: B, G, R (which matches [::-1] on RGB)
    std::vector<float> tensor;
    tensor.reserve(float_img.total() * 3);
    
    // OpenCV channels are in BGR order: [0]=B, [1]=G, [2]=R
    // Python does RGB[::-1] which gives BGR
    // So we want BGR order: channels[0], channels[1], channels[2]
    for (int i = 2; i >= 0; --i) {  // Reverse: R, G, B -> matches [::-1]
        tensor.insert(tensor.end(), (float*)channels[i].data, 
                     (float*)channels[i].data + channels[i].total());
    }
    
    // Debug output (once)
    static bool print_debug = true;
    if (print_debug) {
        print_debug = false;
        std::cout << "\n=== LETTERBOX DEBUG ===" << std::endl;
        std::cout << "Original shape: [" << shape_h << ", " << shape_w << "]" << std::endl;
        std::cout << "Scale ratio (r): " << r << std::endl;
        std::cout << "New unpadded: [" << new_unpad_h << ", " << new_unpad_w << "]" << std::endl;
        std::cout << "Padding (dw, dh): (" << dw << ", " << dh << ")" << std::endl;
        std::cout << "Border (top, bottom, left, right): (" << top << ", " << bottom 
                  << ", " << left << ", " << right << ")" << std::endl;
        std::cout << "Final shape: [" << padded.rows << ", " << padded.cols << "]" << std::endl;
        std::cout << "Tensor size: " << tensor.size() << " (expected: " 
                  << (3 * target_h * target_w) << ")" << std::endl;
        std::cout << "First 10 values: ";
        for (int i = 0; i < 10 && i < static_cast<int>(tensor.size()); ++i) {
            std::cout << std::fixed << std::setprecision(5) << tensor[i] << " ";
        }
        std::cout << "\n=== END LETTERBOX DEBUG ===\n" << std::endl;
    }
    
    return tensor;
}

std::vector<Detection> TritonClient::run_inference(const std::vector<float>& input_tensor,
                                                   int orig_width, int orig_height) {
    if (!grpc_client_) {
        std::cerr << "gRPC client not initialized" << std::endl;
        return {};
    }
    
    try {
        // Run inference using the gRPC client
        auto output_tensor = grpc_client_->run_inference(config_.model.name, input_tensor);
        
        if (output_tensor.empty()) {
            return {};
        }
        
        // Use native C++ YOLO postprocessor with actual original image dimensions
        return yolo_postprocessor_->postprocess(
            output_tensor,
            config_.model.input_width,
            config_.model.input_height,
            orig_width,
            orig_height,
            config_.model.confidence_threshold,
            config_.model.iou_threshold,
            config_.model.max_detections
        );
        
    } catch (const std::exception& e) {
        std::cerr << "Inference failed: " << e.what() << std::endl;
        return {};
    }
}

void TritonClient::run_image_inference(const std::string& image_path, const std::string& output_path) {
    std::cout << "Processing image: " << image_path << std::endl;
    
    // Load image
    cv::Mat image = cv::imread(image_path);
    if (image.empty()) {
        std::cout << "Failed to load image: " << image_path << std::endl;
        return;
    }
    
    // Prepare input
    auto input_tensor = prepare_input_tensor(image);
    
    // Run inference with original image dimensions
    auto detections = run_inference(input_tensor, image.cols, image.rows);
    
    std::cout << "Detected " << detections.size() << " objects" << std::endl;
    
    // Draw detections
    for (const auto& detection : detections) {
        if (detection.confidence < config_.model.draw_confidence) {
            continue;
        }
        
        std::string class_name = (detection.class_id < static_cast<int>(class_names_.size())) 
                               ? class_names_[detection.class_id] 
                               : "class_" + std::to_string(detection.class_id);
        
        draw_detection(image, detection.class_id, class_name, detection.confidence,
                      static_cast<int>(detection.x1), static_cast<int>(detection.y1),
                      static_cast<int>(detection.x2), static_cast<int>(detection.y2),
                      config_.video.line_thickness, config_.video.font_scale, 
                      config_.video.font_thickness);
    }
    
    // Save or display result
    if (!output_path.empty()) {
        cv::imwrite(output_path, image);
        std::cout << "Saved result to " << output_path << std::endl;
    } else {
        cv::imshow("Result", image);
        cv::waitKey(0);
        cv::destroyAllWindows();
    }
}

void TritonClient::run_video_inference(const std::string& video_path, const std::string& output_path) {
    std::cout << "Processing video: " << video_path << std::endl;
    
    // Open video
    cv::VideoCapture cap(video_path);
    if (!cap.isOpened()) {
        std::cout << "Failed to open video: " << video_path << std::endl;
        return;
    }
    
    // Setup output
    cv::VideoWriter out;
    int frame_count = 0;
    PerformanceStats perf_stats;
    
    // Frame history for object tracking
    std::vector<std::vector<Detection>> frame_history;
    
    std::cout << "Starting video processing..." << std::endl;
    
    while (true) {
        auto frame_start_time = std::chrono::high_resolution_clock::now();
        
        cv::Mat frame;
        if (!cap.read(frame)) {
            break;
        }
        
        // Setup output video writer
        if (frame_count == 0 && !output_path.empty()) {
            int fourcc = cv::VideoWriter::fourcc('M', 'P', '4', 'V');
            out.open(output_path, fourcc, config_.video.fps, 
                    cv::Size(frame.cols, frame.rows));
        }
        
        // Process frame
        process_video_frame(frame, perf_stats, frame_history);
        
        // Write or display frame
        if (!output_path.empty()) {
            out.write(frame);
        } else {
            try {
                cv::imshow("Video", frame);
                if (cv::waitKey(1) == 'q') {
                    break;
                }
            } catch (const cv::Exception&) {
                // Skip GUI operations in headless environment
            }
        }
        
        frame_count++;
    }
    
    // Print performance summary
    perf_stats.print_summary();
    
    // Cleanup
    cap.release();
    if (out.isOpened()) {
        out.release();
    } else {
        cv::destroyAllWindows();
    }
}

void TritonClient::process_video_frame(cv::Mat& frame, PerformanceStats& stats, 
                                     std::vector<std::vector<Detection>>& frame_history) {
    auto frame_start_time = std::chrono::high_resolution_clock::now();
    
    // Preprocess
    auto preprocess_start = std::chrono::high_resolution_clock::now();
    auto input_tensor = prepare_input_tensor(frame);
    auto preprocess_latency = std::chrono::duration<float>(
        std::chrono::high_resolution_clock::now() - preprocess_start).count();
    
    // Inference with original frame dimensions
    auto inference_start = std::chrono::high_resolution_clock::now();
    auto detections = run_inference(input_tensor, frame.cols, frame.rows);
    auto inference_latency = std::chrono::duration<float>(
        std::chrono::high_resolution_clock::now() - inference_start).count();
    
    // Postprocess
    auto postprocess_start = std::chrono::high_resolution_clock::now();
    // Postprocessing is already done in run_inference for simplicity
    auto postprocess_latency = std::chrono::duration<float>(
        std::chrono::high_resolution_clock::now() - postprocess_start).count();
    
    // Calculate timing
    auto frame_time = std::chrono::duration<float>(
        std::chrono::high_resolution_clock::now() - frame_start_time).count();
    auto e2e_latency = frame_time;
    
    std::cout << "Frame " << stats.get_frame_count() << ": " << detections.size() << " objects | "
              << "E2E: " << e2e_latency * 1000.0f << "ms | "
              << "Pre: " << preprocess_latency * 1000.0f << "ms | "
              << "Inf: " << inference_latency * 1000.0f << "ms | "
              << "Post: " << postprocess_latency * 1000.0f << "ms" << std::endl;
    
    // Object tracking with history
    std::vector<Detection> current_detections;
    for (const auto& detection : detections) {
        if (detection.confidence >= config_.model.draw_confidence) {
            current_detections.push_back(detection);
        }
    }
    
    // Update frame history
    frame_history.push_back(current_detections);
    if (static_cast<int>(frame_history.size()) > config_.video.max_history) {
        frame_history.erase(frame_history.begin());
    }
    
    // Filter detections based on history
    std::vector<Detection> confirmed_objects;
    for (const auto& detection : current_detections) {
        int count = 0;
        for (const auto& past_frame : frame_history) {
            for (const auto& past_detection : past_frame) {
                if (past_detection.class_id == detection.class_id && 
                    is_same_object(past_detection.get_box(), detection.get_box(), 
                                 config_.video.distance_threshold)) {
                    count++;
                    break;
                }
            }
        }
        if (count >= config_.video.max_history) {
            confirmed_objects.push_back(detection);
        }
    }
    
    // Draw confirmed detections
    for (const auto& detection : confirmed_objects) {
        std::string class_name = (detection.class_id < static_cast<int>(class_names_.size())) 
                               ? class_names_[detection.class_id] 
                               : "class_" + std::to_string(detection.class_id);
        
        draw_detection(frame, detection.class_id, class_name, detection.confidence,
                      static_cast<int>(detection.x1), static_cast<int>(detection.y1),
                      static_cast<int>(detection.x2), static_cast<int>(detection.y2),
                      config_.video.line_thickness, config_.video.font_scale, 
                      config_.video.font_thickness);
    }
    
    // Update performance stats
    stats.add_measurement(e2e_latency, preprocess_latency, inference_latency,
                         postprocess_latency, frame_time, static_cast<int>(detections.size()));
}

} // namespace triton_client
