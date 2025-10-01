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
#include <queue>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <opencv2/imgproc.hpp>

namespace triton_client {

TritonClient::TritonClient(const ClientConfig& config) : config_(config) {
    load_class_names();
    grpc_client_ = std::make_unique<GrpcClient>(config_.server.url);
    yolo_preprocessor_ = std::make_unique<YoloPreprocessor>(
        config_.model.input_width, 
        config_.model.input_height
    );
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
    // Use YoloPreprocessor for LetterBox preprocessing
    // This is now a thin wrapper around the dedicated preprocessor class
    return yolo_preprocessor_->preprocess(image);
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

// Parallel video inference with multi-threaded pipeline
void TritonClient::run_video_inference_parallel(const std::string& video_path, 
                                               const std::string& output_path,
                                               int num_threads) {
    std::cout << "Processing video (PARALLEL): " << video_path << std::endl;
    std::cout << "Using " << num_threads << " inference threads" << std::endl;
    
    // Thread-safe queues
    struct FrameData {
        cv::Mat frame;
        int frame_idx;
        int orig_width;
        int orig_height;
    };
    
    struct PreprocessedData {
        std::vector<float> tensor;
        cv::Mat frame;  // Keep frame for drawing later
        int frame_idx;
        int orig_width;
        int orig_height;
    };
    
    struct InferenceResult {
        std::vector<Detection> detections;
        cv::Mat frame;  // Frame with detections
        int frame_idx;
    };
    
    std::queue<FrameData> read_queue;
    std::queue<PreprocessedData> preprocess_queue;
    std::queue<InferenceResult> inference_queue;
    std::queue<std::pair<cv::Mat, std::vector<Detection>>> draw_queue;
    
    std::mutex read_mutex, preprocess_mutex, inference_mutex, draw_mutex;
    std::condition_variable read_cv, preprocess_cv, inference_cv, draw_cv;
    
    std::atomic<bool> reading_done{false};
    std::atomic<bool> preprocessing_done{false};
    std::atomic<bool> inference_done{false};
    std::atomic<int> frames_read{0};
    std::atomic<int> frames_processed{0};
    
    const int queue_size = num_threads * 2;
    
    cv::VideoCapture cap(video_path);
    if (!cap.isOpened()) {
        std::cerr << "Failed to open video: " << video_path << std::endl;
        return;
    }
    
    int frame_width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int frame_height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    double fps = cap.get(cv::CAP_PROP_FPS);
    int total_frames = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));
    
    std::cout << "Video: " << frame_width << "x" << frame_height 
              << " @ " << fps << " FPS, " << total_frames << " frames" << std::endl;
    
    cv::VideoWriter writer;
    if (!output_path.empty()) {
        int fourcc = cv::VideoWriter::fourcc('m', 'p', '4', 'v');
        writer.open(output_path, fourcc, fps, cv::Size(frame_width, frame_height));
        if (!writer.isOpened()) {
            std::cerr << "Failed to create video writer" << std::endl;
            return;
        }
    }
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Thread 1: Read frames
    std::thread reader([&]() {
        cv::Mat frame;
        int idx = 0;
        while (cap.read(frame)) {
            {
                std::unique_lock<std::mutex> lock(read_mutex);
                read_cv.wait(lock, [&]() { return read_queue.size() < queue_size; });
                read_queue.push({frame.clone(), idx, frame.cols, frame.rows});
                idx++;
                frames_read = idx;  // Update incrementally for progress display
            }
            read_cv.notify_one();
        }
        reading_done = true;
        read_cv.notify_all();
    });
    
    // Thread 2: Preprocess
    std::thread preprocessor([&]() {
        while (true) {
            FrameData data;
            {
                std::unique_lock<std::mutex> lock(read_mutex);
                read_cv.wait(lock, [&]() { return !read_queue.empty() || reading_done; });
                if (read_queue.empty() && reading_done) break;
                if (read_queue.empty()) continue;
                
                data = std::move(read_queue.front());
                read_queue.pop();
            }
            read_cv.notify_one();
            
            auto tensor = prepare_input_tensor(data.frame);
            
            {
                std::unique_lock<std::mutex> lock(preprocess_mutex);
                preprocess_cv.wait(lock, [&]() { return preprocess_queue.size() < queue_size; });
                preprocess_queue.push({std::move(tensor), data.frame, data.frame_idx, data.orig_width, data.orig_height});
            }
            preprocess_cv.notify_one();
        }
        preprocessing_done = true;
        preprocess_cv.notify_all();
    });
    
    // Threads 3-N: Inference workers
    std::vector<std::thread> inference_workers;
    for (int i = 0; i < num_threads; ++i) {
        inference_workers.emplace_back([&]() {
            while (true) {
                PreprocessedData data;
                {
                    std::unique_lock<std::mutex> lock(preprocess_mutex);
                    preprocess_cv.wait(lock, [&]() { return !preprocess_queue.empty() || preprocessing_done; });
                    if (preprocess_queue.empty() && preprocessing_done) break;
                    if (preprocess_queue.empty()) continue;
                    
                    data = std::move(preprocess_queue.front());
                    preprocess_queue.pop();
                }
                preprocess_cv.notify_one();
                
                auto detections = run_inference(data.tensor, data.orig_width, data.orig_height);
                
                {
                    std::unique_lock<std::mutex> lock(inference_mutex);
                    inference_cv.wait(lock, [&]() { return inference_queue.size() < queue_size * 2; });
                    inference_queue.push({std::move(detections), data.frame, data.frame_idx});
                }
                inference_cv.notify_one();
            }
        });
    }
    
    // Thread N+1: Draw and write (must be sequential for video output)
    std::thread drawer([&]() {
        int expected_frame = 0;
        std::map<int, InferenceResult> buffer;
        
        while (true) {
            InferenceResult result;
            {
                std::unique_lock<std::mutex> lock(inference_mutex);
                inference_cv.wait(lock, [&]() { 
                    return !inference_queue.empty() || inference_done; 
                });
                
                // Exit when all inference workers are done AND queue is empty AND buffer is empty
                if (inference_queue.empty() && inference_done && buffer.empty()) {
                    break;
                }
                if (inference_queue.empty()) continue;
                
                result = std::move(inference_queue.front());
                inference_queue.pop();
            }
            inference_cv.notify_one();
            
            buffer[result.frame_idx] = std::move(result);
            
            // Process frames in order
            while (buffer.count(expected_frame)) {
                auto& res = buffer[expected_frame];
                
                // Draw detections on the frame we already have
                cv::Mat frame = res.frame.clone();
                for (const auto& det : res.detections) {
                    if (det.confidence < config_.model.draw_confidence) continue;
                    
                    std::string class_name = (det.class_id < static_cast<int>(class_names_.size()))
                                           ? class_names_[det.class_id]
                                           : "class_" + std::to_string(det.class_id);
                    
                    draw_detection(frame, det.class_id, class_name, det.confidence,
                                 static_cast<int>(det.x1), static_cast<int>(det.y1),
                                 static_cast<int>(det.x2), static_cast<int>(det.y2),
                                 config_.video.line_thickness, config_.video.font_scale,
                                 config_.video.font_thickness);
                }
                
                if (writer.isOpened()) {
                    writer.write(frame);
                }
                
                frames_processed++;
                if (frames_processed % 30 == 0) {
                    auto elapsed = std::chrono::duration<float>(
                        std::chrono::high_resolution_clock::now() - start_time).count();
                    float current_fps = frames_processed / elapsed;
                    std::cout << "Processed " << frames_processed << "/" << frames_read 
                              << " frames | FPS: " << std::fixed << std::setprecision(2) 
                              << current_fps << std::endl;
                }
                
                buffer.erase(expected_frame);
                expected_frame++;
            }
        }
        
        // Process any remaining frames in buffer (shouldn't happen, but safety check)
        while (buffer.count(expected_frame)) {
            auto& res = buffer[expected_frame];
            cv::Mat frame = res.frame.clone();
            for (const auto& det : res.detections) {
                if (det.confidence < config_.model.draw_confidence) continue;
                std::string class_name = (det.class_id < static_cast<int>(class_names_.size()))
                                       ? class_names_[det.class_id]
                                       : "class_" + std::to_string(det.class_id);
                draw_detection(frame, det.class_id, class_name, det.confidence,
                             static_cast<int>(det.x1), static_cast<int>(det.y1),
                             static_cast<int>(det.x2), static_cast<int>(det.y2),
                             config_.video.line_thickness, config_.video.font_scale,
                             config_.video.font_thickness);
            }
            if (writer.isOpened()) {
                writer.write(frame);
            }
            frames_processed++;
            buffer.erase(expected_frame);
            expected_frame++;
        }
    });
    
    // Wait for all threads
    reader.join();
    preprocessor.join();
    for (auto& worker : inference_workers) {
        worker.join();
    }
    inference_done = true;
    inference_cv.notify_all();
    drawer.join();
    
    auto end_time = std::chrono::high_resolution_clock::now();
    float total_time = std::chrono::duration<float>(end_time - start_time).count();
    
    std::cout << "\n============================================================" << std::endl;
    std::cout << "PARALLEL PROCESSING SUMMARY" << std::endl;
    std::cout << "============================================================" << std::endl;
    std::cout << "Total frames: " << frames_processed << std::endl;
    std::cout << "Total time: " << std::fixed << std::setprecision(2) << total_time << "s" << std::endl;
    std::cout << "Average FPS: " << std::fixed << std::setprecision(2) 
              << (frames_processed / total_time) << std::endl;
    std::cout << "Inference threads: " << num_threads << std::endl;
    std::cout << "============================================================" << std::endl;
    
    if (writer.isOpened()) {
        writer.release();
    }
}

} // namespace triton_client
