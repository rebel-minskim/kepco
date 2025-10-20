#include "utils.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <opencv2/imgproc.hpp>

namespace triton_client {

void PerformanceStats::print_summary() const {
    if (frame_count_ == 0) {
        std::cout << "No frames processed" << std::endl;
        return;
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto total_time = std::chrono::duration<float>(end_time - start_time_).count();
    float avg_fps = frame_count_ / total_time;
    
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "PERFORMANCE STATISTICS SUMMARY" << std::endl;
    std::cout << std::string(60, '=') << std::endl;
    std::cout << "Total Frames Processed: " << frame_count_ << std::endl;
    std::cout << "Total Processing Time: " << std::fixed << std::setprecision(2) << total_time << "s" << std::endl;
    std::cout << "Average FPS: " << std::fixed << std::setprecision(2) << avg_fps << std::endl;
    std::cout << "Total Objects Detected: " << total_objects_ << std::endl;
    std::cout << "Average Objects per Frame: " << std::fixed << std::setprecision(2) 
              << static_cast<float>(total_objects_) / frame_count_ << std::endl;
    
    std::cout << "\nLATENCY STATISTICS (ms):" << std::endl;
    std::cout << std::string(40, '-') << std::endl;
    
    float avg_e2e = get_avg_latency(e2e_latencies_) * 1000.0f;
    float avg_pre = get_avg_latency(preprocess_latencies_) * 1000.0f;
    float avg_inf = get_avg_latency(inference_latencies_) * 1000.0f;
    float avg_post = get_avg_latency(postprocess_latencies_) * 1000.0f;
    
    std::cout << std::left << std::setw(15) << "Metric" 
              << std::setw(10) << "Avg" << std::endl;
    std::cout << std::left << std::setw(15) << "E2E Latency" 
              << std::setw(10) << std::fixed << std::setprecision(2) << avg_e2e << std::endl;
    std::cout << std::left << std::setw(15) << "Preprocess" 
              << std::setw(10) << std::fixed << std::setprecision(2) << avg_pre << std::endl;
    std::cout << std::left << std::setw(15) << "Inference" 
              << std::setw(10) << std::fixed << std::setprecision(2) << avg_inf << std::endl;
    std::cout << std::left << std::setw(15) << "Postprocess" 
              << std::setw(10) << std::fixed << std::setprecision(2) << avg_post << std::endl;
    
    std::cout << "\nThroughput: " << std::fixed << std::setprecision(2) << avg_fps << " FPS" << std::endl;
    std::cout << std::string(60, '=') << std::endl;
}

cv::Mat preprocess(const cv::Mat& frame, const std::pair<int, int>& new_shape) {
    cv::Mat resized;
    cv::resize(frame, resized, cv::Size(new_shape.first, new_shape.second));
    
    // Convert BGR to RGB
    cv::Mat rgb;
    cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);
    
    // Convert to float and normalize
    cv::Mat float_img;
    rgb.convertTo(float_img, CV_32F, 1.0 / 255.0);
    
    return float_img;
}

std::vector<Detection> postprocess(const std::vector<float>& outputs, 
                                   const cv::Mat& input_image, 
                                   const cv::Mat& origin_image,
                                   float conf_threshold, 
                                   float iou_threshold, 
                                   int max_detections) {
    std::vector<Detection> detections;
    
    // Simple postprocessing - in a real implementation, you'd want to implement
    // proper NMS and coordinate scaling
    int num_detections = std::min(static_cast<int>(outputs.size() / 6), max_detections);
    
    for (int i = 0; i < num_detections; ++i) {
        int idx = i * 6;
        if (idx + 5 >= static_cast<int>(outputs.size())) break;
        
        float confidence = outputs[idx + 4];
        if (confidence < conf_threshold) continue;
        
        float x1 = outputs[idx];
        float y1 = outputs[idx + 1];
        float x2 = outputs[idx + 2];
        float y2 = outputs[idx + 3];
        int class_id = static_cast<int>(outputs[idx + 5]);
        
        // Scale coordinates back to original image size
        float scale_x = static_cast<float>(origin_image.cols) / input_image.cols;
        float scale_y = static_cast<float>(origin_image.rows) / input_image.rows;
        
        x1 *= scale_x;
        y1 *= scale_y;
        x2 *= scale_x;
        y2 *= scale_y;
        
        detections.emplace_back(class_id, confidence, x1, y1, x2, y2);
    }
    
    return detections;
}

std::pair<float, float> get_center(float x1, float y1, float x2, float y2) {
    return {(x1 + x2) / 2.0f, (y1 + y2) / 2.0f};
}

bool is_same_object(const std::tuple<float, float, float, float>& box1,
                   const std::tuple<float, float, float, float>& box2,
                   float distance_thresh) {
    auto [x1_1, y1_1, x2_1, y2_1] = box1;
    auto [x1_2, y1_2, x2_2, y2_2] = box2;
    
    auto [cx1, cy1] = get_center(x1_1, y1_1, x2_1, y2_1);
    auto [cx2, cy2] = get_center(x1_2, y1_2, x2_2, y2_2);
    
    float distance = std::sqrt((cx1 - cx2) * (cx1 - cx2) + (cy1 - cy2) * (cy1 - cy2));
    return distance < distance_thresh;
}

cv::Scalar get_color(int class_id) {
    // Color palette similar to the Python version
    static const std::vector<cv::Scalar> palette = {
        cv::Scalar(56, 56, 255),   // Red
        cv::Scalar(151, 157, 255), // Light red
        cv::Scalar(31, 112, 255),  // Orange
        cv::Scalar(29, 178, 255),  // Yellow
        cv::Scalar(49, 210, 207),  // Light green
        cv::Scalar(10, 249, 72),   // Green
        cv::Scalar(23, 204, 146),  // Teal
        cv::Scalar(134, 219, 61),  // Light teal
        cv::Scalar(52, 147, 26),   // Dark green
        cv::Scalar(187, 212, 0),   // Cyan
        cv::Scalar(168, 153, 44),  // Blue
        cv::Scalar(255, 194, 0),   // Light blue
        cv::Scalar(147, 69, 52),   // Purple
        cv::Scalar(255, 115, 100), // Light purple
        cv::Scalar(255, 140, 142), // Pink
        cv::Scalar(255, 173, 204), // Light pink
        cv::Scalar(189, 101, 255), // Magenta
        cv::Scalar(188, 50, 255),  // Light magenta
        cv::Scalar(181, 0, 255),   // Violet
        cv::Scalar(190, 60, 135)   // Dark violet
    };
    
    return palette[class_id % palette.size()];
}

void draw_bounding_box(cv::Mat& frame, int x1, int y1, int x2, int y2, 
                      const cv::Scalar& color, int thickness) {
    cv::rectangle(frame, cv::Point(x1, y1), cv::Point(x2, y2), color, thickness);
}

void draw_label(cv::Mat& frame, const std::string& text, const cv::Point& position,
               const cv::Scalar& color, float font_scale, int thickness) {
    cv::putText(frame, text, position, cv::FONT_HERSHEY_SIMPLEX, 
                font_scale, color, thickness, cv::LINE_AA);
}

void draw_detection(cv::Mat& frame, int class_id, const std::string& class_name,
                   float confidence, int x1, int y1, int x2, int y2,
                   int line_thickness, float font_scale, int font_thickness) {
    cv::Scalar color = get_color(class_id);
    
    // Draw bounding box
    draw_bounding_box(frame, x1, y1, x2, y2, color, line_thickness);
    
    // Draw label
    std::string label = class_name + ": " + std::to_string(confidence).substr(0, 4);
    cv::Point label_position(x1, y1 - 10);
    draw_label(frame, label, label_position, color, font_scale, font_thickness);
}

void draw_detections(cv::Mat& frame, const std::vector<Detection>& detections,
                    const std::vector<std::string>& class_names,
                    int line_thickness, float font_scale, int font_thickness) {
    for (const auto& detection : detections) {
        if (detection.confidence < 0.2f) continue;
        
        std::string class_name = (detection.class_id < static_cast<int>(class_names.size())) 
                               ? class_names[detection.class_id] 
                               : "class_" + std::to_string(detection.class_id);
        
        draw_detection(frame, detection.class_id, class_name, detection.confidence,
                      static_cast<int>(detection.x1), static_cast<int>(detection.y1),
                      static_cast<int>(detection.x2), static_cast<int>(detection.y2),
                      line_thickness, font_scale, font_thickness);
    }
}

} // namespace triton_client
