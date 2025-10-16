/**
 * @file utils.h
 * @brief Utility classes and functions for object detection and visualization
 * 
 * Contains:
 * - Detection: Bounding box data structure
 * - PerformanceStats: FPS and latency tracking
 * - Visualization functions: Drawing bounding boxes and labels
 * - Object tracking utilities (for future multi-object tracking)
 */

#pragma once

#include <vector>
#include <string>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>

namespace triton_client {

/**
 * @struct Detection
 * @brief Represents a single object detection with bounding box and metadata
 * 
 * Coordinate format: (x1, y1, x2, y2) in xyxy format
 * - (x1, y1): Top-left corner
 * - (x2, y2): Bottom-right corner
 * 
 * Example:
 *   Detection det(0, 0.95, 100.0, 150.0, 200.0, 300.0);
 *   // Class ID=0 (person), 95% confidence, box at (100,150)-(200,300)
 */
struct Detection {
    int class_id;       ///< Class ID (e.g., 0=person, 1=bicycle, 2=car)
    float confidence;   ///< Detection confidence score [0.0, 1.0]
    float x1, y1;       ///< Top-left corner coordinates
    float x2, y2;       ///< Bottom-right corner coordinates
    
    /**
     * @brief Construct a Detection object
     * @param cls_id Class ID
     * @param conf Confidence score
     * @param x1_ Top-left x coordinate
     * @param y1_ Top-left y coordinate
     * @param x2_ Bottom-right x coordinate
     * @param y2_ Bottom-right y coordinate
     */
    Detection(int cls_id, float conf, float x1_, float y1_, float x2_, float y2_)
        : class_id(cls_id), confidence(conf), x1(x1_), y1(y1_), x2(x2_), y2(y2_) {}
    
    /**
     * @brief Get center point of bounding box
     * @return (center_x, center_y) as std::pair
     */
    std::pair<float, float> get_center() const {
        return {(x1 + x2) / 2.0f, (y1 + y2) / 2.0f};
    }
    
    /**
     * @brief Get bounding box coordinates as tuple
     * @return (x1, y1, x2, y2) as std::tuple
     */
    std::tuple<float, float, float, float> get_box() const {
        return {x1, y1, x2, y2};
    }
};

/**
 * @class PerformanceStats
 * @brief Tracks and reports inference performance metrics
 * 
 * Measures latency at each pipeline stage:
 * - E2E (End-to-End): Total time from frame input to detection output
 * - Preprocessing: Image resize, padding, normalization
 * - Inference: gRPC call + model execution on GPU
 * - Postprocessing: Bbox decode, NMS, coordinate scaling
 * 
 * Also tracks:
 * - FPS (Frames Per Second)
 * - Object detection counts
 * - Per-frame timing
 */
class PerformanceStats {
public:
    /**
     * @brief Initialize performance tracker with current timestamp
     */
    PerformanceStats() : start_time_(std::chrono::high_resolution_clock::now()) {}
    
    /**
     * @brief Add measurements for a single frame
     * @param e2e_lat End-to-end latency (ms)
     * @param pre_lat Preprocessing latency (ms)
     * @param inf_lat Inference latency (ms)
     * @param post_lat Postprocessing latency (ms)
     * @param frame_time Frame processing time in seconds
     * @param obj_count Number of objects detected in this frame
     */
    void add_measurement(float e2e_lat, float pre_lat, float inf_lat, 
                        float post_lat, float frame_time, int obj_count) {
        e2e_latencies_.push_back(e2e_lat);
        preprocess_latencies_.push_back(pre_lat);
        inference_latencies_.push_back(inf_lat);
        postprocess_latencies_.push_back(post_lat);
        frame_times_.push_back(frame_time);
        frame_count_++;
        total_objects_ += obj_count;
    }
    
    /**
     * @brief Calculate average FPS across all frames
     * @return Frames per second (FPS)
     */
    float get_fps() const {
        if (frame_times_.empty()) return 0.0f;
        float total_time = 0.0f;
        for (float time : frame_times_) {
            total_time += time;
        }
        return frame_times_.size() / total_time;
    }
    
    /**
     * @brief Calculate average latency from a latency vector
     * @param latencies Vector of latency measurements (ms)
     * @return Average latency (ms)
     */
    float get_avg_latency(const std::vector<float>& latencies) const {
        if (latencies.empty()) return 0.0f;
        float sum = 0.0f;
        for (float lat : latencies) {
            sum += lat;
        }
        return sum / latencies.size();
    }
    
    /**
     * @brief Print formatted performance summary to console
     * 
     * Outputs:
     * - Total frames processed
     * - Total processing time
     * - Average FPS
     * - Total/average object count
     * - Latency breakdown (E2E, Pre, Inf, Post)
     */
    void print_summary() const;
    
    /**
     * @brief Get total number of frames processed
     * @return Frame count
     */
    int get_frame_count() const { return frame_count_; }
    
private:
    std::vector<float> e2e_latencies_;          ///< End-to-end latencies (ms)
    std::vector<float> preprocess_latencies_;   ///< Preprocessing latencies (ms)
    std::vector<float> inference_latencies_;    ///< Inference latencies (ms)
    std::vector<float> postprocess_latencies_;  ///< Postprocessing latencies (ms)
    std::vector<float> frame_times_;            ///< Per-frame processing times (seconds)
    std::chrono::high_resolution_clock::time_point start_time_;  ///< Session start time
    int frame_count_ = 0;                       ///< Total frames processed
    int total_objects_ = 0;                     ///< Total objects detected
};

// ============================================================
// IMAGE PROCESSING UTILITIES (Legacy - not currently used)
// ============================================================

/**
 * @brief Preprocess frame to target shape (legacy function)
 * @param frame Input image
 * @param new_shape Target (height, width)
 * @return Preprocessed image
 * @note This is not used in current implementation (LetterBox in triton_client.cpp is used instead)
 */
cv::Mat preprocess(const cv::Mat& frame, const std::pair<int, int>& new_shape);

/**
 * @brief Postprocess YOLO outputs (legacy function)
 * @note This is not used in current implementation (YoloPostprocessor class is used instead)
 */
std::vector<Detection> postprocess(const std::vector<float>& outputs, 
                                   const cv::Mat& input_image, 
                                   const cv::Mat& origin_image,
                                   float conf_threshold = 0.25f, 
                                   float iou_threshold = 0.65f, 
                                   int max_detections = 1024);

// ============================================================
// OBJECT TRACKING UTILITIES (For future use)
// ============================================================

/**
 * @brief Calculate center point of bounding box
 * @param x1 Top-left x
 * @param y1 Top-left y
 * @param x2 Bottom-right x
 * @param y2 Bottom-right y
 * @return (center_x, center_y)
 */
std::pair<float, float> get_center(float x1, float y1, float x2, float y2);

/**
 * @brief Check if two bounding boxes represent the same object
 * @param box1 First box (x1, y1, x2, y2)
 * @param box2 Second box (x1, y1, x2, y2)
 * @param distance_thresh Maximum center distance to consider same object (pixels)
 * @return true if boxes likely represent the same object
 * @note Used for simple frame-to-frame object tracking
 */
bool is_same_object(const std::tuple<float, float, float, float>& box1,
                   const std::tuple<float, float, float, float>& box2,
                   float distance_thresh = 50.0f);

// ============================================================
// VISUALIZATION UTILITIES
// ============================================================

/**
 * @brief Get consistent color for a class ID
 * @param class_id Class identifier
 * @return BGR color (cv::Scalar)
 * @note Uses predefined color palette for visual consistency
 */
cv::Scalar get_color(int class_id);

/**
 * @brief Draw a bounding box on frame
 * @param frame Image to draw on (modified in-place)
 * @param x1 Top-left x coordinate
 * @param y1 Top-left y coordinate
 * @param x2 Bottom-right x coordinate
 * @param y2 Bottom-right y coordinate
 * @param color Box color (BGR)
 * @param thickness Line thickness in pixels
 */
void draw_bounding_box(cv::Mat& frame, int x1, int y1, int x2, int y2, 
                      const cv::Scalar& color, int thickness = 4);

/**
 * @brief Draw text label on frame with anti-aliasing
 * @param frame Image to draw on (modified in-place)
 * @param text Label text (e.g., "person: 0.95")
 * @param position Top-left corner of text
 * @param color Text color (BGR)
 * @param font_scale Font size multiplier
 * @param thickness Text thickness
 */
void draw_label(cv::Mat& frame, const std::string& text, const cv::Point& position,
               const cv::Scalar& color, float font_scale = 0.6f, int thickness = 2);

/**
 * @brief Draw complete detection (box + label) on frame
 * @param frame Image to draw on (modified in-place)
 * @param class_id Class ID for color selection
 * @param class_name Human-readable class name (e.g., "person")
 * @param confidence Detection confidence [0.0, 1.0]
 * @param x1 Top-left x coordinate
 * @param y1 Top-left y coordinate
 * @param x2 Bottom-right x coordinate
 * @param y2 Bottom-right y coordinate
 * @param line_thickness Bounding box line thickness
 * @param font_scale Label font size
 * @param font_thickness Label text thickness
 * 
 * Label format: "class_name: 0.XX" (e.g., "person: 0.95")
 * Label position: Above bounding box, slightly offset
 */
void draw_detection(cv::Mat& frame, int class_id, const std::string& class_name,
                   float confidence, int x1, int y1, int x2, int y2,
                   int line_thickness = 4, float font_scale = 0.6f, int font_thickness = 2);

/**
 * @brief Draw multiple detections on frame
 * @param frame Image to draw on (modified in-place)
 * @param detections Vector of Detection objects
 * @param class_names Vector of class name strings (indexed by class_id)
 * @param line_thickness Bounding box line thickness
 * @param font_scale Label font size
 * @param font_thickness Label text thickness
 * 
 * @note Automatically filters out low-confidence detections (< 0.2)
 */
void draw_detections(cv::Mat& frame, const std::vector<Detection>& detections,
                    const std::vector<std::string>& class_names,
                    int line_thickness = 4, float font_scale = 0.6f, int font_thickness = 2);

} // namespace triton_client
