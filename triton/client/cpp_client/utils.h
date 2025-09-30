#pragma once

#include <vector>
#include <string>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>

namespace triton_client {

struct Detection {
    int class_id;
    float confidence;
    float x1, y1, x2, y2;
    
    Detection(int cls_id, float conf, float x1_, float y1_, float x2_, float y2_)
        : class_id(cls_id), confidence(conf), x1(x1_), y1(y1_), x2(x2_), y2(y2_) {}
    
    std::pair<float, float> get_center() const {
        return {(x1 + x2) / 2.0f, (y1 + y2) / 2.0f};
    }
    
    std::tuple<float, float, float, float> get_box() const {
        return {x1, y1, x2, y2};
    }
};

class PerformanceStats {
public:
    PerformanceStats() : start_time_(std::chrono::high_resolution_clock::now()) {}
    
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
    
    float get_fps() const {
        if (frame_times_.empty()) return 0.0f;
        float total_time = 0.0f;
        for (float time : frame_times_) {
            total_time += time;
        }
        return frame_times_.size() / total_time;
    }
    
    float get_avg_latency(const std::vector<float>& latencies) const {
        if (latencies.empty()) return 0.0f;
        float sum = 0.0f;
        for (float lat : latencies) {
            sum += lat;
        }
        return sum / latencies.size();
    }
    
    void print_summary() const;
    int get_frame_count() const { return frame_count_; }
    
private:
    std::vector<float> e2e_latencies_;
    std::vector<float> preprocess_latencies_;
    std::vector<float> inference_latencies_;
    std::vector<float> postprocess_latencies_;
    std::vector<float> frame_times_;
    std::chrono::high_resolution_clock::time_point start_time_;
    int frame_count_ = 0;
    int total_objects_ = 0;
};

// Image processing utilities
cv::Mat preprocess(const cv::Mat& frame, const std::pair<int, int>& new_shape);
std::vector<Detection> postprocess(const std::vector<float>& outputs, 
                                   const cv::Mat& input_image, 
                                   const cv::Mat& origin_image,
                                   float conf_threshold = 0.25f, 
                                   float iou_threshold = 0.65f, 
                                   int max_detections = 1024);

// Object tracking utilities
std::pair<float, float> get_center(float x1, float y1, float x2, float y2);
bool is_same_object(const std::tuple<float, float, float, float>& box1,
                   const std::tuple<float, float, float, float>& box2,
                   float distance_thresh = 50.0f);

// Visualization utilities
cv::Scalar get_color(int class_id);
void draw_bounding_box(cv::Mat& frame, int x1, int y1, int x2, int y2, 
                      const cv::Scalar& color, int thickness = 4);
void draw_label(cv::Mat& frame, const std::string& text, const cv::Point& position,
               const cv::Scalar& color, float font_scale = 0.6f, int thickness = 2);
void draw_detection(cv::Mat& frame, int class_id, const std::string& class_name,
                   float confidence, int x1, int y1, int x2, int y2,
                   int line_thickness = 4, float font_scale = 0.6f, int font_thickness = 2);
void draw_detections(cv::Mat& frame, const std::vector<Detection>& detections,
                    const std::vector<std::string>& class_names,
                    int line_thickness = 4, float font_scale = 0.6f, int font_thickness = 2);

} // namespace triton_client
