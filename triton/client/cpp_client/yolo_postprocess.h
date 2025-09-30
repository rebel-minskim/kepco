#pragma once

#include "utils.h"
#include <vector>
#include <algorithm>
#include <cmath>

namespace triton_client {

class YoloPostprocessor {
public:
    YoloPostprocessor(int num_classes = 9);
    
    /**
     * Postprocess YOLO output with NMS
     * 
     * @param raw_output Raw model output tensor (flattened)
     * @param input_width Model input width (e.g., 800)
     * @param input_height Model input height (e.g., 800)
     * @param orig_width Original image width
     * @param orig_height Original image height
     * @param conf_threshold Confidence threshold (e.g., 0.2)
     * @param iou_threshold IoU threshold for NMS (e.g., 0.65)
     * @param max_detections Maximum number of detections to return
     * @return Vector of detections after NMS
     */
    std::vector<Detection> postprocess(
        const std::vector<float>& raw_output,
        int input_width,
        int input_height,
        int orig_width,
        int orig_height,
        float conf_threshold = 0.2f,
        float iou_threshold = 0.65f,
        int max_detections = 1024
    );

private:
    int num_classes_;
    int num_features_;  // 4 (bbox) + num_classes
    
    struct RawDetection {
        float cx, cy, w, h;
        int class_id;
        float confidence;
        int box_index;
    };
    
    // Helper methods
    float calculate_iou(const Detection& a, const Detection& b);
    std::vector<Detection> apply_nms(std::vector<Detection>& detections, float iou_threshold);
    void scale_coordinates(Detection& det, int input_width, int input_height, 
                          int orig_width, int orig_height);
};

} // namespace triton_client
