#include "yolo_postprocess.h"
#include <iostream>
#include <cmath>

namespace triton_client {

YoloPostprocessor::YoloPostprocessor(int num_classes) 
    : num_classes_(num_classes), num_features_(4 + num_classes) {
}

std::vector<Detection> YoloPostprocessor::postprocess(
    const std::vector<float>& raw_output,
    int input_width,
    int input_height,
    int orig_width,
    int orig_height,
    float conf_threshold,
    float iou_threshold,
    int max_detections) {
    
    // Calculate number of anchor boxes
    int num_boxes = raw_output.size() / num_features_;
    
    if (raw_output.size() != static_cast<size_t>(num_boxes * num_features_)) {
        std::cerr << "Invalid output tensor size: " << raw_output.size() 
                  << " (expected multiple of " << num_features_ << ")" << std::endl;
        return {};
    }
    
    std::vector<Detection> detections;
    detections.reserve(num_boxes / 10);  // Reserve some space
    
    // Debug: Print stats for first few frames
    static int debug_frame_count = 0;
    bool print_debug = (debug_frame_count < 3);
    if (print_debug) debug_frame_count++;
    
    int passed_threshold_count = 0;
    float max_conf_seen = 0.0f;
    float min_conf_passed = 1.0f;
    
    // Debug: Check actual data layout
    if (print_debug && num_boxes > 0) {
        std::cout << "\n=== RAW DATA CHECK ===" << std::endl;
        std::cout << "First 20 values (what we think are cx for first 20 boxes):" << std::endl;
        for (int i = 0; i < 20; ++i) {
            std::cout << raw_output[0 * num_boxes + i] << " ";
        }
        std::cout << "\n\nValues at indices 0-12 (what we think is box 0's all features in row-major):" << std::endl;
        for (int i = 0; i < 13; ++i) {
            std::cout << raw_output[i] << " ";
        }
        std::cout << "\n\nValues at indices 52500-52512 (feature 4, first 13 boxes - should be class 0 confs):" << std::endl;
        for (int i = 0; i < 13; ++i) {
            std::cout << raw_output[52500 + i] << " ";
        }
        std::cout << "\n=== END RAW DATA ===" << std::endl << std::endl;
    }
    
    // Parse each anchor box
    // Following Ultralytics NMS logic from https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/nms.py
    // Input format: [1, num_features, num_boxes] = [1, 13, 13125]
    // After transpose: [1, num_boxes, num_features] = [1, 13125, 13]
    // Each detection: [cx, cy, w, h, class0_conf, class1_conf, ..., class8_conf]
    //
    // In flattened C-order array: [feat0_box0, feat0_box1, ..., feat0_boxN, feat1_box0, ...]
    // So for box i, feature f: index = f * num_boxes + i
    for (int i = 0; i < num_boxes; ++i) {
        // Read features in column-major layout (feature * num_boxes + box_index)
        float cx = raw_output[0 * num_boxes + i];
        float cy = raw_output[1 * num_boxes + i];
        float w = raw_output[2 * num_boxes + i];
        float h = raw_output[3 * num_boxes + i];
        
        // Find class with highest confidence
        // Following: conf, j = cls.max(1, keepdim=True)
        int best_class = 0;
        float confidence = raw_output[4 * num_boxes + i];
        
        for (int c = 1; c < num_classes_; ++c) {
            float conf = raw_output[(4 + c) * num_boxes + i];
            if (conf > confidence) {
                confidence = conf;
                best_class = c;
            }
        }
        
        // Note: This model outputs probabilities directly (0-1 range), not logits
        // So we don't apply sigmoid - the values are already normalized
        
        // Track confidence stats for debugging
        if (confidence > max_conf_seen) max_conf_seen = confidence;
        
        // Filter by confidence threshold
        if (confidence < conf_threshold) {
            continue;
        }
        
        passed_threshold_count++;
        if (confidence < min_conf_passed) min_conf_passed = confidence;
        
        // Following Ultralytics: prediction[..., :4] = xywh2xyxy(prediction[..., :4])
        // Convert from center format (cx, cy, w, h) to corner format (x1, y1, x2, y2)
        float x1 = cx - w / 2.0f;
        float y1 = cy - h / 2.0f;
        float x2 = cx + w / 2.0f;
        float y2 = cy + h / 2.0f;
        
        // Basic sanity check - skip clearly invalid boxes
        if (x2 <= x1 || y2 <= y1 || w <= 0 || h <= 0) {
            continue;
        }
        
        // Skip boxes that are too far out of bounds (but allow some leeway for NMS)
        if (x1 > input_width * 1.5f || y1 > input_height * 1.5f || 
            x2 < -input_width * 0.5f || y2 < -input_height * 0.5f) {
            continue;
        }
        
        // Create detection and scale coordinates to original image size
        Detection det(best_class, confidence, x1, y1, x2, y2);
        scale_coordinates(det, input_width, input_height, orig_width, orig_height);
        
        detections.push_back(det);
    }
    
    // Debug output
    if (print_debug) {
        std::cout << "\n=== YOLO Postprocessing Debug ===" << std::endl;
        std::cout << "Total anchor boxes: " << num_boxes << std::endl;
        std::cout << "Confidence threshold: " << conf_threshold << std::endl;
        std::cout << "IoU threshold: " << iou_threshold << std::endl;
        std::cout << "Max confidence seen: " << max_conf_seen << std::endl;
        std::cout << "Boxes passing threshold: " << passed_threshold_count << std::endl;
        std::cout << "Min confidence that passed: " << min_conf_passed << std::endl;
        std::cout << "After coordinate filtering: " << detections.size() << std::endl;
    }
    
    // Apply NMS to remove overlapping detections
    if (detections.size() > 1) {
        detections = apply_nms(detections, iou_threshold);
    }
    
    if (print_debug) {
        std::cout << "Detections after NMS: " << detections.size() << std::endl;
        std::cout << "=== End Debug ===" << std::endl << std::endl;
    }
    
    // Limit to max detections
    if (static_cast<int>(detections.size()) > max_detections) {
        // Sort by confidence descending
        std::partial_sort(detections.begin(), 
                         detections.begin() + max_detections,
                         detections.end(),
                         [](const Detection& a, const Detection& b) {
                             return a.confidence > b.confidence;
                         });
        detections.erase(detections.begin() + max_detections, detections.end());
    }
    
    return detections;
}

float YoloPostprocessor::calculate_iou(const Detection& a, const Detection& b) {
    // Calculate intersection
    float x1 = std::max(a.x1, b.x1);
    float y1 = std::max(a.y1, b.y1);
    float x2 = std::min(a.x2, b.x2);
    float y2 = std::min(a.y2, b.y2);
    
    if (x2 < x1 || y2 < y1) {
        return 0.0f;  // No intersection
    }
    
    float intersection = (x2 - x1) * (y2 - y1);
    
    // Calculate union
    float area_a = (a.x2 - a.x1) * (a.y2 - a.y1);
    float area_b = (b.x2 - b.x1) * (b.y2 - b.y1);
    float union_area = area_a + area_b - intersection;
    
    if (union_area <= 0.0f) {
        return 0.0f;
    }
    
    return intersection / union_area;
}

std::vector<Detection> YoloPostprocessor::apply_nms(
    std::vector<Detection>& detections, float iou_threshold) {
    
    // Sort by confidence descending
    std::sort(detections.begin(), detections.end(),
             [](const Detection& a, const Detection& b) {
                 return a.confidence > b.confidence;
             });
    
    std::vector<bool> suppressed(detections.size(), false);
    std::vector<Detection> result;
    result.reserve(detections.size());
    
    // For each detection (in order of decreasing confidence)
    for (size_t i = 0; i < detections.size(); ++i) {
        if (suppressed[i]) {
            continue;
        }
        
        result.push_back(detections[i]);
        
        // Suppress all detections with high IoU (across ALL classes)
        // This matches Ultralytics behavior - if boxes overlap significantly,
        // keep only the highest confidence one regardless of class
        for (size_t j = i + 1; j < detections.size(); ++j) {
            if (suppressed[j]) {
                continue;
            }
            
            float iou = calculate_iou(detections[i], detections[j]);
            if (iou > iou_threshold) {
                suppressed[j] = true;
            }
        }
    }
    
    return result;
}

void YoloPostprocessor::scale_coordinates(
    Detection& det, int input_width, int input_height, 
    int orig_width, int orig_height) {
    
    // Ultralytics scale_boxes implementation
    // Reference: ultralytics.utils.ops.scale_boxes
    
    // Calculate gain and padding (same logic as LetterBox)
    // gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])
    float gain = std::min(static_cast<float>(input_height) / static_cast<float>(orig_height),
                          static_cast<float>(input_width) / static_cast<float>(orig_width));
    
    // pad_x = round((img1_shape[1] - img0_shape[1] * gain) / 2 - 0.1)
    // pad_y = round((img1_shape[0] - img0_shape[0] * gain) / 2 - 0.1)
    float pad_x = std::round((input_width - orig_width * gain) / 2.0f - 0.1f);
    float pad_y = std::round((input_height - orig_height * gain) / 2.0f - 0.1f);
    
    // Remove padding from coordinates
    // boxes[..., 0] -= pad_x  # x padding
    // boxes[..., 1] -= pad_y  # y padding
    // boxes[..., 2] -= pad_x  # x padding (for x2 in xyxy format)
    // boxes[..., 3] -= pad_y  # y padding (for y2 in xyxy format)
    det.x1 -= pad_x;
    det.y1 -= pad_y;
    det.x2 -= pad_x;
    det.y2 -= pad_y;
    
    // Scale by inverse gain
    // boxes[..., :4] /= gain
    det.x1 /= gain;
    det.y1 /= gain;
    det.x2 /= gain;
    det.y2 /= gain;
    
    // Clip to image bounds (clip_boxes)
    det.x1 = std::max(0.0f, std::min(static_cast<float>(orig_width), det.x1));
    det.y1 = std::max(0.0f, std::min(static_cast<float>(orig_height), det.y1));
    det.x2 = std::max(0.0f, std::min(static_cast<float>(orig_width), det.x2));
    det.y2 = std::max(0.0f, std::min(static_cast<float>(orig_height), det.y2));
}

} // namespace triton_client
