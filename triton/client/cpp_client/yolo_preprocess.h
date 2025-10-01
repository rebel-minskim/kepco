/**
 * @file yolo_preprocess.h
 * @brief YOLO-specific preprocessing implementation (LetterBox transform)
 * 
 * This file implements the LetterBox preprocessing technique used by Ultralytics
 * YOLO models. LetterBox preserves aspect ratio while resizing images to the
 * target input size by adding gray padding.
 * 
 * Reference Implementation:
 * - ultralytics.data.augment.LetterBox
 * - https://github.com/ultralytics/ultralytics/blob/main/ultralytics/data/augment.py
 * 
 * Key Features:
 * - Aspect ratio preservation (no distortion)
 * - Gray padding (value=114) for letterboxing
 * - Normalization to [0, 1] range
 * - HWC → CHW transpose
 * - Channel reversal for OpenCV BGR compatibility
 * 
 * Why LetterBox?
 * YOLO models are trained with LetterBox preprocessing. Using the same
 * preprocessing at inference ensures optimal detection accuracy.
 */

#pragma once

#include <opencv2/opencv.hpp>
#include <vector>

namespace triton_client {

/**
 * @class YoloPreprocessor
 * @brief Preprocesses images for YOLO inference using LetterBox transform
 * 
 * The LetterBox algorithm:
 * 1. Calculate scale ratio to fit image into target size while preserving aspect ratio
 * 2. Resize image to scaled size
 * 3. Add gray padding (value=114) to reach exact target dimensions
 * 4. Normalize pixel values to [0, 1]
 * 5. Transpose from HWC (Height, Width, Channels) to CHW format
 * 6. Reverse channel order for OpenCV BGR compatibility
 * 
 * Example:
 *   Input: 1920x1080 image → LetterBox → Output: 640x640 tensor
 *   - Scale ratio: 0.333
 *   - Resized: 640x360
 *   - Padding: top=140, bottom=140, left=0, right=0
 */
class YoloPreprocessor {
public:
    /**
     * @brief Construct a new YoloPreprocessor
     * @param input_width Target width for model input (default: 640)
     * @param input_height Target height for model input (default: 640)
     * @param padding_value Gray value for padding (default: 114)
     * @param normalize Whether to normalize to [0,1] (default: true)
     */
    explicit YoloPreprocessor(int input_width = 640, 
                             int input_height = 640,
                             int padding_value = 114,
                             bool normalize = true);
    
    /**
     * @brief Apply LetterBox preprocessing to an image
     * @param image Input image in BGR format (OpenCV default)
     * @return Preprocessed tensor as flat vector [C, H, W] format
     * 
     * Output tensor properties:
     * - Shape: [3, input_height, input_width]
     * - Data type: float32
     * - Value range: [0.0, 1.0] (normalized)
     * - Channel order: RGB (reversed from OpenCV's BGR)
     * - Layout: CHW (channels first)
     * 
     * Performance: ~4ms per frame on 1080p image
     * 
     * @throws std::runtime_error if image is empty or preprocessing fails
     */
    std::vector<float> preprocess(const cv::Mat& image);
    
    /**
     * @brief Get the scale ratio used in last preprocessing
     * @return Scale factor applied to image
     * 
     * Useful for coordinate scaling in postprocessing.
     */
    float get_scale_ratio() const { return last_scale_ratio_; }
    
    /**
     * @brief Get the padding applied in last preprocessing
     * @return (pad_width, pad_height) in pixels
     * 
     * Useful for coordinate scaling in postprocessing.
     */
    std::pair<float, float> get_padding() const { 
        return {last_pad_w_, last_pad_h_}; 
    }
    
    /**
     * @brief Enable or disable debug output
     * @param enabled If true, prints preprocessing details once
     */
    void set_debug(bool enabled) { debug_enabled_ = enabled; }
    
private:
    // Configuration
    int input_width_;        ///< Target width (e.g., 640)
    int input_height_;       ///< Target height (e.g., 640)
    int padding_value_;      ///< Gray value for padding (114)
    bool normalize_;         ///< Normalize to [0, 1]
    
    // State from last preprocessing
    float last_scale_ratio_; ///< Scale factor used
    float last_pad_w_;       ///< Width padding applied
    float last_pad_h_;       ///< Height padding applied
    
    // Debug
    bool debug_enabled_;     ///< Print debug info
    bool debug_printed_;     ///< Debug already printed once
    
    /**
     * @brief Calculate optimal scale ratio for LetterBox
     * @param img_width Original image width
     * @param img_height Original image height
     * @return Scale ratio that fits image into target size
     */
    float calculate_scale_ratio(int img_width, int img_height) const;
    
    /**
     * @brief Print debug information about preprocessing
     * @param original_shape Original image dimensions
     * @param new_unpadded New size after scaling
     * @param padding Padding values
     * @param final_shape Final tensor shape
     * @param tensor_size Total tensor size
     * @param sample_values First few tensor values
     */
    void print_debug_info(const cv::Size& original_shape,
                         const cv::Size& new_unpadded,
                         const cv::Size& padding,
                         const cv::Size& final_shape,
                         size_t tensor_size,
                         const std::vector<float>& sample_values) const;
};

} // namespace triton_client

