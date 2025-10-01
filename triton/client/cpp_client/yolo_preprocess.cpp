/**
 * @file yolo_preprocess.cpp
 * @brief Implementation of YOLO LetterBox preprocessing
 * 
 * Reference: ultralytics.data.augment.LetterBox
 * https://github.com/ultralytics/ultralytics/blob/main/ultralytics/data/augment.py
 */

#include "yolo_preprocess.h"
#include <iostream>
#include <iomanip>
#include <stdexcept>

namespace triton_client {

YoloPreprocessor::YoloPreprocessor(int input_width, int input_height,
                                   int padding_value, bool normalize)
    : input_width_(input_width)
    , input_height_(input_height)
    , padding_value_(padding_value)
    , normalize_(normalize)
    , last_scale_ratio_(1.0f)
    , last_pad_w_(0.0f)
    , last_pad_h_(0.0f)
    , debug_enabled_(false)
    , debug_printed_(false)
{
}

float YoloPreprocessor::calculate_scale_ratio(int img_width, int img_height) const {
    // Calculate scale ratio: r = min(target_h / img_h, target_w / img_w)
    // This ensures the image fits within the target size while preserving aspect ratio
    float ratio_h = static_cast<float>(input_height_) / static_cast<float>(img_height);
    float ratio_w = static_cast<float>(input_width_) / static_cast<float>(img_width);
    return std::min(ratio_h, ratio_w);
}

std::vector<float> YoloPreprocessor::preprocess(const cv::Mat& image) {
    // Validate input
    if (image.empty()) {
        throw std::runtime_error("YoloPreprocessor: Input image is empty");
    }
    
    // ============================================================
    // STEP 1: Calculate scaling and padding
    // ============================================================
    
    int shape_h = image.rows;  // Original height
    int shape_w = image.cols;  // Original width
    
    // Calculate scale ratio to fit into target size
    // Reference: r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    last_scale_ratio_ = calculate_scale_ratio(shape_w, shape_h);
    
    // Compute new unpadded size after scaling
    // Reference: new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    int new_unpad_w = static_cast<int>(std::round(shape_w * last_scale_ratio_));
    int new_unpad_h = static_cast<int>(std::round(shape_h * last_scale_ratio_));
    
    // Compute total padding needed
    // Reference: dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    float dw = input_width_ - new_unpad_w;   // Width padding
    float dh = input_height_ - new_unpad_h;  // Height padding
    
    // Divide padding to both sides (center=True)
    // Reference: dw /= 2, dh /= 2
    dw /= 2.0f;
    dh /= 2.0f;
    
    last_pad_w_ = dw;
    last_pad_h_ = dh;
    
    // ============================================================
    // STEP 2: Resize image with aspect ratio preservation
    // ============================================================
    
    cv::Mat resized;
    if (shape_h != new_unpad_h || shape_w != new_unpad_w) {
        // Resize using bilinear interpolation (cv2.INTER_LINEAR)
        cv::resize(image, resized, cv::Size(new_unpad_w, new_unpad_h), 
                   0, 0, cv::INTER_LINEAR);
    } else {
        // No resize needed, just clone
        resized = image.clone();
    }
    
    // ============================================================
    // STEP 3: Add border padding (LetterBox effect)
    // ============================================================
    
    // Calculate border values for each side
    // Reference: top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    //           left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    int top = static_cast<int>(std::round(dh - 0.1f));
    int bottom = static_cast<int>(std::round(dh + 0.1f));
    int left = static_cast<int>(std::round(dw - 0.1f));
    int right = static_cast<int>(std::round(dw + 0.1f));
    
    // Add border with gray padding (value=114)
    cv::Mat padded;
    cv::copyMakeBorder(resized, padded, top, bottom, left, right,
                       cv::BORDER_CONSTANT, 
                       cv::Scalar(padding_value_, padding_value_, padding_value_));
    
    // ============================================================
    // STEP 4: Normalize to [0, 1]
    // ============================================================
    
    cv::Mat float_img;
    if (normalize_) {
        // Convert to float and normalize: pixel / 255.0
        padded.convertTo(float_img, CV_32F, 1.0 / 255.0);
    } else {
        padded.convertTo(float_img, CV_32F);
    }
    
    // ============================================================
    // STEP 5: Transpose from HWC to CHW format
    // ============================================================
    
    // Split channels: [H, W, C] → 3 separate [H, W] mats
    // OpenCV stores in BGR order: channels[0]=B, channels[1]=G, channels[2]=R
    std::vector<cv::Mat> channels(3);
    cv::split(float_img, channels);
    
    // ============================================================
    // STEP 6: Flatten to CHW format with channel reversal
    // ============================================================
    
    // Reserve memory for efficiency
    std::vector<float> tensor;
    tensor.reserve(float_img.total() * 3);
    
    // Python: img.transpose((2, 0, 1))[::-1]
    // - transpose((2, 0, 1)): HWC → CHW
    // - [::-1]: Reverse channels (RGB → BGR or BGR → RGB)
    // 
    // OpenCV channels are BGR: [0]=B, [1]=G, [2]=R
    // We want to reverse to RGB for YOLO: R, G, B
    for (int i = 2; i >= 0; --i) {  // Iterate: 2, 1, 0 (R, G, B)
        float* data = (float*)channels[i].data;
        tensor.insert(tensor.end(), data, data + channels[i].total());
    }
    
    // ============================================================
    // DEBUG OUTPUT (optional, prints once)
    // ============================================================
    
    if (debug_enabled_ && !debug_printed_) {
        debug_printed_ = true;
        
        std::vector<float> sample_values;
        for (int i = 0; i < 10 && i < static_cast<int>(tensor.size()); ++i) {
            sample_values.push_back(tensor[i]);
        }
        
        print_debug_info(
            cv::Size(shape_w, shape_h),
            cv::Size(new_unpad_w, new_unpad_h),
            cv::Size(left + right, top + bottom),
            cv::Size(padded.cols, padded.rows),
            tensor.size(),
            sample_values
        );
    }
    
    // Verify output size
    size_t expected_size = 3 * input_height_ * input_width_;
    if (tensor.size() != expected_size) {
        throw std::runtime_error(
            "YoloPreprocessor: Output tensor size mismatch. Expected " +
            std::to_string(expected_size) + ", got " + std::to_string(tensor.size())
        );
    }
    
    return tensor;
}

void YoloPreprocessor::print_debug_info(const cv::Size& original_shape,
                                       const cv::Size& new_unpadded,
                                       const cv::Size& padding,
                                       const cv::Size& final_shape,
                                       size_t tensor_size,
                                       const std::vector<float>& sample_values) const {
    std::cout << "\n=== YOLO LETTERBOX PREPROCESSING DEBUG ===" << std::endl;
    std::cout << "Original shape: [H=" << original_shape.height 
              << ", W=" << original_shape.width << "]" << std::endl;
    std::cout << "Scale ratio: " << last_scale_ratio_ << std::endl;
    std::cout << "New unpadded: [H=" << new_unpadded.height 
              << ", W=" << new_unpadded.width << "]" << std::endl;
    std::cout << "Padding (dw, dh): (" << last_pad_w_ << ", " << last_pad_h_ << ")" << std::endl;
    std::cout << "Border padding: (total_w=" << padding.width 
              << ", total_h=" << padding.height << ")" << std::endl;
    std::cout << "Final shape: [H=" << final_shape.height 
              << ", W=" << final_shape.width << "]" << std::endl;
    std::cout << "Tensor size: " << tensor_size 
              << " (expected: " << (3 * input_height_ * input_width_) << ")" << std::endl;
    std::cout << "First 10 values: ";
    for (float val : sample_values) {
        std::cout << std::fixed << std::setprecision(5) << val << " ";
    }
    std::cout << "\n=== END LETTERBOX DEBUG ===\n" << std::endl;
}

} // namespace triton_client

