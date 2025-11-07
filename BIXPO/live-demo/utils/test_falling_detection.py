"""
Test falling detection on a sample image
Visualizes detection results with bounding boxes
"""

import cv2
import numpy as np
import yaml
import sys
import os
import argparse

# Add parent directory to path to import from app_web
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import rebel

# Configuration (same as app_web.py)
CONFIG = {
    'model_path': '../models/yolov11.rbln',
    'confidence': 0.25,
    'iou_threshold': 0.45,
    'model_input_size': 800,
    'class_confidence': {
        8: 0.0001,  # falling - very low threshold
    },
}

# Load class names
def load_class_names(yaml_path="config/data.yaml"):
    try:
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)
        if 'names' in data and isinstance(data['names'], dict):
            num_classes = data.get('nc', len(data['names']))
            class_names = ['unknown'] * num_classes
            for idx, name in data['names'].items():
                if idx < num_classes:
                    class_names[idx] = name
            return class_names
    except:
        pass
    return []

CLASS_NAMES = load_class_names()

# Preprocessing (same as app_web.py)
def preprocess_image(image, input_size=None):
    if input_size is None:
        input_size = (CONFIG['model_input_size'], CONFIG['model_input_size'])
    h, w = image.shape[:2]
    scale = min(input_size[0] / h, input_size[1] / w)
    new_h, new_w = int(h * scale), int(w * scale)
    
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    padded = np.zeros((input_size[0], input_size[1], 3), dtype=np.uint8)
    padded.fill(114)
    top = (input_size[0] - new_h) // 2
    left = (input_size[1] - new_w) // 2
    padded[top:top+new_h, left:left+new_w] = resized
    
    padded = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
    padded = padded.transpose(2, 0, 1).astype(np.float32) / 255.0
    padded = np.expand_dims(padded, axis=0)
    
    return padded, scale, (top, left)

# NMS
def nms(boxes, scores, iou_thresh):
    if len(boxes) == 0:
        return []
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        inds = np.where(iou <= iou_thresh)[0]
        order = order[inds + 1]
    return keep

# Postprocessing
def postprocess_output(output, conf_thresh, iou_thresh, scale, padding, orig_shape, class_confidence_map=None):
    output = output[0]
    if output.shape[0] < output.shape[1]:
        output = output.T
    boxes = output[:, :4]
    class_scores = output[:, 4:]
    max_scores = np.max(class_scores, axis=1)
    class_ids = np.argmax(class_scores, axis=1)
    
    # Apply class-specific confidence thresholds
    if class_confidence_map is not None and len(class_confidence_map) > 0:
        thresholds = np.array([class_confidence_map.get(int(cid), conf_thresh) for cid in class_ids])
        mask = max_scores > thresholds
    else:
        mask = max_scores > conf_thresh
    
    if not mask.any():
        return []
    
    boxes = boxes[mask]
    scores = max_scores[mask]
    class_ids = class_ids[mask]
    
    x_center, y_center, width, height = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x1 = x_center - width / 2
    y1 = y_center - height / 2
    x2 = x_center + width / 2
    y2 = y_center + height / 2
    boxes = np.stack([x1, y1, x2, y2], axis=1)
    
    # Class-wise NMS: Apply NMS separately for each class
    # This allows overlapping detections of different classes
    keep_indices = []
    unique_classes = np.unique(class_ids)
    
    for class_id in unique_classes:
        class_mask = class_ids == class_id
        class_boxes = boxes[class_mask]
        class_scores = scores[class_mask]
        class_indices = np.where(class_mask)[0]
        
        # Apply NMS for this class only
        class_keep = nms(class_boxes, class_scores, iou_thresh)
        
        # Store the original indices
        keep_indices.extend(class_indices[class_keep].tolist())
    
    if len(keep_indices) == 0:
        return []
    
    boxes = boxes[keep_indices]
    scores = scores[keep_indices]
    class_ids = class_ids[keep_indices]
    
    top, left = padding
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - left) / scale
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - top) / scale
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, orig_shape[1])
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, orig_shape[0])
    
    detections = []
    for box, score, class_id in zip(boxes, scores, class_ids):
        x1, y1, x2, y2 = box
        if x2 > x1 and y2 > y1 and x2 > 0 and y2 > 0:
            detections.append({
                'box': box.tolist(),
                'confidence': float(score),
                'class_id': int(class_id),
                'class_name': CLASS_NAMES[int(class_id)] if int(class_id) < len(CLASS_NAMES) else f"class_{int(class_id)}"
            })
    
    return detections

# Draw detections
def draw_detections(image, detections):
    h, w = image.shape[:2]
    result_image = image.copy()
    
    for det in detections:
        box = det['box']
        x1, y1, x2, y2 = map(int, box)
        conf = det['confidence']
        class_name = det['class_name']
        class_id = det['class_id']
        
        if x1 >= x2 or y1 >= y2:
            continue
        
        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(x1 + 1, min(x2, w))
        y2 = max(y1 + 1, min(y2, h))
        
        # Use consistent colors for each class
        np.random.seed(class_id)
        color = tuple(map(int, np.random.randint(0, 255, 3)))
        
        # Draw bounding box
        cv2.rectangle(result_image, (x1, y1), (x2, y2), color, 2)
        
        # Draw label with background
        label = f"{class_name}: {conf:.2%}"
        (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(result_image, (x1, y1 - label_h - 10), (x1 + label_w, y1), color, -1)
        cv2.putText(result_image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return result_image

def test_image(image_path, output_path=None):
    """Test detection on a single image"""
    print(f"\n{'='*70}")
    print(f"Testing falling detection on: {image_path}")
    print(f"{'='*70}\n")
    
    # Load image
    if not os.path.exists(image_path):
        print(f"Error: Image file not found: {image_path}")
        return False
    
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image: {image_path}")
        return False
    
    print(f"Image loaded: {image.shape[1]}x{image.shape[0]}")
    
    # Load model
    print(f"Loading model: {CONFIG['model_path']}")
    try:
        runtime = rebel.Runtime(CONFIG['model_path'])
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        return False
    
    # Preprocess
    print("Preprocessing image...")
    preprocessed, scale, padding = preprocess_image(image)
    
    # Run inference
    print("Running inference...")
    output = runtime.run(preprocessed)
    
    # Postprocess
    print("Postprocessing results...")
    detections = postprocess_output(
        output[0] if isinstance(output, (list, tuple)) else output,
        CONFIG['confidence'],
        CONFIG['iou_threshold'],
        scale,
        padding,
        image.shape[:2],
        CONFIG.get('class_confidence', None)
    )
    
    # Print results
    print(f"\nDetections found: {len(detections)}")
    falling_count = sum(1 for d in detections if d['class_id'] == 8)
    print(f"Falling detections: {falling_count}")
    
    if len(detections) > 0:
        print("\nDetection details:")
        for i, det in enumerate(detections):
            box = det['box']
            print(f"  {i+1}. {det['class_name']}: {det['confidence']:.2%} "
                  f"box=[{box[0]:.1f}, {box[1]:.1f}, {box[2]:.1f}, {box[3]:.1f}]")
    
    # Draw detections
    result_image = draw_detections(image, detections)
    
    # Save result
    if output_path is None:
        # Generate default output path
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        output_path = f"{base_name}_result.jpg"
    
    cv2.imwrite(output_path, result_image)
    print(f"\nResult saved to: {output_path}")
    
    return True

def main():
    parser = argparse.ArgumentParser(description='Test falling detection on sample images')
    parser.add_argument('image', type=str, help='Path to input image')
    parser.add_argument('-o', '--output', type=str, default=None, 
                       help='Path to save output image (default: <image_name>_result.jpg)')
    parser.add_argument('--conf', type=float, default=None,
                       help='Confidence threshold override')
    parser.add_argument('--falling-conf', type=float, default=None,
                       help='Falling class confidence threshold override')
    
    args = parser.parse_args()
    
    # Override config if specified
    if args.conf is not None:
        CONFIG['confidence'] = args.conf
    if args.falling_conf is not None:
        CONFIG['class_confidence'][8] = args.falling_conf
    
    success = test_image(args.image, args.output)
    
    if success:
        print(f"\n{'='*70}")
        print("Test completed successfully!")
        print(f"{'='*70}\n")
    else:
        print(f"\n{'='*70}")
        print("Test failed!")
        print(f"{'='*70}\n")
        sys.exit(1)

if __name__ == "__main__":
    main()

