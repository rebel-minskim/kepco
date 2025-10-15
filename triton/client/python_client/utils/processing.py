"""
Image processing utilities for object detection.
"""
import numpy as np
import torch
from typing import Tuple, List
from ultralytics.data.augment import LetterBox
from ultralytics.utils.nms import non_max_suppression as nms


def preprocess(frame: np.ndarray, new_shape: Tuple[int, int] = (640, 640)) -> np.ndarray:
    """
    Preprocess an image frame for inference.
    
    Args:
        frame: Input image frame
        new_shape: Target shape (width, height)
        
    Returns:
        Preprocessed image array
    """
    img = LetterBox(new_shape=new_shape)(image=frame)
    img = img.transpose((2, 0, 1))[::-1]   
    img = np.ascontiguousarray(img, dtype=np.float32)
    img = img[None] / 255.0  
    return img


def postprocess(outputs: np.ndarray, model_size: tuple, origin_image: np.ndarray,
                conf_threshold: float = 0.25, iou_threshold: float = 0.65, 
                max_detections: int = 1024) -> torch.Tensor:
    """
    Postprocess model outputs with simple proportional scaling.
    For ensemble model with server-side preprocessing (simple resize, no letterbox).
    
    Args:
        outputs: Raw model outputs (1, num_features, num_boxes)
        model_size: Model input size (height, width), e.g., (800, 800)
        origin_image: Original image
        conf_threshold: Confidence threshold
        iou_threshold: IoU threshold for NMS
        max_detections: Maximum number of detections
        
    Returns:
        Tensor of detections with boxes in original image coordinates
    """
    # Make array writable
    outputs = outputs.copy()
    
    # Run NMS
    pred = nms(torch.from_numpy(outputs), conf_threshold, iou_threshold, 
              None, False, max_det=max_detections)[0]
    
    if len(pred) == 0:
        return torch.empty(0, 6)
    
    # Simple proportional scaling from model size to original size
    # Model boxes are in 800x800 space, scale to original image size
    orig_h, orig_w = origin_image.shape[:2]
    model_h, model_w = model_size
    
    # Scale factors
    scale_x = orig_w / model_w
    scale_y = orig_h / model_h
    
    # Scale boxes: [x1, y1, x2, y2]
    pred[:, 0] *= scale_x  # x1
    pred[:, 1] *= scale_y  # y1
    pred[:, 2] *= scale_x  # x2
    pred[:, 3] *= scale_y  # y2
    
    return pred


def get_center(x1: float, y1: float, x2: float, y2: float) -> Tuple[float, float]:
    """Get the center point of a bounding box."""
    return ((x1 + x2) / 2, (y1 + y2) / 2)


def is_same_object(box1: Tuple[float, float, float, float], 
                   box2: Tuple[float, float, float, float], 
                   distance_thresh: float = 50.0) -> bool:
    """
    Check if two bounding boxes represent the same object.
    
    Args:
        box1: First bounding box (x1, y1, x2, y2)
        box2: Second bounding box (x1, y1, x2, y2)
        distance_thresh: Distance threshold for matching
        
    Returns:
        True if boxes represent the same object
    """
    cx1, cy1 = get_center(*box1)
    cx2, cy2 = get_center(*box2)
    return np.hypot(cx1 - cx2, cy1 - cy2) < distance_thresh