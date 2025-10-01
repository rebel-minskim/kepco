"""
Visualization utilities for drawing bounding boxes and labels.
"""
import cv2
from typing import Tuple, List
import numpy as np


def get_color(class_id: int) -> Tuple[int, int, int]:
    """Get color for a class ID."""
    try:
        from ultralytics.utils.plotting import colors as ucolors
        return ucolors(int(class_id), bgr=True)
    except Exception:
        # Fallback color palette
        _PALETTE = [
            (255, 56, 56), (255, 157, 151), (255, 112, 31), (255, 178, 29),
            (207, 210, 49), (72, 249, 10), (146, 204, 23), (61, 219, 134),
            (26, 147, 52), (0, 212, 187), (44, 153, 168), (0, 194, 255),
            (52, 69, 147), (100, 115, 255), (142, 140, 255), (204, 173, 255),
            (255, 101, 189), (255, 50, 188), (255, 0, 181), (135, 60, 190)
        ]
        return _PALETTE[int(class_id) % len(_PALETTE)]


def draw_bounding_box(frame: np.ndarray, x1: int, y1: int, x2: int, y2: int, 
                     color: Tuple[int, int, int], thickness: int = 4) -> np.ndarray:
    """Draw a bounding box on the frame."""
    return cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)


def draw_label(frame: np.ndarray, text: str, position: Tuple[int, int], 
               color: Tuple[int, int, int], font_scale: float = 0.6, 
               thickness: int = 2) -> np.ndarray:
    """Draw a text label on the frame."""
    return cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, 
                      font_scale, color, thickness, cv2.LINE_AA)


def draw_detection(frame: np.ndarray, class_id: int, class_name: str, 
                  confidence: float, x1: int, y1: int, x2: int, y2: int,
                  line_thickness: int = 4, font_scale: float = 0.6, 
                  font_thickness: int = 2) -> np.ndarray:
    """Draw a complete detection (box + label) on the frame."""
    color = get_color(class_id)
    
    # Draw bounding box
    frame = draw_bounding_box(frame, x1, y1, x2, y2, color, line_thickness)
    
    # Draw label
    label = f"{class_name}: {confidence:.2f}"
    label_position = (x1, y1 - 10)
    frame = draw_label(frame, label, label_position, color, font_scale, font_thickness)
    
    return frame


def draw_detections(frame: np.ndarray, detections: List[Tuple[int, str, float, int, int, int, int]],
                   class_names: List[str], line_thickness: int = 4, 
                   font_scale: float = 0.6, font_thickness: int = 2) -> np.ndarray:
    """Draw multiple detections on the frame."""
    for class_id, confidence, x1, y1, x2, y2 in detections:
        if confidence < 0.2:  # Skip low confidence detections
            continue
            
        class_name = class_names[class_id] if class_id < len(class_names) else f"class_{class_id}"
        frame = draw_detection(frame, class_id, class_name, confidence, 
                              x1, y1, x2, y2, line_thickness, font_scale, font_thickness)
    
    return frame
