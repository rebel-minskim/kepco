"""
Data models and classes for the client application.
"""
import time
import statistics
from typing import List, Dict, Any, Tuple
import numpy as np


class PerformanceStats:
    """Performance statistics tracking class."""
    
    def __init__(self):
        self.e2e_latencies: List[float] = []
        self.preprocess_latencies: List[float] = []
        self.inference_latencies: List[float] = []
        self.postprocess_latencies: List[float] = []
        self.frame_times: List[float] = []
        self.start_time: float = None
        self.frame_count: int = 0
        self.total_objects: int = 0
        
    def add_measurement(self, e2e_lat: float, pre_lat: float, inf_lat: float, 
                       post_lat: float, frame_time: float, obj_count: int) -> None:
        """Add a measurement to the statistics."""
        self.e2e_latencies.append(e2e_lat)
        self.preprocess_latencies.append(pre_lat)
        self.inference_latencies.append(inf_lat)
        self.postprocess_latencies.append(post_lat)
        self.frame_times.append(frame_time)
        self.frame_count += 1
        self.total_objects += obj_count
        
    def get_fps(self) -> float:
        """Calculate average FPS."""
        if not self.frame_times:
            return 0
        return len(self.frame_times) / sum(self.frame_times)
    
    def get_avg_latencies(self) -> Dict[str, float]:
        """Get average latencies for all metrics."""
        return {
            'e2e': statistics.mean(self.e2e_latencies) if self.e2e_latencies else 0,
            'preprocess': statistics.mean(self.preprocess_latencies) if self.preprocess_latencies else 0,
            'inference': statistics.mean(self.inference_latencies) if self.inference_latencies else 0,
            'postprocess': statistics.mean(self.postprocess_latencies) if self.postprocess_latencies else 0
        }
    
    def get_percentile_latencies(self, percentile: int = 95) -> Dict[str, float]:
        """Get percentile latencies for all metrics."""
        return {
            'e2e': np.percentile(self.e2e_latencies, percentile) if self.e2e_latencies else 0,
            'preprocess': np.percentile(self.preprocess_latencies, percentile) if self.preprocess_latencies else 0,
            'inference': np.percentile(self.inference_latencies, percentile) if self.inference_latencies else 0,
            'postprocess': np.percentile(self.postprocess_latencies, percentile) if self.postprocess_latencies else 0
        }
    
    def print_summary(self) -> None:
        """Print a formatted summary of performance statistics."""
        if self.frame_count == 0:
            print("No frames processed")
            return
            
        total_time = time.time() - self.start_time if self.start_time else 0
        avg_fps = self.frame_count / total_time if total_time > 0 else 0
        
        print("\n" + "="*60)
        print("PERFORMANCE STATISTICS SUMMARY")
        print("="*60)
        print(f"Total Frames Processed: {self.frame_count}")
        print(f"Total Processing Time: {total_time:.2f}s")
        print(f"Average FPS: {avg_fps:.2f}")
        print(f"Total Objects Detected: {self.total_objects}")
        print(f"Average Objects per Frame: {self.total_objects/self.frame_count:.2f}")
        
        print("\nLATENCY STATISTICS (ms):")
        print("-" * 40)
        avg_lat = self.get_avg_latencies()
        p95_lat = self.get_percentile_latencies(95)
        
        print(f"{'Metric':<15} {'Avg':<10} {'P95':<10}")
        print(f"{'E2E Latency':<15} {avg_lat['e2e']*1000:<10.2f} {p95_lat['e2e']*1000:<10.2f}")
        print(f"{'Preprocess':<15} {avg_lat['preprocess']*1000:<10.2f} {p95_lat['preprocess']*1000:<10.2f}")
        print(f"{'Inference':<15} {avg_lat['inference']*1000:<10.2f} {p95_lat['inference']*1000:<10.2f}")
        print(f"{'Postprocess':<15} {avg_lat['postprocess']*1000:<10.2f} {p95_lat['postprocess']*1000:<10.2f}")
        
        print(f"\nThroughput: {avg_fps:.2f} FPS")
        print("="*60)


class Detection:
    """Represents a single object detection."""
    
    def __init__(self, class_id: int, confidence: float, x1: float, y1: float, x2: float, y2: float):
        self.class_id = class_id
        self.confidence = confidence
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
    
    def get_center(self) -> Tuple[float, float]:
        """Get the center point of the bounding box."""
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)
    
    def get_box(self) -> Tuple[float, float, float, float]:
        """Get the bounding box coordinates."""
        return (self.x1, self.y1, self.x2, self.y2)
