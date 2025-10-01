#!/usr/bin/env python3
"""
Simple concurrent inference client for 90fps.
Uses asyncio for concurrent requests without complex threading.
"""
import argparse
import sys
import time
import asyncio
import numpy as np
import cv2
from typing import Optional, List, Tuple
import concurrent.futures

import tritonclient.grpc as grpcclient
from tritonclient.utils import InferenceServerException

from config import ClientConfig, DEFAULT_CONFIG
from utils import preprocess, postprocess
from utils.visualization import get_color


class ConcurrentSimpleClient:
    """Simple concurrent Triton inference client."""
    
    def __init__(self, config: ClientConfig):
        self.config = config
        self.client = None
        self.class_names = []
        self._load_class_names()
        
        # Concurrent settings
        self.max_concurrent = 4  # Number of concurrent requests
        
    def _load_class_names(self):
        """Load class names from data.yaml."""
        import yaml
        try:
            with open(self.config.paths.data_yaml) as f:
                data = yaml.safe_load(f)
            self.class_names = list(data["names"].values())
        except Exception as e:
            print(f"Warning: Could not load class names from {self.config.paths.data_yaml}: {e}")
            self.class_names = [f"class_{i}" for i in range(100)]  # Fallback
    
    def connect(self) -> bool:
        """Connect to the Triton server."""
        try:
            self.client = grpcclient.InferenceServerClient(
                url=self.config.server.url,
                verbose=self.config.server.verbose
            )
            
            # Health checks
            if not self.client.is_server_live():
                print("FAILED: Server is not live")
                return False
                
            if not self.client.is_server_ready():
                print("FAILED: Server is not ready")
                return False
                
            if not self.client.is_model_ready(self.config.model.name):
                print(f"FAILED: Model {self.config.model.name} is not ready")
                return False
                
            return True
            
        except Exception as e:
            print(f"Connection failed: {e}")
            return False
    
    def _process_frame(self, frame, frame_id, timestamp):
        """Process a single frame."""
        # Preprocess
        input_buffer = preprocess(frame, (self.config.model.input_width, self.config.model.input_height))
        
        # Prepare inputs/outputs
        inputs = [grpcclient.InferInput(
            "INPUT__0", 
            [1, 3, self.config.model.input_height, self.config.model.input_width], 
            "FP32"
        )]
        inputs[0].set_data_from_numpy(input_buffer)
        outputs = [grpcclient.InferRequestedOutput("OUTPUT__0")]
        
        # Run inference
        inference_start = time.time()
        results = self.client.infer(
            model_name=self.config.model.name,
            inputs=inputs,
            outputs=outputs,
            client_timeout=self.config.server.timeout
        )
        inference_time = time.time() - inference_start
        
        # Postprocess
        output_tensor = results.as_numpy("OUTPUT__0")
        detections = postprocess(
            output_tensor, 
            input_buffer, 
            frame,
            self.config.model.confidence_threshold,
            self.config.model.iou_threshold,
            self.config.model.max_detections
        )
        
        return frame, frame_id, timestamp, detections, inference_time
    
    def run_concurrent_simple_inference(self, video_path: str, output_path: Optional[str] = None):
        """Run concurrent simple inference for maximum throughput."""
        print(f"Processing video with concurrent requests: {video_path}")
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Failed to open video: {video_path}")
            return
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Video: {width}x{height}, {fps}fps, {total_frames} frames")
        print(f"Using {self.max_concurrent} concurrent requests")
        
        # Setup output video writer
        out = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Performance tracking
        frame_count = 0
        start_time = time.time()
        inference_times = []
        total_objects = 0
        
        print("Starting concurrent processing...")
        
        # Read all frames first
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        
        cap.release()
        print(f"Loaded {len(frames)} frames")
        
        # Process frames with concurrent requests
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_concurrent) as executor:
            # Submit all frames for processing
            future_to_frame = {}
            for i, frame in enumerate(frames):
                future = executor.submit(self._process_frame, frame, i, time.time())
                future_to_frame[future] = i
            
            # Process results as they complete
            results = []
            for future in concurrent.futures.as_completed(future_to_frame):
                try:
                    result = future.result()
                    results.append(result)
                    frame, frame_id, timestamp, detections, inference_time = result
                    inference_times.append(inference_time)
                    total_objects += len(detections)
                    
                    # Print progress
                    if len(results) % 30 == 0:
                        current_fps = len(results) / (time.time() - start_time)
                        avg_inference = np.mean(inference_times[-30:]) * 1000 if inference_times else 0
                        print(f"Processed {len(results)}/{len(frames)} frames | "
                              f"Objects: {len(detections)} | "
                              f"Inference: {avg_inference:.1f}ms | "
                              f"FPS: {current_fps:.1f}")
                
                except Exception as e:
                    print(f"Error processing frame: {e}")
        
        # Sort results by frame_id to maintain order
        results.sort(key=lambda x: x[1])
        
        # Draw and write frames
        print("Drawing and writing frames...")
        for frame, frame_id, timestamp, detections, inference_time in results:
            # Draw detections
            for *xyxy, conf, cls in detections:
                if conf < self.config.model.draw_confidence:
                    continue
                x1, y1, x2, y2 = map(int, xyxy)
                class_id = int(cls)
                class_name = self.class_names[class_id] if class_id < len(self.class_names) else f"class_{class_id}"
                color = get_color(class_id)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)
                cv2.putText(frame, class_name, (x1, y1 - 2),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
            
            # Write frame
            if out:
                out.write(frame)
        
        # Final stats
        total_time = time.time() - start_time
        avg_fps = len(results) / total_time if total_time > 0 else 0
        avg_inference = np.mean(inference_times) * 1000 if inference_times else 0
        avg_objects = total_objects / len(results) if results else 0
        
        print(f"\n" + "="*60)
        print(f"CONCURRENT PROCESSING RESULTS")
        print(f"="*60)
        print(f"Total frames: {len(results)}")
        print(f"Total time: {total_time:.2f}s")
        print(f"Average FPS: {avg_fps:.1f}")
        print(f"Average inference time: {avg_inference:.1f}ms")
        print(f"Average objects per frame: {avg_objects:.1f}")
        print(f"Concurrent requests: {self.max_concurrent}")
        print(f"Target FPS: 90.0")
        if avg_fps >= 90:
            print(f"✅ SUCCESS: Achieved target FPS!")
        else:
            print(f"❌ TARGET NOT MET: Need 90 FPS, got {avg_fps:.1f} FPS")
            print(f"💡 Inference time: {avg_inference:.1f}ms (target: <11ms for 90fps)")
        print(f"="*60)
        
        # Cleanup
        if out:
            out.release()
        
        print("Processing completed!")


def create_argument_parser() -> argparse.ArgumentParser:
    """Create and configure argument parser."""
    parser = argparse.ArgumentParser(description="Concurrent Simple Triton Inference Client")
    
    # Input/Output
    parser.add_argument('input', type=str, nargs='?', default='media/30sec.mp4',
                       help='Input file path')
    parser.add_argument('-o', '--output', type=str, default='output/result_concurrent_simple.mp4',
                       help='Output file path')
    
    # Model configuration
    parser.add_argument('-m', '--model', type=str, default='yolov11',
                       help='Model name')
    parser.add_argument('--width', type=int, default=800,
                       help='Input width')
    parser.add_argument('--height', type=int, default=800,
                       help='Input height')
    parser.add_argument('--conf', type=float, default=0.20,
                       help='Confidence threshold')
    parser.add_argument('--iou', type=float, default=0.50,
                       help='IoU threshold')
    
    # Server configuration
    parser.add_argument('-u', '--url', type=str, default='localhost:8001',
                       help='Server URL')
    parser.add_argument('-t', '--timeout', type=float, default=None,
                       help='Client timeout')
    parser.add_argument('-v', '--verbose', action='store_true',
                       help='Verbose output')
    
    # Concurrent configuration
    parser.add_argument('--concurrent', type=int, default=4,
                       help='Number of concurrent requests')
    
    return parser


def main():
    """Main entry point."""
    parser = create_argument_parser()
    args = parser.parse_args()
    
    # Create configuration
    config = ClientConfig()
    config.model.name = args.model
    config.model.input_width = args.width
    config.model.input_height = args.height
    config.model.confidence_threshold = args.conf
    config.model.iou_threshold = args.iou
    config.server.url = args.url
    config.server.timeout = args.timeout
    config.server.verbose = args.verbose
    
    # Create concurrent simple client
    client = ConcurrentSimpleClient(config)
    client.max_concurrent = args.concurrent
    
    # Connect to server
    if not client.connect():
        sys.exit(1)
    
    # Run concurrent simple processing
    client.run_concurrent_simple_inference(args.input, args.output)


if __name__ == '__main__':
    main()
