#!/usr/bin/env python3
"""
Concurrent inference client for maximum throughput.
Uses multiple concurrent requests to achieve 90fps with max_batch_size: 1.
"""
import argparse
import sys
import time
import threading
import queue
import numpy as np
import cv2
from typing import Optional, List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import tritonclient.grpc as grpcclient
from tritonclient.utils import InferenceServerException

from config import ClientConfig, DEFAULT_CONFIG
from utils import preprocess, postprocess
from utils.visualization import get_color


class ConcurrentTritonClient:
    """Concurrent Triton inference client for maximum throughput."""
    
    def __init__(self, config: ClientConfig):
        self.config = config
        self.client = None
        self.class_names = []
        self._load_class_names()
        
        # Concurrent processing settings
        self.max_concurrent_requests = 4  # Number of concurrent requests
        self.request_queue = queue.Queue(maxsize=20)  # Buffer for requests
        self.result_queue = queue.Queue(maxsize=20)   # Buffer for results
        self.stop_event = threading.Event()
        
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
    
    def _inference_worker(self, worker_id: int):
        """Worker thread for concurrent inference."""
        print(f"Inference worker {worker_id} started")
        while not self.stop_event.is_set():
            try:
                # Get request from queue with shorter timeout
                request_data = self.request_queue.get(timeout=0.1)
                if request_data is None:  # Shutdown signal
                    print(f"Inference worker {worker_id} shutting down")
                    break
                    
                frame, frame_id, timestamp = request_data
                
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
                
                # Put result in queue
                self.result_queue.put((frame, frame_id, timestamp, detections, inference_time))
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Inference worker {worker_id} error: {e}")
                continue
        
        print(f"Inference worker {worker_id} finished")
    
    def _draw_worker(self, output_writer=None):
        """Worker thread for drawing and output."""
        print("Draw worker started")
        frame_count = 0
        start_time = time.time()
        inference_times = []
        
        while not self.stop_event.is_set():
            try:
                result_data = self.result_queue.get(timeout=0.1)
                if result_data is None:  # Shutdown signal
                    print("Draw worker shutting down")
                    break
                    
                frame, frame_id, timestamp, detections, inference_time = result_data
                inference_times.append(inference_time)
                
                # Calculate timing
                current_time = time.time()
                e2e_latency = current_time - timestamp
                
                # Draw detections (minimal for speed)
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
                if output_writer:
                    output_writer.write(frame)
                
                # Print stats every 30 frames
                if frame_count % 30 == 0:
                    current_fps = frame_count / (current_time - start_time) if current_time > start_time else 0
                    avg_inference = np.mean(inference_times[-30:]) * 1000 if inference_times else 0
                    print(f"Frame {frame_count}: {len(detections)} objects | "
                          f"Inference: {avg_inference:.1f}ms | "
                          f"E2E: {e2e_latency*1000:.1f}ms | "
                          f"FPS: {current_fps:.1f}")
                
                frame_count += 1
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Draw worker error: {e}")
                continue
        
        print("Draw worker finished")
        # Final stats
        total_time = time.time() - start_time
        avg_fps = frame_count / total_time if total_time > 0 else 0
        avg_inference = np.mean(inference_times) * 1000 if inference_times else 0
        
        print(f"\n" + "="*60)
        print(f"CONCURRENT PROCESSING RESULTS")
        print(f"="*60)
        print(f"Total frames processed: {frame_count}")
        print(f"Total time: {total_time:.2f}s")
        print(f"Average FPS: {avg_fps:.1f}")
        print(f"Average inference time: {avg_inference:.1f}ms")
        print(f"Concurrent workers: {self.max_concurrent_requests}")
        print(f"Target FPS: 90.0")
        if avg_fps >= 90:
            print(f"✅ SUCCESS: Achieved target FPS!")
        else:
            print(f"❌ TARGET NOT MET: Need 90 FPS, got {avg_fps:.1f} FPS")
        print(f"="*60)
    
    def run_concurrent_inference(self, video_path: str, output_path: Optional[str] = None, no_drop: bool = False):
        """Run concurrent inference for maximum throughput."""
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
        
        print(f"Video: {width}x{height}, {fps}fps")
        print(f"Using {self.max_concurrent_requests} concurrent workers")
        
        # Setup output video writer
        out = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Start inference workers
        inference_threads = []
        for i in range(self.max_concurrent_requests):
            thread = threading.Thread(target=self._inference_worker, args=(i,))
            thread.daemon = True
            thread.start()
            inference_threads.append(thread)
        
        # Start draw worker
        draw_thread = threading.Thread(target=self._draw_worker, args=(out,))
        draw_thread.daemon = True
        draw_thread.start()
        
        print("Starting concurrent video processing...")
        frame_count = 0
        start_time = time.time()
        dropped_frames = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Add frame to request queue
                if no_drop:
                    # Wait indefinitely for queue space (no frame drops)
                    self.request_queue.put((frame, frame_count, time.time()))
                    frame_count += 1
                else:
                    # Allow frame drops for better performance
                    try:
                        self.request_queue.put((frame, frame_count, time.time()), timeout=1.0)
                        frame_count += 1
                    except queue.Full:
                        # Frame dropped due to queue full
                        dropped_frames += 1
                        if dropped_frames % 50 == 0:
                            print(f"Warning: {dropped_frames} frames dropped due to queue full")
                        continue
        
        except KeyboardInterrupt:
            print("Stopping processing...")
        
        finally:
            print("Stopping workers...")
            # Stop workers
            self.stop_event.set()
            
            # Send shutdown signals (non-blocking)
            for _ in range(self.max_concurrent_requests):
                try:
                    self.request_queue.put_nowait(None)
                except queue.Full:
                    pass  # Skip if queue is full
            try:
                self.result_queue.put_nowait(None)
            except queue.Full:
                pass  # Skip if queue is full
            
            # Wait for threads to finish with timeout
            print("Waiting for inference threads...")
            for i, thread in enumerate(inference_threads):
                thread.join(timeout=2.0)
                if thread.is_alive():
                    print(f"Warning: Inference thread {i} did not terminate gracefully")
            
            print("Waiting for draw thread...")
            draw_thread.join(timeout=2.0)
            if draw_thread.is_alive():
                print("Warning: Draw thread did not terminate gracefully")
            
            # Force clear queues to help threads exit
            while not self.request_queue.empty():
                try:
                    self.request_queue.get_nowait()
                except queue.Empty:
                    break
            while not self.result_queue.empty():
                try:
                    self.result_queue.get_nowait()
                except queue.Empty:
                    break
            
            # Cleanup
            cap.release()
            if out:
                out.release()
            
            # Print final summary with dropped frames
            print(f"\n" + "="*60)
            print(f"FINAL SUMMARY")
            print(f"="*60)
            print(f"Total frames processed: {frame_count}")
            print(f"Dropped frames: {dropped_frames}")
            if dropped_frames > 0:
                print(f"⚠️  WARNING: {dropped_frames} frames were dropped!")
                print(f"💡 Consider using --no-drop or increasing --queue-size")
            else:
                print(f"✅ No frames were dropped!")
            print(f"="*60)
            print("Processing completed!")


def create_argument_parser() -> argparse.ArgumentParser:
    """Create and configure argument parser."""
    parser = argparse.ArgumentParser(description="Concurrent Triton Inference Client")
    
    # Input/Output
    parser.add_argument('input', type=str, nargs='?', default='media/30sec.mp4',
                       help='Input file path')
    parser.add_argument('-o', '--output', type=str, default='output/result_concurrent.mp4',
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
    parser.add_argument('--queue-size', type=int, default=20,
                       help='Maximum queue size')
    parser.add_argument('--no-drop', action='store_true',
                       help='Do not drop frames (wait for queue space)')
    
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
    
    # Create concurrent client
    client = ConcurrentTritonClient(config)
    client.max_concurrent_requests = args.concurrent
    client.request_queue = queue.Queue(maxsize=args.queue_size)
    client.result_queue = queue.Queue(maxsize=args.queue_size)
    
    # Connect to server
    if not client.connect():
        sys.exit(1)
    
    # Run concurrent processing
    client.run_concurrent_inference(args.input, args.output, args.no_drop)


if __name__ == '__main__':
    main()
