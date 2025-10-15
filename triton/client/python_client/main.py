#!/usr/bin/env python3
"""
Main entry point for the client application.
"""
import argparse
import sys
import time
from typing import Optional

import tritonclient.grpc as grpcclient
from tritonclient.utils import InferenceServerException

from config import ClientConfig, DEFAULT_CONFIG
from utils.models import PerformanceStats
from utils import preprocess, postprocess, is_same_object, draw_detections
from utils.visualization import get_color


class TritonClient:
    """Triton inference client wrapper."""
    
    def __init__(self, config: ClientConfig):
        self.config = config
        self.client = None
        self.class_names = []
        self._load_class_names()
        
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
    
    def get_model_info(self):
        """Get and print model information."""
        if not self.client:
            return
            
        try:
            # Model metadata
            metadata = self.client.get_model_metadata(self.config.model.name)
            print("Model Metadata:")
            print(metadata)
            
            # Model configuration
            config = self.client.get_model_config(self.config.model.name)
            print("\nModel Configuration:")
            print(config)
            
            # Inference statistics
            stats = self.client.get_inference_statistics(model_name=self.config.model.name)
            print("\nInference Statistics:")
            print(stats)
            
        except InferenceServerException as ex:
            print(f"Failed to get model info: {ex.message()}")
    
    def run_dummy_inference(self):
        """Run dummy inference with JPEG bytes."""
        print("Running dummy inference...")
        
        import numpy as np
        import cv2
        
        # Create a dummy image and encode as JPEG
        dummy_image = np.random.randint(0, 255, 
            (self.config.model.input_height, self.config.model.input_width, 3), 
            dtype=np.uint8)
        
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
        success, encoded_image = cv2.imencode('.jpg', dummy_image, encode_param)
        if not success:
            print("Failed to encode dummy image")
            return
        
        jpeg_bytes = encoded_image.tobytes()
        jpeg_array = np.frombuffer(jpeg_bytes, dtype=np.uint8)
        
        inputs = []
        outputs = []
        
        # Create input for JPEG ensemble
        inputs.append(grpcclient.InferInput(
            "IMAGE_BYTES", 
            [len(jpeg_bytes)], 
            "UINT8"
        ))
        inputs[0].set_data_from_numpy(jpeg_array)
        
        # Create output
        outputs.append(grpcclient.InferRequestedOutput("OUTPUT__0"))
        
        # Run inference
        results = self.client.infer(
            model_name=self.config.model.name,
            inputs=inputs,
            outputs=outputs,
            client_timeout=self.config.server.timeout
        )
        
        # Print results
        for output_name in ["OUTPUT__0"]:
            result = results.as_numpy(output_name)
            print(f"Received result buffer '{output_name}' of size {result.shape}")
            print(f"Buffer sum: {np.sum(result)}")
    
    def run_image_inference(self, image_path: str, output_path: Optional[str] = None):
        """Run inference on a single image."""
        import cv2
        import numpy as np
        
        print(f"Processing image: {image_path}")
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Failed to load image: {image_path}")
            return
        
        # Resize and JPEG encode (for yolov11_ensemble)
        resized_image = cv2.resize(image, (self.config.model.input_width, self.config.model.input_height))
        
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
        success, encoded_image = cv2.imencode('.jpg', resized_image, encode_param)
        if not success:
            print("Failed to encode image as JPEG")
            return
        
        jpeg_bytes = encoded_image.tobytes()
        jpeg_array = np.frombuffer(jpeg_bytes, dtype=np.uint8)
        print(f"JPEG size: {len(jpeg_bytes)} bytes ({len(jpeg_bytes)/1024:.1f} KB)")
        
        # Prepare inputs/outputs for JPEG ensemble
        inputs = []
        outputs = []
        inputs.append(grpcclient.InferInput(
            "IMAGE_BYTES", 
            [len(jpeg_bytes)], 
            "UINT8"
        ))
        inputs[0].set_data_from_numpy(jpeg_array)
        outputs.append(grpcclient.InferRequestedOutput("OUTPUT__0"))
        
        # Run inference
        results = self.client.infer(
            model_name=self.config.model.name,
            inputs=inputs,
            outputs=outputs,
            client_timeout=self.config.server.timeout
        )
        
        # Postprocess
        output_tensor = results.as_numpy("OUTPUT__0")
        
        # Reshape output tensor for YOLOv11 format
        # Ensemble output is (batch*num_classes, total_boxes) 
        # Need to reshape to (batch, num_classes + coords, total_boxes)
        if output_tensor.shape[0] > 1 and len(output_tensor.shape) == 2:
            # Reshape from (13, 13125) or similar to (1, 84, 8400) format
            batch_size = 1
            num_boxes = output_tensor.shape[1]
            num_features = output_tensor.shape[0]
            output_tensor = output_tensor.reshape(batch_size, num_features, num_boxes)
        
        # For ensemble model: server does simple resize (no letterbox)
        # Use simple proportional scaling for box coordinates
        detections = postprocess(
            output_tensor, 
            (self.config.model.input_height, self.config.model.input_width),  # Model size: 800x800
            image,  # Original image for final box coordinates
            self.config.model.confidence_threshold,
            self.config.model.iou_threshold,
            self.config.model.max_detections
        )
        
        print(f"Detected {len(detections)} objects")
        
        # Draw detections
        for *xyxy, conf, cls in detections:
            if conf < self.config.model.draw_confidence:
                continue
            x1, y1, x2, y2 = map(int, xyxy)
            class_id = int(cls)
            class_name = self.class_names[class_id] if class_id < len(self.class_names) else f"class_{class_id}"
            
            # Draw bounding box and label
            color = get_color(class_id)
            cv2.rectangle(image, (x1, y1), (x2, y2), color, self.config.video.line_thickness)
            cv2.putText(image, f"{class_name}: {conf:.2f}", (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, self.config.video.font_scale, color, 
                       self.config.video.font_thickness, cv2.LINE_AA)
        
        # Save or display result
        if output_path:
            cv2.imwrite(output_path, image)
            print(f"Saved result to {output_path}")
        else:
            cv2.imshow('Result', image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    
    def run_video_inference(self, video_path: str, output_path: Optional[str] = None):
        """Run inference on a video."""
        import cv2
        import numpy as np
        
        print(f"Processing video: {video_path}")
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Failed to open video: {video_path}")
            return
        
        # Setup output
        out = None
        frame_count = 0
        perf_stats = PerformanceStats()
        perf_stats.start_time = time.time()
        
        # Frame history for object tracking
        frame_history = []
        
        print("Starting video processing...")
        
        while True:
            frame_start_time = time.time()
            
            ret, frame = cap.read()
            if not ret:
                break
            
            # Setup output video writer
            if frame_count == 0 and output_path:
                fourcc = cv2.VideoWriter_fourcc('M', 'P', '4', 'V')
                out = cv2.VideoWriter(output_path, fourcc, self.config.video.fps, 
                                    (frame.shape[1], frame.shape[0]))
            
            # Preprocess
            preprocess_start = time.time()
            input_buffer = preprocess(frame, (self.config.model.input_width, self.config.model.input_height))
            preprocess_latency = time.time() - preprocess_start
            
            # Prepare inputs/outputs
            inputs = []
            outputs = []
            inputs.append(grpcclient.InferInput(
                "INPUT__0", 
                [1, 3, self.config.model.input_height, self.config.model.input_width], 
                "FP32"
            ))
            inputs[0].set_data_from_numpy(input_buffer)
            outputs.append(grpcclient.InferRequestedOutput("OUTPUT__0"))
            
            # Inference
            inference_start = time.time()
            results = self.client.infer(
                model_name=self.config.model.name,
                inputs=inputs,
                outputs=outputs,
                client_timeout=self.config.server.timeout
            )
            inference_latency = time.time() - inference_start
            
            # Postprocess
            postprocess_start = time.time()
            output_tensor = results.as_numpy("OUTPUT__0")
            detections = postprocess(
                output_tensor, 
                input_buffer, 
                frame,
                self.config.model.confidence_threshold,
                self.config.model.iou_threshold,
                self.config.model.max_detections
            )
            postprocess_latency = time.time() - postprocess_start
            
            # Calculate timing
            frame_time = time.time() - frame_start_time
            e2e_latency = time.time() - frame_start_time
            
            print(f"Frame {frame_count}: {len(detections)} objects | "
                  f"E2E: {e2e_latency*1000:.1f}ms | "
                  f"Pre: {preprocess_latency*1000:.1f}ms | "
                  f"Inf: {inference_latency*1000:.1f}ms | "
                  f"Post: {postprocess_latency*1000:.1f}ms")
            
            # Object tracking with history
            current_detections = []
            for *xyxy, conf, cls in detections:
                if conf < self.config.model.draw_confidence:
                    continue
                x1, y1, x2, y2 = map(int, xyxy)
                class_id = int(cls)
                current_detections.append((class_id, (x1, y1, x2, y2)))
            
            # Update frame history
            frame_history.append(current_detections)
            if len(frame_history) > self.config.video.max_history:
                frame_history.pop(0)
            
            # Filter detections based on history
            confirmed_objects = []
            for class_id, box in current_detections:
                count = 0
                for past_frame in frame_history:
                    for pid, pbox in past_frame:
                        if pid == class_id and is_same_object(pbox, box, self.config.video.distance_threshold):
                            count += 1
                            break
                if count >= self.config.video.max_history:
                    confirmed_objects.append((class_id, box))
            
            # Draw confirmed detections
            for class_id, (x1, y1, x2, y2) in confirmed_objects:
                class_name = self.class_names[class_id] if class_id < len(self.class_names) else f"class_{class_id}"
                color = get_color(class_id)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, self.config.video.line_thickness)
                cv2.putText(frame, class_name, (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, self.config.video.font_scale, color,
                           self.config.video.font_thickness, cv2.LINE_AA)
            
            # Update performance stats
            perf_stats.add_measurement(
                e2e_latency, preprocess_latency, inference_latency,
                postprocess_latency, frame_time, len(detections)
            )
            
            # Write or display frame
            if output_path:
                out.write(frame)
            else:
                try:
                    cv2.imshow('Video', frame)
                    if cv2.waitKey(1) == ord('q'):
                        break
                except cv2.error:
                    # Skip GUI operations in headless environment
                    pass
            
            frame_count += 1
        
        # Print performance summary
        perf_stats.print_summary()
        
        # Cleanup
        cap.release()
        if out:
            out.release()
        else:
            cv2.destroyAllWindows()


def create_argument_parser() -> argparse.ArgumentParser:
    """Create and configure argument parser."""
    parser = argparse.ArgumentParser(description="Triton Inference Client")
    
    # Mode selection
    parser.add_argument('mode', choices=['dummy', 'image', 'video'], default='video',
                       help='Run mode: dummy (test), image (single image), video (video file)')
    
    # Input/Output
    parser.add_argument('input', type=str, nargs='?', default='media/1.mp4',
                       help='Input file path')
    parser.add_argument('-o', '--output', type=str, default='output/result.mp4',
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
    
    # Video configuration
    parser.add_argument('-f', '--fps', type=float, default=24.0,
                       help='Output video FPS')
    
    # Information
    parser.add_argument('-i', '--model-info', action='store_true',
                       help='Print model information')
    
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
    config.video.fps = args.fps
    
    # Create client
    client = TritonClient(config)
    
    # Connect to server
    if not client.connect():
        sys.exit(1)
    
    # Print model info if requested
    if args.model_info:
        client.get_model_info()
    
    # Run based on mode
    if args.mode == 'dummy':
        client.run_dummy_inference()
    elif args.mode == 'image':
        client.run_image_inference(args.input, args.output)
    elif args.mode == 'video':
        client.run_video_inference(args.input, args.output)


if __name__ == '__main__':
    main()
