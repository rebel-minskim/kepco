"""Triton gRPC Client for Video Object Detection (Decoupled Mode).

This module provides a client interface for interacting with NVIDIA Triton
Inference Server for YOLO object detection, supporting both synchronous and
asynchronous (decoupled) inference modes.
"""
import argparse
import json
import logging
import queue
import sys
import threading
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import cv2
import numpy as np
import tritonclient.grpc as grpcclient
import yaml
from tritonclient.utils import InferenceServerException

# Force unbuffered output for real-time logging
# Python 3: Use line buffering for stdout/stderr
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)

# Module-level logger
logger = logging.getLogger(__name__)

# Constants
DEFAULT_URL = "localhost:8001"
MODEL_NAME = "yolov11"
INPUT_NAME = "INPUT__0"
OUTPUT_NAME = "OUTPUT__0"


def is_display_available() -> bool:
    """Check if display is available for OpenCV windows.

    Returns:
        True if display is available, False otherwise.
    """
    if sys.platform.startswith('linux'):
        import os
        display = os.environ.get('DISPLAY')
        if not display:
            return False
    try:
        test_img = np.zeros((1, 1, 3), dtype=np.uint8)
        cv2.namedWindow("__test__", cv2.WINDOW_NORMAL)
        cv2.destroyWindow("__test__")
        return True
    except cv2.error:
        return False


DISPLAY_AVAILABLE = is_display_available()


@dataclass
class Detection:
    """Detection data class representing a detected object.

    Attributes:
        x1: Left x-coordinate of bounding box.
        y1: Top y-coordinate of bounding box.
        x2: Right x-coordinate of bounding box.
        y2: Bottom y-coordinate of bounding box.
        conf: Confidence score of the detection.
        class_id: Class ID of the detected object.
    """

    x1: float
    y1: float
    x2: float
    y2: float
    conf: float
    class_id: int

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Detection":
        """Create Detection object from server response dictionary.

        Parses server response dictionary flexibly, handling different
        response formats (bbox array or individual keys).

        Args:
            d: Dictionary containing detection data. May contain:
                - "bbox": [x1, y1, x2, y2] array, or
                - Individual "x1", "y1", "x2", "y2" keys
                - "confidence" or "conf" for confidence score
                - "class_id" or "class" for class ID

        Returns:
            Detection object parsed from dictionary.

        Raises:
            ValueError: If bbox length is invalid.
        """
        # Parse coordinates: handle "bbox": [x1, y1, x2, y2] format
        if "bbox" in d:
            bbox = d["bbox"]
            if len(bbox) == 4:
                x1, y1, x2, y2 = bbox
            else:
                logger.error(f"Invalid bbox length: {len(bbox)}, bbox: {bbox}")
                x1, y1, x2, y2 = 0.0, 0.0, 0.0, 0.0
        else:
            # Fallback: individual keys
            x1 = d.get("x1", 0.0)
            y1 = d.get("y1", 0.0)
            x2 = d.get("x2", 0.0)
            y2 = d.get("y2", 0.0)

        # Parse confidence: "confidence" or "conf"
        conf = d.get("confidence", d.get("conf", 0.0))

        # Parse class ID: "class_id" or "class"
        class_id = d.get("class_id", d.get("class", 0))

        return cls(
            x1=float(x1),
            y1=float(y1),
            x2=float(x2),
            y2=float(y2),
            conf=float(conf),
            class_id=int(class_id),
        )


def load_class_names() -> List[str]:
    """Load class names from YAML configuration file.

    Searches for coco128.yaml in multiple possible locations and loads
    class names. Falls back to COCO class names if file not found.

    Returns:
        List of class names (80 COCO classes).
    """
    # COCO class names as fallback
    coco_class_names = [
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
        "truck", "boat", "traffic light", "fire hydrant", "stop sign",
        "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
        "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
        "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
        "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
        "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
        "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
        "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
        "couch", "potted plant", "bed", "dining table", "toilet", "tv",
        "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
        "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
        "scissors", "teddy bear", "hair drier", "toothbrush"
    ]

    base_dir = Path(__file__).parent.absolute()
    # Try multiple possible paths
    possible_paths = [
        base_dir / "coco128.yaml",
        base_dir.parent / "rbln_backend" / "yolov11" / "1" / "coco128.yaml",
        base_dir / "rbln_backend" / "yolov11" / "1" / "coco128.yaml",
        Path("/workspace/kepco/triton/rbln_backend/yolov11/1/coco128.yaml"),
    ]

    for path in possible_paths:
        if path.exists():
            try:
                with path.open() as f:
                    data = yaml.safe_load(f)
                if data and "names" in data:
                    class_names = list(data["names"].values())
                    logger.info(
                        f"Loaded {len(class_names)} class names from {path}"
                    )
                    return class_names
            except Exception as e:
                logger.warning(
                    f"Failed to load class names from {path}: {e}. "
                    f"Using fallback."
                )
                continue

    logger.warning(
        "Class names file not found, using default COCO class names"
    )
    return coco_class_names


def encode_frame_to_jpeg(frame: np.ndarray, quality: int = 80) -> bytes:
    """Encode frame to JPEG format.

    Args:
        frame: Input frame as numpy array (BGR format).
        quality: JPEG quality (1-100). Defaults to 80.

    Returns:
        JPEG-encoded frame as bytes.

    Raises:
        RuntimeError: If encoding fails.
    """
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, quality]
    success, encoded = cv2.imencode(".jpg", frame, encode_params)
    if not success:
        raise RuntimeError("Failed to encode frame to JPEG")
    return encoded.tobytes()


def parse_json_response(json_data: Any) -> List[Detection]:
    """Parse JSON response into list of Detection objects.

    Handles various input formats (numpy array, bytes, string) and converts
    to list of Detection objects. Supports both dictionary format (with
    metadata) and direct list format.

    Args:
        json_data: JSON data in various formats (numpy array, bytes, string).

    Returns:
        List of Detection objects parsed from JSON response.
    """
    try:
        # Convert various formats to string
        if isinstance(json_data, np.ndarray):
            if json_data.dtype == np.object_:
                json_str = json_data.item() if json_data.size == 1 else json_data[0]
            else:
                json_str = json_data.tobytes().decode('utf-8')
        elif isinstance(json_data, bytes):
            json_str = json_data.decode('utf-8')
        else:
            json_str = str(json_data)

        detections_list = json.loads(json_str)

        # Handle dictionary format (with metadata) vs direct list format
        if isinstance(detections_list, dict):
            detections_list = detections_list.get("detections", [])

        return [Detection.from_dict(d) for d in detections_list]

    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error: {e}")
        logger.error(f"Raw data type: {type(json_data)}")
        if isinstance(json_data, np.ndarray):
            logger.error(f"Numpy dtype: {json_data.dtype}, shape: {json_data.shape}")
        return []
    except Exception as e:
        logger.error(f"Parsing error: {e}", exc_info=True)
        return []


def draw_detections(
    frame: np.ndarray,
    detections: List[Detection],
    class_names: List[str],
) -> np.ndarray:
    """Draw bounding boxes and labels on the frame.

    Args:
        frame: Input frame as numpy array (BGR format).
        detections: List of Detection objects to draw.
        class_names: List of class names for labeling.

    Returns:
        Annotated frame with bounding boxes and labels drawn.
    """
    annotated = frame.copy()
    np.random.seed(42)
    colors = np.random.randint(0, 255, size=(len(class_names), 3), dtype=np.uint8)

    # Log frame and detection info for debugging
    if len(detections) > 0:
        h, w = frame.shape[:2]
        det = detections[0]
        logger.debug(
            f"Frame: {w}x{h}, Detection: "
            f"[{det.x1:.1f}, {det.y1:.1f}, {det.x2:.1f}, {det.y2:.1f}]"
        )
        logger.debug(
            f"Detection position: x={det.x1/w:.1%} of width, "
            f"y={det.y1/h:.1%} of height"
        )

    for det in detections:
        # Ensure coordinates are integers and in correct order
        x1, y1, x2, y2 = int(det.x1), int(det.y1), int(det.x2), int(det.y2)
        class_id = det.class_id

        # Validate coordinates
        if x1 < 0 or y1 < 0 or x2 <= x1 or y2 <= y1:
            logger.warning(f"Invalid bbox: ({x1}, {y1}, {x2}, {y2})")
            continue

        # Clip to frame boundaries
        h, w = frame.shape[:2]
        x1 = max(0, min(x1, w))
        y1 = max(0, min(y1, h))
        x2 = max(0, min(x2, w))
        y2 = max(0, min(y2, h))

        if x2 <= x1 or y2 <= y1:
            logger.warning(
                f"Bbox outside frame after clipping: ({x1}, {y1}, {x2}, {y2})"
            )
            continue

        class_name = (
            class_names[class_id]
            if 0 <= class_id < len(class_names)
            else f"class_{class_id}"
        )
        color = (
            tuple(int(c) for c in colors[class_id])
            if 0 <= class_id < len(class_names)
            else (0, 255, 0)
        )

        # Draw rectangle
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

        # Draw label
        label = f"{class_name} {det.conf:.2f}"
        (label_w, label_h), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1
        )

        # Ensure label background stays within frame
        label_y1 = max(y1 - label_h - baseline - 5, 0)
        label_y2 = label_y1 + label_h + baseline + 5

        cv2.rectangle(
            annotated, (x1, label_y1), (x1 + label_w, label_y2), color, -1
        )
        cv2.putText(
            annotated,
            label,
            (x1, label_y2 - baseline - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    return annotated


class TritonGrpcClient:
    """gRPC client for Triton Inference Server (Decoupled Mode support).

    This class provides both synchronous and asynchronous (decoupled mode)
    inference capabilities for object detection.

    Attributes:
        url: Triton server URL.
        verbose: Verbose logging flag.
        client: Underlying gRPC inference client.
        is_decoupled: Whether the model uses decoupled transaction policy.

    Example:
        Basic usage::
            client = TritonGrpcClient("localhost:8001")
            detections = client.infer_sync(jpeg_bytes)
            client.close()
    """

    def __init__(self, url: str = DEFAULT_URL, verbose: bool = False) -> None:
        """Initialize Triton gRPC client.

        Args:
            url: Triton server URL. Defaults to DEFAULT_URL.
            verbose: Enable verbose logging. Defaults to False.

        Raises:
            ConnectionError: If server is not live.
            RuntimeError: If model is not ready.
        """
        self.url = url
        self.verbose = verbose
        self.client = grpcclient.InferenceServerClient(url=url, verbose=verbose)

        if not self.client.is_server_live():
            raise ConnectionError(f"Triton server at {url} is not live")
        if not self.client.is_model_ready(MODEL_NAME):
            raise RuntimeError(f"Model {MODEL_NAME} is not ready")
        
        # Check if model is decoupled
        self.is_decoupled = self._check_decoupled()
        if self.is_decoupled:
            logger.info(
                f"Model {MODEL_NAME} uses decoupled transaction policy. "
                f"Only async/streaming mode is supported."
            )
        else:
            logger.info(f"Model {MODEL_NAME} supports synchronous inference.")
        
        logger.info(f"Connected to Triton server at {url}, model {MODEL_NAME} ready")

    def _check_decoupled(self) -> bool:
        """Check if model uses decoupled transaction policy.

        Returns:
            True if model is decoupled, False otherwise.
        """
        try:
            # Try JSON format first
            try:
                model_config = self.client.get_model_config(MODEL_NAME, as_json=True)
                if isinstance(model_config, dict):
                    transaction_policy = model_config.get('model_transaction_policy', {})
                    decoupled = transaction_policy.get('decoupled', False)
                    if self.verbose:
                        logger.debug(
                            f"Model config (JSON): "
                            f"transaction_policy={transaction_policy}, "
                            f"decoupled={decoupled}"
                        )
                    return decoupled
            except Exception as json_err:
                if self.verbose:
                    logger.debug(f"JSON format failed: {json_err}, trying protobuf")
            
            # Fallback: try protobuf format
            model_config = self.client.get_model_config(MODEL_NAME, as_json=False)
            if hasattr(model_config, 'model_transaction_policy'):
                policy = model_config.model_transaction_policy
                if hasattr(policy, 'decoupled'):
                    decoupled = policy.decoupled
                    if self.verbose:
                        logger.debug(f"Model config (protobuf): decoupled={decoupled}")
                    return decoupled
            
            if self.verbose:
                logger.debug("No decoupled policy found in model config")
            return False
        except Exception as e:
            logger.warning(
                f"Could not check decoupled status: {e}. "
                f"Assuming non-decoupled. Error details: {type(e).__name__}"
            )
            if self.verbose:
                import traceback
                logger.debug(traceback.format_exc())
            return False

    def infer_sync(self, jpeg_bytes: bytes) -> List[Detection]:
        """Perform synchronous inference (non-decoupled mode).

        Args:
            jpeg_bytes: JPEG-encoded image bytes.

        Returns:
            List of Detection objects.

        Raises:
            RuntimeError: If model uses decoupled transaction policy.
        """
        if self.is_decoupled:
            raise RuntimeError(
                f"Model {MODEL_NAME} uses decoupled transaction policy. "
                f"Synchronous inference is not supported. "
                f"Please use async/streaming mode (--async_mode true)."
            )
        
        input_data = np.frombuffer(jpeg_bytes, dtype=np.uint8)
        input_shape = input_data.shape

        inputs = [grpcclient.InferInput(INPUT_NAME, input_shape, "UINT8")]
        inputs[0].set_data_from_numpy(input_data)

        outputs = [grpcclient.InferRequestedOutput(OUTPUT_NAME)]

        try:
            response = self.client.infer(
                model_name=MODEL_NAME, inputs=inputs, outputs=outputs
            )
            json_output = response.as_numpy(OUTPUT_NAME)
            return parse_json_response(json_output)
        except InferenceServerException as e:
            # Check if error is due to decoupled model
            error_msg = str(e)
            if "decoupled transaction policy" in error_msg.lower() or "UNIMPLEMENTED" in error_msg:
                # Update is_decoupled flag for future calls
                self.is_decoupled = True
                raise RuntimeError(
                    f"Model {MODEL_NAME} uses decoupled transaction policy. "
                    f"Synchronous inference is not supported. "
                    f"Please use async/streaming mode (--async_mode true)."
                ) from e
            raise

    def start_stream(self, callback: Callable[..., None]) -> None:
        """Start gRPC bidirectional stream (Decoupled Mode).

        Args:
            callback: Callback function for handling stream responses.
        """
        self.client.start_stream(callback=callback)
        logger.info("Started gRPC stream")

    def stop_stream(self) -> None:
        """Stop gRPC bidirectional stream."""
        self.client.stop_stream()
        logger.info("Stopped gRPC stream")

    def infer_stream(self, jpeg_bytes: bytes, request_id: str) -> None:
        """Send inference request through decoupled stream.

        Args:
            jpeg_bytes: JPEG-encoded image bytes.
            request_id: Request identifier for response matching.
        """
        input_data = np.frombuffer(jpeg_bytes, dtype=np.uint8)
        input_shape = input_data.shape

        inputs = [grpcclient.InferInput(INPUT_NAME, input_shape, "UINT8")]
        inputs[0].set_data_from_numpy(input_data)

        outputs = [grpcclient.InferRequestedOutput(OUTPUT_NAME)]

        # Decoupled mode: use async_stream_infer
        self.client.async_stream_infer(
            model_name=MODEL_NAME,
            inputs=inputs,
            outputs=outputs,
            request_id=request_id,
        )

    def close(self) -> None:
        """Close the client connection."""
        self.client.close()
        logger.info("Closed Triton client connection")


def infer_video(
    video_path: str = "0",
    output_path: Optional[str] = None,
    url: str = DEFAULT_URL,
    verbose: bool = False,
    jpeg_quality: int = 80,
    show_display: Optional[bool] = None,
    save_output: bool = False,
    async_mode: bool = False,
    max_async_requests: int = 16,
    timeout: float = 10.0,
) -> None:
    """Perform object detection on video file or webcam.

    Processes video frames and performs object detection using Triton
    Inference Server. Supports both synchronous and asynchronous
    (decoupled) modes.

    Args:
        video_path: Video file path or "0" for webcam. Defaults to "0".
        output_path: Output video save path. Defaults to None.
        url: Triton server URL. Defaults to DEFAULT_URL.
        verbose: Enable verbose logging. Defaults to False.
        jpeg_quality: JPEG compression quality (1-100). Defaults to 80.
        show_display: Show display window. Defaults to None (auto-detect).
        save_output: Save output video. Defaults to False.
        async_mode: Use async/streaming mode. Defaults to False.
        max_async_requests: Maximum concurrent async requests.
            Defaults to 16.
        timeout: Response timeout in seconds. Defaults to 10.0.

    Raises:
        FileNotFoundError: If video file does not exist.
        ValueError: If video source cannot be opened.
        ConnectionError: If cannot connect to Triton server.
    """
    
    if show_display is None:
        show_display = DISPLAY_AVAILABLE
    elif show_display and not DISPLAY_AVAILABLE:
        logger.warning("Display not available. Disabling show_display.")
        show_display = False

    # Setup video source
    if video_path.isdigit():
        video_source = int(video_path)
        is_webcam = True
    else:
        video_path_obj = Path(video_path)
        if not video_path_obj.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
        video_source = str(video_path_obj)
        is_webcam = False
    
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        raise ValueError(f"Failed to open video source: {video_source}")
    
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = (
        int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if not is_webcam else -1
    )

    logger.info(
        f"Video info: Resolution: {width}x{height}, FPS: {fps}, "
        f"Total frames: {total_frames}"
    )

    # Read and check first frame
    ret, test_frame = cap.read()
    if ret:
        logger.debug(
            f"Actual frame shape from cv2.read(): {test_frame.shape}"
        )
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset to beginning
    else:
        raise ValueError("Failed to read first frame")

    # Setup output video writer
    video_writer = None
    if save_output:
        if output_path is None:
            if is_webcam:
                output_path = "webcam_output.mp4"
            else:
                video_path_obj = Path(video_path)
                output_path = str(
                    video_path_obj.parent
                    / f"{video_path_obj.stem}_output{video_path_obj.suffix}"
                )
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(
            output_path, fourcc, fps, (width, height)
        )
        logger.info(f"Output will be saved to: {output_path}")

    class_names = load_class_names()
    logger.info(f"Connecting to Triton server at {url}...")
    try:
        client = TritonGrpcClient(url=url, verbose=verbose)
        logger.info("Successfully connected to Triton server")
    except Exception as e:
        logger.error(f"Failed to connect to Triton server: {e}", exc_info=True)
        raise

    # Auto-enable async mode for decoupled models
    # If initial check failed, try a test inference to detect decoupled model
    if not client.is_decoupled and not async_mode:
        try:
            # Try a test inference with first frame to detect decoupled model
            test_jpeg = encode_frame_to_jpeg(test_frame, quality=jpeg_quality)
            try:
                client.infer_sync(test_jpeg)
            except RuntimeError as e:
                if "decoupled transaction policy" in str(e):
                    logger.warning(
                        f"Model {MODEL_NAME} uses decoupled transaction policy. "
                        f"Automatically enabling async/streaming mode."
                    )
                    async_mode = True
                    # Reset video capture to beginning
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        except Exception:
            # If test fails for other reasons, continue normally
            pass
    
    if client.is_decoupled and not async_mode:
        logger.warning(
            f"Model {MODEL_NAME} uses decoupled transaction policy. "
            f"Automatically enabling async/streaming mode."
        )
        async_mode = True

    # State management for Decoupled Mode
    pending_frames: Dict[str, tuple[np.ndarray, float, int]] = {}  # frame, time, frame_number
    pending_lock = threading.Lock()
    result_queue: queue.Queue[tuple[str, Optional[str], List[Detection], Optional[Exception]]] = queue.Queue()
    completed_requests: set[str] = set()
    # Frame ordering buffer for sequential output
    frame_buffer: Dict[int, tuple[np.ndarray, List[Detection]]] = {}
    next_frame_to_write = 0
    
    if async_mode:
        # Decoupled streaming callback function
        def stream_callback(
            result: Any, error: Optional[Exception]
        ) -> None:
            """Decoupled mode callback.

            Handles stream responses:
            - Multiple responses may come for each request
            - Last response may have result as None

            Args:
                result: Inference result or None.
                error: Error object if error occurred.
            """
            logger.debug(
                f"Stream callback called - result type: {type(result)}, "
                f"error: {error}"
            )

            if error:
                logger.error(f"Callback error: {error}")
                result_queue.put(("ERROR", None, [], error))
                return

            # None result indicates stream end signal
            if result is None:
                logger.info("Stream end signal received")
                return
            
            try:
                response = result.get_response()
                req_id = response.id

                # Parse actual output data
                detections: List[Detection] = []
                has_output = False

                try:
                    json_output = result.as_numpy(OUTPUT_NAME)
                    logger.debug(
                        f"Output type: {type(json_output)}, "
                        f"dtype: {json_output.dtype if hasattr(json_output, 'dtype') else 'N/A'}"
                    )

                    detections = parse_json_response(json_output)
                    has_output = True

                    if len(detections) > 0:
                        det = detections[0]
                        logger.debug(
                            f"Request {req_id} - First detection bbox: "
                            f"[{det.x1:.1f}, {det.y1:.1f}, {det.x2:.1f}, {det.y2:.1f}]"
                        )

                except Exception as e:
                    logger.warning(
                        f"No output data for request_id: {req_id}, error: {e}"
                    )

                # Add result to queue
                if has_output:
                    result_queue.put(("DATA", req_id, detections, None))
                else:
                    # Treat as completion signal if no data
                    result_queue.put(("COMPLETE", req_id, [], None))

            except Exception as e:
                logger.error(f"Callback exception: {e}", exc_info=True)
                result_queue.put(("ERROR", None, [], e))

        # Start stream
        client.start_stream(callback=stream_callback)
        logger.info(
            f"Async Streaming Mode Started (Decoupled, "
            f"max_requests={max_async_requests})"
        )

    # Main processing loop
    frame_count = 0
    sent_count = 0
    start_time = time.perf_counter()
    last_frame_time: Dict[str, float] = {}

    logger.info("Starting inference... (Press 'q' to quit)")
    
    try:
        while True:
            # 결과 처리 (논블로킹)
            processed_in_loop = 0
            while not result_queue.empty():
                try:
                    msg_type, rid, detections, error = result_queue.get_nowait()
                    processed_in_loop += 1
                    
                    if msg_type == "ERROR":
                        logger.error(f"Inference error: {error}")
                        continue

                    elif msg_type == "DATA":
                        logger.debug(
                            f"Processing DATA for request_id: {rid}, "
                            f"detections: {len(detections)}"
                        )

                        # Process actual detection results
                        orig_frame: Optional[np.ndarray] = None
                        frame_number: Optional[int] = None
                        with pending_lock:
                            if rid in pending_frames:
                                orig_frame, _, frame_number = pending_frames[rid]
                                # Remove after processing
                                pending_frames.pop(rid)
                                if rid in last_frame_time:
                                    del last_frame_time[rid]

                        if orig_frame is not None and frame_number is not None:
                            # Store in buffer for sequential writing
                            with pending_lock:
                                frame_buffer[frame_number] = (orig_frame, detections)
                            
                            # Write frames in order
                            while True:
                                with pending_lock:
                                    if next_frame_to_write in frame_buffer:
                                        buffered_frame, buffered_detections = frame_buffer.pop(next_frame_to_write)
                                        
                                        annotated = draw_detections(
                                            buffered_frame, buffered_detections, class_names
                                        )

                                        if show_display:
                                            cv2.imshow(
                                                "Object Detection (Decoupled)", annotated
                                            )
                                        if video_writer:
                                            video_writer.write(annotated)

                                        frame_count += 1
                                        next_frame_to_write += 1
                                        logger.debug(
                                            f"Frame {frame_count} (frame_number={next_frame_to_write-1}) "
                                            f"written successfully"
                                        )
                                    else:
                                        break
                        else:
                            logger.warning(
                                f"No frame found for request_id: {rid}"
                            )

                    elif msg_type == "COMPLETE":
                        logger.debug(f"COMPLETE signal for request_id: {rid}")
                        # Completion signal - remove from pending
                        with pending_lock:
                            if rid in pending_frames:
                                pending_frames.pop(rid)
                            completed_requests.add(rid)
                            if rid in last_frame_time:
                                del last_frame_time[rid]
                        
                except queue.Empty:
                    break
            
            if processed_in_loop > 0:
                logger.debug(
                    f"Processed {processed_in_loop} messages from queue"
                )

            # Timeout check (clean up requests without responses)
            current_time = time.time()
            with pending_lock:
                timeout_requests = [
                    rid
                    for rid, t in last_frame_time.items()
                    if current_time - t > timeout
                ]
                for rid in timeout_requests:
                    logger.warning(f"Request {rid} timed out")
                    if rid in pending_frames:
                        pending_frames.pop(rid)
                    del last_frame_time[rid]

            # Read and send frames (throttling)
            with pending_lock:
                current_pending = len(pending_frames)

            if async_mode and current_pending >= max_async_requests:
                time.sleep(0.001)
                if show_display and (cv2.waitKey(1) & 0xFF == ord('q')):
                    break
                continue

            ret, frame = cap.read()
            if not ret:
                break

            jpeg_bytes = encode_frame_to_jpeg(frame, quality=jpeg_quality)

            if async_mode:
                # Decoupled streaming send
                req_id = str(sent_count)
                frame_number = sent_count

                with pending_lock:
                    pending_frames[req_id] = (frame.copy(), current_time, frame_number)
                    last_frame_time[req_id] = current_time

                logger.debug(
                    f"Sending frame {sent_count} (request_id: {req_id}, "
                    f"frame_number: {frame_number})"
                )
                client.infer_stream(jpeg_bytes, request_id=req_id)
                sent_count += 1

            else:
                # Synchronous mode
                detections = client.infer_sync(jpeg_bytes)
                annotated = draw_detections(frame, detections, class_names)

                if show_display:
                    cv2.imshow("Object Detection (Sync)", annotated)
                if video_writer:
                    video_writer.write(annotated)

                frame_count += 1
            
            if show_display and (cv2.waitKey(1) & 0xFF == ord('q')):
                break

        # Process remaining responses after loop ends
        if async_mode:
            logger.info(
                f"Waiting for remaining responses... "
                f"(sent={sent_count}, processed={frame_count})"
            )
            wait_start = time.time()

            while True:
                with pending_lock:
                    remaining = len(pending_frames)

                if remaining == 0:
                    logger.info("All responses received.")
                    break

                if time.time() - wait_start > timeout:
                    logger.warning(f"Timeout: {remaining} frames still pending.")
                    break

                try:
                    msg_type, rid, detections, error = result_queue.get(
                        timeout=0.1
                    )

                    if msg_type == "DATA":
                        if verbose:
                            logger.debug(
                                f"Processing DATA for request_id: {rid}"
                            )

                        orig_frame = None
                        frame_number: Optional[int] = None
                        with pending_lock:
                            if rid in pending_frames:
                                orig_frame, _, frame_number = pending_frames[rid]
                                pending_frames.pop(rid)
                                if rid in last_frame_time:
                                    del last_frame_time[rid]

                        if orig_frame is not None and frame_number is not None:
                            # Store in buffer for sequential writing
                            with pending_lock:
                                frame_buffer[frame_number] = (orig_frame, detections)
                            
                            # Write frames in order
                            while True:
                                with pending_lock:
                                    if next_frame_to_write in frame_buffer:
                                        buffered_frame, buffered_detections = frame_buffer.pop(next_frame_to_write)
                                        
                                        annotated = draw_detections(
                                            buffered_frame, buffered_detections, class_names
                                        )
                                        if show_display:
                                            cv2.imshow(
                                                "Object Detection (Decoupled)", annotated
                                            )
                                            cv2.waitKey(1)
                                        if video_writer:
                                            video_writer.write(annotated)
                                        frame_count += 1
                                        next_frame_to_write += 1
                                    else:
                                        break
                        else:
                            if verbose:
                                logger.debug(
                                    f"No frame for request_id: {rid}"
                                )

                    elif msg_type == "COMPLETE":
                        if verbose:
                            logger.debug(
                                f"COMPLETE signal for request_id: {rid}"
                            )
                        with pending_lock:
                            if rid in pending_frames:
                                pending_frames.pop(rid)
                            if rid in last_frame_time:
                                del last_frame_time[rid]

                except queue.Empty:
                    # Check if we can write buffered frames
                    with pending_lock:
                        if next_frame_to_write in frame_buffer:
                            buffered_frame, buffered_detections = frame_buffer.pop(next_frame_to_write)
                            annotated = draw_detections(
                                buffered_frame, buffered_detections, class_names
                            )
                            if show_display:
                                cv2.imshow("Object Detection (Decoupled)", annotated)
                                cv2.waitKey(1)
                            if video_writer:
                                video_writer.write(annotated)
                            frame_count += 1
                            next_frame_to_write += 1
                    continue
            
            # Write any remaining buffered frames in order
            logger.info("Writing remaining buffered frames...")
            while True:
                with pending_lock:
                    if next_frame_to_write in frame_buffer:
                        buffered_frame, buffered_detections = frame_buffer.pop(next_frame_to_write)
                        annotated = draw_detections(
                            buffered_frame, buffered_detections, class_names
                        )
                        if show_display:
                            cv2.imshow("Object Detection (Decoupled)", annotated)
                            cv2.waitKey(1)
                        if video_writer:
                            video_writer.write(annotated)
                        frame_count += 1
                        next_frame_to_write += 1
                    else:
                        break
            
            if len(frame_buffer) > 0:
                logger.warning(
                    f"Skipping {len(frame_buffer)} frames that arrived out of order "
                    f"(next expected: {next_frame_to_write})"
                )

    finally:
        if async_mode:
            client.stop_stream()
        cap.release()
        if video_writer:
            video_writer.release()
        if show_display:
            cv2.destroyAllWindows()
        client.close()

    # Print statistics
    total_time = time.perf_counter() - start_time
    logger.info("=" * 60)
    logger.info("Inference Complete")
    logger.info("=" * 60)
    if async_mode:
        logger.info(f"Frames sent: {sent_count}")
    logger.info(f"Frames processed: {frame_count}")
    logger.info(f"Total time: {total_time:.2f}s")
    logger.info(f"Average FPS: {frame_count/total_time:.2f}")
    logger.info("=" * 60)


def infer_image(
    image_path: str, url: str = DEFAULT_URL, verbose: bool = False
) -> None:
    """Perform object detection on a single image.

    Args:
        image_path: Path to input image file.
        url: Triton server URL. Defaults to DEFAULT_URL.
        verbose: Enable verbose logging. Defaults to False.

    Raises:
        FileNotFoundError: If image file does not exist.
        RuntimeError: If model uses decoupled transaction policy.
    """
    image_path_obj = Path(image_path)
    if not image_path_obj.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")

    image = cv2.imread(str(image_path_obj))
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")

    client = TritonGrpcClient(url=url, verbose=verbose)

    try:
        if client.is_decoupled:
            raise RuntimeError(
                f"Model {MODEL_NAME} uses decoupled transaction policy. "
                f"Image inference is not supported with decoupled models. "
                f"Please use video inference with --async_mode true instead."
            )
        
        jpeg_bytes = encode_frame_to_jpeg(image)
        start = time.time()
        detections = client.infer_sync(jpeg_bytes)
        inference_time = (time.time() - start) * 1000

        logger.info(f"Inference time: {inference_time:.1f}ms")
        logger.info(f"Detected {len(detections)} objects")

        annotated = draw_detections(image, detections, load_class_names())
        out_path = image_path_obj.parent / f"{image_path_obj.stem}_out.jpg"
        cv2.imwrite(str(out_path), annotated)
        logger.info(f"Saved to {out_path}")
    finally:
        client.close()


def health_check(url: str = DEFAULT_URL) -> None:
    """Check Triton server health status.

    Args:
        url: Triton server URL. Defaults to DEFAULT_URL.
    """
    try:
        client = grpcclient.InferenceServerClient(url=url)
        logger.info(f"Connecting to: {url}")
        logger.info(f"Server Live: {client.is_server_live()}")
        logger.info(f"Server Ready: {client.is_server_ready()}")
        logger.info(
            f"Model '{MODEL_NAME}' Ready: "
            f"{client.is_model_ready(MODEL_NAME)}"
        )
    except Exception as e:
        logger.error(f"Connection failed: {e}", exc_info=True)


def _str_to_bool(v: Any) -> bool:
    """Convert string to boolean for argparse.

    Args:
        v: Value to convert.

    Returns:
        Boolean value.

    Raises:
        argparse.ArgumentTypeError: If value cannot be converted.
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError(
            f'Boolean value expected, got: {v}'
        )


def main() -> None:
    """Main entry point for command-line interface."""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    )

    parser = argparse.ArgumentParser(
        description="Triton gRPC Client for Object Detection"
    )
    subparsers = parser.add_subparsers(
        dest="command", help="Available commands"
    )

    # infer_video command
    video_parser = subparsers.add_parser(
        "infer_video", help="Run object detection on video"
    )
    video_parser.add_argument(
        "--video_path",
        type=str,
        default="0",
        help="Video file path or '0' for webcam",
    )
    video_parser.add_argument(
        "--output_path", type=str, default=None, help="Output video path"
    )
    video_parser.add_argument(
        "--url", type=str, default=DEFAULT_URL, help="Triton server URL"
    )
    video_parser.add_argument(
        "--verbose", action="store_true", help="Verbose output"
    )
    video_parser.add_argument(
        "--jpeg_quality",
        type=int,
        default=80,
        help="JPEG compression quality (1-100)",
    )
    video_parser.add_argument(
        "--show_display", action="store_true", help="Show display window"
    )
    video_parser.add_argument(
        "--save_output",
        type=_str_to_bool,
        nargs='?',
        const=True,
        default=False,
        help="Save output video",
    )
    video_parser.add_argument(
        "--async_mode",
        "-async_mode",
        type=_str_to_bool,
        nargs='?',
        const=True,
        default=False,
        help="Use async/streaming mode",
    )
    video_parser.add_argument(
        "--max_async_requests",
        type=int,
        default=16,
        help="Maximum concurrent async requests",
    )
    video_parser.add_argument(
        "--timeout",
        type=float,
        default=10.0,
        help="Response timeout in seconds",
    )

    # infer_image command
    image_parser = subparsers.add_parser(
        "infer_image", help="Run object detection on image"
    )
    image_parser.add_argument(
        "--image_path", type=str, required=True, help="Image file path"
    )
    image_parser.add_argument(
        "--url", type=str, default=DEFAULT_URL, help="Triton server URL"
    )
    image_parser.add_argument(
        "--verbose", action="store_true", help="Verbose output"
    )

    # health_check command
    health_parser = subparsers.add_parser(
        "health_check", help="Check Triton server health"
    )
    health_parser.add_argument(
        "--url", type=str, default=DEFAULT_URL, help="Triton server URL"
    )

    args = parser.parse_args()

    if args.command == "infer_video":
        infer_video(
            video_path=args.video_path,
            output_path=args.output_path,
            url=args.url,
            verbose=args.verbose,
            jpeg_quality=args.jpeg_quality,
            show_display=args.show_display if hasattr(args, 'show_display') else None,
            save_output=args.save_output,
            async_mode=args.async_mode,
            max_async_requests=args.max_async_requests,
            timeout=args.timeout,
        )
    elif args.command == "infer_image":
        infer_image(
            image_path=args.image_path,
            url=args.url,
            verbose=args.verbose,
        )
    elif args.command == "health_check":
        health_check(url=args.url)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()