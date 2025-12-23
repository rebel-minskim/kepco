# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#
# model.py
#
"""Triton Python Backend Model for YOLOv11 with RBLN Runtime.

This module implements a pipelined inference model using Rebellions RBLN
runtime for NPU acceleration. It supports decoupled mode for improved
throughput.
"""
import asyncio
import base64
import json
import queue
import re
import threading
import psutil
import os
from pathlib import Path
from typing import Any, Union

import cv2
import numpy as np
import rebel  # RBLN Runtime
import triton_python_backend_utils as pb_utils
import torch
from ultralytics.data.augment import LetterBox
from ultralytics.utils.nms import non_max_suppression
from ultralytics.utils.ops import scale_boxes

# Constants
NAME_RE = re.compile(r"^(?P<model>.+)_(?P<group>\d+)_(?P<inst>\d+)$")
INPUT_SHAPE = (640, 640)

def postprocess_to_json(
    outputs: np.ndarray,
    origin_shape: tuple[int, int, int],
    class_names: list[str],
    conf_thres: float = 0.25,
    iou_thres: float = 0.45,
    max_det: int = 300,
) -> dict[str, Any]:
    """Postprocess YOLOv11 outputs to JSON format using Ultralytics functions.

    YOLOv11 output format: [1, 84, 8400] or [84, 8400]
    - 84 = 4 (bbox) + 80 (class scores)
    - 8400 = number of anchor boxes

    Args:
        outputs: Model output tensor [1, 84, 8400] or [84, 8400].
        origin_shape: Original image shape (H, W, C).
        class_names: List of class names.
        conf_thres: Confidence threshold. Defaults to 0.25.
        iou_thres: IoU threshold for NMS. Defaults to 0.45.
        max_det: Maximum detections. Defaults to 300.

    Returns:
        Dictionary containing detection results in JSON format.
    """
    
    # Convert to torch tensor for ultralytics ops
    if isinstance(outputs, np.ndarray):
        outputs = torch.from_numpy(outputs)
    
    # Ensure shape is [batch, 84, 8400]
    if outputs.ndim == 2:
        outputs = outputs.unsqueeze(0)  # [84, 8400] -> [1, 84, 8400]
    
    # Use ultralytics non_max_suppression
    # Input: [batch, 84, num_boxes] where first 4 are bbox, rest are class scores
    # Signature: non_max_suppression(prediction, conf_threshold, iou_threshold,
    #                                 classes, agnostic, max_det)
    pred = non_max_suppression(
        outputs,
        conf_thres,
        iou_thres,
        None,  # classes (None = all classes)
        False,  # agnostic (False = class-aware NMS)
        max_det=max_det,
    )

    # pred is a list of tensors, one per batch
    if len(pred) == 0 or len(pred[0]) == 0:
        return {"num_detections": 0, "detections": []}

    # Get first batch result: [num_det, 6] format [x1, y1, x2, y2, conf, cls]
    pred = pred[0].cpu().numpy()

    # Scale boxes back to original image size using ultralytics scale_boxes
    # Convert back to torch for scale_boxes
    pred_torch = torch.from_numpy(pred)

    # scale_boxes expects: img1_shape (preprocessed), boxes, img0_shape (original)
    pred_torch[:, :4] = scale_boxes(
        INPUT_SHAPE,  # (640, 640)
        pred_torch[:, :4],  # boxes in xyxy format
        origin_shape[:2],  # (H, W) of original image
    )

    pred = pred_torch.numpy()

    # Convert to JSON format
    dets = []
    for p in pred:
        cid = int(p[5])
        dets.append({
            "bbox": [float(p[0]), float(p[1]), float(p[2]), float(p[3])],
            "confidence": float(p[4]),
            "class_id": cid,
            "class_name": (
                class_names[cid]
                if cid < len(class_names)
                else str(cid)
            ),
        })

    return {"num_detections": len(dets), "detections": dets}


def decode_and_preprocess(
    inp: Union[str, bytes, np.ndarray],
) -> tuple[np.ndarray, tuple[int, int, int]]:
    """Decode and preprocess input image data using Ultralytics LetterBox.

    Supports multiple input formats:
    - Base64-encoded strings
    - Raw bytes
    - NumPy arrays

    Args:
        inp: Input data (Base64 string, bytes, or numpy array).

    Returns:
        Tuple containing:
            - Preprocessed image tensor [C, H, W] as float32.
            - Original image shape (H, W, C).
    """
    # 1. Robust Data Extraction
    if isinstance(inp, np.ndarray) and inp.dtype == object:
        inp = inp.item()
    
    # 2. Base64 Decoding
    if isinstance(inp, str):
        try:
            inp = base64.b64decode(inp, validate=True)
        except Exception as e:
            pb_utils.Logger.log_error(f"Base64 decode error: {e}")
            return np.zeros((3, 640, 640), dtype=np.float32), (640, 640, 3)
    
    # 3. Convert bytes to numpy array if needed
    if isinstance(inp, bytes):
        inp = np.frombuffer(inp, dtype=np.uint8)
    
    # 4. Decode image
    img = cv2.imdecode(inp, cv2.IMREAD_COLOR)
    if img is None:
        pb_utils.Logger.log_error("Failed to decode image")
        return np.zeros((3, 640, 640), dtype=np.float32), (640, 640, 3)

    # 5. Preprocess using Ultralytics LetterBox
    img_pre = LetterBox(new_shape=INPUT_SHAPE, auto=False, stride=32)(image=img)

    # Convert to CHW format and normalize
    img_pre = img_pre.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    img_pre = np.ascontiguousarray(img_pre, dtype=np.float32) / 255.0
    return img_pre, img.shape


class TritonPythonModel:
    """Pipelined Triton Python Model for YOLOv11 with RBLN Runtime.

    Implements a three-stage pipeline:
    1. Preprocessing (CPU): Decode and resize images
    2. Inference (NPU): Run YOLOv11 model
    3. Postprocessing (CPU): NMS and JSON conversion

    Attributes:
        model_config: Parsed model configuration.
        module: RBLN runtime module.
        request_queue: Queue for incoming requests.
        infer_queue: Queue for preprocessed images.
        post_queue: Queue for inference outputs.
        shutdown_event: Event to signal shutdown.
        threads: List of worker threads.
    """

    def initialize(self, args: dict[str, Any]) -> None:
        """Initialize the model.

        Sets up RBLN runtime, queues, and worker threads for pipelined
        inference.

        Args:
            args: Model initialization arguments from Triton.
        """
        self.model_config = json.loads(args["model_config"])

        # Model Setup - Parse device_id first
        model_dir = Path(args["model_repository"]) / args["model_version"]
        rbln_path = model_dir / f"{args['model_name']}.rbln"
        m = NAME_RE.match(args["model_instance_name"])
        if m is None:
            raise ValueError(
                f"Invalid model instance name: {args['model_instance_name']}"
            )
        device_id = int(m.group("group"))

        # CPU Setup - Assign CPU cores based on device_id
        # Each device_id gets 2 CPU cores:
        proc = psutil.Process(os.getpid())
        
        # Get available CPU cores for this process (may be restricted by Triton)
        try:
            available_cores = proc.cpu_affinity()
        except Exception:
            # If cpu_affinity() fails, get all system CPUs
            available_cores = list(range(psutil.cpu_count(logical=True)))
        
        # Calculate target cores for this device_id (relative to available cores)
        # Map device_id to CPU cores within available range
        cores_per_device = 2
        start_idx = device_id * cores_per_device
        
        # Check if we have enough available cores
        if len(available_cores) < start_idx + cores_per_device:
            pb_utils.Logger.log_warn(
                f"Device {device_id} requested {cores_per_device} CPU cores starting at index {start_idx}, "
                f"but only {len(available_cores)} CPUs available: {available_cores}. "
                f"Skipping CPU affinity."
            )
        else:
            # Select cores from available cores list
            target_cores = available_cores[start_idx:start_idx + cores_per_device]
            
            # Set CPU Affinity
            try:
                proc.cpu_affinity(target_cores)
                pb_utils.Logger.log_info(
                    f"Device {device_id} pinned to CPU cores: {target_cores} "
                    f"(from available: {available_cores})"
                )
            except Exception as e:
                pb_utils.Logger.log_error(
                    f"Failed to set CPU affinity for device {device_id} to {target_cores}: {e}"
                )

        # RBLN AsyncRuntime with parallel execution
        self.module = rebel.AsyncRuntime(str(rbln_path), device=device_id, parallel=2)

        # Queues for Pipelining
        # Increased queue sizes to prevent starvation and maintain high NPU utilization
        self.request_queue: queue.Queue[Any] = queue.Queue(maxsize=512)
        self.infer_queue: queue.Queue[tuple[np.ndarray, tuple[int, int, int], Any]] = queue.Queue(maxsize=128)
        self.post_queue: queue.Queue[tuple[np.ndarray, tuple[int, int, int], Any]] = queue.Queue(maxsize=128)

        # Workers
        self.shutdown_event = threading.Event()
        self.threads: list[threading.Thread] = []

        # [Stage 1] Preprocessing Workers (CPU)
        # Increased to feed infer_queue faster and prevent NPU starvation
        # More workers ensure infer_queue always has data ready
        for _ in range(1):
            t = threading.Thread(target=self.preprocessing_loop)
            t.start()
            self.threads.append(t)

        # [Stage 2] Inference Worker (NPU)
        # Increased threads to maximize NPU utilization
        # With AsyncRuntime parallel=2, more threads ensure NPU stays busy
        # Each thread processes sequentially, but multiple threads run concurrently
        for _ in range(2):
            t_inf = threading.Thread(target=self.inference_loop)
            t_inf.start()
            self.threads.append(t_inf)

        # [Stage 3] Postprocessing Workers (CPU)
        # Increased to prevent post_queue from backing up
        # More workers ensure pipeline doesn't stall
        for _ in range(1):
            t = threading.Thread(target=self.postprocessing_loop)
            t.start()
            self.threads.append(t)

        pb_utils.Logger.log_info(
            f"Pipeline Initialized on Device {device_id}"
        )

    def execute(self, requests: list[Any]) -> None:
        """Execute inference requests (Decoupled Mode).

        In decoupled mode, requests are queued and function returns immediately.
        Responses are sent asynchronously through the pipeline.

        Args:
            requests: List of inference requests.
        """
        # Decoupled mode: queue requests and return immediately
        for request in requests:
            self.request_queue.put(request)
        return None

    def preprocessing_loop(self) -> None:
        """Stage 1: CPU preprocessing loop (Decode & Resize).

        Continuously processes requests from request_queue, decodes images,
        and preprocesses them using LetterBox. Results are sent to
        infer_queue.
        """
        while not self.shutdown_event.is_set():
            try:
                # Reduced timeout to minimize idle time
                req = self.request_queue.get(timeout=0.001)
            except queue.Empty:
                continue
            
            try:
                inp_tensor = pb_utils.get_input_tensor_by_name(req, "INPUT__0")
                if inp_tensor is None:
                    continue

                sender = req.get_response_sender()
                input_np = inp_tensor.as_numpy()
                
                if input_np.dtype == object:
                    raw_data = input_np[0]
                else:
                    raw_data = input_np if input_np.ndim == 1 else input_np[0]
                        
                # CPU preprocessing
                img_pre, shape = decode_and_preprocess(raw_data)

                # Pass to next stage
                self.infer_queue.put((img_pre, shape, sender))

            except Exception as e:
                pb_utils.Logger.log_error(f"Prep Error: {e}")
                if 'sender' in locals():
                    error_response = pb_utils.InferenceResponse(
                        error=pb_utils.TritonError(str(e))
                    )
                    sender.send(error_response, flags=pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL)

    def inference_loop(self) -> None:
        """Stage 2: NPU inference loop.

        Continuously processes preprocessed images from infer_queue,
        runs inference on NPU, and sends results to post_queue.

        Uses sequential processing with run_until_complete for optimal NPU utilization.
        AsyncRuntime's parallel=2 handles internal concurrency, so multiple threads
        with sequential processing achieve better utilization than async task overhead.

        Creates and maintains a dedicated event loop for this thread
        to handle AsyncRuntime coroutines efficiently.
        """
        # Create a dedicated event loop for this thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            while not self.shutdown_event.is_set():
                try:
                    # Reduced timeout to minimize idle time and maximize NPU utilization
                    data = self.infer_queue.get(timeout=0.001)
                except queue.Empty:
                    continue

                img_pre, shape, sender = data

                try:
                    # Expand batch dimension: [C, H, W] -> [1, C, H, W]
                    input_tensor = np.expand_dims(img_pre, axis=0)

                    # Run NPU inference (AsyncRuntime returns AsyncTask coroutine)
                    async_task = self.module.async_run(input_tensor)

                    # Execute coroutine using this thread's event loop
                    # Sequential processing per thread, but multiple threads run concurrently
                    outputs = loop.run_until_complete(async_task)

                    if isinstance(outputs, (list, tuple)):
                        outputs = outputs[0]

                    # Remove batch dimension: [1, 84, 8400] -> [84, 8400]
                    if outputs.ndim == 3 and outputs.shape[0] == 1:
                        outputs = outputs[0]

                    # Pass to next stage
                    self.post_queue.put((outputs, shape, sender))

                except Exception as e:
                    pb_utils.Logger.log_error(f"Infer Error: {e}")
                    error_response = pb_utils.InferenceResponse(
                        error=pb_utils.TritonError(str(e))
                    )
                    sender.send(
                        error_response,
                        flags=pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL,
                    )
        finally:
            # Clean up event loop when thread exits
            loop.close()

    def postprocessing_loop(self) -> None:
        """Stage 3: CPU postprocessing loop (NMS & Response Send).

        Continuously processes inference outputs from post_queue,
        applies NMS, converts to JSON, and sends responses.
        """
        # Load class names (COCO 80 classes)
        class_names = [str(i) for i in range(80)]

        while not self.shutdown_event.is_set():
            try:
                # Reduced timeout to minimize idle time
                data = self.post_queue.get(timeout=0.001)
            except queue.Empty:
                continue

            outputs, shape, sender = data

            try:
                # NMS and JSON conversion (using Ultralytics functions)
                result_json = postprocess_to_json(
                    outputs,
                    shape,
                    class_names,
                    conf_thres=0.25,
                    iou_thres=0.45,
                    max_det=300,
                )

                # Encode JSON to bytes
                out_tensor = pb_utils.Tensor(
                    "OUTPUT__0",
                    np.array(
                        [json.dumps(result_json).encode("utf-8")],
                        dtype=np.bytes_,
                    ),
                )

                # Send response
                response = pb_utils.InferenceResponse(
                    output_tensors=[out_tensor]
                )
                sender.send(response)
                sender.send(flags=pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL)

            except Exception as e:
                pb_utils.Logger.log_error(f"Post Error: {e}")
                error_response = pb_utils.InferenceResponse(
                    error=pb_utils.TritonError(str(e))
                )
                sender.send(
                    error_response,
                    flags=pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL,
                )

    def finalize(self) -> None:
        """Finalize the model.

        Signals shutdown to all worker threads and waits for them to finish.
        """
        self.shutdown_event.set()
        for t in self.threads:
            t.join()