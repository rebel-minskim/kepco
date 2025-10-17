import json
import os
import time
import numpy as np
import torch
import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    def initialize(self, args):
        """Initialize the YOLOv11 model"""
        
        # Parse model configuration
        self.model_config = json.loads(args["model_config"])
        
        # Get output configuration
        output0_config = pb_utils.get_output_config_by_name(self.model_config, "OUTPUT__0")
        self.output0_dtype = pb_utils.triton_string_to_numpy(output0_config["data_type"])
        
        # Get device ID from Triton
        device_id = int(args.get("model_instance_device_id", "0"))
        self.device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
        
        # Instance name for tracking
        self.inst_name = args.get("model_instance_name", "unknown")
        
        # Build path to the yolov11.pt model file
        model_path = os.path.join(
            args["model_repository"],
            args["model_version"],
            "yolov11.pt"
        )
        
        pb_utils.Logger.log_info(f"[{self.inst_name}] Loading YOLOv11 model from: {model_path}")
        pb_utils.Logger.log_info(f"[{self.inst_name}] Using device: {self.device}")
        
        # Load the YOLOv11 PyTorch model
        # weights_only=False is required for models with custom classes (Ultralytics)
        try:
            self.model = torch.load(model_path, map_location=self.device, weights_only=False)
            
            # Handle different model formats (checkpoint dict vs direct model)
            if isinstance(self.model, dict):
                if 'model' in self.model:
                    self.model = self.model['model']
                elif 'ema' in self.model:
                    self.model = self.model['ema']
            
            # Set to evaluation mode
            if hasattr(self.model, 'eval'):
                self.model.eval()
            
            # Convert model to float32 to match input tensor type
            if hasattr(self.model, 'float'):
                self.model.float()
            
            # Disable gradients for inference
            if hasattr(self.model, 'parameters'):
                for param in self.model.parameters():
                    param.requires_grad = False
            
            pb_utils.Logger.log_info(f"[{self.inst_name}] Successfully loaded YOLOv11 model on {self.device}")
            
        except Exception as e:
            pb_utils.Logger.log_error(f"[{self.inst_name}] Failed to load model: {str(e)}")
            raise
        
        # Performance tracking
        self.request_count = 0
        self.total_time = 0
        self.inference_time = 0
        self.data_prep_time = 0
        self.response_time = 0
        self.last_log_time = time.time()

    def execute(self, requests):
        """Execute inference on a batch of requests"""
        
        start_total = time.perf_counter()
        responses = []
        
        for request in requests:
            try:
                # Data preparation timing
                t0 = time.perf_counter()
                in_tensor = pb_utils.get_input_tensor_by_name(request, "INPUT__0")
                input_data = in_tensor.as_numpy()
                
                # Convert to PyTorch tensor and move to device
                input_tensor = torch.from_numpy(input_data).to(self.device)
                
                # Ensure correct shape (add batch dimension if needed)
                if len(input_tensor.shape) == 3:
                    input_tensor = input_tensor.unsqueeze(0)
                
                t1 = time.perf_counter()
                self.data_prep_time += (t1 - t0)
                
                # Run inference timing
                t2 = time.perf_counter()
                with torch.no_grad():
                    output = self.model(input_tensor)
                
                # Synchronize CUDA to get accurate timing
                if self.device.type == 'cuda':
                    torch.cuda.synchronize()
                
                t3 = time.perf_counter()
                self.inference_time += (t3 - t2)
                
                # Response creation timing
                t4 = time.perf_counter()
                
                # Handle different output formats
                if isinstance(output, torch.Tensor):
                    result = output.cpu().numpy()
                elif isinstance(output, (list, tuple)):
                    # YOLOv11 may return multiple outputs
                    result = output[0].cpu().numpy() if len(output) > 0 else output
                else:
                    result = output
                
                # Convert to correct dtype
                result = result.astype(self.output0_dtype)
                
                # Create output tensor
                out_tensor = pb_utils.Tensor("OUTPUT__0", result)
                
                # Create response
                inference_response = pb_utils.InferenceResponse(output_tensors=[out_tensor])
                responses.append(inference_response)
                
                t5 = time.perf_counter()
                self.response_time += (t5 - t4)
                
            except Exception as e:
                # Handle errors
                error_msg = f"[{self.inst_name}] Error during inference: {str(e)}"
                pb_utils.Logger.log_error(error_msg)
                inference_response = pb_utils.InferenceResponse(
                    output_tensors=[],
                    error=pb_utils.TritonError(error_msg)
                )
                responses.append(inference_response)
        
        end_total = time.perf_counter()
        self.total_time += (end_total - start_total)
        self.request_count += len(requests)
        
        # Log performance stats every 5 seconds
        current_time = time.time()
        if current_time - self.last_log_time >= 5.0:
            if self.request_count > 0:
                avg_total = (self.total_time / self.request_count) * 1000
                avg_inference = (self.inference_time / self.request_count) * 1000
                avg_data_prep = (self.data_prep_time / self.request_count) * 1000
                avg_response = (self.response_time / self.request_count) * 1000
                
                pb_utils.Logger.log_info(
                    f"[{self.inst_name}] Performance Stats (ms/request): "
                    f"Total={avg_total:.2f}, Inference={avg_inference:.2f}, "
                    f"DataPrep={avg_data_prep:.2f}, Response={avg_response:.2f}, "
                    f"Requests={self.request_count}, RPS={self.request_count/5.0:.1f}"
                )
                
                # Reset counters
                self.request_count = 0
                self.total_time = 0
                self.inference_time = 0
                self.data_prep_time = 0
                self.response_time = 0
                self.last_log_time = current_time
        
        return responses

    def finalize(self):
        """Cleanup when model is unloaded"""
        pb_utils.Logger.log_info(f"[{self.inst_name}] Cleaning up YOLOv11 model")
