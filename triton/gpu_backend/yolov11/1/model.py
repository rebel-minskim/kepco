import json
import os
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
        
        # Build path to the yolov11.pt model file
        model_path = os.path.join(
            args["model_repository"],
            args["model_version"],
            "yolov11.pt"
        )
        
        pb_utils.Logger.log_info(f"Loading YOLOv11 model from: {model_path}")
        pb_utils.Logger.log_info(f"Using device: {self.device}")
        
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
            
            # Disable gradients for inference
            if hasattr(self.model, 'parameters'):
                for param in self.model.parameters():
                    param.requires_grad = False
            
            # Detect model dtype (float16 or float32)
            self.model_dtype = next(self.model.parameters()).dtype
            pb_utils.Logger.log_info(f"Model dtype: {self.model_dtype}")
            
            pb_utils.Logger.log_info(f"Successfully loaded YOLOv11 model on {self.device}")
            
        except Exception as e:
            pb_utils.Logger.log_error(f"Failed to load model: {str(e)}")
            raise

    def execute(self, requests):
        """Execute inference on a batch of requests"""
        
        responses = []
        
        for request in requests:
            try:
                # Get input tensor
                in_tensor = pb_utils.get_input_tensor_by_name(request, "INPUT__0")
                input_data = in_tensor.as_numpy()
                
                # Convert to PyTorch tensor and move to device
                input_tensor = torch.from_numpy(input_data).to(self.device)
                
                # Ensure correct shape (add batch dimension if needed)
                if len(input_tensor.shape) == 3:
                    input_tensor = input_tensor.unsqueeze(0)
                
                # Convert input to match model dtype (e.g., float16 if model is half precision)
                input_tensor = input_tensor.to(self.model_dtype)
                
                # Run inference
                with torch.no_grad():
                    output = self.model(input_tensor)
                
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
                
            except Exception as e:
                # Handle errors
                error_msg = f"Error during inference: {str(e)}"
                pb_utils.Logger.log_error(error_msg)
                inference_response = pb_utils.InferenceResponse(
                    output_tensors=[],
                    error=pb_utils.TritonError(error_msg)
                )
                responses.append(inference_response)
        
        return responses

    def finalize(self):
        """Cleanup when model is unloaded"""
        pb_utils.Logger.log_info("Cleaning up YOLOv11 model")
