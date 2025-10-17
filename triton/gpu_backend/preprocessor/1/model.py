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

import numpy as np
import cv2
import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    def initialize(self, args):
        """`initialize` is called only once when the model is being loaded.

        Parameters
        ----------
        args : dict
          Both keys and values are strings. The dictionary keys and values are:
          * model_config: A JSON string containing the model configuration
          * model_instance_kind: A string containing model instance kind
          * model_instance_device_id: A string containing model instance device ID
          * model_repository: Model repository path
          * model_version: Model version
          * model_name: Model name
        """
        pb_utils.Logger.log_info("Preprocessor model initialized")

    def execute(self, requests):
        """`execute` MUST be implemented in every Python model. `execute`
        function receives a list of pb_utils.InferenceRequest as the only
        argument. This function is called when an inference request is made
        for this model.

        Parameters
        ----------
        requests : list
          A list of pb_utils.InferenceRequest

        Returns
        -------
        list
          A list of pb_utils.InferenceResponse. The length of this list must
          be the same as `requests`
        """
        responses = []

        for request in requests:
            try:
                # Get JPEG bytes from input
                image_bytes_tensor = pb_utils.get_input_tensor_by_name(request, "IMAGE_BYTES")
                image_bytes = image_bytes_tensor.as_numpy()
                
                # Decode JPEG
                # image_bytes is 1D array of uint8, need to decode it
                image_array = np.frombuffer(image_bytes.tobytes(), dtype=np.uint8)
                image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
                
                if image is None:
                    error = pb_utils.TritonError("Failed to decode JPEG image")
                    responses.append(pb_utils.InferenceResponse(error=error))
                    continue
                
                # Simple resize to 800x800 (no letterbox)
                # Box coordinates will be scaled proportionally by client
                image = cv2.resize(image, (800, 800), interpolation=cv2.INTER_LINEAR)
                
                # Convert BGR to RGB
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # Normalize to [0, 1] and convert to float32
                image = image.astype(np.float32) / 255.0
                
                # Transpose from HWC to CHW format
                image = np.transpose(image, (2, 0, 1))
                
                # Add batch dimension: (3, 800, 800) -> (1, 3, 800, 800)
                image = np.expand_dims(image, axis=0)
                
                # Create output tensor
                out_tensor = pb_utils.Tensor("PREPROCESSED_IMAGE", image)
                
                # Create response
                inference_response = pb_utils.InferenceResponse(output_tensors=[out_tensor])
                responses.append(inference_response)
                
            except Exception as e:
                error = pb_utils.TritonError(f"Preprocessing error: {str(e)}")
                responses.append(pb_utils.InferenceResponse(error=error))

        return responses

    def finalize(self):
        """`finalize` is called only once when the model is being unloaded.
        """
        pb_utils.Logger.log_info("Preprocessor model finalized")


