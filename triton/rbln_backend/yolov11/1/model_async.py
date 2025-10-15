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
# model_async.py - Enhanced async version of the RBLN model
#
import json
import os
import asyncio
import concurrent.futures
import rebel  # RBLN Runtime
import triton_python_backend_utils as pb_utils

# Number of devices to allocate.
# Available device numbers can be found through `rbln-stat` command.
NUM_OF_DEVICES = 1


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
          * model_instance_name: A string containing model instance name in form of <model_name>_<instance_group_id>_<instance_id>
          * model_repository: Model repository path
          * model_version: Model version
          * model_name: Model name
        """

        self.model_config = model_config = json.loads(args["model_config"])
        instance_group_config = model_config["instance_group"][0]
        instance_count = instance_group_config["count"]
        instance_idx = 0
        # Get `instance_idx` for multiple instances.
        # instance_group's count should be bigger than 1 in config.pbtxt.
        if instance_count > 1:
            instance_name_parts = args["model_instance_name"].split("_")
            if not instance_name_parts[-1].isnumeric():
                raise pb_utils.TritonModelException(
                    "model instance name should end with '_<instance_idx>', got {}".format(
                        args["model_instance_name"]
                    )
                )
            instance_idx = int(instance_name_parts[-1])

        output0_config = pb_utils.get_output_config_by_name(model_config, "OUTPUT__0")
        self.output0_dtype = pb_utils.triton_string_to_numpy(
            output0_config["data_type"]
        )

        # Path to rbln compiled model file
        rbln_path = os.path.join(
            args["model_repository"],
            args["model_version"],
            f"{args['model_name']}.rbln",
        )

        # Create rbln async runtime module with optimized settings
        self.module = rebel.AsyncRuntime(rbln_path, parallel=2, device=instance_idx % NUM_OF_DEVICES)  # Enable double buffering
        self.module.num_threads = 8  # Optimize thread count
        
        # Create thread pool for async operations
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)
        
        print(f"Initialized async RBLN model: {rbln_path}")
        print(f"Parallel processing enabled: {2}")
        print(f"Thread count: {8}")

    def execute(self, requests):
        """`execute` with enhanced async processing capabilities.
        
        This version processes multiple requests concurrently when possible,
        and uses the AsyncRuntime's async_run method for better performance.

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
        output0_dtype = self.output0_dtype
        
        # For single request, use direct processing
        if len(requests) == 1:
            return self._process_single_request(requests[0], output0_dtype)
        
        # For multiple requests, use async batch processing
        return self._process_batch_requests(requests, output0_dtype)

    def _process_single_request(self, request, output0_dtype):
        """Process a single request efficiently"""
        in_0 = pb_utils.get_input_tensor_by_name(request, "INPUT__0")
        
        # Use async_run for single request (runs in event loop internally)
        try:
            # Create async task and run it
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                result = loop.run_until_complete(
                    self.module.async_run(in_0.as_numpy())
                )
            finally:
                loop.close()
        except Exception as e:
            print(f"Async inference failed, falling back to sync: {e}")
            # Fallback to synchronous run
            result = self.module.run(in_0.as_numpy())
        
        out_tensor_0 = pb_utils.Tensor("OUTPUT__0", result[0].astype(output0_dtype))
        inference_response = pb_utils.InferenceResponse(
            output_tensors=[out_tensor_0]
        )
        return [inference_response]

    def _process_batch_requests(self, requests, output0_dtype):
        """Process multiple requests with async batch processing"""
        responses = []
        
        # Extract input tensors
        input_tensors = []
        for request in requests:
            in_0 = pb_utils.get_input_tensor_by_name(request, "INPUT__0")
            input_tensors.append(in_0.as_numpy())
        
        try:
            # Process batch asynchronously
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                # Create async tasks for all inputs
                tasks = [self.module.async_run(input_tensor) for input_tensor in input_tensors]
                results = loop.run_until_complete(asyncio.gather(*tasks))
            finally:
                loop.close()
        except Exception as e:
            print(f"Async batch processing failed, falling back to sync: {e}")
            # Fallback to synchronous processing
            results = []
            for input_tensor in input_tensors:
                result = self.module.run(input_tensor)
                results.append(result)
        
        # Create responses
        for result in results:
            out_tensor_0 = pb_utils.Tensor("OUTPUT__0", result[0].astype(output0_dtype))
            inference_response = pb_utils.InferenceResponse(
                output_tensors=[out_tensor_0]
            )
            responses.append(inference_response)
        
        return responses

    def finalize(self):
        """`finalize` is called only once when the model is being unloaded.
        Implementing `finalize` function is optional. This function allows
        the model to perform any necessary clean ups before exit.
        """
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=True)
        print("Async RBLN model finalized")
