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
import json
import os
import re
import time
import rebel  # RBLN Runtime
import triton_python_backend_utils as pb_utils

# Number of devices to allocate.
# Available device numbers can be found through `rbln-stat` command.
NAME_RE = re.compile(r"^(?P<model>.+)_(?P<group>\d+)_(?P<inst>\d+)$")
NUM_OF_DEVICES = 4


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

        inst_name = args.get("model_instance_name", "")
        m = NAME_RE.match(inst_name)
        if not m:
            raise RuntimeError(f"Unexpected model_instance_name: {inst_name}")

        device_id = int(m.group("group"))

        #Triton logger generation for debug
        pb_utils.Logger.log_info(
            f"[{inst_name}] -> RBLN_DEVICE={device_id}"
        )

        # Create rbln runtime module
        self.module = rebel.Runtime(rbln_path, device=device_id)
        # Minimal threads since inference is hardware-bound, not CPU-bound
        self.module.num_threads = 8
        
        # Performance tracking
        self.inst_name = inst_name
        self.request_count = 0
        self.total_time = 0
        self.inference_time = 0
        self.data_prep_time = 0
        self.response_time = 0
        self.last_log_time = time.time()

    def execute(self, requests):
        """`execute` MUST be implemented in every Python model. `execute`
        function receives a list of pb_utils.InferenceRequest as the only
        argument. This function is called when an inference request is made
        for this model. Depending on the batching configuration (e.g. Dynamic
        Batching) used, `requests` may contain multiple requests. Every
        Python model, must create one pb_utils.InferenceResponse for every
        pb_utils.InferenceRequest in `requests`. If there is an error, you can
        set the error argument when creating a pb_utils.InferenceResponse

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
        start_total = time.perf_counter()
        output0_dtype = self.output0_dtype
        responses = []

        for request in requests:
            # Data preparation timing
            t0 = time.perf_counter()
            in_0 = pb_utils.get_input_tensor_by_name(request, "INPUT__0")
            input_data = in_0.as_numpy()
            t1 = time.perf_counter()
            self.data_prep_time += (t1 - t0)

            # Run inference timing
            t2 = time.perf_counter()
            result = self.module.run(input_data)
            t3 = time.perf_counter()
            self.inference_time += (t3 - t2)

            # Response creation timing
            t4 = time.perf_counter()
            out_tensor_0 = pb_utils.Tensor("OUTPUT__0", result[0].astype(output0_dtype))
            inference_response = pb_utils.InferenceResponse(
                output_tensors=[out_tensor_0]
            )
            responses.append(inference_response)
            t5 = time.perf_counter()
            self.response_time += (t5 - t4)

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