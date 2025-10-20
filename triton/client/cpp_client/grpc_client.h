/**
 * @file grpc_client.h
 * @brief Low-level gRPC client for Triton Inference Server communication
 * 
 * This class handles all gRPC protocol communication with NVIDIA Triton
 * Inference Server, including health checks, model queries, and inference
 * requests. It uses Protobuf-generated stubs for type-safe RPC calls.
 * 
 * Protocol: gRPC (HTTP/2 + Protocol Buffers)
 * Generated from: grpc_service.proto (Triton inference service definition)
 * 
 * Key Responsibilities:
 * - Channel creation and connection management
 * - Server health/readiness checks (ServerLive, ServerReady, ModelReady)
 * - Inference request/response handling via ModelInfer RPC
 * - Error handling and timeout management
 */

#pragma once

#include <grpcpp/grpcpp.h>
#include <grpcpp/channel.h>
#include <grpcpp/client_context.h>
#include <grpcpp/create_channel.h>
#include <grpcpp/security/credentials.h>
#include "grpc_service.grpc.pb.h"
#include "grpc_service.pb.h"
#include <memory>
#include <string>
#include <vector>

namespace triton_client {

/**
 * @class GrpcClient
 * @brief Low-level gRPC communication layer for Triton Inference Server
 * 
 * Encapsulates all gRPC protocol details, including:
 * - Channel creation with insecure credentials (for local/trusted networks)
 * - Connection timeout handling (5 seconds default)
 * - RPC call execution (ServerLive, ServerReady, ModelReady, ModelInfer)
 * - Protobuf serialization/deserialization
 * 
 * This is the **only class** that directly interacts with gRPC and Triton.
 * Higher-level classes (TritonClient) use this as a thin abstraction layer.
 * 
 * Example Usage:
 * @code
 *   GrpcClient client("localhost:8001");
 *   if (client.is_server_ready()) {
 *       auto output = client.run_inference("yolo11n", input_tensor);
 *   }
 * @endcode
 */
class GrpcClient {
public:
    /**
     * @brief Construct a new gRPC client and connect to Triton server
     * @param server_url Server address in "host:port" format (e.g., "localhost:8001")
     * @throws std::runtime_error if channel or stub creation fails
     * 
     * Creates an insecure gRPC channel and waits up to 5 seconds for connection.
     * Note: Uses InsecureChannelCredentials (no TLS) - suitable for local/internal use.
     */
    explicit GrpcClient(const std::string& server_url);
    
    /**
     * @brief Destructor - automatically cleans up gRPC resources
     */
    ~GrpcClient() = default;
    
    // ============================================================
    // CONNECTION MANAGEMENT
    // ============================================================
    
    /**
     * @brief Check if Triton server process is alive
     * @return true if server responds to ServerLive RPC
     * 
     * This is the most basic health check. A "live" server means the process
     * is running, but it may not be ready to serve inference requests yet.
     * 
     * RPC: ServerLive() -> ServerLiveResponse {bool live}
     */
    bool is_server_live();
    
    /**
     * @brief Check if Triton server is ready to accept inference requests
     * @return true if server responds to ServerReady RPC with ready=true
     * 
     * A "ready" server has fully initialized, loaded models, and is ready
     * to process inference requests. This is stronger than is_server_live().
     * 
     * RPC: ServerReady() -> ServerReadyResponse {bool ready}
     */
    bool is_server_ready();
    
    /**
     * @brief Check if a specific model is loaded and ready for inference
     * @param model_name Name of the model to check (e.g., "yolo11n")
     * @return true if model is loaded and ready
     * 
     * Verifies that the specified model:
     * - Exists in the model repository
     * - Has been successfully loaded
     * - Is ready to accept inference requests
     * 
     * RPC: ModelReady(model_name) -> ModelReadyResponse {bool ready}
     */
    bool is_model_ready(const std::string& model_name);
    
    // ============================================================
    // MODEL INFORMATION
    // ============================================================
    
    /**
     * @brief Get model metadata (version, inputs, outputs, platform)
     * @param model_name Name of the model
     * @return JSON string with model metadata
     * 
     * Returns information about:
     * - Model name and version
     * - Input tensors (name, shape, datatype)
     * - Output tensors (name, shape, datatype)
     * - Backend platform (e.g., "onnxruntime_onnx")
     * 
     * RPC: ModelMetadata(model_name) -> ModelMetadataResponse
     */
    std::string get_model_metadata(const std::string& model_name);
    
    /**
     * @brief Get model configuration details
     * @param model_name Name of the model
     * @return JSON string with model configuration
     * 
     * Returns config.pbtxt information:
     * - Max batch size
     * - Dynamic batching settings
     * - Instance group (GPU allocation)
     * - Optimization settings
     * 
     * RPC: ModelConfig(model_name) -> ModelConfigResponse
     */
    std::string get_model_config(const std::string& model_name);
    
    // ============================================================
    // INFERENCE EXECUTION
    // ============================================================
    
    /**
     * @brief Execute inference on Triton server
     * @param model_name Name of the model to run (e.g., "yolo11n")
     * @param input_tensor Input data as flat vector of floats
     * @param input_name Name of model input tensor (default: "INPUT__0")
     * @param output_name Name of model output tensor (default: "OUTPUT__0")
     * @return Output tensor as flat vector of floats
     * 
     * **This is the core inference function!**
     * 
     * Process:
     * 1. Create ModelInferRequest protobuf message
     * 2. Populate with model name, input tensor shape and data
     * 3. Send gRPC request to Triton server
     * 4. Wait for ModelInferResponse
     * 5. Extract output tensor data
     * 6. Return as std::vector<float>
     * 
     * Input/Output Format:
     * - Input: [1, 3, 640, 640] = 1,228,800 floats (flattened)
     * - Output: [1, 84, 8400] = 705,600 floats (flattened)
     * 
     * RPC: ModelInfer(request) -> ModelInferResponse
     * 
     * @throws std::runtime_error if RPC fails or output parsing fails
     */
    std::vector<float> run_inference(const std::string& model_name,
                                    const std::vector<float>& input_tensor,
                                    const std::string& input_name = "INPUT__0",
                                    const std::string& output_name = "OUTPUT__0");
    
private:
    // ============================================================
    // PRIVATE MEMBERS
    // ============================================================
    
    std::shared_ptr<grpc::Channel> channel_;                                ///< gRPC channel to Triton server
    std::unique_ptr<inference::GRPCInferenceService::Stub> stub_;          ///< Protobuf-generated RPC stub
    
    // ============================================================
    // HELPER METHODS
    // ============================================================
    
    /**
     * @brief Create and connect gRPC channel to Triton server
     * @param server_url Server address in "host:port" format
     * @return Shared pointer to gRPC channel, or nullptr on failure
     * 
     * Creates an insecure channel (no TLS) and waits up to 5 seconds
     * for connection establishment. Prints connection status to console.
     */
    std::shared_ptr<grpc::Channel> create_channel(const std::string& server_url);
};

} // namespace triton_client
