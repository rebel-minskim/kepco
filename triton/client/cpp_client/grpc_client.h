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

class GrpcClient {
public:
    explicit GrpcClient(const std::string& server_url);
    ~GrpcClient() = default;
    
    // Connection management
    bool is_server_live();
    bool is_server_ready();
    bool is_model_ready(const std::string& model_name);
    
    // Model information
    std::string get_model_metadata(const std::string& model_name);
    std::string get_model_config(const std::string& model_name);
    
    // Inference
    std::vector<float> run_inference(const std::string& model_name,
                                    const std::vector<float>& input_tensor,
                                    const std::string& input_name = "INPUT__0",
                                    const std::string& output_name = "OUTPUT__0");
    
private:
    std::shared_ptr<grpc::Channel> channel_;
    std::unique_ptr<inference::GRPCInferenceService::Stub> stub_;
    
    // Helper methods
    std::shared_ptr<grpc::Channel> create_channel(const std::string& server_url);
};

} // namespace triton_client
