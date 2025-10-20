#include "grpc_client.h"
#include <iostream>
#include <chrono>
#include <thread>
#include <cstring>

namespace triton_client {

GrpcClient::GrpcClient(const std::string& server_url) {
    channel_ = create_channel(server_url);
    if (!channel_) {
        throw std::runtime_error("Failed to create gRPC channel");
    }
    
    // Create the gRPC stub for Triton Inference Service
    stub_ = inference::GRPCInferenceService::NewStub(channel_);
    if (!stub_) {
        throw std::runtime_error("Failed to create gRPC stub");
    }
    
    std::cout << "gRPC client initialized for server: " << server_url << std::endl;
}

std::shared_ptr<grpc::Channel> GrpcClient::create_channel(const std::string& server_url) {
    try {
        // Create insecure channel
        auto channel = grpc::CreateChannel(server_url, grpc::InsecureChannelCredentials());
        
        // Wait for channel to be ready (with timeout)
        auto deadline = std::chrono::system_clock::now() + std::chrono::seconds(5);
        bool connected = channel->WaitForConnected(deadline);
        
        if (connected) {
            std::cout << "✅ gRPC channel connected to server" << std::endl;
        } else {
            std::cout << "⚠️ gRPC channel created but connection not verified" << std::endl;
            std::cout << "   (Server may still be starting or not fully ready)" << std::endl;
        }
        
        return channel;
    } catch (const std::exception& e) {
        std::cerr << "Failed to create gRPC channel: " << e.what() << std::endl;
        return nullptr;
    }
}

bool GrpcClient::is_server_live() {
    try {
        // Create request and response objects
        inference::ServerLiveRequest request;
        inference::ServerLiveResponse response;
        grpc::ClientContext context;
        
        // Set timeout
        context.set_deadline(std::chrono::system_clock::now() + std::chrono::seconds(5));
        
        // Make the actual RPC call to Triton server
        grpc::Status status = stub_->ServerLive(&context, request, &response);
        
        if (status.ok()) {
            return response.live();
        } else {
            std::cerr << "ServerLive RPC failed: " << status.error_message() << std::endl;
            return false;
        }
    } catch (const std::exception& e) {
        std::cerr << "Error checking server liveness: " << e.what() << std::endl;
        return false;
    }
}

bool GrpcClient::is_server_ready() {
    try {
        // Create request and response objects
        inference::ServerReadyRequest request;
        inference::ServerReadyResponse response;
        grpc::ClientContext context;
        
        // Set timeout
        context.set_deadline(std::chrono::system_clock::now() + std::chrono::seconds(5));
        
        // Make the actual RPC call to Triton server
        grpc::Status status = stub_->ServerReady(&context, request, &response);
        
        if (status.ok()) {
            return response.ready();
        } else {
            std::cerr << "ServerReady RPC failed: " << status.error_message() << std::endl;
            return false;
        }
    } catch (const std::exception& e) {
        std::cerr << "Error checking server readiness: " << e.what() << std::endl;
        return false;
    }
}

bool GrpcClient::is_model_ready(const std::string& model_name) {
    try {
        // Create request and response objects
        inference::ModelReadyRequest request;
        inference::ModelReadyResponse response;
        grpc::ClientContext context;
        
        // Set model name
        request.set_name(model_name);
        
        // Set timeout
        context.set_deadline(std::chrono::system_clock::now() + std::chrono::seconds(5));
        
        // Make the actual RPC call to Triton server
        grpc::Status status = stub_->ModelReady(&context, request, &response);
        
        if (status.ok()) {
            return response.ready();
        } else {
            std::cerr << "ModelReady RPC failed: " << status.error_message() << std::endl;
            return false;
        }
    } catch (const std::exception& e) {
        std::cerr << "Error checking model readiness: " << e.what() << std::endl;
        return false;
    }
}

std::string GrpcClient::get_model_metadata(const std::string& model_name) {
    try {
        // In a real implementation, this would call the ModelMetadata RPC
        return "Model metadata for " + model_name + " (simulated)";
    } catch (const std::exception& e) {
        std::cerr << "Error getting model metadata: " << e.what() << std::endl;
        return "";
    }
}

std::string GrpcClient::get_model_config(const std::string& model_name) {
    try {
        // In a real implementation, this would call the ModelConfig RPC
        return "Model config for " + model_name + " (simulated)";
    } catch (const std::exception& e) {
        std::cerr << "Error getting model config: " << e.what() << std::endl;
        return "";
    }
}

std::vector<float> GrpcClient::run_inference(const std::string& model_name,
                                            const std::vector<float>& input_tensor,
                                            const std::string& input_name,
                                            const std::string& output_name) {
    try {
        // Create request and response objects
        inference::ModelInferRequest request;
        inference::ModelInferResponse response;
        grpc::ClientContext context;
        
        // Set model name and version
        request.set_model_name(model_name);
        request.set_model_version("");  // Use latest version
        
        // Add input tensor
        auto* input = request.add_inputs();
        input->set_name(input_name);
        input->set_datatype("FP32");
        input->add_shape(1);  // batch size
        input->add_shape(3);  // channels
        input->add_shape(800); // height
        input->add_shape(800); // width
        
        // Add output request
        auto* output = request.add_outputs();
        output->set_name(output_name);
        
        // Set input data as raw bytes
        request.mutable_raw_input_contents()->Add(
            std::string(reinterpret_cast<const char*>(input_tensor.data()),
                       input_tensor.size() * sizeof(float))
        );
        
        // Set timeout
        context.set_deadline(std::chrono::system_clock::now() + std::chrono::seconds(30));
        
        // Make the actual RPC call to Triton server
        grpc::Status status = stub_->ModelInfer(&context, request, &response);
        
        if (!status.ok()) {
            std::cerr << "ModelInfer RPC failed: " << status.error_message() << std::endl;
            return {};
        }
        
        // Extract output data
        if (response.raw_output_contents_size() > 0) {
            const std::string& raw_output = response.raw_output_contents(0);
            std::vector<float> output_data;
            output_data.resize(raw_output.size() / sizeof(float));
            std::memcpy(output_data.data(), raw_output.data(), raw_output.size());
            
            return output_data;
        }
        
        std::cerr << "No output data received from server" << std::endl;
        return {};
        
    } catch (const std::exception& e) {
        std::cerr << "Error running inference: " << e.what() << std::endl;
        return {};
    }
}

} // namespace triton_client
