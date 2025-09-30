#include <iostream>
#include <vector>
#include <thread>
#include <chrono>
#include <random>
#include <atomic>
#include <future>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <memory>

#include <grpcpp/grpcpp.h>
#include <grpcpp/health_check_service_interface.h>
#include <grpcpp/ext/proto_server_reflection_plugin.h>

#include "grpc_service.grpc.pb.h"

using grpc::Channel;
using grpc::ClientContext;
using grpc::Status;
using grpc::ClientAsyncResponseReader;

using namespace inference;

class TritonCppClient {
private:
    std::unique_ptr<GRPCInferenceService::Stub> stub_;
    std::string model_name_;
    int input_width_;
    int input_height_;
    
    // Performance tracking
    std::atomic<int> total_requests_{0};
    std::atomic<double> total_inference_time_{0.0};
    std::atomic<double> total_e2e_time_{0.0};
    std::mutex stats_mutex_;
    
    // Thread pool
    std::vector<std::thread> workers_;
    std::queue<std::function<void()>> task_queue_;
    std::mutex queue_mutex_;
    std::condition_variable queue_cv_;
    std::atomic<bool> stop_flag_{false};
    
    // Request tracking
    std::atomic<int> completed_requests_{0};
    std::atomic<int> target_requests_{0};
    
public:
    TritonCppClient(const std::string& server_url, 
                   const std::string& model_name,
                   int input_width = 800, 
                   int input_height = 800)
        : model_name_(model_name), input_width_(input_width), input_height_(input_height) {
        
        // Create gRPC channel
        auto channel = grpc::CreateChannel(server_url, grpc::InsecureChannelCredentials());
        stub_ = GRPCInferenceService::NewStub(channel);
        
        // Start worker threads
        start_workers();
    }
    
    ~TritonCppClient() {
        stop_workers();
    }
    
    void start_workers() {
        int num_workers = std::thread::hardware_concurrency();
        std::cout << "Starting " << num_workers << " worker threads" << std::endl;
        
        for (int i = 0; i < num_workers; ++i) {
            workers_.emplace_back([this, i]() {
                worker_thread(i);
            });
        }
    }
    
    void stop_workers() {
        stop_flag_ = true;
        queue_cv_.notify_all();
        
        for (auto& worker : workers_) {
            if (worker.joinable()) {
                worker.join();
            }
        }
    }
    
    void worker_thread(int worker_id) {
        while (!stop_flag_) {
            std::function<void()> task;
            {
                std::unique_lock<std::mutex> lock(queue_mutex_);
                queue_cv_.wait(lock, [this] { return !task_queue_.empty() || stop_flag_; });
                
                if (stop_flag_) break;
                
                if (!task_queue_.empty()) {
                    task = task_queue_.front();
                    task_queue_.pop();
                }
            }
            
            if (task) {
                task();
            }
        }
    }
    
    void add_task(std::function<void()> task) {
        {
            std::lock_guard<std::mutex> lock(queue_mutex_);
            task_queue_.push(task);
        }
        queue_cv_.notify_one();
    }
    
    // Create dummy input data (like perf_analyzer)
    std::vector<float> create_dummy_input() {
        std::vector<float> data(input_width_ * input_height_ * 3);
        
        static std::random_device rd;
        static std::mt19937 gen(rd());
        static std::uniform_real_distribution<float> dis(0.0f, 1.0f);
        
        for (auto& val : data) {
            val = dis(gen);
        }
        
        return data;
    }
    
    // Single inference request
    bool single_inference(int request_id) {
        auto start_time = std::chrono::high_resolution_clock::now();
        
        // Create request
        ModelInferRequest request;
        request.set_model_name(model_name_);
        
        // Add input
        auto* input = request.add_inputs();
        input->set_name("INPUT__0");
        input->set_datatype("FP32");
        
        // Set shape
        input->add_shape(1);
        input->add_shape(3);
        input->add_shape(input_height_);
        input->add_shape(input_width_);
        
        // Create dummy data
        auto dummy_data = create_dummy_input();
        
        // Add raw input data
        auto* raw_input = request.add_raw_input_contents();
        raw_input->assign(reinterpret_cast<const char*>(dummy_data.data()), 
                         dummy_data.size() * sizeof(float));
        
        // Add output
        auto* output = request.add_outputs();
        output->set_name("OUTPUT__0");
        
        // Send request
        ModelInferResponse response;
        ClientContext context;
        
        auto inference_start = std::chrono::high_resolution_clock::now();
        Status status = stub_->ModelInfer(&context, request, &response);
        auto inference_end = std::chrono::high_resolution_clock::now();
        
        auto end_time = std::chrono::high_resolution_clock::now();
        
        if (status.ok()) {
            auto inference_time = std::chrono::duration<double, std::milli>(inference_end - inference_start).count();
            auto e2e_time = std::chrono::duration<double, std::milli>(end_time - start_time).count();
            
            {
                std::lock_guard<std::mutex> lock(stats_mutex_);
                total_inference_time_ = total_inference_time_.load() + inference_time;
                total_e2e_time_ = total_e2e_time_.load() + e2e_time;
            }
            total_requests_++;
            completed_requests_++;
            
            if (completed_requests_ % 100 == 0) {
                double current_fps = completed_requests_.load() / 
                    (std::chrono::duration<double>(end_time - start_time).count());
                std::cout << "Processed " << completed_requests_ << "/" << target_requests_ 
                         << " requests | Inference: " << inference_time << "ms | FPS: " << current_fps << std::endl;
            }
            
            return true;
        } else {
            std::cerr << "Request " << request_id << " failed: " << status.error_message() << std::endl;
            return false;
        }
    }
    
    // Run performance test
    void run_performance_test(int num_requests, int request_rate) {
        std::cout << "Running C++ performance test:" << std::endl;
        std::cout << "  Requests: " << num_requests << std::endl;
        std::cout << "  Request rate: " << request_rate << " req/s" << std::endl;
        std::cout << "  Target: 90 FPS" << std::endl;
        
        target_requests_ = num_requests;
        completed_requests_ = 0;
        total_requests_ = 0;
        total_inference_time_ = 0.0;
        total_e2e_time_ = 0.0;
        
        auto start_time = std::chrono::high_resolution_clock::now();
        
        // Submit requests at controlled rate
        for (int i = 0; i < num_requests; ++i) {
            add_task([this, i]() {
                single_inference(i);
            });
            
            // Control request rate
            auto elapsed = std::chrono::high_resolution_clock::now() - start_time;
            auto target_time = std::chrono::milliseconds(i * 1000 / request_rate);
            if (elapsed < target_time) {
                std::this_thread::sleep_for(target_time - elapsed);
            }
        }
        
        // Wait for all requests to complete
        while (completed_requests_ < num_requests) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto total_time = std::chrono::duration<double>(end_time - start_time).count();
        
        // Calculate final statistics
        double avg_fps = total_requests_ / total_time;
        double avg_inference = total_inference_time_ / total_requests_;
        double avg_e2e = total_e2e_time_ / total_requests_;
        
        std::cout << "\n" << std::string(60, '=') << std::endl;
        std::cout << "C++ CLIENT PERFORMANCE RESULTS" << std::endl;
        std::cout << std::string(60, '=') << std::endl;
        std::cout << "Total requests: " << total_requests_ << std::endl;
        std::cout << "Total time: " << total_time << "s" << std::endl;
        std::cout << "Average FPS: " << avg_fps << std::endl;
        std::cout << "Average inference time: " << avg_inference << "ms" << std::endl;
        std::cout << "Average E2E time: " << avg_e2e << "ms" << std::endl;
        std::cout << "Request rate: " << request_rate << " req/s" << std::endl;
        std::cout << "Target FPS: 90.0" << std::endl;
        
        if (avg_fps >= 90.0) {
            std::cout << "SUCCESS: Achieved target FPS!" << std::endl;
        } else {
            std::cout << "TARGET NOT MET: Need 90 FPS, got " << avg_fps << " FPS" << std::endl;
        }
        std::cout << std::string(60, '=') << std::endl;
    }
};

int main(int argc, char* argv[]) {
    // Parse command line arguments
    std::string server_url = "localhost:8001";
    std::string model_name = "yolov11";
    int num_requests = 900;
    int request_rate = 90;
    int input_width = 800;
    int input_height = 800;
    
    for (int i = 1; i < argc; i += 2) {
        if (i + 1 >= argc) break;
        
        std::string arg = argv[i];
        if (arg == "--url" || arg == "-u") {
            server_url = argv[i + 1];
        } else if (arg == "--model" || arg == "-m") {
            model_name = argv[i + 1];
        } else if (arg == "--requests") {
            num_requests = std::stoi(argv[i + 1]);
        } else if (arg == "--rate") {
            request_rate = std::stoi(argv[i + 1]);
        } else if (arg == "--width") {
            input_width = std::stoi(argv[i + 1]);
        } else if (arg == "--height") {
            input_height = std::stoi(argv[i + 1]);
        }
    }
    
    std::cout << "C++ Triton Client - Performance Test" << std::endl;
    std::cout << "Server: " << server_url << std::endl;
    std::cout << "Model: " << model_name << std::endl;
    std::cout << "Input size: " << input_width << "x" << input_height << std::endl;
    
    try {
        // Create client
        TritonCppClient client(server_url, model_name, input_width, input_height);
        
        // Run performance test
        client.run_performance_test(num_requests, request_rate);
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
