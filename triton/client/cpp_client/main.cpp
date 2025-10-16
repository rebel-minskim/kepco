/**
 * @file main.cpp
 * @brief Main entry point for the C++ Triton Inference Client
 * 
 * This client supports multiple inference modes:
 * - dummy: Test mode with synthetic data (no actual video/image input)
 * - image: Single image inference with detection visualization
 * - video: Sequential video processing (single-threaded, ~35 FPS)
 * - parallel: Multi-threaded video processing pipeline (4-8 threads, ~88 FPS)
 * 
 * Performance Benchmark (1.mp4, 440 frames):
 * - Single-threaded: 35.06 FPS (12.66s total)
 * - Multi-threaded (4 threads): 88.92 FPS (4.95s total) - 2.5x speedup
 * 
 * Usage Examples:
 *   ./triton_client dummy
 *   ./triton_client image input.jpg output.jpg
 *   ./triton_client video input.mp4 output.mp4
 *   ./triton_client parallel input.mp4 output.mp4 4  # 4 inference threads
 */

#include "triton_client.h"
#include "config.h"
#include <iostream>
#include <string>

int main(int argc, char* argv[]) {
    try {
        // ============================================================
        // COMMAND LINE ARGUMENT PARSING
        // ============================================================
        // Default values for quick testing
        std::string mode = "video";                    // Inference mode
        std::string input = "media/1.mp4";            // Input file path
        std::string output = "output/result.mp4";     // Output file path
        
        // Parse command line arguments
        if (argc > 1) {
            mode = argv[1];      // Mode: dummy, image, video, parallel
        }
        if (argc > 2) {
            input = argv[2];     // Input path (image or video file)
        }
        if (argc > 3) {
            output = argv[3];    // Output path (for saving results)
        }
        
        // ============================================================
        // CLIENT INITIALIZATION
        // ============================================================
        // Create configuration with default settings
        // (server URL, model name, confidence thresholds, etc.)
        triton_client::ClientConfig config;
        
        // Initialize Triton client with configuration
        triton_client::TritonClient client(config);
        
        // Establish gRPC connection to Triton Inference Server
        // This performs health checks (ServerLive, ServerReady, ModelReady)
        if (!client.connect()) {
            std::cerr << "Failed to connect to server" << std::endl;
            return 1;
        }
        
        // ============================================================
        // INFERENCE MODE SELECTION
        // ============================================================
        
        if (mode == "dummy") {
            // Test mode: Generate synthetic input tensor for quick testing
            client.run_dummy_inference();
        }
        else if (mode == "image") {
            // Single image mode: Load image → Preprocess → Infer → Visualize
            client.run_image_inference(input, output);
        }
        else if (mode == "video") {
            // Single-threaded video mode: Sequential frame processing
            // Performance: ~35 FPS (slower but simpler)
            client.run_video_inference(input, output);
        }
        else if (mode == "video_parallel" || mode == "parallel") {
            // Multi-threaded parallel mode: Pipeline processing for high FPS
            // Pipeline: Read → Preprocess → Inference (multi-threaded) → Draw → Write
            // Performance: ~88 FPS with 4 threads (2.5x faster than single-threaded)
            
            int num_threads = 4; // Default: 4 inference worker threads
            if (argc > 4) {
                num_threads = std::atoi(argv[4]);  // Override from command line
            }
            client.run_video_inference_parallel(input, output, num_threads);
        }
        else {
            // Unknown mode - print usage information
            std::cerr << "Unknown mode: " << mode << std::endl;
            std::cerr << "Usage: " << argv[0] << " <mode> <input> <output> [num_threads]" << std::endl;
            std::cerr << "Modes: dummy, image, video, parallel" << std::endl;
            std::cerr << "\nExamples:" << std::endl;
            std::cerr << "  " << argv[0] << " parallel ../media/1.mp4 output.mp4 4" << std::endl;
            std::cerr << "  " << argv[0] << " video test.mp4 result.mp4" << std::endl;
            std::cerr << "  " << argv[0] << " image photo.jpg detected.jpg" << std::endl;
            return 1;
        }
        
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}