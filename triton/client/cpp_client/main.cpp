#include "triton_client.h"
#include "config.h"
#include <iostream>
#include <string>

int main(int argc, char* argv[]) {
    try {
        // Simple argument parsing for now
        std::string mode = "video";
        std::string input = "media/1.mp4";
        std::string output = "output/result.mp4";
        
        if (argc > 1) {
            mode = argv[1];
        }
        if (argc > 2) {
            input = argv[2];
        }
        if (argc > 3) {
            output = argv[3];
        }
        
        // Create configuration
        triton_client::ClientConfig config;
        
        // Create client
        triton_client::TritonClient client(config);
        
        // Connect to server
        if (!client.connect()) {
            std::cerr << "Failed to connect to server" << std::endl;
            return 1;
        }
        
        // Run based on mode
        if (mode == "dummy") {
            client.run_dummy_inference();
        }
        else if (mode == "image") {
            client.run_image_inference(input, output);
        }
        else if (mode == "video") {
            client.run_video_inference(input, output);
        }
        
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}