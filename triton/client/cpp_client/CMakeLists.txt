cmake_minimum_required(VERSION 3.15)
project(TritonCppClient)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
find_package(PkgConfig REQUIRED)
find_package(Protobuf REQUIRED)
find_package(gRPC REQUIRED)

# Find OpenCV (optional, for video processing)
find_package(OpenCV QUIET)

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# Add executable
add_executable(triton_cpp_client 
    main.cpp
    grpc_service.pb.cc
    grpc_service.grpc.pb.cc
)

# Link libraries
target_link_libraries(triton_cpp_client
    gRPC::grpc++
    protobuf::libprotobuf
    pthread
)

# Optional: Link OpenCV if available
if(OpenCV_FOUND)
    target_link_libraries(triton_cpp_client ${OpenCV_LIBS})
    target_compile_definitions(triton_cpp_client PRIVATE OPENCV_AVAILABLE)
endif()

# Compiler flags for optimization
target_compile_options(triton_cpp_client PRIVATE
    -O3
    -march=native
    -mtune=native
    -flto
)

# Link time optimization
set_target_properties(triton_cpp_client PROPERTIES
    LINK_FLAGS "-flto"
)
