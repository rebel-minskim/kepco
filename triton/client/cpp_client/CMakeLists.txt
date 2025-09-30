cmake_minimum_required(VERSION 3.16)
project(triton_client_cpp)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
find_package(OpenCV REQUIRED)
find_package(PkgConfig REQUIRED)

# Find gRPC and protobuf
find_package(PkgConfig REQUIRED)
pkg_check_modules(GRPC REQUIRED grpc++)

# Include directories
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${GRPC_INCLUDE_DIRS})

# Source files
set(SOURCES
    main.cpp
    triton_client.cpp
    utils.cpp
    grpc_client.cpp
    yolo_postprocess.cpp
    grpc_service.pb.cc
    grpc_service.grpc.pb.cc
)

# Create executable
add_executable(triton_client ${SOURCES})

# Link libraries
target_link_libraries(triton_client 
    ${OpenCV_LIBS}
    ${GRPC_LIBRARIES}
    protobuf
    pthread
)

# Compiler flags
target_compile_options(triton_client PRIVATE ${GRPC_CFLAGS_OTHER})

# Set output directory
set_target_properties(triton_client PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# Create directories
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/output)
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/media)

# Copy data.yaml if it exists
if(EXISTS ${CMAKE_SOURCE_DIR}/data.yaml)
    configure_file(${CMAKE_SOURCE_DIR}/data.yaml ${CMAKE_BINARY_DIR}/data.yaml COPYONLY)
endif()

# Print configuration info
message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "OpenCV libraries: ${OpenCV_LIBS}")
message(STATUS "gRPC libraries: ${GRPC_LIBRARIES}")