#!/bin/bash

# Build script for C++ Triton client
set -e

echo "Building C++ Triton Client..."

# Create build directory
mkdir -p build
cd build

# Configure with CMake
echo "Configuring with CMake..."
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_FLAGS="-O3 -march=native -mtune=native" \
    -DCMAKE_EXE_LINKER_FLAGS="-flto"

# Build
echo "Building..."
make -j$(nproc)

echo "Build completed successfully!"
echo "Executable: build/triton_cpp_client"
