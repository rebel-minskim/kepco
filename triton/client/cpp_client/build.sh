#!/bin/bash

# Build script for C++ Triton client
set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
BUILD_TYPE="Release"
CLEAN_BUILD=false
VERBOSE=false
JOBS=$(nproc)
INSTALL_PREFIX=""

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to show usage
show_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -h, --help              Show this help message"
    echo "  -c, --clean             Clean build directory before building"
    echo "  -d, --debug             Build in Debug mode (default: Release)"
    echo "  -v, --verbose           Verbose build output"
    echo "  -j, --jobs N            Number of parallel jobs (default: $(nproc))"
    echo "  -i, --install PREFIX    Install prefix (optional)"
    echo "  --check-deps            Check dependencies and exit"
    echo ""
    echo "Examples:"
    echo "  $0                      # Standard release build"
    echo "  $0 --clean --debug      # Clean debug build"
    echo "  $0 --verbose -j4        # Verbose build with 4 jobs"
}

# Function to check dependencies
check_dependencies() {
    print_status "Checking dependencies..."
    
    local missing_deps=()
    
    # Check for required tools
    if ! command -v cmake &> /dev/null; then
        missing_deps+=("cmake")
    fi
    
    if ! command -v make &> /dev/null && ! command -v ninja &> /dev/null; then
        missing_deps+=("make or ninja")
    fi
    
    if ! command -v pkg-config &> /dev/null; then
        missing_deps+=("pkg-config")
    fi
    
    # Check for OpenCV
    if ! pkg-config --exists opencv4 && ! pkg-config --exists opencv; then
        missing_deps+=("OpenCV (libopencv-dev)")
    fi
    
    # Check for gRPC
    if ! pkg-config --exists grpc++; then
        missing_deps+=("gRPC (libgrpc++-dev)")
    fi
    
    # Check for protobuf
    if ! pkg-config --exists protobuf; then
        missing_deps+=("protobuf (libprotobuf-dev)")
    fi
    
    if [ ${#missing_deps[@]} -eq 0 ]; then
        print_success "All dependencies found!"
        return 0
    else
        print_error "Missing dependencies:"
        for dep in "${missing_deps[@]}"; do
            echo "  - $dep"
        done
        echo ""
        echo "Install missing dependencies:"
        echo "  Ubuntu/Debian: sudo apt-get install build-essential cmake pkg-config libopencv-dev libgrpc++-dev libprotobuf-dev"
        echo "  CentOS/RHEL:   sudo yum install gcc-c++ cmake pkgconfig opencv-devel grpc-devel protobuf-devel"
        return 1
    fi
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_usage
            exit 0
            ;;
        -c|--clean)
            CLEAN_BUILD=true
            shift
            ;;
        -d|--debug)
            BUILD_TYPE="Debug"
            shift
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        -j|--jobs)
            JOBS="$2"
            shift 2
            ;;
        -i|--install)
            INSTALL_PREFIX="$2"
            shift 2
            ;;
        --check-deps)
            check_dependencies
            exit $?
            ;;
        *)
            print_error "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

# Check dependencies
if ! check_dependencies; then
    exit 1
fi

print_status "Building C++ Triton Client..."
print_status "Build type: $BUILD_TYPE"
print_status "Jobs: $JOBS"

# Clean build directory if requested
if [ "$CLEAN_BUILD" = true ]; then
    print_status "Cleaning build directory..."
    rm -rf build
fi

# Create build directory
mkdir -p build
cd build

# Configure with CMake
print_status "Configuring with CMake..."

CMAKE_ARGS=(
    -DCMAKE_BUILD_TYPE=$BUILD_TYPE
    -DCMAKE_CXX_STANDARD=17
    -DCMAKE_CXX_STANDARD_REQUIRED=ON
)

# Add optimization flags for Release build
if [ "$BUILD_TYPE" = "Release" ]; then
    CMAKE_ARGS+=(
        -DCMAKE_CXX_FLAGS="-O3 -march=native -mtune=native -DNDEBUG"
        -DCMAKE_EXE_LINKER_FLAGS="-flto"
    )
fi

# Add install prefix if specified
if [ -n "$INSTALL_PREFIX" ]; then
    CMAKE_ARGS+=(-DCMAKE_INSTALL_PREFIX="$INSTALL_PREFIX")
fi

# Add verbose flag if requested
if [ "$VERBOSE" = true ]; then
    CMAKE_ARGS+=(-DCMAKE_VERBOSE_MAKEFILE=ON)
fi

# Run CMake configuration
if ! cmake .. "${CMAKE_ARGS[@]}"; then
    print_error "CMake configuration failed!"
    exit 1
fi

# Build
print_status "Building with $JOBS parallel jobs..."
if [ "$VERBOSE" = true ]; then
    if ! make -j$JOBS VERBOSE=1; then
        print_error "Build failed!"
        exit 1
    fi
else
    if ! make -j$JOBS; then
        print_error "Build failed!"
        exit 1
    fi
fi

# Check if executable was created
if [ -f "bin/triton_client" ]; then
    print_success "Build completed successfully!"
    print_success "Executable: $(pwd)/bin/triton_client"
    
    # Show executable info
    echo ""
    print_status "Executable information:"
    ls -lh bin/triton_client
    echo ""
    
    # Test if executable runs
    print_status "Testing executable..."
    if ./bin/triton_client --help 2>/dev/null || ./bin/triton_client -h 2>/dev/null; then
        print_success "Executable is working!"
    else
        print_warning "Executable created but may have issues running"
    fi
    
    echo ""
    print_status "Usage examples:"
    echo "  ./bin/triton_client dummy"
    echo "  ./bin/triton_client image input.jpg output.jpg"
    echo "  ./bin/triton_client video input.mp4 output.mp4"
    
else
    print_error "Executable not found after build!"
    exit 1
fi
