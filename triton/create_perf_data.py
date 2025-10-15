#!/usr/bin/env python3
"""
Create binary input data directory for perf_analyzer with JPEG images
"""
import numpy as np
import cv2
import os
import argparse

def create_jpeg_image(quality=90):
    """Create a test JPEG image"""
    image = np.random.randint(0, 255, (800, 800, 3), dtype=np.uint8)
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    success, encoded_image = cv2.imencode('.jpg', image, encode_param)
    if not success:
        raise ValueError("Failed to encode image")
    return encoded_image.tobytes()

def create_input_directory(output_dir: str, quality: int = 90):
    """
    Create directory with binary file for perf_analyzer
    
    perf_analyzer expects:
    - Directory name as --input-data argument
    - Binary file named after the input tensor (IMAGE_BYTES)
    - File contains raw bytes for batch-1 request
    """
    # Create directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate JPEG
    print(f"Creating JPEG image (quality={quality})...")
    jpeg_bytes = create_jpeg_image(quality)
    
    # Save as binary file named after the input
    input_file = os.path.join(output_dir, "IMAGE_BYTES")
    with open(input_file, 'wb') as f:
        f.write(jpeg_bytes)
    
    print(f"✓ Created {input_file}")
    print(f"  JPEG size: {len(jpeg_bytes)} bytes ({len(jpeg_bytes)/1024:.1f} KB)")
    print(f"\nUsage:")
    print(f"  perf_analyzer -m yolov11_ensemble --input-data {output_dir} \\")
    print(f"    -u localhost:8001 -i grpc")
    print(f"\nExamples:")
    print(f"  # Basic test")
    print(f"  perf_analyzer -m yolov11_ensemble --input-data {output_dir} -u localhost:8001 -i grpc")
    print(f"")
    print(f"  # Request rate range (10~200 RPS)")
    print(f"  perf_analyzer -m yolov11_ensemble --input-data {output_dir} \\")
    print(f"    -u localhost:8001 -i grpc --request-rate-range 10:200:10")
    print(f"")
    print(f"  # Concurrency range (1~16)")
    print(f"  perf_analyzer -m yolov11_ensemble --input-data {output_dir} \\")
    print(f"    -u localhost:8001 -i grpc --concurrency-range 1:16:1")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create perf_analyzer input directory with JPEG data')
    parser.add_argument('-o', '--output-dir', type=str, default='perf_data',
                        help='Output directory (default: perf_data)')
    parser.add_argument('-q', '--quality', type=int, default=90,
                        help='JPEG quality 1-100 (default: 90)')
    
    args = parser.parse_args()
    
    create_input_directory(args.output_dir, args.quality)

