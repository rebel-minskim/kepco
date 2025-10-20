"""
Test script to verify NPU inference is working correctly
Compares GPU and NPU inference on a sample image
"""

import cv2
import numpy as np
import time
from PIL import Image

def test_gpu_inference():
    """Test GPU inference using Ultralytics YOLO"""
    print("\n" + "="*50)
    print("Testing GPU/CPU Inference")
    print("="*50)
    
    try:
        from ultralytics import YOLO
        
        # Load model
        print("Loading YOLO model...")
        model = YOLO('yolov11.pt')
        
        # Create a test image
        test_image = np.random.randint(0, 255, (800, 800, 3), dtype=np.uint8)
        
        # Warm-up
        print("Warming up...")
        _ = model.predict(test_image, conf=0.25, verbose=False)
        
        # Timed inference
        print("Running inference...")
        start_time = time.time()
        results = model.predict(test_image, conf=0.25, verbose=False)
        elapsed = time.time() - start_time
        
        detections = results[0].boxes
        print(f"✅ GPU Inference successful!")
        print(f"   - Time: {elapsed*1000:.2f} ms")
        print(f"   - Detections: {len(detections)}")
        
        return True
        
    except Exception as e:
        print(f"❌ GPU Inference failed: {str(e)}")
        return False

def test_npu_inference():
    """Test NPU inference using RBLN Runtime"""
    print("\n" + "="*50)
    print("Testing NPU Inference")
    print("="*50)
    
    try:
        import rebel
        
        # Load compiled model
        print("Loading RBLN model...")
        runtime = rebel.Runtime('yolov11.rbln')
        
        # Create a test image (preprocessed format)
        test_image = np.random.rand(1, 3, 800, 800).astype(np.float32)
        
        # Warm-up
        print("Warming up...")
        _ = runtime.run(test_image)
        
        # Timed inference
        print("Running inference...")
        start_time = time.time()
        output = runtime.run(test_image)
        elapsed = time.time() - start_time
        
        print(f"✅ NPU Inference successful!")
        print(f"   - Time: {elapsed*1000:.2f} ms")
        print(f"   - Output shape: {output[0].shape if isinstance(output, (list, tuple)) else output.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ NPU Inference failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("\n" + "="*50)
    print("YOLO11 Inference Comparison Test")
    print("="*50)
    
    # Test GPU
    gpu_success = test_gpu_inference()
    
    # Test NPU
    npu_success = test_npu_inference()
    
    # Summary
    print("\n" + "="*50)
    print("Test Summary")
    print("="*50)
    print(f"GPU/CPU: {'✅ PASS' if gpu_success else '❌ FAIL'}")
    print(f"NPU:     {'✅ PASS' if npu_success else '❌ FAIL'}")
    
    if gpu_success and npu_success:
        print("\n🎉 Both inference methods are working!")
        print("You can now run:")
        print("  - GPU version: streamlit run app.py")
        print("  - NPU version: streamlit run app_npu.py")
    elif npu_success:
        print("\n⚠️  NPU inference is working, but GPU inference failed.")
        print("You can run the NPU version: streamlit run app_npu.py")
    elif gpu_success:
        print("\n⚠️  GPU inference is working, but NPU inference failed.")
        print("Please check your RBLN installation and compiled model.")
    else:
        print("\n❌ Both inference methods failed. Please check your setup.")
    
    print("="*50 + "\n")

if __name__ == "__main__":
    main()

