#!/usr/bin/env python3
"""
Camera Performance Diagnostic Tool
Tests different camera configurations to find the fastest setup
"""

import cv2
import time
import numpy as np

def test_camera_config(width, height, backend=None, fourcc=None, backend_name="Default"):
    """Test a specific camera configuration"""
    print(f"\n{'='*60}")
    print(f"Testing: {width}x{height} @ {backend_name}")
    if fourcc:
        print(f"Format: {fourcc}")
    print('='*60)
    
    try:
        # Open camera
        if backend is not None:
            cap = cv2.VideoCapture(0, backend)
        else:
            cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("❌ Failed to open camera")
            return None
        
        # Set properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        if fourcc:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
        
        # Verify actual resolution
        actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        print(f"✅ Camera opened: {int(actual_width)}x{int(actual_height)}")
        
        # Warm up
        for _ in range(5):
            cap.read()
        
        # Time 30 frame captures
        times = []
        print("📊 Capturing 30 frames...")
        for i in range(30):
            start = time.time()
            ret, frame = cap.read()
            elapsed = (time.time() - start) * 1000
            times.append(elapsed)
            
            if not ret:
                print(f"❌ Failed to read frame {i}")
                break
        
        cap.release()
        
        if len(times) > 0:
            avg_time = np.mean(times)
            min_time = np.min(times)
            max_time = np.max(times)
            std_time = np.std(times)
            
            print(f"\n📈 Results:")
            print(f"   Average: {avg_time:.1f}ms per frame")
            print(f"   Min: {min_time:.1f}ms")
            print(f"   Max: {max_time:.1f}ms")
            print(f"   Std Dev: {std_time:.1f}ms")
            print(f"   Theoretical FPS: {1000/avg_time:.1f}")
            
            return {
                'config': f"{width}x{height} @ {backend_name}",
                'avg_time': avg_time,
                'fps': 1000/avg_time,
                'fourcc': fourcc
            }
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def main():
    print("\n" + "="*60)
    print("🎥 CAMERA PERFORMANCE DIAGNOSTIC")
    print("="*60)
    print("\nThis will test different camera configurations to find")
    print("the fastest setup for your hardware.\n")
    
    results = []
    
    # Test different resolutions
    resolutions = [
        (640, 480, "VGA"),
        (800, 600, "SVGA"),
        (960, 540, "qHD"),
        (1280, 720, "HD 720p"),
    ]
    
    # Test different backends
    backends = [
        (None, "Default"),
        (cv2.CAP_V4L2, "V4L2"),
    ]
    
    # Test different formats
    formats = [None, 'MJPG', 'YUYV']
    
    print("\n🔍 Testing different configurations...")
    print("This will take about 1-2 minutes...\n")
    
    for width, height, res_name in resolutions:
        for backend, backend_name in backends:
            for fmt in formats:
                if fmt:
                    config_name = f"{backend_name} + {fmt}"
                else:
                    config_name = backend_name
                
                result = test_camera_config(width, height, backend, fmt, config_name)
                if result:
                    results.append(result)
                
                time.sleep(0.5)  # Let camera settle
    
    # Print summary
    print("\n" + "="*60)
    print("📊 SUMMARY - FASTEST CONFIGURATIONS")
    print("="*60)
    
    if results:
        # Sort by FPS (descending)
        results.sort(key=lambda x: x['fps'], reverse=True)
        
        print(f"\n{'Rank':<6} {'Configuration':<30} {'Avg Time':<12} {'FPS':<8}")
        print("-" * 60)
        
        for i, r in enumerate(results[:10], 1):  # Top 10
            print(f"{i:<6} {r['config']:<30} {r['avg_time']:>8.1f}ms   {r['fps']:>6.1f}")
        
        # Recommendations
        best = results[0]
        print("\n" + "="*60)
        print("💡 RECOMMENDATION")
        print("="*60)
        print(f"\n✅ Use: {best['config']}")
        print(f"   Expected FPS: {best['fps']:.1f}")
        print(f"   Camera read time: {best['avg_time']:.1f}ms")
        
        # Extract resolution
        config = best['config']
        if '640x480' in config:
            print(f"\n📝 Update app_web.py CONFIG:")
            print(f"   'camera_width': 640,")
            print(f"   'camera_height': 480,")
        elif '800x600' in config:
            print(f"\n📝 Update app_web.py CONFIG:")
            print(f"   'camera_width': 800,")
            print(f"   'camera_height': 600,")
        elif '1280x720' in config:
            print(f"\n📝 Update app_web.py CONFIG:")
            print(f"   'camera_width': 1280,")
            print(f"   'camera_height': 720,")
        
        if 'V4L2' in config:
            print("   Use V4L2 backend: cv2.VideoCapture(0, cv2.CAP_V4L2)")
        
        if best['fourcc']:
            print(f"   Use FOURCC: {best['fourcc']}")
    
    else:
        print("\n❌ No successful camera configurations found!")
        print("Check if camera is connected and accessible.")
    
    print("\n" + "="*60)

if __name__ == '__main__':
    main()

