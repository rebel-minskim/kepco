#!/usr/bin/env python3
"""
Camera Diagnostic Tool - Find and fix blur issues
"""

import cv2
import numpy as np
import time
import sys

def measure_sharpness(frame):
    """Measure image sharpness using Laplacian variance"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def diagnose_camera():
    print("🔍 Camera Diagnostic Tool")
    print("=" * 60)
    
    # Open camera
    print("\n📷 Opening camera...")
    camera = cv2.VideoCapture(0)
    
    if not camera.isOpened():
        print("❌ Failed to open camera!")
        return 1
    
    # Try different settings
    print("\n🔧 Testing different camera settings...\n")
    
    # Get default settings
    width = camera.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = camera.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = camera.get(cv2.CAP_PROP_FPS)
    autofocus = camera.get(cv2.CAP_PROP_AUTOFOCUS)
    focus = camera.get(cv2.CAP_PROP_FOCUS)
    exposure = camera.get(cv2.CAP_PROP_EXPOSURE)
    
    print(f"Default Settings:")
    print(f"  Resolution: {int(width)}x{int(height)}")
    print(f"  FPS: {int(fps)}")
    print(f"  Autofocus: {autofocus}")
    print(f"  Focus: {focus}")
    print(f"  Exposure: {exposure}")
    
    # Set to 960x540
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)
    camera.set(cv2.CAP_PROP_FPS, 30)
    camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    # Enable autofocus
    camera.set(cv2.CAP_PROP_AUTOFOCUS, 1)
    
    print(f"\n📊 Testing frame quality over time...")
    print(f"{'Frame':<8} {'Time(s)':<10} {'Sharpness':<12} {'Status'}")
    print("-" * 60)
    
    start_time = time.time()
    sharpness_values = []
    
    for i in range(20):
        ret, frame = camera.read()
        if not ret:
            print(f"❌ Failed to read frame {i+1}")
            continue
        
        elapsed = time.time() - start_time
        sharpness = measure_sharpness(frame)
        sharpness_values.append(sharpness)
        
        # Determine status
        if sharpness < 50:
            status = "🔴 Very Blurry"
        elif sharpness < 100:
            status = "🟡 Blurry"
        elif sharpness < 200:
            status = "🟢 Acceptable"
        else:
            status = "✅ Sharp"
        
        print(f"{i+1:<8} {elapsed:<10.2f} {sharpness:<12.1f} {status}")
        
        # Save first and last frame for comparison
        if i == 0:
            cv2.imwrite('/tmp/camera_frame_first.jpg', frame)
        elif i == 19:
            cv2.imwrite('/tmp/camera_frame_last.jpg', frame)
        
        time.sleep(0.2)
    
    camera.release()
    
    print("\n" + "=" * 60)
    print("📈 Analysis:")
    print(f"  First frame sharpness: {sharpness_values[0]:.1f}")
    print(f"  Last frame sharpness: {sharpness_values[-1]:.1f}")
    print(f"  Average sharpness: {np.mean(sharpness_values):.1f}")
    print(f"  Improvement: {sharpness_values[-1] - sharpness_values[0]:.1f}")
    
    # Calculate when it stabilizes
    avg_last_5 = np.mean(sharpness_values[-5:])
    for i, val in enumerate(sharpness_values):
        if val >= avg_last_5 * 0.9:  # Within 90% of stable value
            print(f"  Stabilized at frame: {i+1} ({i*0.2:.1f}s)")
            break
    
    print(f"\n💡 Recommendation:")
    if sharpness_values[-1] < 100:
        print("  ⚠️  Camera appears to be out of focus!")
        print("  Try:")
        print("    1. Clean the camera lens")
        print("    2. Check if there's a physical focus ring to adjust")
        print("    3. Increase lighting in the room")
        print("    4. Try manual focus if auto-focus isn't working")
    elif sharpness_values[0] < 100 and sharpness_values[-1] > 150:
        stable_frame = next((i for i, v in enumerate(sharpness_values) if v >= avg_last_5 * 0.9), 10)
        stable_time = stable_frame * 0.2
        print(f"  ✅ Camera needs ~{stable_time:.1f}s to stabilize")
        print(f"  Increase buffer clearing to {stable_frame+2} frames with 0.2s delays")
    else:
        print("  ✅ Camera appears to be working well!")
    
    print(f"\n📁 Sample frames saved:")
    print(f"  First frame: /tmp/camera_frame_first.jpg")
    print(f"  Last frame:  /tmp/camera_frame_last.jpg")
    print(f"\nCompare these images to see the difference!")
    
    return 0

if __name__ == '__main__':
    try:
        sys.exit(diagnose_camera())
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        sys.exit(1)
