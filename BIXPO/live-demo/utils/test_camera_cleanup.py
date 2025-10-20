#!/usr/bin/env python3
"""
Test script to verify camera cleanup works properly
"""

import cv2
import time
import sys
import signal

camera = None

def cleanup():
    """Clean up camera"""
    global camera
    if camera is not None:
        print("Releasing camera...")
        camera.release()
        camera = None
        cv2.destroyAllWindows()
        time.sleep(0.5)
        print("Camera released!")

def signal_handler(sig, frame):
    """Handle Ctrl+C"""
    print("\nInterrupted, cleaning up...")
    cleanup()
    sys.exit(0)

def main():
    global camera
    
    # Register signal handler
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    print("Opening camera...")
    camera = cv2.VideoCapture(0)
    
    if not camera.isOpened():
        print("Failed to open camera!")
        return 1
    
    # Set resolution
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    # Clear buffer
    print("Clearing camera buffer...")
    for _ in range(5):
        camera.read()
        time.sleep(0.1)
    
    print("Camera opened successfully!")
    print("Reading frames for 5 seconds...")
    print("   Press Ctrl+C to test cleanup")
    
    start_time = time.time()
    frame_count = 0
    
    try:
        while time.time() - start_time < 5:
            ret, frame = camera.read()
            if ret:
                frame_count += 1
                # Just count frames, don't display
            else:
                print("Failed to read frame")
                break
            time.sleep(0.1)
        
        print(f"\nRead {frame_count} frames successfully")
        
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        cleanup()
    
    print("\nChecking if camera device is free...")
    time.sleep(1)
    
    # Try to open camera again to verify it was released
    test_camera = cv2.VideoCapture(0)
    if test_camera.isOpened():
        print("Camera device is free! (can be reopened)")
        test_camera.release()
        return 0
    else:
        print("Camera device is still locked!")
        return 1

if __name__ == '__main__':
    sys.exit(main())

