# Camera Resource Management Fix

## Issues Identified

### 1. **Blurred Screen on Startup**
**Problem**: When starting the app with `start_app.sh`, the camera shows a blurred screen.

**Root Cause**: 
- Camera buffer accumulates old frames when the device is opened
- When running in background with `nohup`, the camera initialization doesn't have time to stabilize
- OpenCV's internal buffer holds stale frames from previous sessions

**Fix**:
- Added buffer clearing mechanism: read and discard 5 frames after camera initialization
- Added 0.1s delay between each frame read to allow camera to stabilize
- Improved camera initialization check before proceeding

### 2. **Camera Stays Active (Green Light On) After Stop**
**Problem**: When using `stop_app.sh`, the camera's green light stays on, indicating it's still in use.

**Root Cause**:
- `cleanup_camera()` wasn't being called reliably when Flask was terminated
- Signal handlers (SIGTERM/SIGINT) don't work properly with background processes started via `nohup`
- The `generate_frames()` generator's `finally` block wasn't executing when process was killed

**Fix**:
- Added graceful shutdown API endpoint (`/shutdown`) 
- Modified `stop_app.sh` to call the shutdown API first before using signals
- Added `cv2.destroyAllWindows()` call in cleanup to force release
- Added sleep delays to ensure camera hardware has time to release
- Added timeout waits in cleanup to ensure proper resource release

### 3. **/dev/video0 Not Detected**
**Problem**: Sometimes the camera device disappears completely after improper shutdown.

**Root Cause**:
- Camera device driver enters error state when not released properly
- Multiple processes or zombie processes holding the device
- USB camera module (uvcvideo) in bad state

**Fix**:
- Created `utils/reset_camera.sh` script to:
  - Kill all processes using camera devices
  - Unload and reload camera kernel modules (uvcvideo)
  - Verify camera device availability
- Added proper cleanup in all exit paths
- Added error handling in camera release operations

## Changes Made

### app_web.py

#### 1. Enhanced `cleanup_camera()`:
```python
def cleanup_camera():
    """Release camera resources"""
    global camera
    with camera_lock:
        if camera is not None:
            print("Releasing camera...")
            try:
                camera.release()
            except Exception as e:
                print(f"Error releasing camera: {e}")
            camera = None
            # Wait a bit for camera to fully release
            time.sleep(0.5)
            print("Camera released!")
    
    # Force release using cv2
    cv2.destroyAllWindows()
    time.sleep(0.2)
```

**Improvements**:
- Added try-except for safe release
- Added sleep delays for hardware release
- Added `cv2.destroyAllWindows()` to force cleanup

#### 2. Improved `get_camera()`:
```python
def get_camera():
    """Get or initialize camera"""
    global camera
    with camera_lock:
        if camera is None or not camera.isOpened():
            print("Initializing camera...")
            
            # Release any existing camera first
            if camera is not None:
                try:
                    camera.release()
                except:
                    pass
                camera = None
                time.sleep(0.5)
            
            camera = cv2.VideoCapture(CONFIG['camera_index'])
            
            if not camera.isOpened():
                print("Failed to open camera!")
                return None
            
            camera.set(cv2.CAP_PROP_FRAME_WIDTH, CONFIG['camera_width'])
            camera.set(cv2.CAP_PROP_FRAME_HEIGHT, CONFIG['camera_height'])
            camera.set(cv2.CAP_PROP_FPS, 30)
            camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            # Clear any buffered frames by reading a few
            print("Clearing camera buffer...")
            for _ in range(5):
                camera.read()
                time.sleep(0.1)
            
            print("Camera initialized!")
        return camera
```

**Improvements**:
- Added pre-release check to ensure clean state
- Added camera open verification
- **Added buffer clearing**: reads 5 frames to clear stale data
- Added proper error handling

#### 3. New `/shutdown` API Endpoint:
```python
@app.route('/shutdown', methods=['POST'])
def shutdown():
    """Graceful shutdown endpoint"""
    global stop_streaming
    print("Shutdown requested via API...")
    stop_streaming = True
    
    # Cleanup camera
    cleanup_camera()
    
    # Shutdown Flask
    func = request.environ.get('werkzeug.server.shutdown')
    if func is None:
        return jsonify({'status': 'shutting_down'})
    func()
    return jsonify({'status': 'shutdown_complete'})
```

**Purpose**: Allows graceful shutdown that properly releases camera before process termination

#### 4. Enhanced `signal_handler()`:
```python
def signal_handler(sig, frame):
    """Handle termination signals"""
    global stop_streaming
    print(f"\nReceived signal {sig}, shutting down...")
    stop_streaming = True
    
    # Give time for active streams to finish
    time.sleep(1)
    
    cleanup_camera()
    print("Cleanup complete!")
    import sys
    sys.exit(0)
```

**Improvements**:
- Added delay to allow active streams to finish
- Ensures cleanup happens before exit

#### 5. Improved `generate_frames()`:
```python
def generate_frames():
    """Generator function for MJPEG streaming"""
    cap = get_camera()
    
    if cap is None:
        print("Failed to initialize camera for streaming")
        return
    
    # ... rest of function ...
```

**Improvements**:
- Added None check for camera
- Better error messages

### stop_app.sh

#### Added Graceful Shutdown via API:
```bash
# Try graceful API shutdown first
echo "Attempting graceful shutdown via API..."
SHUTDOWN_RESPONSE=$(curl -s -X POST http://localhost:5000/shutdown 2>/dev/null)
if [ $? -eq 0 ]; then
    echo "Shutdown API called successfully"
    sleep 2
    
    # Check if process stopped
    PIDS=$(ps aux | grep "[p]ython.*app_web.py" | awk '{print $2}')
    if [ -z "$PIDS" ]; then
        echo "Flask app stopped gracefully"
        # ... port cleanup ...
        exit 0
    fi
else
    echo "Shutdown API not available, proceeding with signal-based shutdown..."
fi
```

**Improvements**:
- Tries API shutdown first (cleanest method)
- Falls back to signal-based shutdown if API fails
- Better feedback to user

## New Scripts

### 1. utils/reset_camera.sh
Emergency camera reset script that:
- Stops the Flask app
- Kills any processes using camera devices
- Unloads and reloads camera kernel modules
- Verifies camera availability

**Usage**:
```bash
./utils/reset_camera.sh
```

### 2. utils/test_camera_cleanup.py
Test script to verify camera cleanup works properly:
- Opens camera
- Reads frames for 5 seconds
- Tests cleanup on normal exit and interrupt
- Verifies camera can be reopened

**Usage**:
```bash
python3 utils/test_camera_cleanup.py
```

## Usage Instructions

### Normal Operation

1. **Start the app**:
   ```bash
   ./start_app.sh
   ```

2. **Stop the app**:
   ```bash
   ./stop_app.sh
   ```

3. **If camera gets stuck**:
   ```bash
   ./utils/reset_camera.sh
   ```

### Verification

After stopping the app, verify camera is released:

```bash
# Check if camera device is available
ls -l /dev/video0

# Check if any process is using the camera
lsof /dev/video0

# Test camera with cheese (should work immediately)
cheese
```

### Testing

Run the test script to verify cleanup:
```bash
python3 utils/test_camera_cleanup.py
```

Expected output:
```
Opening camera...
Clearing camera buffer...
Camera opened successfully!
Reading frames for 5 seconds...
   Press Ctrl+C to test cleanup

Read 50 frames successfully
Releasing camera...
Camera released!

Checking if camera device is free...
Camera device is free! (can be reopened)
```

## Why Cheese Works But Your App Didn't

**Cheese**:
- Properly handles camera lifecycle
- Has mature cleanup handlers
- Waits for device to fully release before exiting
- Has explicit signal handling for terminal environments

**Your App (before fixes)**:
- Background process (`nohup`) bypasses normal signal handling
- Generator cleanup wasn't guaranteed to run
- No buffer clearing on initialization
- Insufficient delays for hardware release

## Technical Details

### Camera Buffer Issue
OpenCV's `VideoCapture` maintains an internal buffer of frames. When you first open the camera:
1. Old frames from previous session may still be in hardware buffer
2. First few frames are often blurred/out-of-focus
3. Camera needs time to adjust exposure and focus

**Solution**: Read and discard 5-10 frames after opening camera.

### Background Process Issues
When running with `nohup`:
- Process is detached from terminal
- Signal handlers may not work reliably
- `atexit` handlers may not be called
- Need alternative shutdown mechanism (API endpoint)

### Camera Hardware Release
Camera hardware needs time to:
1. Flush internal buffers
2. Reset hardware state
3. Release USB device locks
4. Update kernel driver state

**Solution**: Add 0.5-1s delays after `camera.release()`.

## Troubleshooting

### Green light stays on after stop
1. Run `./utils/reset_camera.sh`
2. If that doesn't work: `sudo rmmod uvcvideo && sudo modprobe uvcvideo`
3. Check for zombie processes: `ps aux | grep python`

### /dev/video0 not found
1. Run `./utils/reset_camera.sh`
2. Check USB connection: `lsusb`
3. Check kernel logs: `dmesg | grep video`
4. Replug camera

### Blurred screen on start
1. This should be fixed with buffer clearing
2. If persists, increase clear count in `get_camera()` from 5 to 10
3. Add longer delays between frame reads (0.2s instead of 0.1s)

### Port 5000 still in use
1. Wait 30-60 seconds for TIME_WAIT state to clear
2. Or use: `sudo fuser -k 5000/tcp`
3. Script already handles this automatically

## Summary

The fixes ensure proper camera resource management by:
1. Clearing camera buffer on initialization (fixes blurred screen)
2. Adding graceful shutdown API (ensures cleanup)
3. Improving signal handlers (better background process handling)
4. Adding delays for hardware release (ensures green light turns off)
5. Adding emergency reset script (recovers from bad states)
6. Adding comprehensive error handling (prevents crashes)

Your app should now handle cameras as reliably as `cheese`!

