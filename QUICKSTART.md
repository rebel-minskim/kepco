# Quick Start Guide - Webcam Live Detection

## Getting Started with Live Webcam Streaming

The application supports both **continuous live streaming** and **snapshot modes** for object detection.

### Step 1: Run the Application

```bash
# Activate virtual environment
source venv/bin/activate

# Run the application
streamlit run app.py
```

### Step 2: Select "Live Webcam Stream"

1. In the sidebar, select **"Live Webcam Stream"**
2. Click the **"▶️ Start Webcam"** button
3. Your webcam will start streaming with real-time detection!

### Step 3: Adjust Settings (Optional)

- **Confidence Threshold**: How confident the model should be (default: 0.25)
- **IoU Threshold**: Overlapping detection threshold (default: 0.45)
- **Max FPS**: Frame rate for streaming (default: 15)
- **Camera Index**: Which camera to use (0 = default, 1 = second camera, etc.)

## Available Features

| Feature | Available |
|---------|-----------|
| **Live Webcam Stream** | ✓ |
| **Webcam Snapshot** | ✓ |
| **Video Upload** | ✓ |
| **Image Upload** | ✓ |
| **FPS Control** | ✓ |
| **Start/Stop Control** | ✓ |
| **Camera Selection** | ✓ |

## Troubleshooting

### Webcam Still Not Showing

1. **Check Camera Index**
   - Try changing "Camera Index" from 0 to 1 or 2
   - Some systems have multiple camera devices

2. **Browser Permissions**
   - Chrome/Firefox: Look for camera permission popup
   - Grant permission when prompted

3. **Camera in Use**
   - Close other applications using the webcam
   - Close other browser tabs with camera access

4. **Check Camera Connection**
   ```bash
   # On Linux, check available cameras
   ls /dev/video*
   ```

5. **Restart the Application**
   - Press `Ctrl+C` in terminal
   - Run `streamlit run app.py` again

### Stream is Laggy

- Lower the **Max FPS** slider (try 10 or 5)
- Reduce **Confidence Threshold** (fewer detections = faster processing)
- Close other heavy applications

## What You Should See

When working correctly:
- **Left column**: Live webcam feed (original)
- **Right column**: Detection results with bounding boxes
- **Stats**: Frame count and detected objects
- **Real-time updates**: Detections update as you move objects

## Example Usage

1. Start the webcam with the "▶️ Start Webcam" button
2. Hold up common objects (phone, cup, laptop, etc.)
3. Watch the model detect them in real-time!
4. Adjust confidence threshold to fine-tune results
5. Click "⏹️ Stop Webcam" when done

---

## Other Input Modes

The application also supports:
- **Webcam Snapshot**: Taking single photos for detection
- **Upload Video**: Processing pre-recorded videos frame-by-frame
- **Upload Image**: Analyzing static images

Switch between modes using the sidebar radio buttons!

