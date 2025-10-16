# KEPCO - Streamlit Live Inference Application

A real-time object detection application using YOLOv11 and Streamlit, based on the [Ultralytics Streamlit Live Inference Guide](https://docs.ultralytics.com/guides/streamlit-live-inference/).

## Features

- 🎥 **Live Webcam Inference**: Real-time object detection from your webcam
- 📹 **Video Upload**: Upload and process video files
- 🖼️ **Image Upload**: Detect objects in static images
- ⚙️ **Adjustable Parameters**: Control confidence threshold, IoU threshold, and more
- 📊 **Real-time Visualization**: See detections with bounding boxes and labels

## Prerequisites

- Python 3.8 or higher
- Webcam (for live inference)
- Virtual environment (recommended)

## Installation

1. **Activate the virtual environment**:
   ```bash
   source venv/bin/activate
   ```

2. **Install dependencies** (if not already installed):
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Run the Application

```bash
streamlit run app.py
```

The application will automatically open in your default web browser at `http://localhost:8501`.

### Using the Interface

1. **Select Input Source**:
   - **Live Webcam Stream**: Continuous real-time detection (click Start/Stop buttons)
   - **Webcam Snapshot**: Single frame capture for detection
   - **Upload Video**: Process pre-recorded video files
   - **Upload Image**: Detect objects in static images

2. **Adjust Detection Parameters**:
   - **Confidence Threshold**: Minimum confidence for detections (0.0 - 1.0)
   - **IoU Threshold**: Intersection over Union threshold for NMS
   - **Max FPS**: Control frame rate for live streaming
   - **Camera Index**: Select camera device (0 for default, 1+ for additional cameras)

3. **View Results**:
   - Original and annotated frames displayed side-by-side
   - Detection statistics and frame rate information
   - Bounding boxes with class labels and confidence scores

## Models

This project includes:
- `yolov11.pt`: YOLOv11 PyTorch model
- `yolov11.rbln`: YOLOv11 Rebellions optimized model

You can use different models by modifying the `model` parameter in `app.py`.

## Project Structure

```
kepco/
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
├── yolov11.pt         # YOLO model file
├── yolov11.rbln       # Rebellions optimized model
├── venv/              # Virtual environment
└── logs/              # Application logs
```

## Customization

You can customize the application by modifying `app.py`:

```python
inf = solutions.Inference(
    model="yolov11.pt",      # Change model path
    # Add more parameters as needed
)
```

## Troubleshooting

### Webcam Not Visible / Not Working

- Select **"Live Webcam Stream"** from the sidebar and click **"▶️ Start Webcam"** button
- Ensure your webcam is not being used by another application
- Try different camera indices (0, 1, 2) if default doesn't work
- Grant browser permissions for camera access when prompted
- Try a different browser (Chrome/Firefox recommended)
- If stream freezes, click "⏹️ Stop Webcam" and restart
- For single snapshots instead of streaming, use "Webcam Snapshot" mode

### Model Loading Errors
- Verify the model file exists in the project directory
- Check that the model path in `app.py` is correct
- Ensure sufficient disk space for model loading

### Performance Issues
- Lower the input resolution
- Reduce the confidence threshold
- Use a smaller YOLO model (e.g., yolo11n.pt)
- Close other resource-intensive applications

## Additional Resources

- [Ultralytics Documentation](https://docs.ultralytics.com/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [YOLOv11 Guide](https://docs.ultralytics.com/models/yolo11/)

## License

This project uses Ultralytics YOLO models. Please refer to the [Ultralytics License](https://github.com/ultralytics/ultralytics/blob/main/LICENSE) for usage terms.

