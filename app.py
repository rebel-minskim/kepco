"""
Streamlit Live Inference Application with YOLO11 - Continuous Webcam Stream
Based on Ultralytics Guide: https://docs.ultralytics.com/guides/streamlit-live-inference/
"""

import streamlit as st
import cv2
from ultralytics import YOLO
from PIL import Image
import numpy as np
import tempfile
import time

# Page configuration
st.set_page_config(
    page_title="YOLO11 Live Inference",
    page_icon="🎥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title and description
st.title("🎥 YOLO11 Live Inference Application")
st.markdown("Real-time object detection using YOLOv11")

# Sidebar for settings
st.sidebar.header("⚙️ Settings")

# Model selection
model_path = st.sidebar.text_input("Model Path", value="yolov11.pt")

# Confidence threshold
confidence = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.25, 0.05)

# IoU threshold
iou_threshold = st.sidebar.slider("IoU Threshold", 0.0, 1.0, 0.45, 0.05)

# Input source selection
source_option = st.sidebar.radio(
    "Select Input Source",
    ["Live Webcam Stream", "Webcam Snapshot", "Upload Video", "Upload Image"]
)

# Load YOLO model
@st.cache_resource
def load_model(model_path):
    try:
        model = YOLO(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

model = load_model(model_path)

if model is None:
    st.error("Failed to load the model. Please check the model path.")
    st.stop()

# Main content area
col1, col2 = st.columns(2)

# Live Webcam Stream (continuous)
if source_option == "Live Webcam Stream":
    st.info("🎥 Click 'Start Webcam' to begin live detection")
    
    # Webcam controls
    webcam_col1, webcam_col2 = st.columns(2)
    
    with webcam_col1:
        start_webcam = st.button("▶️ Start Webcam", type="primary")
    
    with webcam_col2:
        stop_webcam = st.button("⏹️ Stop Webcam")
    
    # Camera index selection
    camera_index = st.sidebar.number_input("Camera Index", min_value=0, max_value=10, value=0, step=1)
    
    # FPS control
    max_fps = st.sidebar.slider("Max FPS", 1, 30, 15, 1)
    
    # Initialize session state
    if 'webcam_running' not in st.session_state:
        st.session_state.webcam_running = False
    
    if start_webcam:
        st.session_state.webcam_running = True
    
    if stop_webcam:
        st.session_state.webcam_running = False
    
    if st.session_state.webcam_running:
        # Create placeholders
        with col1:
            st.subheader("📹 Live Feed")
            frame_placeholder_input = st.empty()
        
        with col2:
            st.subheader("🎯 Detection Results")
            frame_placeholder_output = st.empty()
            stats_placeholder = st.empty()
        
        # Open webcam
        cap = cv2.VideoCapture(camera_index)
        
        if not cap.isOpened():
            st.error(f"Cannot open camera with index {camera_index}. Please check your camera connection.")
            st.session_state.webcam_running = False
        else:
            frame_delay = 1.0 / max_fps
            frame_count = 0
            
            while st.session_state.webcam_running:
                ret, frame = cap.read()
                
                if not ret:
                    st.error("Failed to capture frame from webcam")
                    break
                
                # Run inference
                results = model.predict(
                    source=frame,
                    conf=confidence,
                    iou=iou_threshold,
                    show=False,
                    save=False,
                    verbose=False
                )
                
                # Get annotated frame
                annotated_frame = results[0].plot()
                
                # Display frames
                frame_placeholder_input.image(frame, channels="BGR", use_container_width=True)
                frame_placeholder_output.image(annotated_frame, channels="BGR", use_container_width=True)
                
                # Display stats
                detections = results[0].boxes
                frame_count += 1
                
                stats_text = f"**Frame:** {frame_count} | **Objects Detected:** {len(detections)}\n\n"
                if len(detections) > 0:
                    stats_text += "**Detected Classes:**\n"
                    class_counts = {}
                    for box in detections:
                        class_id = int(box.cls[0])
                        class_name = model.names[class_id]
                        conf = float(box.conf[0])
                        
                        if class_name not in class_counts:
                            class_counts[class_name] = []
                        class_counts[class_name].append(conf)
                    
                    for class_name, confs in class_counts.items():
                        avg_conf = sum(confs) / len(confs)
                        stats_text += f"- {class_name}: {len(confs)} ({avg_conf:.2%})\n"
                
                stats_placeholder.markdown(stats_text)
                
                # Frame rate control
                time.sleep(frame_delay)
            
            cap.release()
            st.info("Webcam stopped")
    else:
        st.warning("Click 'Start Webcam' to begin live detection")

# Webcam Snapshot
elif source_option == "Webcam Snapshot":
    with col1:
        st.subheader("📸 Camera Input")
    with col2:
        st.subheader("🎯 Detection Results")
    
    st.info("📸 Enable your webcam using the camera input below")
    
    camera_input = st.camera_input("Take a picture")
    
    if camera_input is not None:
        # Convert to PIL Image
        image = Image.open(camera_input)
        
        # Convert to numpy array
        img_array = np.array(image)
        
        # Display original image
        with col1:
            st.image(image, caption="Original Image", use_container_width=True)
        
        # Run inference
        results = model.predict(
            source=img_array,
            conf=confidence,
            iou=iou_threshold,
            show=False,
            save=False
        )
        
        # Get annotated image
        annotated_frame = results[0].plot()
        
        # Display result
        with col2:
            st.image(annotated_frame, caption="Detection Results", use_container_width=True, channels="BGR")
            
            # Display detection statistics
            detections = results[0].boxes
            st.metric("Objects Detected", len(detections))
            
            # Show detected classes
            if len(detections) > 0:
                st.write("**Detected Objects:**")
                for box in detections:
                    class_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    class_name = model.names[class_id]
                    st.write(f"- {class_name}: {conf:.2%}")

# Video upload
elif source_option == "Upload Video":
    with col1:
        st.subheader("📹 Original Video")
    with col2:
        st.subheader("🎯 Detection Results")
    
    video_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov", "mkv"])
    
    if video_file is not None:
        # Save uploaded video to temporary file
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(video_file.read())
        
        # Display original video
        with col1:
            st.video(video_file)
        
        # Process video button
        if st.button("🚀 Process Video"):
            with st.spinner("Processing video..."):
                # Open video
                cap = cv2.VideoCapture(tfile.name)
                
                # Get video properties
                fps = int(cap.get(cv2.CAP_PROP_FPS))
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                
                # Create placeholder for processed frame
                with col2:
                    frame_placeholder = st.empty()
                    progress_bar = st.progress(0)
                    stats_placeholder = st.empty()
                
                frame_count = 0
                
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    # Run inference
                    results = model.predict(
                        source=frame,
                        conf=confidence,
                        iou=iou_threshold,
                        show=False,
                        save=False,
                        verbose=False
                    )
                    
                    # Get annotated frame
                    annotated_frame = results[0].plot()
                    
                    # Display frame
                    frame_placeholder.image(annotated_frame, channels="BGR", use_container_width=True)
                    
                    # Update progress
                    frame_count += 1
                    progress = frame_count / total_frames
                    progress_bar.progress(progress)
                    
                    # Display stats
                    detections = results[0].boxes
                    stats_placeholder.write(f"Frame {frame_count}/{total_frames} | Objects: {len(detections)}")
                
                cap.release()
                st.success("✅ Video processing complete!")

# Image upload
elif source_option == "Upload Image":
    with col1:
        st.subheader("📷 Original Image")
    with col2:
        st.subheader("🎯 Detection Results")
    
    image_file = st.file_uploader("Upload an image file", type=["jpg", "jpeg", "png", "bmp"])
    
    if image_file is not None:
        # Load image
        image = Image.open(image_file)
        img_array = np.array(image)
        
        # Display original image
        with col1:
            st.image(image, caption="Original Image", use_container_width=True)
        
        # Run inference
        with st.spinner("Running detection..."):
            results = model.predict(
                source=img_array,
                conf=confidence,
                iou=iou_threshold,
                show=False,
                save=False
            )
        
        # Get annotated image
        annotated_frame = results[0].plot()
        
        # Display result
        with col2:
            st.image(annotated_frame, caption="Detection Results", use_container_width=True, channels="BGR")
            
            # Display detection statistics
            detections = results[0].boxes
            st.metric("Objects Detected", len(detections))
            
            # Show detected classes with details
            if len(detections) > 0:
                st.write("**Detection Details:**")
                for i, box in enumerate(detections):
                    class_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    class_name = model.names[class_id]
                    coords = box.xyxy[0].tolist()
                    
                    with st.expander(f"{i+1}. {class_name} - {conf:.2%}"):
                        st.write(f"**Confidence:** {conf:.2%}")
                        st.write(f"**Bounding Box:** [{coords[0]:.1f}, {coords[1]:.1f}, {coords[2]:.1f}, {coords[3]:.1f}]")

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("### About")
st.sidebar.info(
    "This application uses YOLOv11 for real-time object detection. "
    "Adjust the confidence and IoU thresholds to fine-tune the detection results."
)

st.sidebar.markdown("### Tips")
st.sidebar.markdown("""
- **Live Webcam Stream**: Continuous detection (requires rerun to update)
- **Webcam Snapshot**: Single frame capture
- **Upload Video**: Process pre-recorded videos
- **Upload Image**: Detect objects in images
""")

