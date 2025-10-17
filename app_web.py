"""
Flask-based Web Interface for YOLO11 NPU Detection
Much faster than Streamlit - uses MJPEG streaming for real-time video
"""

from flask import Flask, render_template, Response, request, jsonify
import cv2
import numpy as np
import rebel
import yaml
import time
import json
import webbrowser
import threading
import signal
import atexit
from typing import List, Tuple
from queue import Queue, Empty

app = Flask(__name__)

# Global camera object for proper cleanup
camera = None
camera_lock = threading.Lock()
stop_streaming = False

# Configuration
CONFIG = {
    'model_path': 'yolov11.rbln',
    'confidence': 0.25,
    'iou_threshold': 0.45,
    'camera_index': 0,
    'skip_frames': 2,  # Process every 2nd frame (balance FPS vs NPU util)
    'camera_width': 960,   # qHD resolution - good balance of speed and quality
    'camera_height': 540,  # 960x540 (qHD) 16:9 aspect ratio
    'jpeg_quality': 90,  # Reduced for faster encoding
    'model_input_size': 800,  # Model input size
}

# Load class names
def load_class_names(yaml_path="data.yaml"):
    try:
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)
        if 'names' in data and isinstance(data['names'], dict):
            num_classes = data.get('nc', len(data['names']))
            class_names = ['unknown'] * num_classes
            for idx, name in data['names'].items():
                if idx < num_classes:
                    class_names[idx] = name
            return class_names
    except:
        pass
    return []

CLASS_NAMES = load_class_names()

# Load model with Runtime for better performance (AsyncRuntime has overhead)
print("Loading RBLN model with Runtime...")
runtime = rebel.Runtime(CONFIG['model_path'])
print("✅ Runtime loaded!")

# Preprocessing (highly optimized for speed)
def preprocess_image(image, input_size=None):
    if input_size is None:
        input_size = (CONFIG['model_input_size'], CONFIG['model_input_size'])
    h, w = image.shape[:2]
    scale = min(input_size[0] / h, input_size[1] / w)
    new_h, new_w = int(h * scale), int(w * scale)
    
    # Use INTER_LINEAR (faster than INTER_AREA for downsampling on camera frames)
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    # Create padded image with pre-allocated array (faster than np.full)
    padded = np.zeros((input_size[0], input_size[1], 3), dtype=np.uint8)
    padded.fill(114)  # Slightly faster than np.full
    top = (input_size[0] - new_h) // 2
    left = (input_size[1] - new_w) // 2
    padded[top:top+new_h, left:left+new_w] = resized
    
    # Convert BGR to RGB
    padded = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
    
    # Transpose and normalize in one go (more cache-friendly)
    padded = padded.transpose(2, 0, 1).astype(np.float32) / 255.0
    padded = np.expand_dims(padded, axis=0)
    
    return padded, scale, (top, left)

# NMS
def nms(boxes, scores, iou_thresh):
    if len(boxes) == 0:
        return []
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        inds = np.where(iou <= iou_thresh)[0]
        order = order[inds + 1]
    return keep

# Postprocessing
def postprocess_output(output, conf_thresh, iou_thresh, scale, padding, orig_shape):
    output = output[0]
    if output.shape[0] < output.shape[1]:
        output = output.T
    boxes = output[:, :4]
    class_scores = output[:, 4:]
    max_scores = np.max(class_scores, axis=1)
    class_ids = np.argmax(class_scores, axis=1)
    mask = max_scores > conf_thresh
    if not mask.any():
        return []
    boxes = boxes[mask]
    scores = max_scores[mask]
    class_ids = class_ids[mask]
    x_center, y_center, width, height = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x1 = x_center - width / 2
    y1 = y_center - height / 2
    x2 = x_center + width / 2
    y2 = y_center + height / 2
    boxes = np.stack([x1, y1, x2, y2], axis=1)
    keep_indices = nms(boxes, scores, iou_thresh)
    if len(keep_indices) == 0:
        return []
    boxes = boxes[keep_indices]
    scores = scores[keep_indices]
    class_ids = class_ids[keep_indices]
    top, left = padding
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - left) / scale
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - top) / scale
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, orig_shape[1])
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, orig_shape[0])
    detections = []
    for box, score, class_id in zip(boxes, scores, class_ids):
        detections.append({
            'box': box.tolist(),
            'confidence': float(score),
            'class_id': int(class_id),
            'class_name': CLASS_NAMES[int(class_id)] if int(class_id) < len(CLASS_NAMES) else f"class_{int(class_id)}"
        })
    return detections

# Draw detections
def draw_detections(image, detections):
    for det in detections:
        box = det['box']
        x1, y1, x2, y2 = map(int, box)
        conf = det['confidence']
        class_name = det['class_name']
        np.random.seed(det['class_id'])
        color = tuple(map(int, np.random.randint(0, 255, 3)))
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        label = f"{class_name}: {conf:.2%}"
        (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(image, (x1, y1 - label_h - 10), (x1 + label_w, y1), color, -1)
        cv2.putText(image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return image

# Global stats
stats = {
    'fps': 0,
    'inference_time': 0,
    'detections': 0,
    'frame_count': 0,
    'npu_utilization': 0,  # Track NPU busy time vs total time
    'preprocess_time': 0,
    'postprocess_time': 0,
    # Bottleneck profiling
    'camera_time': 0,
    'drawing_time': 0,
    'jpeg_encode_time': 0,
    'network_yield_time': 0,
    'total_loop_time': 0,
}

def cleanup_camera():
    """Release camera resources"""
    global camera
    with camera_lock:
        if camera is not None:
            print("🎥 Releasing camera...")
            try:
                camera.release()
            except Exception as e:
                print(f"⚠️  Error releasing camera: {e}")
            camera = None
            # Wait a bit for camera to fully release
            time.sleep(0.5)
            print("✅ Camera released!")
    
    # Force release using cv2
    cv2.destroyAllWindows()
    time.sleep(0.2)

def get_camera():
    """Get or initialize camera"""
    global camera
    with camera_lock:
        if camera is None or not camera.isOpened():
            print("🎥 Initializing camera...")
            
            # Release any existing camera first
            if camera is not None:
                try:
                    camera.release()
                except:
                    pass
                camera = None
                time.sleep(0.5)
            
            # Use default backend with MJPEG format (user preference: 960x540 @ Default + MJPG)
            camera = cv2.VideoCapture(CONFIG['camera_index'])
            
            if not camera.isOpened():
                print("❌ Failed to open camera!")
                return None
            
            # Set camera properties for fast capture
            camera.set(cv2.CAP_PROP_FRAME_WIDTH, CONFIG['camera_width'])
            camera.set(cv2.CAP_PROP_FRAME_HEIGHT, CONFIG['camera_height'])
            camera.set(cv2.CAP_PROP_FPS, 30)
            camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimal buffering
            
            # Use MJPEG format for faster capture
            camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
            print("📷 Using MJPEG format for camera capture")
            
            # Warm up camera with a few quick reads (no sleep!)
            print("🔄 Warming up camera...")
            for _ in range(3):
                camera.read()
            
            print("✅ Camera initialized!")
        return camera

def signal_handler(sig, frame):
    """Handle termination signals"""
    global stop_streaming
    print(f"\n⚠️  Received signal {sig}, shutting down...")
    stop_streaming = True
    
    # Give time for active streams to finish
    time.sleep(1)
    
    cleanup_camera()
    print("✅ Cleanup complete!")
    import sys
    sys.exit(0)

# Register cleanup handlers
atexit.register(cleanup_camera)
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

def generate_frames():
    """Generator function for MJPEG streaming with NPU utilization tracking"""
    cap = get_camera()
    
    if cap is None:
        print("❌ Failed to initialize camera for streaming")
        return
    
    frame_count = 0
    fps_start_time = time.time()
    last_detections = []
    total_npu_time = 0
    total_preprocess_time = 0
    total_postprocess_time = 0
    npu_call_count = 0
    
    try:
        while not stop_streaming:
            loop_start = time.time()
            
            # BOTTLENECK CHECK 1: Camera capture time
            camera_start = time.time()
            ret, frame = cap.read()
            camera_time = (time.time() - camera_start) * 1000
            stats['camera_time'] = camera_time
            
            if not ret:
                print("⚠️  Failed to read frame from camera")
                break
            
            # No cropping needed - 1280x720 is already 16:9
            frame_count += 1
            
            # Process every Nth frame
            if frame_count % CONFIG['skip_frames'] == 0:
                # Measure preprocessing time
                preprocess_start = time.time()
                preprocessed, scale, padding = preprocess_image(frame)
                preprocess_time = (time.time() - preprocess_start) * 1000
                total_preprocess_time += preprocess_time
                
                # Measure NPU inference time ONLY (not including pre/post processing)
                npu_start = time.time()
                output = runtime.run(preprocessed)
                npu_time = (time.time() - npu_start) * 1000
                total_npu_time += npu_time
                npu_call_count += 1
                
                # Measure postprocessing time
                postprocess_start = time.time()
                last_detections = postprocess_output(
                    output[0] if isinstance(output, (list, tuple)) else output,
                    CONFIG['confidence'],
                    CONFIG['iou_threshold'],
                    scale,
                    padding,
                    frame.shape[:2]
                )
                postprocess_time = (time.time() - postprocess_start) * 1000
                total_postprocess_time += postprocess_time
                
                # Calculate total pipeline time
                total_inference_time = preprocess_time + npu_time + postprocess_time
                
                # Update stats
                stats['inference_time'] = npu_time  # Pure NPU time
                stats['preprocess_time'] = preprocess_time
                stats['postprocess_time'] = postprocess_time
                stats['detections'] = len(last_detections)
                
                # Calculate NPU utilization (NPU busy time / total elapsed time)
                elapsed_total = time.time() - fps_start_time
                if elapsed_total > 0 and npu_call_count > 0:
                    stats['npu_utilization'] = (total_npu_time / 1000.0) / elapsed_total * 100
            
            # BOTTLENECK CHECK 2: Drawing time
            drawing_start = time.time()
            annotated_frame = draw_detections(frame, last_detections)
            
            # Calculate FPS
            elapsed = time.time() - fps_start_time
            current_fps = frame_count / elapsed if elapsed > 0 else 0
            stats['fps'] = current_fps
            stats['frame_count'] = frame_count
            
            # Draw performance stats on frame
            cv2.putText(annotated_frame, f"FPS: {current_fps:.1f}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(annotated_frame, f"Detections: {len(last_detections)}", (10, 70), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            drawing_time = (time.time() - drawing_start) * 1000
            stats['drawing_time'] = drawing_time
            
            # BOTTLENECK CHECK 3: JPEG encoding time
            encode_start = time.time()
            ret, buffer = cv2.imencode('.jpg', annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, CONFIG['jpeg_quality']])
            frame_bytes = buffer.tobytes()
            encode_time = (time.time() - encode_start) * 1000
            stats['jpeg_encode_time'] = encode_time
            
            # BOTTLENECK CHECK 4: Network yield time
            yield_start = time.time()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            yield_time = (time.time() - yield_start) * 1000
            stats['network_yield_time'] = yield_time
            
            # Total loop time
            total_loop = (time.time() - loop_start) * 1000
            stats['total_loop_time'] = total_loop
    
    except GeneratorExit:
        print("📹 Browser closed, stream ended")
    except Exception as e:
        print(f"❌ Error in video stream: {e}")
    finally:
        # Release camera when stream ends to turn off the green light
        cleanup_camera()
        print("📹 Video stream generator exited")

@app.before_request
def check_shutdown():
    """Check if we're shutting down"""
    if stop_streaming:
        return "Server shutting down", 503

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html', config=CONFIG, classes=CLASS_NAMES)

@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stats')
def get_stats():
    """API endpoint for stats"""
    return jsonify(stats)

@app.route('/bottleneck')
def bottleneck_analysis():
    """Detailed bottleneck analysis"""
    total = stats.get('total_loop_time', 1)  # Avoid division by zero
    
    analysis = {
        'timings_ms': {
            'camera_capture': stats.get('camera_time', 0),
            'preprocess': stats.get('preprocess_time', 0),
            'npu_inference': stats.get('inference_time', 0),
            'postprocess': stats.get('postprocess_time', 0),
            'drawing': stats.get('drawing_time', 0),
            'jpeg_encode': stats.get('jpeg_encode_time', 0),
            'network_yield': stats.get('network_yield_time', 0),
            'total_loop': total,
        },
        'percentages': {
            'camera_capture': (stats.get('camera_time', 0) / total * 100) if total > 0 else 0,
            'preprocess': (stats.get('preprocess_time', 0) / total * 100) if total > 0 else 0,
            'npu_inference': (stats.get('inference_time', 0) / total * 100) if total > 0 else 0,
            'postprocess': (stats.get('postprocess_time', 0) / total * 100) if total > 0 else 0,
            'drawing': (stats.get('drawing_time', 0) / total * 100) if total > 0 else 0,
            'jpeg_encode': (stats.get('jpeg_encode_time', 0) / total * 100) if total > 0 else 0,
            'network_yield': (stats.get('network_yield_time', 0) / total * 100) if total > 0 else 0,
        },
        'bottleneck': 'Calculating...',
        'recommendation': 'Run for a few seconds to get accurate data'
    }
    
    # Identify the biggest bottleneck
    if total > 10:  # Only if we have real data
        timings = analysis['timings_ms']
        max_time_key = max(timings, key=lambda k: timings[k] if k != 'total_loop' else 0)
        max_time = timings[max_time_key]
        
        analysis['bottleneck'] = f"{max_time_key}: {max_time:.2f}ms ({timings[max_time_key]/total*100:.1f}%)"
        
        # Recommendations
        if max_time_key == 'camera_capture':
            analysis['recommendation'] = "Camera I/O is the bottleneck. Consider: lower resolution, different camera, or parallel pipeline."
        elif max_time_key == 'npu_inference':
            analysis['recommendation'] = "NPU inference is the bottleneck. Consider: smaller model input, model optimization, or batch processing."
        elif max_time_key == 'jpeg_encode':
            analysis['recommendation'] = "JPEG encoding is slow. Consider: lower JPEG quality, smaller frame size, or hardware encoding."
        elif max_time_key == 'network_yield':
            analysis['recommendation'] = "Network streaming is slow. Check: network bandwidth, browser performance, or reduce frame size."
        elif max_time_key in ['preprocess', 'postprocess']:
            analysis['recommendation'] = f"{max_time_key} is slow. Optimize the algorithm or use faster libraries."
        elif max_time_key == 'drawing':
            analysis['recommendation'] = "Drawing/text overlay is slow. Reduce number of boxes or text operations."
    
    return jsonify(analysis)

@app.route('/config', methods=['GET', 'POST'])
def config():
    """API endpoint for configuration"""
    if request.method == 'POST':
        data = request.json
        CONFIG.update(data)
        return jsonify({'status': 'success', 'config': CONFIG})
    return jsonify(CONFIG)

@app.route('/shutdown', methods=['POST'])
def shutdown():
    """Graceful shutdown endpoint"""
    global stop_streaming
    print("📴 Shutdown requested via API...")
    stop_streaming = True
    
    # Cleanup camera
    cleanup_camera()
    
    # Shutdown Flask
    func = request.environ.get('werkzeug.server.shutdown')
    if func is None:
        # For production servers, just return success
        # The actual shutdown will be handled by stop_app.sh
        return jsonify({'status': 'shutting_down'})
    func()
    return jsonify({'status': 'shutdown_complete'})

def open_browser():
    """Open browser after a short delay"""
    time.sleep(1.5)  # Wait for Flask to start
    webbrowser.open('http://localhost:5000')

if __name__ == '__main__':
    print("\n" + "="*70)
    print("🚀 YOLO11 NPU Web Interface - Bottleneck Profiling Enabled")
    print("="*70)
    print(f"\n✅ Model loaded: {CONFIG['model_path']}")
    print(f"✅ Classes: {len(CLASS_NAMES)}")
    print(f"✅ Camera: {CONFIG['camera_index']}")
    print(f"✅ Resolution: {CONFIG['camera_width']}x{CONFIG['camera_height']} (qHD)")
    print(f"✅ Format: MJPEG (Default backend)")
    print(f"✅ Model input: {CONFIG['model_input_size']}x{CONFIG['model_input_size']}")
    print(f"✅ Skip frames: {CONFIG['skip_frames']} (process every 2nd frame)")
    print(f"\n🌐 Opening browser automatically to: http://localhost:5000")
    print("\n🔍 BOTTLENECK PROFILING ACTIVE:")
    print("   Every operation is timed to find the slowest component!")
    print("   - Camera capture time")
    print("   - Preprocessing time")
    print("   - NPU inference time")
    print("   - Postprocessing time")
    print("   - Drawing/text overlay time")
    print("   - JPEG encoding time")
    print("   - Network streaming time")
    print("\n📊 Analysis Endpoints:")
    print("   - http://localhost:5000/stats - All timing stats")
    print("   - http://localhost:5000/bottleneck - Detailed bottleneck analysis")
    print("\n💡 After running for a few seconds, check:")
    print("   curl http://localhost:5000/bottleneck | jq")
    print("="*70 + "\n")
    
    # Open browser in background thread
    threading.Thread(target=open_browser, daemon=True).start()
    
    # Run Flask app
    app.run(host='0.0.0.0', port=5000, threaded=True, debug=False)

