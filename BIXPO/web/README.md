# NPU vs GPU Performance Comparison Dashboard

A real-time web dashboard comparing ATOM™-Max NPU and NVIDIA L40S GPU performance metrics for AI video processing workloads.

![Dashboard Preview](/BIXPO/web/assets/images/image.png)

![Status](https://img.shields.io/badge/Status-Live-green)

## Features

- **Dual Video Comparison**: Side-by-side video playback showing NPU and GPU processing results
- **Real-time Power Monitoring**: Animated gauge charts showing current power consumption
- **Performance Efficiency Tracking**: Live scrolling graph comparing FPS per Watt
- **Power Usage Analysis**: Real-time power draw visualization in Watts
- **Auto-calculated Metrics**: Automatically computes efficiency multipliers from data

## Dashboard Components

### 1. Video Panels (Top)
- **Left**: ATOM™-Max NPU processing output (36 Imgs/s)
- **Right**: L40S GPU processing output (24 Imgs/s)
- Both videos play in sync

### 2. Power Consumption (Bottom Left)
- Gauge charts showing real-time power usage
- Updates dynamically as data streams
- Shows current power and maximum capacity

### 3. Performance Efficiency (Bottom Center)
- Line graph showing FPS/Watt over time
- Scrolls from right to left
- Shows efficiency advantage multiplier (e.g., "6.3x")

### 4. Power Usage (Bottom Right)
- Line graph showing power draw over time
- Real-time watts measurement
- Compares power consumption between devices

## Quick Start

### Prerequisites
- Modern web browser (Chrome, Firefox, Safari, Edge)
- Video files: `assets/videos/output_npu.mp4` and `assets/videos/output_gpu.mp4`
- Data files: JSON files with power and FPS metrics

### Two Ways to Use:

#### Option 1: Standalone (Recommended for Offline Use)

**No server needed! Works completely offline.**

1. **Build the standalone version:**
   ```bash
   cd cctv_solution/frontend
   python3 build_standalone.py
   ```

2. **Open the file:**
   - **Double-click** `index_standalone.html`, or
   - Right-click → Open With → Your Browser

3. **Done!** Works offline, no internet needed.

**When to use:**
- Presenting offline (no internet)
- Don't want to start a server every time
- Simple double-click to open
- Share single HTML file (+ videos)

#### Option 2: With Web Server

**Better for development and frequent data updates.**

1. **Navigate to the frontend directory:**
   ```bash
   cd cctv_solution/frontend
   ```

2. **Start a local web server:**
   ```bash
   python3 -m http.server 8080
   ```

3. **Open your browser:**
   ```
   http://localhost:8080/index.html
   ```

**When to use:**
- Developing and testing
- Frequently updating data
- Don't want to rebuild after each change

### Alternative: Using Node.js

If you prefer Node.js:
```bash
npx http-server -p 8080
```

### Required Files

```
web/
├── index.html                      # Original (needs server)
├── index_standalone.html           # Standalone (no server)
├── build_standalone.py             # Build script
├── assets/
│   ├── css/style.css
│   ├── js/script.js
│   ├── images/ (logos)
│   └── videos/ (output_npu.mp4, output_gpu.mp4)
└── data/
    ├── npu_data.json
    └── gpu_data.json
```

## How to Use

1. **Load the Page**: Open `http://localhost:8080/index.html` in your browser

2. **Start Animation**: Click play on any video to start:
   - Videos play in sync
   - Charts begin animating
   - Power gauges update in real-time

3. **Observe Metrics**:
   - Watch power consumption change on gauge charts
   - See efficiency graphs scroll from right to left
   - Compare NPU (green) vs GPU (purple) lines

4. **Pause/Resume**: Click pause on video to freeze all animations

5. **Loop**: Videos and data automatically loop for continuous monitoring

## Updating Data (Standalone Version)

If you're using the standalone version and want to update your data:

```bash
# 1. Update your JSON data files
# Edit npu_power.json, gpu_power.json, npu_fps.json, gpu_fps.json

# 2. Rebuild the standalone file
python3 build_standalone.py

# 3. Open the new standalone file
# Double-click index_standalone.html
```

**That's it!** All your new data is embedded in the HTML file.

## Customizing with Your Data

### Step 1: Prepare Your Video Files

Convert your videos to web-compatible format:
```bash
# For NPU video
ffmpeg -i your_npu_video.mp4 -c:v libx264 -preset fast -crf 23 \
  -movflags +faststart assets/videos/output_npu.mp4

# For GPU video
ffmpeg -i your_gpu_video.mp4 -c:v libx264 -preset fast -crf 23 \
  -movflags +faststart assets/videos/output_gpu.mp4
```

### Step 2: Create Power Data JSON

Format: `npu_power.json` and `gpu_power.json`

```json
{
  "metadata": {
    "total_samples": 41,
    "duration_seconds": 23.9,
    "device": "Device Name"
  },
  "samples": [
    {
      "timestamp": 1760605347.19,
      "power": 50.5,
      "relative_time": 0.0
    },
    {
      "timestamp": 1760605347.73,
      "power": 51.2,
      "relative_time": 0.54
    }
    // ... more samples
  ]
}
```

**Key Fields:**
- `samples[].power`: Power in Watts
- `samples[].timestamp`: Unix timestamp
- `samples[].relative_time`: Seconds from start

### Step 3: Create FPS Data JSON

Format: `npu_fps.json` and `gpu_fps.json`

```json
{
  "metadata": {
    "total_samples": 180
  },
  "summary": {
    "total_frames": 1800,
    "total_time_seconds": 23.36,
    "average_fps": 90.0,
    "video_info": {
      "width": 1920,
      "height": 1080
    }
  },
  "statistics": {
    "inference_time": {
      "mean_ms": 28.0,
      "min_ms": 18.0,
      "max_ms": 45.0
    }
  },
  "frames": []
}
```

**Key Fields:**
- `summary.average_fps`: Average frames per second
- `summary.total_time_seconds`: Processing duration

### Step 4: Update Device Names (Optional)

Edit `index.html` to change device names:
```html
<!-- Line 19 -->
<div class="device-name">Your NPU Name</div>

<!-- Line 43 -->
<div class="device-name">Your GPU Name</div>
```

### Step 5: Refresh and View

Reload the page to see your data!

## Calculated Metrics

The dashboard automatically calculates:

### Performance Efficiency
```
Efficiency = FPS / Average Power (Watts)
Example: 90 FPS / 50W = 1.8 FPS/Watt
```

### Energy Per Frame
```
Energy = Average Power / FPS
Example: 50W / 90 FPS = 0.556 Joules/frame
```

### Efficiency Multiplier
```
Multiplier = NPU Efficiency / GPU Efficiency
Shows how many times more efficient NPU is
```

## Troubleshooting

### Issue: CORS Error (Cross-Origin Request Blocked)

**Problem**: Opening `index.html` directly shows CORS errors

**Solution**: Always use a web server
```bash
# Use Python
python3 -m http.server 8080

# Or use Node.js
npx http-server -p 8080
```

### Issue: Videos Not Playing

**Problem**: Videos show black screen or don't load

**Solutions**:
1. Check video codec: Must be H.264
   ```bash
   ffmpeg -i video.mp4 -c:v libx264 output.mp4
   ```

2. Check file paths in HTML match your video filenames

3. Check browser console (F12) for error messages

### Issue: Charts Not Displaying

**Problem**: Blank areas where charts should be

**Solutions**:
1. Open browser console (F12) and check for errors
2. Verify JSON files are valid (use jsonlint.com)
3. Check that Chart.js loaded: Look for "Script loaded" in console
4. Ensure data files exist and have correct names

### Issue: Charts Not Animating

**Problem**: Charts visible but not moving

**Solutions**:
1. Click play on the videos
2. Check console for "Animation loop started" message
3. Verify data files have multiple samples (at least 10+)
4. Check that `samples` array exists in JSON files

### Issue: Wrong Metrics Displayed

**Problem**: Efficiency multiplier shows incorrect values

**Solutions**:
1. Verify `average_fps` is in `summary` section of FPS JSON
2. Check power values are in Watts (not milliwatts)
3. Ensure sample counts are greater than 0
4. Recalculate: Open console and check logged metrics

## Data Collection Tips

### Collecting Power Data

**For NVIDIA GPUs:**
```bash
# Continuous monitoring
nvidia-smi --query-gpu=timestamp,power.draw \
  --format=csv,noheader,nounits --loop-ms=500 > gpu_power.txt
```

**For NPU:** Use your device's monitoring tool to collect similar data

### Collecting FPS Data

Track these metrics during video processing:
- Total frames processed
- Total processing time
- Average FPS: `total_frames / total_time`
- Inference time per frame

### Syncing Data with Video

For best results:
- Start power monitoring when video processing starts
- Stop power monitoring when processing ends
- Use same video file for demo as was processed
- Match data duration to video duration

## Customization

### Changing Colors

Edit `assets/css/style.css`:
```css
/* NPU color (green) */
.legend-item.atom .legend-dot { background: #76ff03; }

/* GPU color (purple) */
.legend-item.nvidia .legend-dot { background: #b794f6; }
```

### Adjusting Animation Speed

Edit `assets/js/script.js`:
```javascript
// Line 350: Change update interval calculation
let updateInterval = 500; // milliseconds
```

### Changing Data Window Size

Edit `assets/js/script.js`:
```javascript
// Line 12: Number of data points visible
maxDataPoints: 60  // Show last 60 samples
```

## File Structure

```
web/
├── index.html                  # Main HTML page
├── index_standalone.html       # Standalone version (generated)
├── build_standalone.py         # Script to build standalone version
├── README.md                   # This file (English)
├── README_ko.md                # Korean documentation
│
├── assets/
│   ├── css/
│   │   └── style.css          # Styling and layout
│   ├── js/
│   │   └── script.js          # JavaScript logic & animations
│   ├── images/
│   │   ├── logo_rebellions.svg # Rebellions logo
│   │   ├── logo_nvidia.svg     # NVIDIA logo
│   │   └── image.png           # Dashboard preview
│   └── videos/
│       ├── output_npu.mp4      # NPU processing video
│       └── output_gpu.mp4      # GPU processing video
│
└── data/
    ├── npu_data.json           # NPU performance data
    └── gpu_data.json           # GPU performance data
```

## Browser Console Commands

Open console (F12) and try:

```javascript
// Check loaded data
console.log(gpuPowerData);
console.log(gpuFpsData);

// Check metrics
console.log(calculateMetrics());

// Manually stop animation
stopAnimationLoop();

// Manually start animation
startAnimationLoop();
```

## Requirements

- **Browser**: Modern browser with ES6 support
- **Web Server**: Python 3.x, Node.js, or any HTTP server
- **Video Codec**: H.264 (MP4)
- **Data Format**: JSON files with specified structure

## Limitations

- Requires local web server (cannot open file:// directly)
- Videos must be H.264 encoded
- Maximum recommended data points: ~1000 samples
- Best viewed on desktop/laptop (responsive design limited)

## Tips for Best Results

1. **Video Quality**: Use high-quality H.264 encoded videos
2. **Data Sync**: Ensure data duration matches video duration
3. **Sample Rate**: 2-5 samples per second provides smooth animation
4. **File Size**: Keep videos under 100MB for better loading
5. **Testing**: Always test locally before deploying

## Support

**Debug Checklist:**
- [ ] Web server is running
- [ ] All files exist in frontend directory
- [ ] JSON files are valid
- [ ] Videos are H.264 encoded
- [ ] Browser console shows no errors
- [ ] Port is not blocked by firewall

**Still having issues?**
1. Check browser console (F12) for error messages
2. Verify file names match exactly (case-sensitive)
3. Test with sample data first
4. Ensure all JSON files have correct structure

## License

Copyright © 2025 Rebellions Inc.

---

**Quick Commands Reference:**

```bash
# Start server
python3 -m http.server 8080

# Convert video
ffmpeg -i input.mp4 -c:v libx264 -crf 23 output.mp4

# Validate JSON
python3 -m json.tool data.json

# Open in browser
open http://localhost:8080/index.html
```
