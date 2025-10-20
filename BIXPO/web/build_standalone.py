#!/usr/bin/env python3
"""
Build standalone HTML file with embedded data.
This creates a version that works offline without a web server.
"""

import json
import os

def main():
    # Read data files
    print("Reading data files...")
    
    # Load NPU and GPU data
    with open('data/npu_data.json', 'r') as f:
        npu_data = json.load(f)
    print(f"NPU data loaded: {len(npu_data['frame_data'])} frames")
    
    with open('data/gpu_data.json', 'r') as f:
        gpu_data = json.load(f)
    print(f"GPU data loaded: {len(gpu_data['frame_data'])} frames")
    
    # Read template
    with open('index.html', 'r') as f:
        html_content = f.read()
    
    # Read script
    with open('assets/js/script.js', 'r') as f:
        script_content = f.read()
    
    # Modify script to use embedded data instead of fetch
    script_standalone = script_content.replace(
        'async function loadDataFiles() {',
        'async function loadDataFiles() {\n    // Load from embedded data\n    const embeddedData = JSON.parse(document.getElementById("embedded-data").textContent);'
    )
    
    # Replace fetch calls with embedded data access
    script_standalone = script_standalone.replace(
        '''// Load NPU data
        const npuDataResponse = await fetch('data/npu_data.json');
        npuData = await npuDataResponse.json();
        console.log('NPU data loaded:', npuData.frame_data.length, 'frames');
        
        // Load GPU data
        const gpuDataResponse = await fetch('data/gpu_data.json');
        gpuData = await gpuDataResponse.json();
        console.log('GPU data loaded:', gpuData.frame_data.length, 'frames');''',
        '''npuData = embeddedData.npu_data;
        gpuData = embeddedData.gpu_data;
        
        console.log('NPU data loaded:', npuData.frame_data.length, 'frames');
        console.log('GPU data loaded:', gpuData.frame_data.length, 'frames');'''
    )
    
    # Remove try-catch wrapper
    script_standalone = script_standalone.replace('try {', '').replace(
        '''} catch (error) {
        console.error('Error loading data files:', error);
    }''', '')
    
    # Create embedded data JSON
    embedded_data = {
        "npu_data": npu_data,
        "gpu_data": gpu_data
    }
    
    embedded_json = json.dumps(embedded_data, separators=(',', ':'))
    
    # Build standalone HTML
    html_standalone = html_content.replace('</body>', f'''
    <!-- Embedded Data -->
    <script id="embedded-data" type="application/json">
    {embedded_json}
    </script>
    
    <script>
    {script_standalone}
    </script>
</body>''')
    
    # Remove original script tag
    html_standalone = html_standalone.replace('<script src="assets/js/script.js"></script>', '')
    
    # Write standalone file
    with open('index_standalone.html', 'w') as f:
        f.write(html_standalone)
    
    print("\nCreated: index_standalone.html")
    print("\nYou can now:")
    print("1. Double-click index_standalone.html to open in browser")
    print("2. No web server needed!")
    print("3. Works completely offline")
    print("\nNote: Videos must be in the same directory")

if __name__ == '__main__':
    main()

