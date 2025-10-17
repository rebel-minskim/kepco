// Data storage
let npuData = null;  // NPU data (npu_data.json)
let gpuData = null;  // GPU data (gpu_data.json)
let charts = {};

// Animation state
let animationState = {
    currentIndex: 0,
    isRunning: false,
    intervalId: null,
    maxDataPoints: 50
};

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', async function() {
    console.log('Initializing...');
    
    // Load data files
    await loadDataFiles();
    
    // Create charts first
    createFpsChart();
    createPerformanceChart();
    createEnergyChart();
    
    // Then sync video playback
    syncVideos();
    
    console.log('Initialization complete');
});

// Load JSON data files
async function loadDataFiles() {
    try {
        // Load NPU data
        const npuDataResponse = await fetch('npu_data.json');
        npuData = await npuDataResponse.json();
        console.log('✓ NPU data loaded:', npuData.frame_data.length, 'frames');
        
        // Load GPU data
        const gpuDataResponse = await fetch('gpu_data.json');
        gpuData = await gpuDataResponse.json();
        console.log('✓ GPU data loaded:', gpuData.frame_data.length, 'frames');
        
    } catch (error) {
        console.error('❌ Error loading data files:', error);
    }
}

// Calculate metrics
function calculateMetrics() {
    const metrics = {
        npu: {
            avgPower: 50,
            avgFps: 36,
            maxPower: 55
        },
        gpu: {
            avgPower: 0,
            avgFps: 0,
            maxPower: 0
        }
    };
    
    // Calculate NPU metrics from npu_data.json
    if (npuData) {
        metrics.npu.avgPower = npuData.statistics.power_w.mean;
        metrics.npu.maxPower = npuData.statistics.power_w.max;
        metrics.npu.avgFps = npuData.statistics.fps.mean;
    }
    
    // Calculate GPU metrics from gpu_data.json
    if (gpuData) {
        metrics.gpu.avgPower = gpuData.statistics.power_w.mean;
        metrics.gpu.maxPower = gpuData.statistics.power_w.max;
        metrics.gpu.avgFps = gpuData.statistics.fps.mean;
    }
    
    // Calculate efficiency
    metrics.npu.efficiency = metrics.npu.avgFps / metrics.npu.avgPower;
    metrics.gpu.efficiency = metrics.gpu.avgFps / metrics.gpu.avgPower;
    metrics.npu.energyPerFrame = metrics.npu.avgPower / metrics.npu.avgFps;
    metrics.gpu.energyPerFrame = metrics.gpu.avgPower / metrics.gpu.avgFps;
    metrics.efficiencyMultiplier = (metrics.npu.efficiency / metrics.gpu.efficiency).toFixed(1);
    metrics.energyMultiplier = (metrics.gpu.energyPerFrame / metrics.npu.energyPerFrame).toFixed(1);
    
    return metrics;
}

// Sync videos
function syncVideos() {
    const video1 = document.getElementById('videoPlayer1');
    const video2 = document.getElementById('videoPlayer2');
    
    if (!video1 || !video2) {
        console.log('Videos not found');
        return;
    }
    
    video1.addEventListener('play', () => {
        video2.play();
        startAnimationLoop();
    });
    
    video1.addEventListener('pause', () => {
        video2.pause();
        stopAnimationLoop();
    });
    
    video1.addEventListener('seeked', () => {
        video2.currentTime = video1.currentTime;
        animationState.currentIndex = 0;
    });
    
    video1.addEventListener('ended', () => {
        if (video1.loop) {
            animationState.currentIndex = 0;
        }
    });
    
    video1.play().catch(err => console.log('Video autoplay prevented'));
    video2.play().catch(err => console.log('Video autoplay prevented'));
}

// Create FPS chart
function createFpsChart() {
    // Use fixed values for display
    const npuFps = 353;  // Fixed NPU FPS
    const gpuFps = 195;  // Fixed GPU FPS
    
    // Create FPS chart
    const ctx = document.getElementById('fpsChart');
    if (!ctx) {
        console.log('FPS chart canvas not found');
        return;
    }
    
    // Update FPS advantage display
    const fpsDisplay = document.querySelector('#fpsChart').closest('.chart-container')?.querySelector('.metric-value');
    if (fpsDisplay) {
        const fpsRatio = (npuFps / gpuFps).toFixed(1);
        fpsDisplay.textContent = `${fpsRatio}x`;
    }
    
    // Create static bar chart
    charts.fps = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: ['NPU', 'GPU'],
            datasets: [{
                label: 'Throughput (Imgs/s)',
                data: [npuFps, gpuFps],
                backgroundColor: ['#76ff03', '#b794f6'],
                borderWidth: 0,
                barThickness: 80
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            animation: {
                duration: 1000,
                easing: 'easeInOutQuart'
            },
            plugins: {
                legend: { display: false },
                tooltip: { 
                    enabled: true,
                    callbacks: {
                        label: function(context) {
                            return context.parsed.y + ' Imgs/s';
                        }
                    }
                }
            },
            scales: {
                x: {
                    grid: { display: false },
                    ticks: {
                        color: '#ffffff',
                        font: { size: 14, weight: 'bold' }
                    }
                },
                y: {
                    display: true,
                    min: 0,
                    max: 400,
                    grid: { 
                        color: 'rgba(255, 255, 255, 0.1)',
                        drawBorder: false
                    },
                    ticks: {
                        color: '#888888',
                        font: { size: 12 }
                    }
                }
            }
        },
        plugins: [{
            afterDatasetsDraw: function(chart) {
                const ctx = chart.ctx;
                chart.data.datasets.forEach((dataset, i) => {
                    const meta = chart.getDatasetMeta(i);
                    meta.data.forEach((bar, index) => {
                        const data = dataset.data[index];
                        ctx.fillStyle = '#ffffff';
                        ctx.font = 'bold 16px Arial';
                        ctx.textAlign = 'center';
                        ctx.textBaseline = 'bottom';
                        ctx.fillText(data + ' Imgs/s', bar.x, bar.y - 5);
                    });
                });
            }
        }]
    });
    console.log('✓ FPS chart created');
}

// Create performance chart
function createPerformanceChart() {
    const ctx = document.getElementById('performanceChart');
    if (!ctx) {
        console.log('Performance chart canvas not found');
        return;
    }
    
    const metrics = calculateMetrics();
    
    // Update multiplier display
    const efficiencyDisplay = document.querySelector('#performanceChart').closest('.chart-container')?.querySelector('.metric-value');
    if (efficiencyDisplay) {
        efficiencyDisplay.textContent = `${metrics.efficiencyMultiplier}x`;
    }
    
    // Create empty chart
    const labels = Array(animationState.maxDataPoints).fill('');
    
    charts.performance = new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [
                {
                    label: `ATOM NPU (${metrics.npu.efficiency.toFixed(2)} FPS/W)`,
                    data: Array(animationState.maxDataPoints).fill(null),
                    borderColor: '#76ff03',
                    backgroundColor: 'rgba(118, 255, 3, 0.1)',
                    borderWidth: 2,
                    fill: true,
                    tension: 0.4,
                    pointRadius: 0,
                    pointHoverRadius: 0,
                    pointHitRadius: 0
                },
                {
                    label: `L40S GPU (${metrics.gpu.efficiency.toFixed(2)} FPS/W)`,
                    data: Array(animationState.maxDataPoints).fill(null),
                    borderColor: '#b794f6',
                    backgroundColor: 'rgba(183, 148, 246, 0.1)',
                    borderWidth: 2,
                    fill: true,
                    tension: 0.4,
                    pointRadius: 0,
                    pointHoverRadius: 0,
                    pointHitRadius: 0
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            animation: false,
            interaction: {
                mode: null
            },
            plugins: {
                legend: { display: false },
                tooltip: { enabled: false }
            },
            scales: {
                x: {
                    display: false,
                    grid: { display: false }
                },
                y: {
                    display: false,
                    min: 0,
                    max: 0.2,
                    ticks: {
                        stepSize: 0.00000003
                    },
                    grid: { display: false }
                }
            }
        }
    });
    console.log('✓ Performance chart created');
}

// Create energy gauges
function createEnergyChart() {
    const maxPower = 300; // W for gauge scale
    
    // ATOM NPU Power Gauge
    const atomCtx = document.getElementById('atomEnergyGauge');
    if (atomCtx) {
        charts.atomEnergy = new Chart(atomCtx, {
            type: 'doughnut',
            data: {
                datasets: [{
                    data: [0, maxPower],
                    backgroundColor: ['#76ff03', 'rgba(255, 255, 255, 0.1)'],
                    borderWidth: 0,
                    circumference: 270,
                    rotation: 225
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: true,
                cutout: '75%',
                plugins: {
                    legend: { display: false },
                    tooltip: { enabled: false }
                }
            }
        });
        console.log('✓ ATOM power gauge created');
    }
    
    // NVIDIA GPU Power Gauge
    const nvidiaCtx = document.getElementById('nvidiaEnergyGauge');
    if (nvidiaCtx) {
        charts.nvidiaEnergy = new Chart(nvidiaCtx, {
            type: 'doughnut',
            data: {
                datasets: [{
                    data: [0, maxPower],
                    backgroundColor: ['#b794f6', 'rgba(255, 255, 255, 0.1)'],
                    borderWidth: 0,
                    circumference: 270,
                    rotation: 225
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: true,
                cutout: '75%',
                plugins: {
                    legend: { display: false },
                    tooltip: { enabled: false }
                }
            }
        });
        console.log('✓ NVIDIA power gauge created');
    }
}

// Start animation loop
function startAnimationLoop() {
    if (animationState.isRunning) return;
    if (!npuData || !gpuData) {
        console.log('Cannot start animation: Data not loaded');
        return;
    }
    
    animationState.isRunning = true;
    animationState.currentIndex = 0;
    
    const metrics = calculateMetrics();
    const maxPower = 300; // W for gauge scale
    
    // Fixed interval: update every 0.5 seconds
    const updateInterval = 500; // 500ms = 0.5 seconds
    
    console.log('▶ Starting animation with interval:', updateInterval, 'ms');
    
    animationState.intervalId = setInterval(() => {
        const idx = animationState.currentIndex;
        
        // Get current frame data (same index for both charts to ensure sync)
        const gpuIdx = Math.min(idx, gpuData.frame_data.length - 1);
        const npuIdx = Math.min(idx, npuData.frame_data.length - 1);
        
        const gpuPower = gpuData.frame_data[gpuIdx].power_w;
        const gpuFps = gpuData.frame_data[gpuIdx].fps;
        const npuPower = npuData.frame_data[npuIdx].power_w;
        const npuFps = npuData.frame_data[npuIdx].fps;
        
        // Calculate real-time efficiency (FPS/Watt)
        const npuEfficiency = npuFps / npuPower;
        const gpuEfficiency = gpuFps / gpuPower;
        
        // Calculate and update NPU Advantage
        const npuAdvantage = npuEfficiency / gpuEfficiency;
        const performanceAdvantageDisplay = document.querySelector('#performanceChart')?.closest('.chart-container')?.querySelector('.metric-value');
        if (performanceAdvantageDisplay) {
            performanceAdvantageDisplay.textContent = `${npuAdvantage.toFixed(1)}x`;
        }
        
        // Update Energy Savings display (Power ratio)
        const powerSavings = gpuPower / npuPower;
        const energyDisplay = document.querySelector('#atomEnergyGauge')?.closest('.chart-container')?.querySelector('.metric-value');
        if (energyDisplay) {
            energyDisplay.textContent = `${powerSavings.toFixed(1)}x`;
        }
        
        // Update NPU power gauge (real-time power consumption)
        if (charts.atomEnergy) {
            const atomGaugeValue = document.querySelector('.atom-gauge .gauge-value');
            if (atomGaugeValue) {
                atomGaugeValue.innerHTML = `${Math.round(npuPower)}<span class="unit">W</span>`;
            }
            charts.atomEnergy.data.datasets[0].data = [
                npuPower, 
                Math.max(0, maxPower - npuPower)
            ];
            charts.atomEnergy.update('none');
        }
        
        // Update GPU power gauge (real-time power consumption)
        if (charts.nvidiaEnergy) {
            const nvidiaGaugeValue = document.querySelector('.nvidia-gauge .gauge-value');
            if (nvidiaGaugeValue) {
                nvidiaGaugeValue.innerHTML = `${Math.round(gpuPower)}<span class="unit">W</span>`;
            }
            charts.nvidiaEnergy.data.datasets[0].data = [
                gpuPower, 
                Math.max(0, maxPower - gpuPower)
            ];
            charts.nvidiaEnergy.update('none');
        }
        
        // Update performance chart (efficiency) - shift left and add new data
        if (charts.performance) {
            const perfData0 = charts.performance.data.datasets[0].data;
            const perfData1 = charts.performance.data.datasets[1].data;
            
            perfData0.shift();
            perfData0.push(npuEfficiency);
            perfData1.shift();
            perfData1.push(gpuEfficiency);
            
            charts.performance.update('none');
        }
        
        // Increment and loop
        animationState.currentIndex++;
        const maxFrames = Math.min(npuData.frame_data.length, gpuData.frame_data.length);
        if (animationState.currentIndex >= maxFrames) {
            animationState.currentIndex = 0;
        }
    }, updateInterval);
}

// Stop animation
function stopAnimationLoop() {
    if (animationState.intervalId) {
        clearInterval(animationState.intervalId);
        animationState.isRunning = false;
        console.log('⏸ Animation stopped');
    }
}

console.log('🚀 Script loaded');
