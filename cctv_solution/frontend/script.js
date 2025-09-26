// Global variables
let loopCount = 0;
let isPlaying = false;
let detectionInterval;
let alertCount = 0;

// AI Detection System
const detectionSystem = {
    // Detection rules
    rules: {
        noHelmet: {
            threshold: 0.6,
            duration: 2000, // 2 seconds
            cooldown: 60000 // 60 seconds
        },
        restrictedZone: {
            polygon: [
                { x: 0.6, y: 0.2 }, // Top-right corner
                { x: 0.8, y: 0.2 },
                { x: 0.8, y: 0.5 },
                { x: 0.6, y: 0.5 }
            ],
            cooldown: 60000
        },
        fall: {
            aspectRatioThreshold: 0.3, // Height/Width ratio
            stationaryTime: 3000, // 3 seconds
            cooldown: 60000
        }
    },
    
    // Active detections
    activeDetections: new Map(),
    
    // Cooldown tracking
    cooldowns: new Map(),
    
    // Event history
    events: [],
    
    // Statistics
    stats: {
        helmetCompliance: 95,
        incidentCount: 0,
        fallCount: 0,
        zoneViolations: 0,
        personCount: 0
    }
};

// Initialize application
document.addEventListener('DOMContentLoaded', function() {
    initializeVideoControls();
    initializeDashboard();
    startAIDetection();
    updateCurrentTime();
    setInterval(updateCurrentTime, 1000);
});

// Initialize video controls
function initializeVideoControls() {
    const videoPlayer = document.getElementById('videoPlayer');
    const playBtn = document.getElementById('playBtn');
    const restartBtn = document.getElementById('restartBtn');
    const loopBtn = document.getElementById('loopBtn');
    const fullscreenBtn = document.getElementById('fullscreenBtn');
    const emergencyBtn = document.getElementById('emergencyBtn');
    const videoOverlay = document.getElementById('videoOverlay');
    
    // Play/Pause button
    if (playBtn) {
        playBtn.addEventListener('click', () => {
            if (videoPlayer.paused) {
                videoPlayer.play();
                isPlaying = true;
            } else {
                videoPlayer.pause();
                isPlaying = false;
            }
        });
    }
    
    // Restart button
    if (restartBtn) {
        restartBtn.addEventListener('click', () => {
            videoPlayer.currentTime = 0;
            videoPlayer.play();
            isPlaying = true;
        });
    }

    // Loop toggle button
    if (loopBtn) {
        loopBtn.addEventListener('click', () => {
            videoPlayer.loop = !videoPlayer.loop;
            loopBtn.innerHTML = videoPlayer.loop ? 
                '<i class="fas fa-sync"></i><span>Loop: ON</span>' : 
                '<i class="fas fa-sync"></i><span>Loop: OFF</span>';
            loopBtn.classList.toggle('active', videoPlayer.loop);
        });
    }
    
    // Fullscreen button
    if (fullscreenBtn) {
        fullscreenBtn.addEventListener('click', () => {
            if (videoPlayer.requestFullscreen) {
                videoPlayer.requestFullscreen();
            } else if (videoPlayer.webkitRequestFullscreen) {
                videoPlayer.webkitRequestFullscreen();
            } else if (videoPlayer.msRequestFullscreen) {
                videoPlayer.msRequestFullscreen();
            }
        });
    }
    
    // Emergency button
    if (emergencyBtn) {
        emergencyBtn.addEventListener('click', () => {
            triggerEmergencyAlert();
        });
    }
    
    videoPlayer.addEventListener('play', () => {
        if (videoOverlay) {
            videoOverlay.innerHTML = '<i class="fas fa-circle"></i><span>LIVE</span>';
        }
        if (playBtn) {
            playBtn.innerHTML = '<i class="fas fa-pause"></i><span>Pause</span>';
        }
        isPlaying = true;
    });
    
    videoPlayer.addEventListener('pause', () => {
        if (videoOverlay) {
            videoOverlay.innerHTML = '<i class="fas fa-pause"></i><span>PAUSED</span>';
        }
        if (playBtn) {
            playBtn.innerHTML = '<i class="fas fa-play"></i><span>Play</span>';
        }
        isPlaying = false;
    });
    
    videoPlayer.addEventListener('ended', () => {
        if (videoPlayer.loop) {
            if (videoOverlay) {
                videoOverlay.innerHTML = '<i class="fas fa-sync"></i><span>RESTARTING...</span>';
            }
            loopCount++;
        } else {
            if (videoOverlay) {
                videoOverlay.innerHTML = '<i class="fas fa-stop"></i><span>ENDED</span>';
            }
            isPlaying = false;
        }
    });

    // Handle video loading
    videoPlayer.addEventListener('loadeddata', () => {
        console.log('📹 Video loaded and ready to play');
        if (videoOverlay) {
            videoOverlay.innerHTML = '<i class="fas fa-circle"></i><span>LIVE</span>';
        }
    });

    videoPlayer.addEventListener('error', (e) => {
        console.error('❌ Video error:', e);
        if (videoOverlay) {
            videoOverlay.innerHTML = '<i class="fas fa-exclamation-triangle"></i><span>ERROR</span>';
        }
    });
}

// Initialize dashboard
function initializeDashboard() {
    setupFilters();
    addSampleEvents();
}

// Setup filter functionality
function setupFilters() {
    const cameraFilter = document.getElementById('cameraFilter');
    const eventFilter = document.getElementById('eventFilter');
    const timeFilter = document.getElementById('timeFilter');
    const clearFilters = document.getElementById('clearFilters');
    
    [cameraFilter, eventFilter, timeFilter].forEach(filter => {
        filter.addEventListener('change', applyFilters);
    });
    
    clearFilters.addEventListener('click', () => {
        cameraFilter.value = 'all';
        eventFilter.value = 'all';
        timeFilter.value = 'all';
        applyFilters();
    });
}

// Apply filters to events
function applyFilters() {
    const cameraFilter = document.getElementById('cameraFilter').value;
    const eventFilter = document.getElementById('eventFilter').value;
    const timeFilter = document.getElementById('timeFilter').value;
    
    const eventsList = document.getElementById('eventsList');
    const allEvents = eventsList.querySelectorAll('.event-item');
    
    allEvents.forEach(event => {
        let show = true;
        
        // Camera filter
        if (cameraFilter !== 'all') {
            const eventCamera = event.dataset.camera;
            if (eventCamera !== cameraFilter) show = false;
        }
        
        // Event type filter
        if (eventFilter !== 'all') {
            const eventType = event.dataset.type;
            if (eventType !== eventFilter) show = false;
        }
        
        // Time filter
        if (timeFilter !== 'all') {
            const eventTime = new Date(event.dataset.timestamp);
            const now = new Date();
            const timeDiff = now - eventTime;
            
            switch (timeFilter) {
                case '1h':
                    if (timeDiff > 3600000) show = false;
                    break;
                case '6h':
                    if (timeDiff > 21600000) show = false;
                    break;
                case '24h':
                    if (timeDiff > 86400000) show = false;
                    break;
            }
        }
        
        event.style.display = show ? 'flex' : 'none';
    });
    
    updateEventCount();
}

// Update event count
function updateEventCount() {
    const eventsList = document.getElementById('eventsList');
    const visibleEvents = eventsList.querySelectorAll('.event-item[style*="flex"], .event-item:not([style])');
    document.getElementById('eventCount').textContent = `${visibleEvents.length} events`;
}

// Add sample events for demo
function addSampleEvents() {
    const sampleEvents = [
        {
            type: 'no_helmet',
            title: 'No Helmet Detected',
            time: new Date(Date.now() - 2 * 60 * 1000), // 2 minutes ago
            location: 'CAM-001',
            camera: 'cam001'
        },
        {
            type: 'fall',
            title: 'Fall Detection Alert',
            time: new Date(Date.now() - 5 * 60 * 1000), // 5 minutes ago
            location: 'CAM-002',
            camera: 'cam002'
        },
        {
            type: 'no_helmet',
            title: 'No Helmet Detected',
            time: new Date(Date.now() - 12 * 60 * 1000), // 12 minutes ago
            location: 'CAM-003',
            camera: 'cam003'
        },
        {
            type: 'fall',
            title: 'Fall Detection Alert',
            time: new Date(Date.now() - 15 * 60 * 1000), // 15 minutes ago
            location: 'CAM-001',
            camera: 'cam001'
        },
        {
            type: 'no_helmet',
            title: 'No Helmet Detected',
            time: new Date(Date.now() - 22 * 60 * 1000), // 22 minutes ago
            location: 'CAM-001',
            camera: 'cam001'
        },
        {
            type: 'fall',
            title: 'Fall Detection Alert',
            time: new Date(Date.now() - 25 * 60 * 1000), // 25 minutes ago
            location: 'CAM-003',
            camera: 'cam003'
        }
    ];
    
    console.log('Adding sample events:', sampleEvents.length);
    sampleEvents.forEach(event => addEventToList(event));
    updateEventCount();
    console.log('Sample events added successfully');
}

// Add event to list
function addEventToList(eventData) {
    const eventsList = document.getElementById('eventsList');
    console.log('Adding event:', eventData.title, 'to eventsList:', eventsList);
    const eventItem = document.createElement('div');
    eventItem.className = `event-item ${eventData.type}`;
    eventItem.dataset.type = eventData.type;
    eventItem.dataset.camera = eventData.camera;
    eventItem.dataset.timestamp = eventData.time.getTime();
    
    const timeString = eventData.time.toLocaleTimeString('ko-KR', { 
        hour: '2-digit', 
        minute: '2-digit' 
    });
    
    const icons = {
        no_helmet: 'material-icons',
        restricted_zone: 'material-icons',
        fall: 'material-icons'
    };
    
    const iconNames = {
        no_helmet: 'construction',
        restricted_zone: 'block',
        fall: 'person_off'
    };
    
    eventItem.innerHTML = `
        <div class="event-thumbnail">
            <span class="${icons[eventData.type]}">${iconNames[eventData.type]}</span>
        </div>
        <div class="event-content">
            <div class="event-title">${eventData.title}</div>
            <div class="event-time">${timeString}</div>
            <div class="event-location">${eventData.location}</div>
        </div>
        <button class="event-play-btn" onclick="playEvent('${eventData.type}')">
            <i class="fas fa-play"></i>
        </button>
    `;
    
    eventsList.insertBefore(eventItem, eventsList.firstChild);
    
    // Keep only last 20 events
    while (eventsList.children.length > 20) {
        eventsList.removeChild(eventsList.lastChild);
    }
    
    updateEventCount();
}

// Play event (simulate jumping to event time)
function playEvent(eventType) {
    const videoPlayer = document.getElementById('videoPlayer');
    // Simulate jumping to a specific time for the event
    videoPlayer.currentTime = Math.random() * videoPlayer.duration;
    videoPlayer.play();
}

// Start AI detection simulation
function startAIDetection() {
    detectionInterval = setInterval(() => {
        simulateAIDetection();
    }, 1000);
}

// Simulate AI detection
function simulateAIDetection() {
    // Simulate person detection
    const personCount = Math.floor(Math.random() * 4) + 1;
    detectionSystem.stats.personCount = personCount;
    document.getElementById('personCount').textContent = personCount;
    
    // Simulate detections based on rules
    if (Math.random() < 0.1) { // 10% chance per second
        const detectionType = ['no_helmet', 'restricted_zone', 'fall'][Math.floor(Math.random() * 3)];
        triggerDetection(detectionType);
    }
    
    // Update live bounding boxes
    updateLiveDetections();
}

// Trigger specific detection
function triggerDetection(type) {
    const personId = `person_${Date.now()}`;
    const cooldownKey = `${type}_${personId}`;
    
    // Check cooldown
    if (detectionSystem.cooldowns.has(cooldownKey)) {
        return;
    }
    
    // Set cooldown
    detectionSystem.cooldowns.set(cooldownKey, Date.now());
    setTimeout(() => {
        detectionSystem.cooldowns.delete(cooldownKey);
    }, detectionSystem.rules[type].cooldown);
    
    // Create detection
    const detection = {
        id: personId,
        type: type,
        timestamp: Date.now(),
        bbox: generateRandomBbox(),
        score: 0.7 + Math.random() * 0.3
    };
    
    detectionSystem.activeDetections.set(personId, detection);
    
    // Add to events list
    const eventData = {
        type: type,
        title: getDetectionTitle(type),
        time: new Date(),
        location: 'CAM-001',
        camera: 'cam001'
    };
    
    addEventToList(eventData);
    
    // Update statistics
    updateStatsForDetection(type);
    
    // Show detection overlay
    showDetectionOverlay(detection);
    
    // Auto-hide after 5 seconds
    setTimeout(() => {
        hideDetectionOverlay(personId);
    }, 5000);
}

// Generate random bounding box
function generateRandomBbox() {
    return {
        x: Math.random() * 0.6 + 0.1,
        y: Math.random() * 0.6 + 0.1,
        width: 0.1 + Math.random() * 0.2,
        height: 0.15 + Math.random() * 0.3
    };
}

// Get detection title
function getDetectionTitle(type) {
    const titles = {
        no_helmet: 'No Helmet Detected',
        restricted_zone: 'Restricted Zone Violation',
        fall: 'Fall Detection Alert'
    };
    return titles[type] || 'Unknown Detection';
}

// Update statistics for detection
function updateStatsForDetection(type) {
    // Statistics are now handled by the detection classes display
    // This function is kept for compatibility but doesn't update UI
}

// Show detection overlay
function showDetectionOverlay(detection) {
    const overlay = document.getElementById('detectionOverlay');
    const bbox = detection.bbox;
    
    const bboxElement = document.createElement('div');
    bboxElement.className = `person-bbox ${detection.type}`;
    bboxElement.id = `bbox_${detection.id}`;
    bboxElement.style.left = (bbox.x * 100) + '%';
    bboxElement.style.top = (bbox.y * 100) + '%';
    bboxElement.style.width = (bbox.width * 100) + '%';
    bboxElement.style.height = (bbox.height * 100) + '%';
    
    const label = document.createElement('div');
    label.className = `person-label ${detection.type}`;
    label.textContent = getDetectionTitle(detection.type);
    bboxElement.appendChild(label);
    
    overlay.appendChild(bboxElement);
}

// Hide detection overlay
function hideDetectionOverlay(personId) {
    const bboxElement = document.getElementById(`bbox_${personId}`);
    if (bboxElement) {
        bboxElement.remove();
    }
    detectionSystem.activeDetections.delete(personId);
}

// Update live detections
function updateLiveDetections() {
    // Update existing detection positions (simulate movement)
    detectionSystem.activeDetections.forEach((detection, personId) => {
        const bboxElement = document.getElementById(`bbox_${personId}`);
        if (bboxElement) {
            // Slight movement simulation
            const currentBbox = detection.bbox;
            currentBbox.x += (Math.random() - 0.5) * 0.01;
            currentBbox.y += (Math.random() - 0.5) * 0.01;
            
            // Keep within bounds
            currentBbox.x = Math.max(0, Math.min(0.8, currentBbox.x));
            currentBbox.y = Math.max(0, Math.min(0.8, currentBbox.y));
            
            bboxElement.style.left = (currentBbox.x * 100) + '%';
            bboxElement.style.top = (currentBbox.y * 100) + '%';
        }
    });
}

// Update current time
function updateCurrentTime() {
    const now = new Date();
    const timeString = now.toLocaleTimeString('ko-KR', { 
        hour: '2-digit', 
        minute: '2-digit' 
    });
    document.getElementById('currentTime').textContent = timeString;
}

// Emergency alert function
function triggerEmergencyAlert() {
    const eventData = {
        type: 'fall',
        title: 'EMERGENCY: Fall Detection',
        time: new Date(),
        location: 'CAM-001',
        camera: 'cam001'
    };
    
    addEventToList(eventData);
    updateStatsForDetection('fall');
    
    // Flash the emergency button
    const emergencyBtn = document.getElementById('emergencyBtn');
    emergencyBtn.style.animation = 'blink 0.5s infinite';
    
    setTimeout(() => {
        emergencyBtn.style.animation = '';
    }, 5000);
    
    // Show notification
    if (navigator.permissions && Notification.permission === 'granted') {
        new Notification('Emergency Alert', {
            body: 'Fall incident detected - Emergency response required!',
            icon: '/favicon.ico'
        });
    }
}

console.log('🚀 BIXPO 2024 - AI Safety Monitoring System initialized successfully!');