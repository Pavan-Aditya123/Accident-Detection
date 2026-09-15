/**
 * National Highways AI Smart Accident Detection Control Center Frontend Application
 */

let state = {
    cameras: [],
    selectedCamera: null,
    selectedFile: null,
    sampleFilename: null,
    isSampleVideo: false,
    alerts: [],
    activeView: 'highway',
    map: null,
    markers: {}
};

// Initialize application on DOM load
document.addEventListener('DOMContentLoaded', async () => {
    await fetchCameras();
    await fetchSampleVideos();
    initMap();
    renderCameraMatrix();
    await fetchAlerts();
    
    // Poll for alerts every 3 seconds to keep real-time sync
    setInterval(fetchAlerts, 3000);
});

// View Switcher between Highway Authority & Hospital Dashboards
function switchDashboardView(view) {
    state.activeView = view;
    const highwayView = document.getElementById('view-highway-dashboard');
    const hospitalView = document.getElementById('view-hospital-dashboard');
    const btnHighway = document.getElementById('nav-btn-highway');
    const btnHospital = document.getElementById('nav-btn-hospital');

    if (view === 'highway') {
        highwayView.classList.remove('hidden');
        hospitalView.classList.add('hidden');
        
        btnHighway.className = "px-4 py-2 rounded-lg text-xs font-semibold flex items-center gap-2 transition-all bg-blue-600 text-white shadow-md shadow-blue-600/30";
        btnHospital.className = "px-4 py-2 rounded-lg text-xs font-semibold flex items-center gap-2 transition-all text-slate-400 hover:text-slate-200 hover:bg-slate-800/60 relative";
        
        // Invalidate map size when switching back to highway view
        if (state.map) {
            setTimeout(() => state.map.invalidateSize(), 100);
        }
    } else {
        highwayView.classList.add('hidden');
        hospitalView.classList.remove('hidden');
        
        btnHospital.className = "px-4 py-2 rounded-lg text-xs font-semibold flex items-center gap-2 transition-all bg-rose-600 text-white shadow-md shadow-rose-600/30 relative";
        btnHighway.className = "px-4 py-2 rounded-lg text-xs font-semibold flex items-center gap-2 transition-all text-slate-400 hover:text-slate-200 hover:bg-slate-800/60";
        
        renderHospitalAlertsFeed();
    }
}

// Fetch Demo Cameras from API
async function fetchCameras() {
    try {
        const res = await fetch('/api/cameras');
        const data = await res.json();
        if (data.status === 'success') {
            state.cameras = data.cameras;
            if (state.cameras.length > 0 && !state.selectedCamera) {
                // Default select C03 as per prompt requirements
                state.selectedCamera = state.cameras.find(c => c.id === 'C03') || state.cameras[0];
            }
            updateCameraTelemetryStats();
        }
    } catch (err) {
        console.error('Error fetching cameras:', err);
    }
}

// Fetch Sample Test Videos from API
async function fetchSampleVideos() {
    try {
        const res = await fetch('/api/sample-videos');
        const data = await res.json();
        const container = document.getElementById('sample-video-buttons');
        if (!container) return;

        if (data.status === 'success' && data.samples.length > 0) {
            container.innerHTML = data.samples.map(s => {
                let badge = "";
                if (s.filename === '1.mp4') badge = " (Test Video 1)";
                if (s.filename === '5.mp4') badge = " (Test Video 5)";
                return `
                    <button onclick="selectSampleVideo('${s.filename}')" 
                        class="px-2.5 py-1.5 rounded-lg border border-slate-700 bg-slate-900/80 hover:border-blue-500 hover:bg-blue-600/10 text-xs text-slate-300 font-mono text-left transition-colors truncate flex items-center justify-between">
                        <span>📹 ${s.filename}</span>
                        <span class="text-[10px] text-blue-400">${badge}</span>
                    </button>
                `;
            }).join('');
        }
    } catch (err) {
        console.error('Error fetching sample videos:', err);
    }
}

// Initialize Leaflet Map
function initMap() {
    const mapElement = document.getElementById('highway-map');
    if (!mapElement) return;

    // Center map around Northern/Central India highway corridor
    state.map = L.map('highway-map', {
        center: [28.5, 77.2],
        zoom: 7,
        zoomControl: true
    });

    // Standard OpenStreetMap tile layer (no API key required)
    L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png', {
        attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
        maxZoom: 19
    }).addTo(state.map);

    // Add Camera Markers to Map
    state.cameras.forEach(cam => {
        const markerClass = cam.status.toLowerCase();
        const customIcon = L.divIcon({
            className: 'custom-camera-marker',
            html: `<div class="marker-pin ${markerClass}"></div><div style="margin-top: 36px;" class="bg-slate-900/90 text-[10px] font-bold text-slate-200 px-1.5 py-0.5 rounded border border-slate-700 shadow font-mono">${cam.id}</div>`,
            iconSize: [30, 42],
            iconAnchor: [15, 21]
        });

        const marker = L.marker([cam.lat, cam.lng], { icon: customIcon }).addTo(state.map);
        marker.bindPopup(`
            <div class="p-2 text-slate-900">
                <div class="font-bold text-sm">${cam.name} (${cam.highway})</div>
                <div class="text-xs text-slate-600">${cam.location}</div>
                <div class="mt-2 text-xs font-semibold text-emerald-600">Status: ${cam.status}</div>
            </div>
        `);

        marker.on('click', () => {
            selectCamera(cam.id);
        });

        state.markers[cam.id] = marker;
    });

    // Sync drawer to default selected camera
    if (state.selectedCamera) {
        selectCamera(state.selectedCamera.id);
    }
}

// Update Camera Drawer Details
function selectCamera(cameraId) {
    const cam = state.cameras.find(c => c.id === cameraId);
    if (!cam) return;
    
    state.selectedCamera = cam;
    
    // Update drawer UI elements
    document.getElementById('panel-camera-id').innerText = cam.name.toUpperCase();
    document.getElementById('panel-camera-highway').innerText = cam.highway;
    document.getElementById('panel-camera-location').innerText = cam.location;
    document.getElementById('panel-camera-coords').innerText = `${cam.lat.toFixed(4)}° N, ${cam.lng.toFixed(4)}° E`;

    // Status badge
    const badge = document.getElementById('panel-camera-status-badge');
    if (cam.status === 'ACCIDENT') {
        badge.className = "px-2.5 py-1 rounded-full text-xs font-bold bg-rose-500/10 text-rose-400 border border-rose-500/30 flex items-center gap-1.5";
        badge.innerHTML = `<span class="w-2 h-2 rounded-full bg-rose-500 animate-ping"></span> ACCIDENT`;
    } else if (cam.status === 'PROCESSING') {
        badge.className = "px-2.5 py-1 rounded-full text-xs font-bold bg-amber-500/10 text-amber-400 border border-amber-500/30 flex items-center gap-1.5";
        badge.innerHTML = `<span class="w-2 h-2 rounded-full bg-amber-400 animate-spin"></span> PROCESSING`;
    } else {
        badge.className = "px-2.5 py-1 rounded-full text-xs font-bold bg-emerald-500/10 text-emerald-400 border border-emerald-500/30 flex items-center gap-1.5";
        badge.innerHTML = `<span class="w-2 h-2 rounded-full bg-emerald-400"></span> ONLINE`;
    }

    // Pan map to camera
    if (state.map) {
        state.map.flyTo([cam.lat, cam.lng], 9, { duration: 1.2 });
    }

    renderCameraMatrix();
}

// Render Camera Matrix Grid Cards
function renderCameraMatrix() {
    const container = document.getElementById('camera-matrix-grid');
    if (!container) return;

    container.innerHTML = state.cameras.map(cam => {
        const isSelected = state.selectedCamera && state.selectedCamera.id === cam.id;
        let border = isSelected ? "border-blue-500 bg-blue-950/20" : "border-slate-800 bg-slate-900/60 hover:border-slate-700";
        let statusColor = "bg-emerald-400";
        if (cam.status === 'ACCIDENT') statusColor = "bg-rose-500 animate-ping";
        if (cam.status === 'PROCESSING') statusColor = "bg-amber-400 animate-spin";
        if (cam.status === 'OFFLINE') statusColor = "bg-slate-500";

        return `
            <div onclick="selectCamera('${cam.id}')" class="p-3 rounded-xl border ${border} cursor-pointer transition-all flex flex-col justify-between space-y-2 group">
                <div class="flex items-center justify-between">
                    <span class="text-xs font-bold text-slate-200 group-hover:text-blue-400">${cam.id}</span>
                    <span class="w-2 h-2 rounded-full ${statusColor}"></span>
                </div>
                <div class="text-[11px] text-slate-400 font-medium truncate">${cam.highway}</div>
                <div class="text-[10px] text-slate-500 truncate">${cam.location}</div>
            </div>
        `;
    }).join('');
}

// File Upload Handler
function handleFileSelect(event) {
    const file = event.target.files[0];
    if (!file) return;

    state.selectedFile = file;
    state.sampleFilename = null;
    state.isSampleVideo = false;

    document.getElementById('upload-filename-text').innerText = `Selected: ${file.name}`;
    document.getElementById('selected-video-name').innerText = file.name;

    const previewPlayer = document.getElementById('cctv-preview-player');
    previewPlayer.src = URL.createObjectURL(file);
    document.getElementById('video-preview-wrapper').classList.remove('hidden');

    enableAnalyzeButton();
}

// Sample Video Select Handler
function selectSampleVideo(filename) {
    state.selectedFile = null;
    state.sampleFilename = filename;
    state.isSampleVideo = true;

    document.getElementById('upload-filename-text').innerText = `Sample: ${filename}`;
    document.getElementById('selected-video-name').innerText = filename;

    const previewPlayer = document.getElementById('cctv-preview-player');
    previewPlayer.src = `/samples/${filename}`;
    document.getElementById('video-preview-wrapper').classList.remove('hidden');

    enableAnalyzeButton();
}

function enableAnalyzeButton() {
    const btn = document.getElementById('btn-analyze');
    btn.disabled = false;
    btn.className = "w-full py-3 px-4 rounded-xl text-xs font-bold uppercase tracking-wider flex items-center justify-center gap-2 transition-all bg-blue-600 hover:bg-blue-500 text-white cursor-pointer shadow-lg shadow-blue-600/30 border border-blue-500";
}

// Trigger AI Model Inference
async function triggerAccidentAnalysis() {
    if (!state.selectedCamera) return;
    if (!state.selectedFile && !state.sampleFilename) return;

    const btn = document.getElementById('btn-analyze');
    btn.disabled = true;
    btn.innerHTML = `<i data-lucide="loader" class="w-4 h-4 animate-spin"></i> RUNNING AI INFERENCE...`;

    // Set camera status to PROCESSING
    state.selectedCamera.status = 'PROCESSING';
    selectCamera(state.selectedCamera.id);

    // Show HUD Progress Card
    const hudCard = document.getElementById('analysis-hud-card');
    hudCard.classList.remove('hidden');
    document.getElementById('ai-result-container').innerHTML = '';

    // Simulate step-by-step processing HUD updates
    updateHudStep('hud-step-night', 'Processing...', 'text-amber-400');
    await new Promise(r => setTimeout(r, 600));
    updateHudStep('hud-step-night', '✓', 'text-emerald-400');

    updateHudStep('hud-step-accident', 'Processing...', 'text-amber-400');
    await new Promise(r => setTimeout(r, 700));
    updateHudStep('hud-step-accident', '✓', 'text-emerald-400');

    updateHudStep('hud-step-temporal', 'Verifying (5-frame)...', 'text-amber-400');

    // Build Form Data for API call
    const formData = new FormData();
    formData.append('camera_id', state.selectedCamera.id);
    formData.append('location', state.selectedCamera.location);

    if (state.selectedFile) {
        formData.append('file', state.selectedFile);
    } else if (state.sampleFilename) {
        formData.append('sample_filename', state.sampleFilename);
    }

    try {
        const res = await fetch('/api/analyze', {
            method: 'POST',
            body: formData
        });

        const data = await res.json();
        hudCard.classList.add('hidden');

        if (data.status === 'success' && data.result) {
            renderAiResult(data.result);
        } else {
            alert('Analysis failed: ' + (data.detail || 'Unknown error'));
        }

    } catch (err) {
        console.error('Inference API Error:', err);
        hudCard.classList.add('hidden');
        alert('AI Inference pipeline encountered an error: ' + err.message);
    } finally {
        btn.disabled = false;
        btn.innerHTML = `<i data-lucide="play-circle" class="w-4 h-4"></i> ANALYZE FOR ACCIDENT`;
        lucide.createIcons();
    }
}

function updateHudStep(elementId, statusText, colorClass) {
    const el = document.getElementById(elementId);
    if (!el) return;
    const statusSpan = el.querySelector('.hud-status');
    statusSpan.className = `hud-status font-bold ${colorClass}`;
    statusSpan.innerText = statusText;
}

// Render AI Inference Results Card
function renderAiResult(result) {
    const container = document.getElementById('ai-result-container');
    const isAccident = result.accident_verified;

    // Update Camera Status on Map
    if (state.selectedCamera) {
        state.selectedCamera.status = isAccident ? 'ACCIDENT' : 'ONLINE';
        selectCamera(state.selectedCamera.id);
    }

    if (!isAccident) {
        // NO ACCIDENT DETECTED RESULT CARD
        container.innerHTML = `
            <div class="bg-emerald-950/30 border border-emerald-500/40 rounded-2xl p-5 shadow-2xl space-y-4">
                <div class="flex items-center space-x-3 text-emerald-400 border-b border-emerald-500/20 pb-3">
                    <div class="w-10 h-10 rounded-xl bg-emerald-500/20 flex items-center justify-center border border-emerald-500/40">
                        <i data-lucide="check-circle-2" class="w-6 h-6"></i>
                    </div>
                    <div>
                        <h4 class="text-base font-black tracking-wide">✓ NO ACCIDENT DETECTED</h4>
                        <p class="text-xs text-emerald-300/80">CCTV video analysis completed without incidents.</p>
                    </div>
                </div>

                <div class="grid grid-cols-2 gap-2 text-xs bg-slate-900/80 p-3 rounded-xl border border-slate-800">
                    <div><span class="text-slate-500">Camera ID:</span> <span class="font-bold text-slate-200">${result.camera_id}</span></div>
                    <div><span class="text-slate-500">Highway:</span> <span class="font-bold text-slate-200">${state.selectedCamera.highway}</span></div>
                    <div><span class="text-slate-500">Location:</span> <span class="font-medium text-slate-300 truncate">${result.location}</span></div>
                    <div><span class="text-slate-500">Decision:</span> <span class="font-bold text-emerald-400">NORMAL TRAFFIC</span></div>
                </div>

                <div class="text-xs space-y-1.5">
                    <span class="text-slate-400 font-semibold">Processed Output Video Stream:</span>
                    <div class="rounded-xl overflow-hidden border border-slate-800 bg-black aspect-video">
                        <video src="${result.output_video}" controls autoplay muted class="w-full h-full object-contain"></video>
                    </div>
                </div>
            </div>
        `;
    } else {
        // ACCIDENT CONFIRMED RESULT CARD
        const sevColor = result.severity === 'HIGH' ? 'text-rose-500 bg-rose-500/10 border-rose-500/40' : 'text-amber-400 bg-amber-500/10 border-amber-500/40';
        
        container.innerHTML = `
            <div class="bg-rose-950/40 border-2 border-rose-600/80 rounded-2xl p-5 shadow-2xl space-y-4 animate-pulse">
                <div class="flex items-center justify-between border-b border-rose-600/40 pb-3">
                    <div class="flex items-center space-x-3">
                        <div class="w-10 h-10 rounded-xl bg-rose-600/30 flex items-center justify-center text-rose-500 border border-rose-500/50 shadow-lg shadow-rose-600/30">
                            <i data-lucide="alert-triangle" class="w-6 h-6"></i>
                        </div>
                        <div>
                            <h4 class="text-lg font-black tracking-tight text-rose-400 flex items-center gap-2">
                                🔴 ACCIDENT CONFIRMED
                            </h4>
                            <p class="text-xs text-slate-300">Verified by 5-Frame Temporal Model & Severity Classifier</p>
                        </div>
                    </div>
                    <span class="px-3 py-1 rounded-full text-xs font-black uppercase border ${sevColor}">
                        SEVERITY: ${result.severity}
                    </span>
                </div>

                <!-- Incident Telemetry Details -->
                <div class="grid grid-cols-2 gap-3 text-xs bg-slate-900/90 p-3 rounded-xl border border-slate-800">
                    <div><span class="text-slate-400 font-medium">Camera:</span> <span class="font-bold text-slate-100">${result.camera_id}</span></div>
                    <div><span class="text-slate-400 font-medium">Accident Type:</span> <span class="font-bold text-rose-400 font-mono">${result.accident_type}</span></div>
                    <div><span class="text-slate-400 font-medium">Location:</span> <span class="font-medium text-slate-200 truncate">${result.location}</span></div>
                    <div><span class="text-slate-400 font-medium">Confidence:</span> <span class="font-bold text-emerald-400">${(result.average_confidence * 100).toFixed(1)}%</span></div>
                    <div><span class="text-slate-400 font-medium">First Detection:</span> <span class="font-mono text-slate-300">Frame ${result.first_detection}</span></div>
                    <div><span class="text-slate-400 font-medium">Confirmation:</span> <span class="font-mono text-slate-300">Frame ${result.confirmation_frame}</span></div>
                </div>

                <!-- Processed AI Video Player -->
                <div class="space-y-1.5" id="accident-video-section">
                    <span class="text-xs text-slate-300 font-bold uppercase tracking-wider block">Annotated Accident Result Video:</span>
                    <div class="rounded-xl overflow-hidden border border-rose-600/40 bg-black aspect-video shadow-2xl">
                        <video src="${result.output_video}" controls autoplay muted class="w-full h-full object-contain"></video>
                    </div>
                </div>

                <!-- Action Buttons -->
                <div class="grid grid-cols-3 gap-2 pt-2">
                    <button onclick="scrollToVideo()" class="py-2 px-2 rounded-lg bg-slate-800 hover:bg-slate-700 text-xs font-bold text-slate-200 border border-slate-700 flex items-center justify-center gap-1">
                        <i data-lucide="play" class="w-3.5 h-3.5"></i> VIDEO
                    </button>
                    <button onclick="focusCameraMap('${result.camera_id}')" class="py-2 px-2 rounded-lg bg-slate-800 hover:bg-slate-700 text-xs font-bold text-slate-200 border border-slate-700 flex items-center justify-center gap-1">
                        <i data-lucide="map-pin" class="w-3.5 h-3.5"></i> MAP
                    </button>
                    <button onclick="sendEmergencyAlertFromAi('${encodeURIComponent(JSON.stringify(result))}')" class="py-2 px-2 rounded-lg bg-rose-600 hover:bg-rose-500 text-xs font-bold text-white shadow-lg shadow-rose-600/30 flex items-center justify-center gap-1">
                        <i data-lucide="send" class="w-3.5 h-3.5"></i> ALERT
                    </button>
                </div>
            </div>
        `;
    }

    updateCameraTelemetryStats();
    lucide.createIcons();
}

function scrollToVideo() {
    const el = document.getElementById('accident-video-section');
    if (el) el.scrollIntoView({ behavior: 'smooth' });
}

function focusCameraMap(cameraId) {
    selectCamera(cameraId);
    const mapEl = document.getElementById('highway-map');
    if (mapEl) mapEl.scrollIntoView({ behavior: 'smooth' });
}

// Send Emergency Alert to Hospital Dashboard
async function sendEmergencyAlertFromAi(resultJsonEncoded) {
    const result = JSON.parse(decodeURIComponent(resultJsonEncoded));
    
    const alertData = {
        from: "National Highways Authority",
        alert: "Road Accident Detected",
        camera_id: result.camera_id,
        location: result.location,
        accident_type: result.accident_type,
        severity: result.severity,
        confidence: result.average_confidence,
        video_url: result.output_video,
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
    };

    try {
        const res = await fetch('/api/alerts', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(alertData)
        });

        const data = await res.json();
        if (data.status === 'success') {
            await fetchAlerts();
            alert(`🚨 EMERGENCY ALERT GENERATED & DISPATCHED!\n\nSent to:\n• Nearby General Hospital\n• Highway Police Station\n\nAlert ID: ${data.alert.id}`);
            switchDashboardView('hospital');
        }
    } catch (err) {
        console.error('Error sending alert:', err);
    }
}

// Fetch Alerts from Server
async function fetchAlerts() {
    try {
        const res = await fetch('/api/alerts');
        const data = await res.json();
        if (data.status === 'success') {
            state.alerts = data.alerts;
            updateAlertTelemetryStats();
            if (state.activeView === 'hospital') {
                renderHospitalAlertsFeed();
            }
        }
    } catch (err) {
        console.error('Error fetching alerts:', err);
    }
}

// Update Global Telemetry Counters
function updateCameraTelemetryStats() {
    document.getElementById('stat-cameras-count').innerText = state.cameras.length;
    const active = state.cameras.filter(c => c.status === 'ONLINE' || c.status === 'ACCIDENT').length;
    document.getElementById('stat-active-count').innerText = active;
}

function updateAlertTelemetryStats() {
    const totalAlerts = state.alerts.length;
    document.getElementById('stat-alerts-count').innerText = totalAlerts;
    document.getElementById('stat-incidents-count').innerText = totalAlerts;

    const badge = document.getElementById('hospital-badge-count');
    if (badge) {
        if (totalAlerts > 0) {
            badge.innerText = totalAlerts;
            badge.classList.remove('hidden');
        } else {
            badge.classList.add('hidden');
        }
    }
}

// Render Hospital Emergency Dashboard Alerts Feed
function renderHospitalAlertsFeed() {
    const feedContainer = document.getElementById('hospital-alerts-feed');
    const feedCount = document.getElementById('hospital-alert-feed-count');
    if (!feedContainer) return;

    if (feedCount) feedCount.innerText = `${state.alerts.length} Alerts Received`;

    if (state.alerts.length === 0) {
        feedContainer.innerHTML = `
            <div class="bg-[#111827] border border-slate-800 rounded-2xl p-12 text-center flex flex-col items-center justify-center space-y-3">
                <div class="w-14 h-14 rounded-full bg-slate-900 border border-slate-800 flex items-center justify-center text-slate-600">
                    <i data-lucide="bell-off" class="w-7 h-7"></i>
                </div>
                <h4 class="text-base font-bold text-slate-300">NO ACCIDENT ALERTS RECEIVED YET</h4>
                <p class="text-xs text-slate-500 max-w-md">When an accident is confirmed on the Highway Authority Dashboard, an emergency dispatch alert will appear here in real-time.</p>
            </div>
        `;
        lucide.createIcons();
        return;
    }

    feedContainer.innerHTML = state.alerts.map(alt => {
        let statusBadge = "";
        if (alt.hospital_status === 'ACKNOWLEDGED') {
            statusBadge = `<span class="px-3 py-1 rounded-full text-xs font-bold bg-emerald-500/20 text-emerald-400 border border-emerald-500/40 flex items-center gap-1.5"><i data-lucide="check" class="w-3.5 h-3.5"></i> ALERT ACKNOWLEDGED</span>`;
        } else if (alt.hospital_status === 'DISPATCHED') {
            statusBadge = `<span class="px-3 py-1 rounded-full text-xs font-bold bg-cyan-500/20 text-cyan-300 border border-cyan-500/40 flex items-center gap-1.5"><i data-lucide="ambulance" class="w-3.5 h-3.5"></i> EMERGENCY TEAM DISPATCHED</span>`;
        } else {
            statusBadge = `<span class="px-3 py-1 rounded-full text-xs font-bold bg-rose-500/20 text-rose-400 border border-rose-500/40 flex items-center gap-1.5 animate-pulse"><i data-lucide="bell-ring" class="w-3.5 h-3.5"></i> NEW ALERT RECEIVED</span>`;
        }

        const sevBadgeClass = alt.severity === 'HIGH' ? 'bg-rose-500/20 text-rose-400 border-rose-500/40' : 'bg-amber-500/20 text-amber-400 border-amber-500/40';

        return `
            <div class="bg-[#111827] border-2 border-rose-900/60 hover:border-rose-600/80 rounded-2xl p-5 shadow-2xl space-y-4 transition-colors">
                
                <div class="flex flex-col md:flex-row md:items-center justify-between gap-3 border-b border-slate-800 pb-3">
                    <div class="flex items-center space-x-3">
                        <div class="w-10 h-10 rounded-xl bg-rose-600/20 border border-rose-500/40 flex items-center justify-center text-rose-500">
                            <i data-lucide="siren" class="w-6 h-6"></i>
                        </div>
                        <div>
                            <div class="flex items-center space-x-2">
                                <span class="text-xs font-bold text-rose-400 uppercase">🔴 NEW ACCIDENT ALERT</span>
                                <span class="text-[10px] font-mono text-slate-500">ID: ${alt.id}</span>
                            </div>
                            <p class="text-xs text-slate-300">From: <strong class="text-slate-100">${alt.from}</strong> &bull; Received: <strong class="text-slate-200">${alt.timestamp}</strong></p>
                        </div>
                    </div>
                    <div>${statusBadge}</div>
                </div>

                <!-- Alert Information Grid -->
                <div class="grid grid-cols-2 md:grid-cols-4 gap-3 text-xs bg-slate-900/90 p-3 rounded-xl border border-slate-800">
                    <div><span class="text-slate-500 block">Camera ID</span> <span class="font-bold text-slate-200 text-sm">${alt.camera_id}</span></div>
                    <div><span class="text-slate-500 block">Highway Location</span> <span class="font-medium text-slate-300 truncate">${alt.location}</span></div>
                    <div><span class="text-slate-500 block">Accident Type</span> <span class="font-mono text-rose-400 font-bold">${alt.accident_type}</span></div>
                    <div><span class="text-slate-500 block">Severity Level</span> <span class="font-bold px-2 py-0.5 rounded text-[11px] inline-block border ${sevBadgeClass}">${alt.severity}</span></div>
                </div>

                <!-- Processed Accident Video Player -->
                <div class="space-y-1.5">
                    <span class="text-xs font-semibold text-slate-300 flex items-center gap-1.5">
                        <i data-lucide="video" class="w-3.5 h-3.5 text-blue-400"></i> ACCIDENT FOOTAGE EVIDENCE
                    </span>
                    <div class="rounded-xl overflow-hidden border border-slate-800 bg-black aspect-video max-h-[320px] flex items-center justify-center">
                        <video src="${alt.video_url}" controls autoplay muted playsinline class="w-full h-full object-contain"
                            onerror="this.onerror=null; this.parentNode.innerHTML='<div class=\'p-6 text-center text-xs text-rose-400 font-medium flex items-center justify-center gap-2\'><i data-lucide=\'alert-triangle\' class=\'w-4 h-4\'></i> Unable to load accident footage video.</div>'; lucide.createIcons();"></video>
                    </div>
                </div>

                <!-- Action Controls -->
                <div class="flex flex-wrap items-center justify-between gap-3 pt-2 border-t border-slate-800">
                    <div class="text-xs text-slate-400 flex items-center gap-2">
                        <i data-lucide="users" class="w-3.5 h-3.5 text-slate-500"></i>
                        <span>Notified: <strong>General Hospital Trauma Unit</strong>, <strong>Highway Patrol</strong></span>
                    </div>

                    <div class="flex items-center space-x-2">
                        <button onclick="updateAlertStatus('${alt.id}', 'ACKNOWLEDGED')" 
                            ${alt.hospital_status !== 'PENDING' ? 'disabled' : ''}
                            class="px-4 py-2 rounded-xl text-xs font-bold transition-all flex items-center gap-1.5 ${alt.hospital_status !== 'PENDING' ? 'bg-slate-800 text-slate-500 cursor-not-allowed border border-slate-700' : 'bg-emerald-600 hover:bg-emerald-500 text-white shadow-lg shadow-emerald-600/20 border border-emerald-500'}">
                            <i data-lucide="check-circle-2" class="w-4 h-4"></i>
                            ACKNOWLEDGE ALERT
                        </button>
                        <button onclick="updateAlertStatus('${alt.id}', 'DISPATCHED')" 
                            ${alt.hospital_status === 'DISPATCHED' ? 'disabled' : ''}
                            class="px-4 py-2 rounded-xl text-xs font-bold transition-all flex items-center gap-1.5 ${alt.hospital_status === 'DISPATCHED' ? 'bg-slate-800 text-slate-500 cursor-not-allowed border border-slate-700' : 'bg-rose-600 hover:bg-rose-500 text-white shadow-lg shadow-rose-600/30 border border-rose-500'}">
                            <i data-lucide="ambulance" class="w-4 h-4"></i>
                            DISPATCH EMERGENCY TEAM
                        </button>
                    </div>
                </div>

            </div>
        `;
    }).join('');

    lucide.createIcons();
}

// Update Alert Status in Backend
async function updateAlertStatus(alertId, newStatus) {
    try {
        const res = await fetch(`/api/alerts/${alertId}`, {
            method: 'PATCH',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ status: newStatus })
        });
        const data = await res.json();
        if (data.status === 'success') {
            await fetchAlerts();
        }
    } catch (err) {
        console.error('Error updating alert status:', err);
    }
}
