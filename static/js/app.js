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
    map: null,                 // Highway CCTV Monitoring Map
    markers: {},               // Highway Camera Markers
    hospitalMap: null,         // Dedicated Hospital Emergency Route Map
    hospitalMapMarkers: [],   // Hospital Map Markers
    hospitalRoutePolyline: null,// OSRM Route Polyline
    activeIncidents: {},       // Map of alert_id -> accidentResult object for active incident cards
    pendingAiResult: null      // Transient AI result before SEND ALERT is clicked
};

// Initialize application on DOM load
document.addEventListener('DOMContentLoaded', async () => {
    await fetchCameras();
    await fetchSampleVideos();
    initMap();
    renderCameraMatrix();
    await fetchAlerts();
    
    // Poll for alerts every 1.5 seconds to keep real-time sync across dashboards
    setInterval(fetchAlerts, 1500);
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
        
        if (state.map) {
            setTimeout(() => state.map.invalidateSize(), 100);
        }
        renderHighwayAlertsList();
        renderActiveIncidentCards();
    } else {
        highwayView.classList.add('hidden');
        hospitalView.classList.remove('hidden');
        
        btnHospital.className = "px-4 py-2 rounded-lg text-xs font-semibold flex items-center gap-2 transition-all bg-rose-600 text-white shadow-md shadow-rose-600/30 relative";
        btnHighway.className = "px-4 py-2 rounded-lg text-xs font-semibold flex items-center gap-2 transition-all text-slate-400 hover:text-slate-200 hover:bg-slate-800/60";
        
        initHospitalMap();
        if (state.hospitalMap) {
            setTimeout(() => state.hospitalMap.invalidateSize(), 100);
        }
        renderHospitalAlertsFeed();
    }
}

// Reset Entire Emergency Workflow via Backend Endpoint
async function resetDemoWorkflow() {
    if (!confirm("Are you sure you want to reset the demo? This will clear all active emergency alerts and reset dashboard states.")) {
        return;
    }

    try {
        const res = await fetch('/api/reset', { method: 'POST' });
        const data = await res.json();
        
        if (data.status === 'success') {
            // Remove Hospital Map layers
            if (state.hospitalMap) {
                if (state.hospitalMapMarkers) {
                    state.hospitalMapMarkers.forEach(m => state.hospitalMap.removeLayer(m));
                }
                if (state.hospitalRoutePolyline) {
                    state.hospitalMap.removeLayer(state.hospitalRoutePolyline);
                }
            }
            state.hospitalMapMarkers = [];
            state.hospitalRoutePolyline = null;

            const mapContainer = document.getElementById('hospital-map-container');
            if (mapContainer) mapContainer.classList.add('hidden');

            // Reset camera statuses
            state.cameras.forEach(c => c.status = 'ONLINE');
            
            // Clear local states
            state.alerts = [];
            state.activeIncidents = {};
            state.pendingAiResult = null;
            state.selectedFile = null;
            state.sampleFilename = null;

            // Clear preview players & result containers
            const previewPlayer = document.getElementById('cctv-preview-player');
            if (previewPlayer) previewPlayer.src = '';
            document.getElementById('video-preview-wrapper').classList.add('hidden');
            document.getElementById('ai-result-container').innerHTML = '';
            document.getElementById('upload-filename-text').innerText = 'Click or Drag CCTV Video (.mp4 / .avi)';

            const btnAnalyze = document.getElementById('btn-analyze');
            btnAnalyze.disabled = true;
            btnAnalyze.className = "w-full py-3 px-4 rounded-xl text-xs font-bold uppercase tracking-wider flex items-center justify-center gap-2 transition-all bg-slate-800 text-slate-500 cursor-not-allowed border border-slate-700";

            // Refresh UI
            await fetchCameras();
            selectCamera('C03');
            updateAlertTelemetryStats();
            renderHighwayAlertsList();
            renderHospitalAlertsFeed();

            alert('✓ Demo environment successfully reset to initial state.');
        }
    } catch (err) {
        console.error('Error resetting demo:', err);
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

// Initialize Highway Authority CCTV Map
function initMap() {
    const mapElement = document.getElementById('highway-map');
    if (!mapElement) return;

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

    // Add Camera Markers to Highway Map
    updateHighwayMapMarkers();

    if (state.selectedCamera) {
        selectCamera(state.selectedCamera.id);
    }
}

// Update Highway Map Camera Markers
function updateHighwayMapMarkers() {
    if (!state.map) return;

    state.cameras.forEach(cam => {
        const markerClass = cam.status.toLowerCase();
        const customIcon = L.divIcon({
            className: 'custom-camera-marker',
            html: `<div class="marker-pin ${markerClass}"></div><div style="margin-top: 36px;" class="bg-slate-900/90 text-[10px] font-bold text-slate-200 px-1.5 py-0.5 rounded border border-slate-700 shadow font-mono">${cam.id}</div>`,
            iconSize: [30, 42],
            iconAnchor: [15, 21]
        });

        if (state.markers[cam.id]) {
            state.markers[cam.id].setIcon(customIcon);
        } else {
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
        }
    });
}

// Initialize Hospital Emergency Route Map
function initHospitalMap() {
    const mapElement = document.getElementById('hospital-emergency-map');
    if (!mapElement || state.hospitalMap) return;

    state.hospitalMap = L.map('hospital-emergency-map', {
        center: [28.5, 77.2],
        zoom: 9,
        zoomControl: true
    });

    L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png', {
        attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
        maxZoom: 19
    }).addTo(state.hospitalMap);
}

// Fetch OSRM Road-Following Route
async function fetchAndRenderOsrmRoute(h_lat, h_lng, c_lat, c_lng) {
    const url = `https://router.project-osrm.org/route/v1/driving/${h_lng},${h_lat};${c_lng},${c_lat}?overview=full&geometries=geojson`;
    try {
        const res = await fetch(url);
        const data = await res.json();
        if (data.code === 'Ok' && data.routes && data.routes.length > 0) {
            const route = data.routes[0];
            const coords = route.geometry.coordinates.map(pt => [pt[1], pt[0]]);
            const distKm = (route.distance / 1000).toFixed(1);
            const durationMin = Math.ceil(route.duration / 60);

            return { coords, distKm, durationMin };
        }
    } catch (err) {
        console.warn('OSRM routing request failed, falling back to direct line:', err);
    }

    // Fallback if OSRM endpoint fails
    const distKm = calculateHaversineDistance(h_lat, h_lng, c_lat, c_lng).toFixed(1);
    const durationMin = Math.ceil(parseFloat(distKm) * 2);
    return {
        coords: [[h_lat, h_lng], [c_lat, c_lng]],
        distKm,
        durationMin
    };
}

// Calculate Haversine Distance
function calculateHaversineDistance(lat1, lon1, lat2, lon2) {
    const R = 6371;
    const dLat = (lat2 - lat1) * Math.PI / 180;
    const dLon = (lon2 - lon1) * Math.PI / 180;
    const a = Math.sin(dLat/2) * Math.sin(dLat/2) +
              Math.cos(lat1 * Math.PI / 180) * Math.cos(lat2 * Math.PI / 180) *
              Math.sin(dLon/2) * Math.sin(dLon/2);
    const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1-a));
    return R * c;
}

// Update Hospital Emergency Map for Active Alerts
async function updateHospitalMapForAlerts() {
    const card = document.getElementById('hospital-map-container');
    if (!card) return;

    if (state.alerts.length === 0) {
        card.classList.add('hidden');
        return;
    }

    card.classList.remove('hidden');
    initHospitalMap();

    if (state.hospitalMap) {
        setTimeout(() => state.hospitalMap.invalidateSize(), 50);
    }

    // Clear previous hospital map markers & polylines
    if (state.hospitalMapMarkers) {
        state.hospitalMapMarkers.forEach(m => {
            if (state.hospitalMap) state.hospitalMap.removeLayer(m);
        });
    }
    state.hospitalMapMarkers = [];

    if (state.hospitalRoutePolyline && state.hospitalMap) {
        state.hospitalMap.removeLayer(state.hospitalRoutePolyline);
        state.hospitalRoutePolyline = null;
    }

    const latestAlert = state.alerts[0]; // Most recent alert
    const cam = state.cameras.find(c => c.id === latestAlert.camera_id) || { lat: 29.6857, lng: 76.9905 };
    const h = latestAlert.hospital;

    if (!h || !h.latitude || !h.longitude) return;

    const c_lat = cam.lat;
    const c_lng = cam.lng;
    const h_lat = h.latitude;
    const h_lng = h.longitude;

    // 1. Hospital Marker 🏥
    const hIcon = L.divIcon({
        className: 'custom-camera-marker',
        html: `<div class="marker-pin hospital"></div><div style="margin-top: 36px;" class="bg-rose-950/90 text-[10px] font-bold text-rose-200 px-2 py-0.5 rounded border border-rose-600 shadow font-mono truncate max-w-[140px]">🏥 ${h.name}</div>`,
        iconSize: [30, 42],
        iconAnchor: [15, 21]
    });
    const hMarker = L.marker([h_lat, h_lng], { icon: hIcon }).addTo(state.hospitalMap);
    hMarker.bindPopup(`
        <div class="p-2 text-slate-900">
            <div class="font-bold text-xs uppercase text-rose-600">Dispatched Hospital</div>
            <div class="font-bold text-sm text-slate-900">${h.name}</div>
            <div class="text-xs text-slate-600 mt-1">Distance to Incident: <strong>${h.distance_km} km</strong></div>
        </div>
    `);
    state.hospitalMapMarkers.push(hMarker);

    // 2. Accident Location Marker 🚨
    const accIcon = L.divIcon({
        className: 'custom-camera-marker',
        html: `<div class="marker-pin accident"></div><div style="margin-top: 36px;" class="bg-rose-600 text-[10px] font-bold text-white px-2 py-0.5 rounded border border-rose-400 shadow font-mono truncate">🚨 ACCIDENT (${latestAlert.camera_id})</div>`,
        iconSize: [30, 42],
        iconAnchor: [15, 21]
    });
    const accMarker = L.marker([c_lat, c_lng], { icon: accIcon }).addTo(state.hospitalMap);
    accMarker.bindPopup(`
        <div class="p-2 text-slate-900">
            <div class="font-bold text-xs uppercase text-rose-600">Accident Location</div>
            <div class="font-bold text-sm text-slate-900">${latestAlert.location}</div>
            <div class="text-xs text-rose-600 font-bold">${latestAlert.accident_type} (${latestAlert.severity})</div>
        </div>
    `);
    state.hospitalMapMarkers.push(accMarker);

    // 3. Stationary Ambulance Marker 🚑 (Only when dispatched)
    if (latestAlert.dispatch_status === 'dispatched') {
        const ambIcon = L.divIcon({
            className: 'custom-camera-marker',
            html: `<div class="marker-pin ambulance"></div><div style="margin-top: 36px;" class="bg-cyan-950/90 text-[10px] font-bold text-cyan-300 px-1.5 py-0.5 rounded border border-cyan-500 shadow font-mono">🚑 AMBULANCE</div>`,
            iconSize: [30, 42],
            iconAnchor: [15, 21]
        });
        const ambMarker = L.marker([h_lat + 0.0002, h_lng + 0.0002], { icon: ambIcon }).addTo(state.hospitalMap);
        ambMarker.bindPopup(`
            <div class="p-2 text-slate-900">
                <div class="font-bold text-xs uppercase text-cyan-600">Dispatched Ambulance</div>
                <div class="font-bold text-sm text-slate-900">Location: ${h.name} Base</div>
                <div class="text-xs text-slate-600">Status: Stationary (Live GPS tracking not active)</div>
            </div>
        `);
        state.hospitalMapMarkers.push(ambMarker);
    }

    // 4. Fetch and draw OSRM Road Route
    const routeInfo = await fetchAndRenderOsrmRoute(h_lat, h_lng, c_lat, c_lng);

    state.hospitalRoutePolyline = L.polyline(routeInfo.coords, {
        color: '#06b6d4',
        weight: 5,
        opacity: 0.9
    }).addTo(state.hospitalMap);

    state.hospitalMap.fitBounds(state.hospitalRoutePolyline.getBounds().pad(0.3));

    const distEl = document.getElementById('hospital-route-distance');
    const etaEl = document.getElementById('hospital-route-eta');
    if (distEl) distEl.innerText = `Road Distance: ${routeInfo.distKm} km`;
    if (etaEl) etaEl.innerText = `Est. Travel Time: ${routeInfo.durationMin} min`;

    latestAlert.osrm_distance_km = routeInfo.distKm;
    latestAlert.osrm_duration_min = routeInfo.durationMin;
}

// Update Camera Drawer Details
function selectCamera(cameraId) {
    const cam = state.cameras.find(c => c.id === cameraId);
    if (!cam) return;
    
    state.selectedCamera = cam;
    
    document.getElementById('panel-camera-id').innerText = cam.name.toUpperCase();
    document.getElementById('panel-camera-highway').innerText = cam.highway;
    document.getElementById('panel-camera-location').innerText = cam.location;
    document.getElementById('panel-camera-coords').innerText = `${cam.lat.toFixed(4)}° N, ${cam.lng.toFixed(4)}° E`;

    const badge = document.getElementById('panel-camera-status-badge');
    if (cam.status === 'ACCIDENT') {
        badge.className = "px-2.5 py-1 rounded-full text-xs font-bold bg-rose-500/10 text-rose-400 border border-rose-500/30 flex items-center gap-1.5";
        badge.innerHTML = `<span class="w-2 h-2 rounded-full bg-rose-500"></span> ACCIDENT`;
    } else if (cam.status === 'PROCESSING') {
        badge.className = "px-2.5 py-1 rounded-full text-xs font-bold bg-amber-500/10 text-amber-400 border border-amber-500/30 flex items-center gap-1.5";
        badge.innerHTML = `<span class="w-2 h-2 rounded-full bg-amber-400 animate-spin"></span> PROCESSING`;
    } else {
        badge.className = "px-2.5 py-1 rounded-full text-xs font-bold bg-emerald-500/10 text-emerald-400 border border-emerald-500/30 flex items-center gap-1.5";
        badge.innerHTML = `<span class="w-2 h-2 rounded-full bg-emerald-400"></span> ONLINE`;
    }

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
        if (cam.status === 'ACCIDENT') statusColor = "bg-rose-500";
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
    btn.innerHTML = `<i data-lucide="loader" class="w-4 h-4 animate-spin"></i> ANALYZING FOOTAGE...`;

    state.selectedCamera.status = 'PROCESSING';
    selectCamera(state.selectedCamera.id);

    const hudCard = document.getElementById('analysis-hud-card');
    hudCard.classList.remove('hidden');
    document.getElementById('ai-result-container').innerHTML = '';

    updateHudStep('hud-step-night', 'Processing...', 'text-amber-400');
    await new Promise(r => setTimeout(r, 600));
    updateHudStep('hud-step-night', '✓', 'text-emerald-400');

    updateHudStep('hud-step-accident', 'Processing...', 'text-amber-400');
    await new Promise(r => setTimeout(r, 700));
    updateHudStep('hud-step-accident', '✓', 'text-emerald-400');

    updateHudStep('hud-step-temporal', 'Verification in progress...', 'text-amber-400');

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
        alert('Video analysis pipeline encountered an error: ' + err.message);
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
    const isAccident = result.accident_verified;

    if (state.selectedCamera) {
        state.selectedCamera.status = isAccident ? 'ACCIDENT' : 'ONLINE';
        selectCamera(state.selectedCamera.id);
        updateHighwayMapMarkers();
    }

    if (!isAccident) {
        state.pendingAiResult = null;
        const container = document.getElementById('ai-result-container');
        if (container) {
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
        }
    } else {
        // Store transient result until SEND ALERT is clicked
        state.pendingAiResult = result;
        renderActiveIncidentCards();
    }

    updateCameraTelemetryStats();
    lucide.createIcons();
}

// Helper for logging video source & state telemetry
function logVideoState(videoEl, label) {
    if (!videoEl) return;
    console.log(`[VIDEO TELEMETRY] ${label} -> SOURCE: currentSrc="${videoEl.currentSrc}", src="${videoEl.src}"`);
    console.log(`[VIDEO TELEMETRY] ${label} -> STATE: currentTime=${videoEl.currentTime}, duration=${videoEl.duration}, paused=${videoEl.paused}, readyState=${videoEl.readyState}`);
}

// Render Active Incident Cards linked with alert_id (DOM-preserving to prevent video reset during polling)
function renderActiveIncidentCards() {
    const container = document.getElementById('ai-result-container');
    if (!container) return;

    // Active unacknowledged alert IDs from state.alerts
    const activeAlertIds = new Set(state.alerts.filter(a => !a.acknowledged).map(a => a.id || a.alert_id));

    // Purge any activeIncident whose alert_id has been acknowledged!
    Object.keys(state.activeIncidents).forEach(alertId => {
        if (!activeAlertIds.has(alertId)) {
            delete state.activeIncidents[alertId];
        }
    });

    const incidentKeys = Object.keys(state.activeIncidents);

    // Remove any DOM card for alerts that are no longer active/unacknowledged
    const existingCards = container.querySelectorAll('[data-alert-card="true"]');
    existingCards.forEach(cardEl => {
        const cardAlertId = cardEl.getAttribute('data-alert-id');
        if (cardAlertId && !state.activeIncidents[cardAlertId]) {
            cardEl.remove();
        }
    });

    // If pending result is gone, remove pending card
    const pendingCardEl = document.getElementById('pending-ai-result-card');
    if (!state.pendingAiResult && pendingCardEl) {
        pendingCardEl.remove();
    }

    // If no active cards and no pending result, check camera status and clean up if needed
    if (incidentKeys.length === 0 && !state.pendingAiResult) {
        if (state.selectedCamera && state.selectedCamera.status === 'ACCIDENT') {
            const camHasUnackAlert = state.alerts.some(a => !a.acknowledged && a.camera_id === state.selectedCamera.id);
            if (!camHasUnackAlert) {
                state.selectedCamera.status = 'ONLINE';
                selectCamera(state.selectedCamera.id);
                updateHighwayMapMarkers();
            }
        }
        if (!container.querySelector('#no-accident-card') && container.children.length === 0) {
            container.innerHTML = '';
        }
        return;
    }

    // If there is a pending AI result, render or keep existing pending card
    if (state.pendingAiResult) {
        if (!document.getElementById('pending-ai-result-card')) {
            const result = state.pendingAiResult;
            const sevColor = result.severity === 'HIGH' ? 'text-rose-500 bg-rose-500/10 border-rose-500/40' : 'text-amber-400 bg-amber-500/10 border-amber-500/40';
            const hospitalData = result.hospital;
            let hospitalText = (hospitalData && hospitalData.name) ? `${hospitalData.name} (${hospitalData.distance_km} km — ${hospitalData.source})` : "Nearest hospital location unavailable";

            const div = document.createElement('div');
            div.id = 'pending-ai-result-card';
            div.className = 'bg-rose-950/40 border-2 border-rose-600/80 rounded-2xl p-5 shadow-2xl space-y-4 mb-4';
            div.innerHTML = `
                <div class="flex items-center justify-between border-b border-rose-600/40 pb-3">
                    <div class="flex items-center space-x-3">
                        <div class="w-10 h-10 rounded-xl bg-rose-600/30 flex items-center justify-center text-rose-500 border border-rose-500/50 shadow-lg shadow-rose-600/30">
                            <i data-lucide="alert-triangle" class="w-6 h-6"></i>
                        </div>
                        <div>
                            <h4 class="text-lg font-black tracking-tight text-rose-400 flex items-center gap-2">
                                🔴 ACCIDENT CONFIRMED
                            </h4>
                            <p class="text-xs text-slate-300">Verified by Intelligent Incident Assessment</p>
                        </div>
                    </div>
                    <span class="px-3 py-1 rounded-full text-xs font-black uppercase border ${sevColor}">
                        SEVERITY: ${result.severity}
                    </span>
                </div>

                <div class="grid grid-cols-2 gap-3 text-xs bg-slate-900/90 p-3 rounded-xl border border-slate-800">
                    <div><span class="text-slate-400 font-medium">Camera:</span> <span class="font-bold text-slate-100">${result.camera_id}</span></div>
                    <div><span class="text-slate-400 font-medium">Accident Type:</span> <span class="font-bold text-rose-400 font-mono">${result.accident_type}</span></div>
                    <div class="col-span-2"><span class="text-slate-400 font-medium">Location:</span> <span class="font-medium text-slate-200 truncate">${result.location}</span></div>
                    <div class="col-span-2"><span class="text-slate-400 font-medium">Nearest Hospital:</span> <span class="font-bold text-rose-400">${hospitalText}</span></div>
                    <div><span class="text-slate-400 font-medium">Confidence:</span> <span class="font-bold text-emerald-400">${(result.average_confidence * 100).toFixed(1)}%</span></div>
                    <div><span class="text-slate-400 font-medium">Verification Status:</span> <span class="font-semibold text-emerald-400">Confirmed</span></div>
                </div>

                <div class="space-y-1.5" id="accident-video-section">
                    <span class="text-xs text-slate-300 font-bold uppercase tracking-wider block">Accident Analysis Video Stream:</span>
                    <div class="rounded-xl overflow-hidden border border-rose-600/40 bg-black aspect-video shadow-2xl">
                        <video src="${result.output_video}" controls autoplay muted class="w-full h-full object-contain"></video>
                    </div>
                </div>

                <div class="grid grid-cols-3 gap-2 pt-2">
                    <button onclick="scrollToVideo('')" class="py-2 px-2 rounded-lg bg-slate-800 hover:bg-slate-700 text-xs font-bold text-slate-200 border border-slate-700 flex items-center justify-center gap-1">
                        <i data-lucide="play" class="w-3.5 h-3.5"></i> VIDEO
                    </button>
                    <button onclick="focusCameraMap('${result.camera_id}')" class="py-2 px-2 rounded-lg bg-slate-800 hover:bg-slate-700 text-xs font-bold text-slate-200 border border-slate-700 flex items-center justify-center gap-1">
                        <i data-lucide="map-pin" class="w-3.5 h-3.5"></i> MAP
                    </button>
                    <button onclick="sendEmergencyAlertFromAi('${encodeURIComponent(JSON.stringify(result))}')" class="py-2 px-2 rounded-lg bg-rose-600 hover:bg-rose-500 text-xs font-bold text-white shadow-lg shadow-rose-600/30 flex items-center justify-center gap-1">
                        <i data-lucide="send" class="w-3.5 h-3.5"></i> SEND ALERT
                    </button>
                </div>
            `;
            container.prepend(div);
            lucide.createIcons();
            const vid = div.querySelector('video');
            if (vid) {
                logVideoState(vid, "Highway Pending Card Created");
                vid.addEventListener('play', () => logVideoState(vid, "Highway Pending Card Play"));
            }
        }
    }

    // Render active incident cards bound to alert_id (only create if not already existing)
    incidentKeys.forEach(alertId => {
        const cardId = `ai-result-card-${alertId}`;
        if (!document.getElementById(cardId)) {
            const result = state.activeIncidents[alertId];
            const sevColor = result.severity === 'HIGH' ? 'text-rose-500 bg-rose-500/10 border-rose-500/40' : 'text-amber-400 bg-amber-500/10 border-amber-500/40';
            const hospitalData = result.hospital;
            let hospitalText = (hospitalData && hospitalData.name) ? `${hospitalData.name} (${hospitalData.distance_km} km — ${hospitalData.source})` : "Nearest hospital location unavailable";

            const div = document.createElement('div');
            div.id = cardId;
            div.setAttribute('data-alert-card', 'true');
            div.setAttribute('data-alert-id', alertId);
            div.className = 'bg-rose-950/40 border-2 border-rose-600/80 rounded-2xl p-5 shadow-2xl space-y-4 mb-4';
            div.innerHTML = `
                <div class="flex items-center justify-between border-b border-rose-600/40 pb-3">
                    <div class="flex items-center space-x-3">
                        <div class="w-10 h-10 rounded-xl bg-rose-600/30 flex items-center justify-center text-rose-500 border border-rose-500/50 shadow-lg shadow-rose-600/30">
                            <i data-lucide="alert-triangle" class="w-6 h-6"></i>
                        </div>
                        <div>
                            <h4 class="text-lg font-black tracking-tight text-rose-400 flex items-center gap-2">
                                🔴 ACCIDENT CONFIRMED
                            </h4>
                            <p class="text-xs text-slate-300">Verified by Intelligent Incident Assessment</p>
                        </div>
                    </div>
                    <span class="px-3 py-1 rounded-full text-xs font-black uppercase border ${sevColor}">
                        SEVERITY: ${result.severity}
                    </span>
                </div>

                <div class="grid grid-cols-2 gap-3 text-xs bg-slate-900/90 p-3 rounded-xl border border-slate-800">
                    <div><span class="text-slate-400 font-medium">Alert ID:</span> <span class="font-bold text-rose-400 font-mono">${alertId}</span></div>
                    <div><span class="text-slate-400 font-medium">Camera:</span> <span class="font-bold text-slate-100">${result.camera_id}</span></div>
                    <div><span class="text-slate-400 font-medium">Accident Type:</span> <span class="font-bold text-rose-400 font-mono">${result.accident_type}</span></div>
                    <div><span class="text-slate-400 font-medium">Confidence:</span> <span class="font-bold text-emerald-400">${(result.average_confidence * 100).toFixed(1)}%</span></div>
                    <div class="col-span-2"><span class="text-slate-400 font-medium">Location:</span> <span class="font-medium text-slate-200 truncate">${result.location}</span></div>
                    <div class="col-span-2"><span class="text-slate-400 font-medium">Nearest Hospital:</span> <span class="font-bold text-rose-400">${hospitalText}</span></div>
                </div>

                <div class="space-y-1.5" id="accident-video-section-${alertId}">
                    <span class="text-xs text-slate-300 font-bold uppercase tracking-wider block">Accident Analysis Video Stream:</span>
                    <div class="rounded-xl overflow-hidden border border-rose-600/40 bg-black aspect-video shadow-2xl">
                        <video src="${result.output_video}" controls autoplay muted class="w-full h-full object-contain"></video>
                    </div>
                </div>

                <div class="grid grid-cols-2 gap-2 pt-2">
                    <button onclick="scrollToVideo('${alertId}')" class="py-2 px-2 rounded-lg bg-slate-800 hover:bg-slate-700 text-xs font-bold text-slate-200 border border-slate-700 flex items-center justify-center gap-1">
                        <i data-lucide="play" class="w-3.5 h-3.5"></i> VIDEO
                    </button>
                    <button onclick="focusCameraMap('${result.camera_id}')" class="py-2 px-2 rounded-lg bg-slate-800 hover:bg-slate-700 text-xs font-bold text-slate-200 border border-slate-700 flex items-center justify-center gap-1">
                        <i data-lucide="map-pin" class="w-3.5 h-3.5"></i> MAP
                    </button>
                </div>
            `;
            container.appendChild(div);
            lucide.createIcons();
            const vid = div.querySelector('video');
            if (vid) {
                logVideoState(vid, `Highway Incident Card Created (${alertId})`);
                vid.addEventListener('play', () => logVideoState(vid, `Highway Incident Card Play (${alertId})`));
            }
        }
    });
}

function scrollToVideo(alertId) {
    const targetId = alertId ? `accident-video-section-${alertId}` : 'accident-video-section';
    const el = document.getElementById(targetId);
    if (el) el.scrollIntoView({ behavior: 'smooth' });
}

function focusCameraMap(cameraId) {
    selectCamera(cameraId);
    const mapEl = document.getElementById('highway-map');
    if (mapEl) mapEl.scrollIntoView({ behavior: 'smooth' });
}

// Send Emergency Alert to Backend & Hospital Dashboard
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
        hospital: result.hospital,
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
    };

    try {
        const res = await fetch('/api/alerts', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(alertData)
        });

        const data = await res.json();
        if (data.status === 'success' || data.success === true) {
            const alertId = data.alert.id || data.alert.alert_id;
            result.alert_id = alertId;
            state.activeIncidents[alertId] = result;
            state.pendingAiResult = null;
            
            await fetchAlerts();
            renderActiveIncidentCards();

            alert(`🚨 EMERGENCY ALERT GENERATED!\n\nDestination: ${result.hospital && result.hospital.name ? result.hospital.name : 'Nearest Trauma Unit'}\n\nAlert ID: ${alertId}`);
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
        if (data.status === 'success' || data.success === true) {
            state.alerts = data.alerts;
            updateAlertTelemetryStats();
            renderHighwayAlertsList();
            renderActiveIncidentCards(); // Purge acknowledged accident result cards automatically
            if (state.activeView === 'hospital') {
                renderHospitalAlertsFeed();
            }
        }
    } catch (err) {
        console.error('Error fetching alerts:', err);
    }
}

// Render Highway Authority Active Alerts Telemetry List
// Requirements: Acknowledged alerts MUST be automatically filtered out/removed from Highway Authority active alerts list
function renderHighwayAlertsList() {
    const container = document.getElementById('highway-alerts-list');
    const badgeCount = document.getElementById('stat-alerts-count');
    const highwayBadge = document.getElementById('highway-alerts-count-badge');
    if (!container) return;

    // Filter out acknowledged alerts so they leave the active Highway list
    const activeAlerts = state.alerts.filter(alt => !alt.acknowledged);

    if (badgeCount) badgeCount.innerText = activeAlerts.length;
    if (highwayBadge) highwayBadge.innerText = `${activeAlerts.length} Active Alert${activeAlerts.length !== 1 ? 's' : ''}`;

    if (activeAlerts.length === 0) {
        container.innerHTML = `
            <div class="text-xs text-slate-500 p-3 bg-slate-900/40 rounded-xl text-center border border-slate-800">
                NO ACTIVE ACCIDENT ALERTS
            </div>
        `;
        return;
    }

    container.innerHTML = activeAlerts.map(alt => {
        const hName = alt.hospital && alt.hospital.name ? alt.hospital.name : 'Nearest Hospital';
        
        let statusBadge = `
            <div class="p-2 rounded-lg bg-amber-950/40 border border-amber-500/40 text-amber-400 font-bold text-xs flex items-center justify-between">
                <span class="flex items-center gap-1.5"><i data-lucide="clock" class="w-4 h-4 text-amber-400"></i> AWAITING HOSPITAL ACKNOWLEDGEMENT</span>
                <span class="text-[10px] font-mono text-slate-400">Sent: ${alt.timestamp}</span>
            </div>
        `;

        return `
            <div class="p-3 rounded-xl border border-slate-800 bg-slate-900/60 space-y-2">
                <div class="flex items-center justify-between text-xs">
                    <span class="font-bold text-slate-200">Alert ID: ${alt.id} (${alt.camera_id})</span>
                    <span class="text-rose-400 font-mono font-bold">${alt.accident_type}</span>
                </div>
                <div class="text-xs text-slate-400 flex items-center justify-between">
                    <span>Location: ${alt.location}</span>
                    <span class="text-slate-300">Hospital: <strong class="text-rose-400">${hName}</strong></span>
                </div>
                ${statusBadge}
            </div>
        `;
    }).join('');

    lucide.createIcons();
}

// Update Global Telemetry Counters
function updateCameraTelemetryStats() {
    document.getElementById('stat-cameras-count').innerText = state.cameras.length;
    const active = state.cameras.filter(c => c.status === 'ONLINE' || c.status === 'ACCIDENT').length;
    document.getElementById('stat-active-count').innerText = active;
}

function updateAlertTelemetryStats() {
    const activeCount = state.alerts.filter(a => !a.acknowledged).length;
    document.getElementById('stat-alerts-count').innerText = activeCount;
    document.getElementById('stat-incidents-count').innerText = state.alerts.length;

    const badge = document.getElementById('hospital-badge-count');
    if (badge) {
        const unackCount = state.alerts.filter(a => !a.acknowledged).length;
        if (unackCount > 0) {
            badge.innerText = unackCount;
            badge.classList.remove('hidden');
        } else {
            badge.classList.add('hidden');
        }
    }
}

/// Render Hospital Emergency Dashboard Alerts Feed (DOM-preserving to prevent video resets during polling)
function renderHospitalAlertsFeed() {
    const feedContainer = document.getElementById('hospital-alerts-feed');
    const feedCount = document.getElementById('hospital-alert-feed-count');
    if (!feedContainer) return;

    if (feedCount) feedCount.innerText = `${state.alerts.length} Alerts Received`;

    if (state.alerts.length === 0) {
        feedContainer.innerHTML = `
            <div id="hospital-empty-state" class="bg-[#111827] border border-slate-800 rounded-2xl p-12 text-center flex flex-col items-center justify-center space-y-3">
                <div class="w-14 h-14 rounded-full bg-slate-900 border border-slate-800 flex items-center justify-center text-slate-600">
                    <i data-lucide="bell-off" class="w-7 h-7"></i>
                </div>
                <h4 class="text-base font-bold text-slate-300">NO INCOMING HIGHWAY ACCIDENT ALERTS</h4>
                <p class="text-xs text-slate-500 max-w-md">When an accident is confirmed on the Highway Authority Dashboard, an emergency dispatch alert will appear here in real-time.</p>
            </div>
        `;
        const card = document.getElementById('hospital-map-container');
        if (card) card.classList.add('hidden');
        lucide.createIcons();
        return;
    }

    // If alerts exist, remove empty state if present
    const emptyState = document.getElementById('hospital-empty-state');
    if (emptyState) emptyState.remove();

    // Update Emergency Response Map on Hospital Dashboard
    updateHospitalMapForAlerts();

    const headerTitle = document.getElementById('hospital-header-title');
    if (headerTitle && state.alerts.length > 0) {
        const latestHospital = state.alerts[0].hospital;
        if (latestHospital && latestHospital.name) {
            headerTitle.innerText = `${latestHospital.name.toUpperCase()} — EMERGENCY DEPT`;
        }
    }

    const currentAlertIds = new Set(state.alerts.map(a => a.id || a.alert_id));

    // Remove cards for alerts that are no longer in state.alerts
    const existingCards = feedContainer.querySelectorAll('[data-hospital-card="true"]');
    existingCards.forEach(cardEl => {
        const cardAlertId = cardEl.getAttribute('data-alert-id');
        if (cardAlertId && !currentAlertIds.has(cardAlertId)) {
            cardEl.remove();
        }
    });

    state.alerts.forEach(alt => {
        const alertId = alt.id || alt.alert_id;
        const cardId = `hospital-alert-card-${alertId}`;
        let cardEl = document.getElementById(cardId);

        let statusBadge = "";
        if (alt.dispatch_status === 'dispatched') {
            statusBadge = `<span class="px-3 py-1 rounded-full text-xs font-bold bg-cyan-500/20 text-cyan-300 border border-cyan-500/40 flex items-center gap-1.5"><i data-lucide="ambulance" class="w-3.5 h-3.5"></i> EMERGENCY TEAM DISPATCHED</span>`;
        } else if (alt.acknowledged) {
            statusBadge = `<span class="px-3 py-1 rounded-full text-xs font-bold bg-emerald-500/20 text-emerald-400 border border-emerald-500/40 flex items-center gap-1.5"><i data-lucide="check-circle-2" class="w-3.5 h-3.5"></i> ALERT ACKNOWLEDGED</span>`;
        } else {
            statusBadge = `<span class="px-3 py-1 rounded-full text-xs font-bold bg-rose-500/20 text-rose-400 border border-rose-500/40 flex items-center gap-1.5"><i data-lucide="bell-ring" class="w-3.5 h-3.5"></i> NEW ALERT RECEIVED</span>`;
        }

        const sevBadgeClass = alt.severity === 'HIGH' ? 'bg-rose-500/20 text-rose-400 border-rose-500/40' : 'bg-amber-500/20 text-amber-400 border-amber-500/40';
        const hData = alt.hospital;
        const hName = (hData && hData.name) ? `${hData.name} (${hData.distance_km} km)` : "Nearest hospital location unavailable";

        if (cardEl) {
            // Card already exists in DOM — UPDATE IN PLACE WITHOUT TOUCHING VIDEO PLAYER!
            const badgeContainer = cardEl.querySelector(`#hospital-status-badge-container-${alertId}`);
            if (badgeContainer) badgeContainer.innerHTML = statusBadge;

            const dispatchBox = cardEl.querySelector(`#hospital-dispatch-info-${alertId}`);
            if (dispatchBox) {
                if (alt.dispatch_status === 'dispatched') {
                    dispatchBox.classList.remove('hidden');
                    const etaSpan = cardEl.querySelector(`#dispatch-eta-${alertId}`);
                    if (etaSpan) etaSpan.innerText = `Road Travel Time: ${alt.osrm_duration_min || alt.eta_minutes || 5} min`;
                    const distSpan = cardEl.querySelector(`#dispatch-dist-${alertId}`);
                    if (distSpan) distSpan.innerText = `${alt.osrm_distance_km || (hData && hData.distance_km) || '2.5'} km`;
                } else {
                    dispatchBox.classList.add('hidden');
                }
            }

            const btnAck = cardEl.querySelector(`#btn-ack-${alertId}`);
            if (btnAck) {
                btnAck.disabled = !!alt.acknowledged;
                btnAck.className = `px-4 py-2 rounded-xl text-xs font-bold transition-all flex items-center gap-1.5 ${alt.acknowledged ? 'bg-slate-800 text-slate-500 cursor-not-allowed border border-slate-700' : 'bg-emerald-600 hover:bg-emerald-500 text-white shadow-lg shadow-emerald-600/20 border border-emerald-500'}`;
                const span = btnAck.querySelector('span');
                if (span) span.innerText = alt.acknowledged ? '✓ ALERT ACKNOWLEDGED' : 'ACKNOWLEDGE ALERT';
            }

            const btnDispatch = cardEl.querySelector(`#btn-dispatch-${alertId}`);
            if (btnDispatch) {
                btnDispatch.disabled = (alt.dispatch_status === 'dispatched');
                btnDispatch.className = `px-4 py-2 rounded-xl text-xs font-bold transition-all flex items-center gap-1.5 ${alt.dispatch_status === 'dispatched' ? 'bg-slate-800 text-slate-500 cursor-not-allowed border border-slate-700' : 'bg-rose-600 hover:bg-rose-500 text-white shadow-lg shadow-rose-600/30 border border-rose-500'}`;
                const span = btnDispatch.querySelector('span');
                if (span) span.innerText = (alt.dispatch_status === 'dispatched') ? '🚑 EMERGENCY TEAM DISPATCHED' : 'DISPATCH EMERGENCY TEAM';
            }

            lucide.createIcons();
        } else {
            // Create brand new card node
            cardEl = document.createElement('div');
            cardEl.id = cardId;
            cardEl.setAttribute('data-hospital-card', 'true');
            cardEl.setAttribute('data-alert-id', alertId);
            cardEl.className = 'bg-[#111827] border-2 border-rose-900/60 hover:border-rose-600/80 rounded-2xl p-5 shadow-2xl space-y-4 transition-colors mb-4';

            cardEl.innerHTML = `
                <div class="flex flex-col md:flex-row md:items-center justify-between gap-3 border-b border-slate-800 pb-3">
                    <div class="flex items-center space-x-3">
                        <div class="w-10 h-10 rounded-xl bg-rose-600/20 border border-rose-500/40 flex items-center justify-center text-rose-500">
                            <i data-lucide="siren" class="w-6 h-6"></i>
                        </div>
                        <div>
                            <div class="flex items-center space-x-2">
                                <span class="text-xs font-bold text-rose-400 uppercase">🔴 NEW HIGHWAY ACCIDENT ALERT</span>
                                <span class="text-[10px] font-mono text-slate-500">ID: ${alertId}</span>
                            </div>
                            <p class="text-xs text-slate-300">From: <strong class="text-slate-100">${alt.from}</strong> &bull; Received: <strong class="text-slate-200">${alt.timestamp}</strong></p>
                        </div>
                    </div>
                    <div id="hospital-status-badge-container-${alertId}">${statusBadge}</div>
                </div>

                <!-- Alert Information Grid -->
                <div class="grid grid-cols-2 md:grid-cols-4 gap-3 text-xs bg-slate-900/90 p-3 rounded-xl border border-slate-800">
                    <div><span class="text-slate-500 block">Camera ID</span> <span class="font-bold text-slate-200 text-sm">${alt.camera_id}</span></div>
                    <div><span class="text-slate-500 block">Highway Location</span> <span class="font-medium text-slate-300 truncate">${alt.location}</span></div>
                    <div><span class="text-slate-500 block">Accident Type</span> <span class="font-mono text-rose-400 font-bold">${alt.accident_type}</span></div>
                    <div><span class="text-slate-500 block">Severity Level</span> <span class="font-bold px-2 py-0.5 rounded text-[11px] inline-block border ${sevBadgeClass}">${alt.severity}</span></div>
                    <div class="col-span-2 md:col-span-4 border-t border-slate-800/80 pt-2"><span class="text-slate-500">Nearest OpenStreetMap Hospital Destination:</span> <span class="font-bold text-rose-400">${hName}</span></div>
                </div>

                <!-- Processed Accident Video Player -->
                <div class="space-y-1.5">
                    <span class="text-xs font-semibold text-slate-300 flex items-center gap-1.5">
                        <i data-lucide="video" class="w-3.5 h-3.5 text-blue-400"></i> ACCIDENT FOOTAGE EVIDENCE
                    </span>
                    <div class="rounded-xl overflow-hidden border border-slate-800 bg-black aspect-video flex items-center justify-center">
                        <video src="${alt.video_url}" controls autoplay muted playsinline class="w-full h-full object-contain"
                            onerror="this.onerror=null; this.parentNode.innerHTML='<div class=\\'p-6 text-center text-xs text-rose-400 font-medium flex items-center justify-center gap-2\\'><i data-lucide=\\'alert-triangle\\' class=\\'w-4 h-4\\'></i> Unable to load accident footage video.</div>'; lucide.createIcons();"></video>
                    </div>
                </div>

                <!-- Dispatch Info Box -->
                <div id="hospital-dispatch-info-${alertId}" class="${alt.dispatch_status === 'dispatched' ? '' : 'hidden'}">
                    <div class="p-3 bg-cyan-950/40 border border-cyan-500/40 rounded-xl space-y-1 text-xs">
                        <div class="flex items-center justify-between font-bold text-cyan-300">
                            <span class="flex items-center gap-2"><i data-lucide="ambulance" class="w-4 h-4 text-cyan-400"></i> 🚑 EMERGENCY TEAM DISPATCHED</span>
                            <span id="dispatch-eta-${alertId}">Road Travel Time: ${alt.osrm_duration_min || alt.eta_minutes || 5} min</span>
                        </div>
                        <div class="text-slate-300 text-[11px] space-y-0.5">
                            <div><strong>AMBULANCE TRACKING:</strong> <span class="text-amber-400 font-mono">LIVE TRACKING NOT ACTIVE</span> — Current Location: Hospital</div>
                            <div>Hospital Base: <strong>${(hData && hData.name) ? hData.name : 'Hospital'}</strong> &bull; Road Distance: <strong id="dispatch-dist-${alertId}">${alt.osrm_distance_km || (hData && hData.distance_km) || '2.5'} km</strong></div>
                        </div>
                    </div>
                </div>

                <!-- Action Controls -->
                <div class="flex flex-wrap items-center justify-between gap-3 pt-2 border-t border-slate-800">
                    <div class="text-xs text-slate-400 flex items-center gap-2">
                        <i data-lucide="users" class="w-3.5 h-3.5 text-slate-500"></i>
                        <span>Recipients: <strong>${(hData && hData.name) ? hData.name : 'Emergency Hospital'}</strong>, <strong>Highway Patrol</strong></span>
                    </div>

                    <div class="flex items-center space-x-2">
                        <button id="btn-ack-${alertId}" onclick="updateAlertStatus('${alertId}', 'ACKNOWLEDGED')" 
                            ${alt.acknowledged ? 'disabled' : ''}
                            class="px-4 py-2 rounded-xl text-xs font-bold transition-all flex items-center gap-1.5 ${alt.acknowledged ? 'bg-slate-800 text-slate-500 cursor-not-allowed border border-slate-700' : 'bg-emerald-600 hover:bg-emerald-500 text-white shadow-lg shadow-emerald-600/20 border border-emerald-500'}">
                            <i data-lucide="check-circle-2" class="w-4 h-4"></i>
                            <span>${alt.acknowledged ? '✓ ALERT ACKNOWLEDGED' : 'ACKNOWLEDGE ALERT'}</span>
                        </button>
                        <button id="btn-dispatch-${alertId}" onclick="updateAlertStatus('${alertId}', 'DISPATCHED')" 
                            ${alt.dispatch_status === 'dispatched' ? 'disabled' : ''}
                            class="px-4 py-2 rounded-xl text-xs font-bold transition-all flex items-center gap-1.5 ${alt.dispatch_status === 'dispatched' ? 'bg-slate-800 text-slate-500 cursor-not-allowed border border-slate-700' : 'bg-rose-600 hover:bg-rose-500 text-white shadow-lg shadow-rose-600/30 border border-rose-500'}">
                            <i data-lucide="ambulance" class="w-4 h-4"></i>
                            <span>${alt.dispatch_status === 'dispatched' ? '🚑 EMERGENCY TEAM DISPATCHED' : 'DISPATCH EMERGENCY TEAM'}</span>
                        </button>
                    </div>
                </div>
            `;
            feedContainer.appendChild(cardEl);
            lucide.createIcons();
            const vid = cardEl.querySelector('video');
            if (vid) {
                logVideoState(vid, `Hospital Alert Card Created (${alertId})`);
                vid.addEventListener('play', () => logVideoState(vid, `Hospital Alert Card Play (${alertId})`));
            }
        }
    });
}

// Update Alert Status in Backend (Acknowledge / Dispatch)
async function updateAlertStatus(alertId, action) {
    let endpoint = `/api/alerts/${alertId}/acknowledge`;
    if (action === 'DISPATCHED') {
        endpoint = `/api/alerts/${alertId}/dispatch`;
    }

    try {
        const res = await fetch(endpoint, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' }
        });
        const data = await res.json();
        if (res.ok && (data.status === 'success' || data.success === true)) {
            // Immediately update backend alert object in local state
            if (data.alert) {
                const idx = state.alerts.findIndex(a => (a.id === alertId || a.alert_id === alertId));
                if (idx !== -1) {
                    state.alerts[idx] = data.alert;
                }
            }
            await fetchAlerts();
            renderHighwayAlertsList();
            renderActiveIncidentCards();
            renderHospitalAlertsFeed();
        } else {
            alert("Unable to acknowledge alert. Please retry.");
        }
    } catch (err) {
        console.error(`Error updating alert ${action}:`, err);
        alert("Unable to acknowledge alert. Please retry.");
    }
}
