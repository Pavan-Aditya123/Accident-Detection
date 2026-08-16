'use strict';

const API_BASE = 'http://127.0.0.1:8000';

// ── DOM refs ──────────────────────────────────────────────────────────────────
const dropZone      = document.getElementById('drop-zone');
const fileInput     = document.getElementById('file-input');
const fileInfo      = document.getElementById('file-info');
const fileNameLabel = document.getElementById('file-name-label');
const btnClear      = document.getElementById('btn-clear');
const btnUpload     = document.getElementById('btn-upload');

const statusCard    = document.getElementById('status-card');
const statusDot     = document.getElementById('status-dot');
const statusText    = document.getElementById('status-text');
const progressWrap  = document.getElementById('progress-wrap');

const resultsCard   = document.getElementById('results-card');
const accidentBanner= document.getElementById('accident-banner');
const bannerIcon    = document.getElementById('banner-icon');
const bannerText    = document.getElementById('banner-text');

const severityPanel = document.getElementById('severity-panel');
const sevClass      = document.getElementById('sev-class');
const sevImpact     = document.getElementById('sev-impact');
const sevLevel      = document.getElementById('sev-level');
const sevPriority   = document.getElementById('sev-priority');
const sevReason     = document.getElementById('sev-reason');

const statFrames    = document.getElementById('stat-frames');
const statPossible  = document.getElementById('stat-possible');
const statConfirmed = document.getElementById('stat-confirmed');
const statFps       = document.getElementById('stat-fps');

const resInput      = document.getElementById('res-input');
const resOutput     = document.getElementById('res-output');
const videoSection  = document.getElementById('video-section');
const outputVideo   = document.getElementById('output-video');
const btnReset      = document.getElementById('btn-reset');

const errorCard     = document.getElementById('error-card');
const errorMessage  = document.getElementById('error-message');
const btnErrorReset = document.getElementById('btn-error-reset');

// ── State ─────────────────────────────────────────────────────────────────────
let selectedFile = null;

// ── File selection ────────────────────────────────────────────────────────────
fileInput.addEventListener('change', () => handleFileSelect(fileInput.files[0]));

dropZone.addEventListener('dragover', (e) => {
  e.preventDefault();
  dropZone.classList.add('drag-over');
});
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag-over'));
dropZone.addEventListener('drop', (e) => {
  e.preventDefault();
  dropZone.classList.remove('drag-over');
  const f = e.dataTransfer.files[0];
  if (f) handleFileSelect(f);
});

function handleFileSelect(file) {
  if (!file) return;
  const allowed = ['.mp4', '.mov', '.avi'];
  const ext = '.' + file.name.split('.').pop().toLowerCase();
  if (!allowed.includes(ext)) {
    showError(`Invalid file type "${ext}". Please upload a .mp4, .mov, or .avi video.`);
    return;
  }
  selectedFile = file;
  fileNameLabel.textContent = file.name;
  fileInfo.classList.remove('hidden');
  btnUpload.disabled = false;
}

btnClear.addEventListener('click', clearFile);

function clearFile() {
  selectedFile = null;
  fileInput.value = '';
  fileNameLabel.textContent = '';
  fileInfo.classList.add('hidden');
  btnUpload.disabled = true;
}

// ── Upload & inference ────────────────────────────────────────────────────────
btnUpload.addEventListener('click', () => {
  if (!selectedFile) {
    showError('No file selected. Please choose a video file first.');
    return;
  }
  uploadAndProcess(selectedFile);
});

async function uploadAndProcess(file) {
  setStatus('processing', 'Uploading and processing… this may take several minutes.');
  statusCard.classList.remove('hidden');
  resultsCard.classList.add('hidden');
  errorCard.classList.add('hidden');
  progressWrap.classList.remove('hidden');
  btnUpload.disabled = true;

  const formData = new FormData();
  formData.append('file', file);

  try {
    const response = await fetch(`${API_BASE}/upload`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      let detail = `Server error (HTTP ${response.status})`;
      try {
        const errBody = await response.json();
        if (errBody.detail) detail = errBody.detail;
      } catch (_) { /* ignore */ }
      throw new Error(detail);
    }

    const data = await response.json();
    showResults(data);

  } catch (err) {
    if (err instanceof TypeError && err.message.toLowerCase().includes('fetch')) {
      showError(`Cannot reach the backend at ${API_BASE}. Make sure it is running.`);
    } else {
      showError(err.message || 'An unexpected error occurred.');
    }
  }
}

// ── Results ───────────────────────────────────────────────────────────────────
function showResults(data) {
  setStatus('completed', 'Processing complete.');
  progressWrap.classList.add('hidden');

  // ── Accident banner ───────────────────────────────────────────────────────
  const detected = data.accident_detected === true;
  accidentBanner.className = 'accident-banner ' + (detected ? 'detected' : 'none');
  bannerIcon.textContent   = detected ? '🚨' : '✅';
  bannerText.textContent   = detected
    ? `Accident detected — ${data.confirmed_accidents} confirmed event(s)`
    : 'No accident confirmed in this video';

  // ── Severity panel ────────────────────────────────────────────────────────
  const sev = data.severity;
  if (detected && sev) {
    sevClass.textContent = sev.accident_class || '—';
    sevImpact.textContent = sev.impact_level  || '—';

    sevLevel.textContent = sev.severity || '—';
    sevLevel.dataset.sev = sev.severity || '';

    sevPriority.textContent = sev.priority || '—';
    sevPriority.dataset.pri = sev.priority || '';

    sevReason.textContent = sev.reason || '';
    severityPanel.classList.remove('hidden');
  } else {
    severityPanel.classList.add('hidden');
  }

  // ── Stats ─────────────────────────────────────────────────────────────────
  statFrames.textContent    = data.frames_processed   ?? '—';
  statPossible.textContent  = data.possible_accidents ?? '—';
  statConfirmed.textContent = data.confirmed_accidents ?? '—';
  statFps.textContent       = data.avg_fps != null ? data.avg_fps.toFixed(1) : '—';

  // ── File info ─────────────────────────────────────────────────────────────
  resInput.textContent  = data.input_video  || '—';
  resOutput.textContent = data.output_video || '—';

  // ── Video player ──────────────────────────────────────────────────────────
  // The backend exposes GET /video/{filename} to stream the output file.
  if (data.output_filename) {
    outputVideo.src = `${API_BASE}/video/${encodeURIComponent(data.output_filename)}`;
    outputVideo.load();
    videoSection.classList.remove('hidden');
  } else {
    outputVideo.src = '';
    videoSection.classList.add('hidden');
  }

  resultsCard.classList.remove('hidden');
}

// ── Status helper ─────────────────────────────────────────────────────────────
function setStatus(state, message) {
  statusDot.className    = 'status-dot ' + state;
  statusText.textContent = message;
}

// ── Error display ─────────────────────────────────────────────────────────────
function showError(message) {
  setStatus('error', 'An error occurred.');
  progressWrap.classList.add('hidden');
  statusCard.classList.remove('hidden');
  errorMessage.textContent = message;
  errorCard.classList.remove('hidden');
  resultsCard.classList.add('hidden');
  btnUpload.disabled = (selectedFile === null);
}

// ── Reset ─────────────────────────────────────────────────────────────────────
function resetUI() {
  clearFile();
  statusCard.classList.add('hidden');
  resultsCard.classList.add('hidden');
  errorCard.classList.add('hidden');
  progressWrap.classList.add('hidden');
  severityPanel.classList.add('hidden');
  outputVideo.src = '';
  setStatus('waiting', 'Waiting…');
}

btnReset.addEventListener('click', resetUI);
btnErrorReset.addEventListener('click', resetUI);

// ── Backend health check on load ──────────────────────────────────────────────
(async () => {
  try {
    const res = await fetch(`${API_BASE}/health`, {
      signal: AbortSignal.timeout(3000),
    });
    if (!res.ok) throw new Error('unhealthy');
  } catch (_) {
    console.warn('[dashboard] Backend not reachable at', API_BASE);
    const sub = document.querySelector('#upload-card .card-sub');
    if (sub) {
      sub.innerHTML += '&nbsp;<span style="color:#e84040;font-weight:600">⚠ Backend offline</span>';
    }
  }
})();
