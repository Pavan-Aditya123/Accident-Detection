"""
FastAPI Server for AI Smart Accident Detection System Frontend Demo.
Provides API endpoints for camera management, video analysis via Python AI pipeline,
dynamic OpenStreetMap hospital lookup, and two-way emergency alert dispatch & acknowledgement workflow.
"""

import sys
import os
import shutil
from pathlib import Path
from typing import Optional, List
import uuid
from datetime import datetime

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Query, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))

from inference_api import analyze_video, UPLOADS_DIR, INTEGRATION_RESULTS_DIR
from hospital_service import find_nearest_hospital

app = FastAPI(title="National Highways AI Smart Accident Detection System")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Demo CCTV Camera Locations
DEMO_CAMERAS = [
    {
        "id": "C01",
        "name": "Camera C01",
        "highway": "NH-44",
        "location": "NH-44 KM 142 — Murthal Toll Plaza",
        "lat": 28.9892,
        "lng": 77.0706,
        "status": "ONLINE"
    },
    {
        "id": "C02",
        "name": "Camera C02",
        "highway": "NH-44",
        "location": "NH-44 KM 188 — Panipat Flyover North",
        "lat": 29.3909,
        "lng": 76.9635,
        "status": "ONLINE"
    },
    {
        "id": "C03",
        "name": "Camera C03",
        "highway": "NH-44",
        "location": "NH-44 KM 230 — Karnal Highway Junction",
        "lat": 29.6857,
        "lng": 76.9905,
        "status": "ONLINE"
    },
    {
        "id": "C04",
        "name": "Camera C04",
        "highway": "NH-48",
        "location": "NH-48 KM 45 — Gurgaon Expressway Interchange",
        "lat": 28.4595,
        "lng": 77.0266,
        "status": "ONLINE"
    },
    {
        "id": "C05",
        "name": "Camera C05",
        "highway": "NH-16",
        "location": "NH-16 KM 88 — Vizag Port Connector",
        "lat": 17.6868,
        "lng": 83.2185,
        "status": "ONLINE"
    },
    {
        "id": "C06",
        "name": "Camera C06",
        "highway": "NH-65",
        "location": "NH-65 KM 110 — Solapur Highway Bypass",
        "lat": 17.6599,
        "lng": 75.9064,
        "status": "ONLINE"
    }
]

# Shared In-Memory Emergency Alerts Store
EMERGENCY_ALERTS: List[dict] = []

SAMPLE_VIDEOS_DIR = ROOT_DIR / "runs" / "detect" / "results" / "video_test"


@app.get("/api/cameras")
def get_cameras():
    """Returns list of CCTV demo cameras."""
    return {"status": "success", "cameras": DEMO_CAMERAS}


@app.get("/api/sample-videos")
def get_sample_videos():
    """Returns list of available sample CCTV test videos."""
    samples = []
    if SAMPLE_VIDEOS_DIR.exists():
        for f in SAMPLE_VIDEOS_DIR.glob("*.mp4"):
            samples.append({
                "filename": f.name,
                "url": f"/samples/{f.name}",
                "description": f"Sample Video {f.name}"
            })
    return {"status": "success", "samples": sorted(samples, key=lambda x: x["filename"])}


@app.post("/api/analyze")
async def analyze_cctv(
    file: Optional[UploadFile] = File(None),
    sample_filename: Optional[str] = Form(None),
    camera_id: str = Form("C03"),
    location: str = Form("NH-44 Demo Location")
):
    """
    Processes video using the existing Python AI pipeline.
    If an accident is confirmed, dynamically queries OpenStreetMap for the nearest hospital.
    """
    if file and file.filename:
        req_id = str(uuid.uuid4())[:8]
        saved_filename = f"upload_{req_id}_{file.filename}"
        input_path = UPLOADS_DIR / saved_filename
        with open(input_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    elif sample_filename:
        input_path = SAMPLE_VIDEOS_DIR / sample_filename
        if not input_path.exists():
            raise HTTPException(status_code=404, detail=f"Sample video '{sample_filename}' not found.")
    else:
        raise HTTPException(status_code=400, detail="Please provide a video file upload or select a sample video.")

    try:
        # Run real AI inference
        result = analyze_video(
            input_video_path=str(input_path),
            camera_id=camera_id,
            location=location
        )

        # Dynamic OpenStreetMap Hospital selection on verified accident
        target_cam = next((c for c in DEMO_CAMERAS if c["id"] == camera_id), DEMO_CAMERAS[2])

        if result.get("accident_verified"):
            hospital_info = find_nearest_hospital(target_cam["lat"], target_cam["lng"])
            result["hospital"] = hospital_info
        else:
            result["hospital"] = None

        return {"status": "success", "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI Pipeline Error: {str(e)}")


@app.get("/api/alerts")
def get_alerts():
    """Returns active emergency alerts for Hospital Dashboard and Highway Authority."""
    return {"status": "success", "success": True, "alerts": EMERGENCY_ALERTS}


@app.post("/api/alerts")
def create_alert(alert_data: dict):
    """Creates a demo emergency alert sent to Hospital & Police."""
    alert_id = f"ALT-{str(uuid.uuid4())[:6].upper()}"
    hospital_data = alert_data.get("hospital")
    now_str = datetime.now().strftime("%I:%M %p")

    alert_record = {
        "id": alert_id,
        "alert_id": alert_id,
        "from": alert_data.get("from", "National Highways Authority"),
        "alert": alert_data.get("alert", "Road Accident Detected"),
        "camera_id": alert_data.get("camera_id", "C03"),
        "location": alert_data.get("location", "Demo Location"),
        "accident_type": alert_data.get("accident_type", "car_bike_accident"),
        "severity": alert_data.get("severity", "HIGH"),
        "confidence": alert_data.get("confidence", 0.0),
        "video_url": alert_data.get("video_url", ""),
        "timestamp": alert_data.get("timestamp", now_str),
        "hospital": hospital_data,
        "acknowledged": False,
        "acknowledged_at": None,
        "acknowledged_by": None,
        "dispatch_status": "pending",  # pending | dispatched
        "dispatched_at": None,
        "dispatched_by": None,
        "eta_minutes": None,
        "hospital_status": "PENDING",  # PENDING | ACKNOWLEDGED | DISPATCHED
        "recipients": [
            hospital_data["name"] if hospital_data and hospital_data.get("name") else "Emergency Trauma Unit",
            "Highway Police Station"
        ]
    }
    EMERGENCY_ALERTS.insert(0, alert_record)
    return {"status": "success", "success": True, "alert": alert_record}


@app.post("/api/alerts/{alert_id}/acknowledge")
@app.patch("/api/alerts/{alert_id}/acknowledge")
def acknowledge_alert_api(alert_id: str):
    """Marks alert as acknowledged by hospital."""
    now_str = datetime.now().strftime("%I:%M %p")
    for alert in EMERGENCY_ALERTS:
        if alert.get("id") == alert_id or alert.get("alert_id") == alert_id:
            alert["acknowledged"] = True
            alert["acknowledged_at"] = now_str
            h_name = alert["hospital"]["name"] if alert.get("hospital") and alert["hospital"].get("name") else "Emergency Hospital"
            alert["acknowledged_by"] = h_name
            alert["hospital_status"] = "ACKNOWLEDGED"
            return {"status": "success", "success": True, "alert": alert}
    raise HTTPException(status_code=404, detail=f"Alert '{alert_id}' not found.")


@app.post("/api/alerts/{alert_id}/dispatch")
@app.patch("/api/alerts/{alert_id}/dispatch")
def dispatch_alert_api(alert_id: str):
    """Marks emergency response team as dispatched for alert."""
    now_str = datetime.now().strftime("%I:%M %p")
    for alert in EMERGENCY_ALERTS:
        if alert.get("id") == alert_id or alert.get("alert_id") == alert_id:
            alert["dispatch_status"] = "dispatched"
            alert["dispatched_at"] = now_str
            h_name = alert["hospital"]["name"] if alert.get("hospital") and alert["hospital"].get("name") else "Emergency Hospital"
            alert["dispatched_by"] = h_name
            alert["hospital_status"] = "DISPATCHED"
            dist = alert["hospital"]["distance_km"] if alert.get("hospital") and alert["hospital"].get("distance_km") else 3.5
            alert["eta_minutes"] = max(2, int(dist * 2.5))
            return {"status": "success", "success": True, "alert": alert}
    raise HTTPException(status_code=404, detail=f"Alert '{alert_id}' not found.")


@app.patch("/api/alerts/{alert_id}")
def update_alert_status(alert_id: str, payload: dict):
    """Generic status updater endpoint."""
    new_status = payload.get("status")
    if new_status == "ACKNOWLEDGED":
        return acknowledge_alert_api(alert_id)
    elif new_status == "DISPATCHED":
        return dispatch_alert_api(alert_id)
    else:
        raise HTTPException(status_code=400, detail="Invalid status value.")


@app.post("/api/reset")
def reset_demo():
    """Resets shared in-memory emergency alerts and telemetry state."""
    global EMERGENCY_ALERTS
    EMERGENCY_ALERTS.clear()
    for cam in DEMO_CAMERAS:
        cam["status"] = "ONLINE"
    return {"status": "success", "message": "Demo state reset successfully."}


# Mount Static directory
STATIC_DIR = ROOT_DIR / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

INTEGRATION_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/integration-results", StaticFiles(directory=str(INTEGRATION_RESULTS_DIR)), name="integration-results")

if SAMPLE_VIDEOS_DIR.exists():
    app.mount("/samples", StaticFiles(directory=str(SAMPLE_VIDEOS_DIR)), name="samples")



@app.get("/")
def read_root():
    """Serves the main frontend dashboard HTML."""
    index_path = STATIC_DIR / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    return HTMLResponse("<h1>National Highways AI Smart Accident Detection System API Server</h1>")


if __name__ == "__main__":
    print("[+] Starting National Highways AI Control Center Server on http://localhost:8000")
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)
