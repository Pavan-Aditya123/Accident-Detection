"""
FastAPI Server for AI Smart Accident Detection System Frontend Demo.
Provides API endpoints for camera management, video analysis via Python AI pipeline,
and emergency alert dispatch between Highway Authority and Hospital Emergency response.
"""

import sys
import os
import shutil
from pathlib import Path
from typing import Optional, List
import uuid

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Query, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))

from inference_api import analyze_video, UPLOADS_DIR, OUTPUT_DIR

app = FastAPI(title="National Highways AI Smart Accident Detection System")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Demo CCTV Camera Locations (Fictional Highway Locations)
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
    Accepts either an uploaded CCTV file or a sample CCTV video filename.
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
        return {"status": "success", "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI Pipeline Error: {str(e)}")


@app.get("/api/alerts")
def get_alerts():
    """Returns active emergency alerts for Hospital Dashboard."""
    return {"status": "success", "alerts": EMERGENCY_ALERTS}


@app.post("/api/alerts")
def create_alert(alert_data: dict):
    """Creates a demo emergency alert sent to Hospital & Police."""
    alert_id = f"ALT-{str(uuid.uuid4())[:6].upper()}"
    alert_record = {
        "id": alert_id,
        "from": alert_data.get("from", "National Highways Authority"),
        "alert": alert_data.get("alert", "Road Accident Detected"),
        "camera_id": alert_data.get("camera_id", "C03"),
        "location": alert_data.get("location", "Demo Location"),
        "accident_type": alert_data.get("accident_type", "car_bike_accident"),
        "severity": alert_data.get("severity", "HIGH"),
        "confidence": alert_data.get("confidence", 0.0),
        "video_url": alert_data.get("video_url", ""),
        "timestamp": alert_data.get("timestamp", "Just now"),
        "hospital_status": "PENDING",  # PENDING | ACKNOWLEDGED | DISPATCHED
        "recipients": ["Demo General Hospital", "Highway Police Station"]
    }
    EMERGENCY_ALERTS.insert(0, alert_record)
    return {"status": "success", "alert": alert_record}


@app.patch("/api/alerts/{alert_id}")
def update_alert_status(alert_id: str, payload: dict):
    """Updates emergency alert status (ACKNOWLEDGED / DISPATCHED)."""
    new_status = payload.get("status")
    if new_status not in ["ACKNOWLEDGED", "DISPATCHED"]:
        raise HTTPException(status_code=400, detail="Invalid status value.")
    
    for alert in EMERGENCY_ALERTS:
        if alert["id"] == alert_id:
            alert["hospital_status"] = new_status
            return {"status": "success", "alert": alert}
            
    raise HTTPException(status_code=404, detail="Alert not found.")


# Mount Static directory
STATIC_DIR = ROOT_DIR / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

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
