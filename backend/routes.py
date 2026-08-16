"""
API routes for AI Smart Accident Detection System.
"""

from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from pathlib import Path
from typing import Dict, Any
import shutil

from backend.config import UPLOAD_DIR, ROOT_DIR
from backend.inference_runner import InferenceRunner

router = APIRouter()

_runner = None

ALLOWED_EXTENSIONS = {".mp4", ".mov", ".avi"}

RESULTS_DIR = ROOT_DIR / "results" / "enhanced_detection_severity"


def _get_runner():
    global _runner
    if _runner is None:
        _runner = InferenceRunner()
    return _runner


@router.post("/upload", tags=["Inference"])
async def upload_video(file: UploadFile = File(...)):
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type '{file_ext}'. Allowed: {', '.join(ALLOWED_EXTENSIONS)}",
        )

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    upload_path = UPLOAD_DIR / file.filename
    with open(upload_path, "wb") as buf:
        shutil.copyfileobj(file.file, buf)

    try:
        result = _get_runner().run_inference(input_video_path=str(upload_path))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {e}")

    if not result["success"]:
        raise HTTPException(status_code=500, detail=f"Inference failed: {result['error']}")

    stats = result.get("stats", {})
    output_path = result["output_video_path"]
    output_filename = Path(output_path).name if output_path else None
    severity = stats.get("last_severity")

    return {
        "status":              "success",
        "input_video":         file.filename,
        "output_video":        output_path,
        "output_filename":     output_filename,
        "accident_detected":   (stats.get("confirmed_accidents", 0) > 0),
        "confirmed_accidents": stats.get("confirmed_accidents", 0),
        "possible_accidents":  stats.get("possible_accidents", 0),
        "frames_processed":    stats.get("frames_processed", 0),
        "avg_fps":             stats.get("avg_fps", 0),
        "avg_ms_per_frame":    stats.get("avg_ms_per_frame", 0),
        "severity":            severity,
    }


@router.get("/video/{filename}", tags=["Inference"])
async def serve_video(filename: str):
    safe_name = Path(filename).name
    video_path = RESULTS_DIR / safe_name
    if not video_path.exists():
        raise HTTPException(status_code=404, detail=f"Video not found: {safe_name}")
    return FileResponse(path=str(video_path), media_type="video/mp4", filename=safe_name)


@router.get("/inference/status", tags=["Inference"])
async def inference_status():
    return {
        "status":              "ready",
        "system":              "AI Smart Accident Detection System",
        "upload_dir":          str(UPLOAD_DIR),
        "allowed_extensions":  list(ALLOWED_EXTENSIONS),
    }
