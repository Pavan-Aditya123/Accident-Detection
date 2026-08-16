"""
Backend configuration.
All paths are relative to the project root (one level above this file).
"""

from pathlib import Path

# Project root — two levels up from this file (backend/config.py → backend/ → root)
ROOT_DIR: Path = Path(__file__).resolve().parent.parent

# Upload directory for incoming video/image files
UPLOAD_DIR: Path = ROOT_DIR / "backend" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# Model paths (not loaded here — referenced only)
YOLO_MODEL_PATH:  Path = ROOT_DIR / "results" / "detection" / "yolo26_baseline" / "weights" / "best.pt"
ZDCE_MODEL_PATH:  Path = ROOT_DIR / "models" / "enhancement" / "zero_dce" / "best_zero_dce.pth"

# API settings
API_TITLE:   str = "AI Smart Accident Detection System"
API_VERSION: str = "1.0.0"
API_DESCRIPTION: str = (
    "Backend API for real-time accident detection using Zero-DCE "
    "low-light enhancement and YOLO26 object detection."
)

# CORS — allow all origins for development; tighten for production
CORS_ORIGINS: list[str] = ["*"]
