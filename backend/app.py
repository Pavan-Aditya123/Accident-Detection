"""
FastAPI application — AI Smart Accident Detection System.

Integrated with AI inference pipeline.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.config import (
    API_TITLE,
    API_VERSION,
    API_DESCRIPTION,
    CORS_ORIGINS,
    UPLOAD_DIR,
    YOLO_MODEL_PATH,
    ZDCE_MODEL_PATH,
)
from backend.routes import router

app = FastAPI(
    title=API_TITLE,
    version=API_VERSION,
    description=API_DESCRIPTION,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include inference routes
app.include_router(router)


@app.get("/", tags=["Health"])
async def root():
    return {
        "status": "running",
        "system": "AI Smart Accident Detection System",
    }


@app.get("/health", tags=["Health"])
async def health():
    return {
        "status": "running",
        "system": "AI Smart Accident Detection System",
        "version": API_VERSION,
        "upload_dir": str(UPLOAD_DIR),
        "models": {
            "yolo26":   str(YOLO_MODEL_PATH),
            "zero_dce": str(ZDCE_MODEL_PATH),
        },
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.app:app", host="0.0.0.0", port=8000, reload=True)
