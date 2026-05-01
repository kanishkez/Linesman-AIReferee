"""
FastAPI Server for Football AI VAR

Provides REST API endpoints for uploading videos and retrieving analysis results.
Serves the web frontend as static files.
"""

import os
import re
import uuid
import threading
from pathlib import Path
from dotenv import load_dotenv

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from .models import AnalysisResult, AnalysisStatus
from .pipeline import VARPipeline

# Load environment variables
load_dotenv()

# ─── App Setup ────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Football AI VAR",
    description="AI-powered Video Assistant Referee for football foul detection",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Directories ──────────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).resolve().parent.parent
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "outputs"
STATIC_DIR = BASE_DIR / "static"

UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# ─── Global State ─────────────────────────────────────────────────────────────

# In-memory job store (for prototype; would use Redis/DB in production)
jobs: dict[str, AnalysisResult] = {}

# Pipeline instance (initialized lazily)
pipeline: VARPipeline | None = None


def get_pipeline() -> VARPipeline:
    """Get or create the pipeline instance."""
    global pipeline
    if pipeline is None:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY not set. Copy .env.example to .env and add your key."
            )
        pipeline = VARPipeline(api_key=api_key, output_base_dir=str(OUTPUT_DIR))
    return pipeline


# ─── API Routes ───────────────────────────────────────────────────────────────

@app.post("/api/analyze")
async def analyze_video(video: UploadFile = File(...)):
    """
    Upload a football video clip for VAR analysis.

    The analysis runs in a background thread and can be polled via /api/status/{job_id}.
    """
    # Validate file type
    allowed_types = {"video/mp4", "video/quicktime", "video/x-msvideo", "video/webm"}
    if video.content_type and video.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported video type: {video.content_type}. Accepted: MP4, MOV, AVI, WebM",
        )

    # Generate job ID and save file with ASCII-safe filename
    job_id = str(uuid.uuid4())[:8]
    # Sanitize filename: remove non-ASCII characters to prevent encoding errors
    safe_original = re.sub(r'[^\x00-\x7F]+', '', video.filename or 'video.mp4').strip()
    if not safe_original:
        safe_original = 'video.mp4'
    filename = f"{job_id}_{safe_original}"
    filepath = UPLOAD_DIR / filename

    with open(filepath, "wb") as f:
        content = await video.read()
        f.write(content)

    print(f"[API] Video uploaded: {filename} ({len(content) / 1024 / 1024:.1f} MB)")

    # Create the result object
    result = AnalysisResult(
        job_id=job_id,
        status=AnalysisStatus.PENDING,
        video_filename=filename,
    )
    jobs[job_id] = result

    # Run pipeline in background thread
    def run_analysis():
        try:
            pipe = get_pipeline()
            pipe.run(
                video_path=str(filepath),
                job_id=job_id,
                result=result,
            )
        except Exception as e:
            result.status = AnalysisStatus.ERROR
            result.error_message = str(e)
            print(f"[API] Pipeline error for job {job_id}: {e}")

    thread = threading.Thread(target=run_analysis, daemon=True)
    thread.start()

    return {"job_id": job_id, "status": "pending", "message": "Analysis started"}


@app.get("/api/status/{job_id}")
async def get_status(job_id: str):
    """Check the status of an analysis job."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    result = jobs[job_id]
    return {
        "job_id": job_id,
        "status": result.status.value,
        "error_message": result.error_message,
    }


@app.get("/api/results/{job_id}")
async def get_results(job_id: str):
    """Get the full results of a completed analysis."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    result = jobs[job_id]

    # Build response
    response = {
        "job_id": result.job_id,
        "status": result.status.value,
        "video_filename": result.video_filename,
        "processing_time_sec": result.processing_time_sec,
        "error_message": result.error_message,
    }

    if result.yolo_analysis:
        response["yolo"] = {
            "total_frames": result.yolo_analysis.total_frames,
            "fps": result.yolo_analysis.fps,
            "duration_sec": result.yolo_analysis.duration_sec,
            "max_players_detected": result.yolo_analysis.max_players_detected,
            "contact_frames_count": len(result.yolo_analysis.key_contact_frames),
            "key_contact_frames": result.yolo_analysis.key_contact_frames[:20],
        }

    if result.gemini_analysis:
        response["gemini"] = result.gemini_analysis.model_dump()

    if result.var_decision:
        response["decision"] = result.var_decision.model_dump()

    return JSONResponse(content=response)


@app.get("/api/annotated-video/{job_id}")
async def get_annotated_video(job_id: str):
    """Serve the annotated video with YOLO overlays."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    result = jobs[job_id]
    if result.annotated_video_path and os.path.exists(result.annotated_video_path):
        return FileResponse(
            result.annotated_video_path,
            media_type="video/mp4",
            filename=f"var_annotated_{job_id}.mp4",
        )

    raise HTTPException(status_code=404, detail="Annotated video not yet available")


# ─── Static Files & Frontend ─────────────────────────────────────────────────

# Mount static files
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
async def serve_frontend():
    """Serve the main web UI."""
    index_path = STATIC_DIR / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    return {"message": "Football AI VAR API is running. Frontend not found at /static/index.html"}
