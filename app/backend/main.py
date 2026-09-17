"""
LICENSE VISION AI — FastAPI backend.

Run:
    uvicorn main:app --reload --port 8000

Endpoints mirror the brief exactly:
    GET  /api/health
    GET  /api/dashboard
    GET  /api/batch/results
    GET  /api/batch/analytics
    GET  /api/video/summary
    GET  /api/video/detections
    GET  /api/plates
    GET  /api/model
    GET  /api/system
    POST /api/detect/image
    POST /api/process/video
    GET  /api/process/video/{job_id}

All read-only endpoints serve REAL data from the files in ./data.
No metric is fabricated: if a source file is missing, the endpoint
returns null/"N/A" fields rather than invented numbers.
"""

from __future__ import annotations

import time
import traceback
import uuid
from typing import Optional

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from services import data_loader, history_log

app = FastAPI(title="License Vision AI API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve video_results/ (annotated_video.mp4 etc.) as static files so the
# frontend can play the real annotated video without proxying bytes through
# a Python endpoint. Mounted even if the directory doesn't exist yet on
# disk — FastAPI creates the route, requests just 404 until a video lands.
if data_loader.VIDEO_RESULTS_DIR.exists():
    app.mount(
        "/media/video",
        StaticFiles(directory=str(data_loader.VIDEO_RESULTS_DIR)),
        name="video_media",
    )


# ============================================================
# GLOBAL ERROR HANDLING — never leak a raw traceback to the client
# ============================================================

@app.exception_handler(Exception)
async def unhandled_exception_handler(request, exc):  # noqa: ANN001
    traceback.print_exc()
    return JSONResponse(
        status_code=500,
        content={
            "error": "Something went wrong processing that request.",
            "detail": str(exc) if app.debug else None,
        },
    )


# ============================================================
# HEALTH
# ============================================================

@app.get("/api/health")
def health():
    from services.detection_service import models_ready

    return {
        "status": "online",
        "models_loaded": models_ready(),
        "sources": data_loader.files_status(),
    }


# ============================================================
# DASHBOARD (Overview page)
# ============================================================

@app.get("/api/dashboard")
def dashboard():
    dash = data_loader.get_dashboard_data()
    video = data_loader.get_video_summary()

    if dash is None:
        raise HTTPException(
            status_code=404,
            detail="dashboard_data.json not found. Run the batch pipeline first.",
        )

    unique_plates = None
    if video:
        unique_plates = (
            video.get("detection", {}).get("unique_plates")
            or video.get("detection", {}).get("unique_plate_clusters")
        )

    return {
        "total_images": dash.get("dataset", {}).get("total_images"),
        "plates_detected": dash.get("dataset", {}).get("total_plates"),
        "ocr_success_rate": dash.get("ocr", {}).get("success_rate"),
        "mean_yolo_confidence": dash.get("detection", {}).get("mean_yolo_confidence"),
        "mean_ocr_confidence": dash.get("ocr", {}).get("mean_confidence"),
        "unique_plates_in_video": unique_plates,
        "multi_plate_images": dash.get("detection", {}).get("multi_plate_images"),
        "no_plate_images": dash.get("no_plate", {}).get("count"),
        "status_distribution": dash.get("status_distribution", []),
        "ocr_confidence_distribution": dash.get("ocr_confidence_distribution", []),
        "plate_distribution": dash.get("plate_distribution", []),
        "video_available": video is not None,
    }


# ============================================================
# BATCH INTELLIGENCE
# ============================================================

@app.get("/api/batch/results")
def batch_results(
    status: Optional[str] = Query(None, description="Filter: SUCCESS, OCR_FAILED, NO_PLATE, ..."),
    min_confidence: float = Query(0.0, ge=0.0, le=1.0),
    search: Optional[str] = Query(None, description="Search image name or plate text"),
    page: int = Query(1, ge=1),
    page_size: int = Query(25, ge=1, le=200),
):
    rows = data_loader.get_batch_rows()

    def keep(row: dict) -> bool:
        if status and row.get("status") != status:
            return False
        try:
            conf = float(row.get("best_yolo_confidence") or 0)
        except ValueError:
            conf = 0.0
        if conf < min_confidence:
            return False
        if search:
            needle = search.lower()
            if needle not in (row.get("image") or "").lower() and needle not in (
                row.get("plate_text") or ""
            ).lower():
                return False
        return True

    filtered = [r for r in rows if keep(r)]
    total = len(filtered)
    start = (page - 1) * page_size
    page_rows = filtered[start : start + page_size]

    return {
        "total": total,
        "page": page,
        "page_size": page_size,
        "results": page_rows,
    }


@app.get("/api/batch/analytics")
def batch_analytics():
    dash = data_loader.get_dashboard_data()
    if dash is None:
        raise HTTPException(status_code=404, detail="dashboard_data.json not found.")
    return dash


# ============================================================
# VIDEO ANALYTICS
# ============================================================

@app.get("/api/video/summary")
def video_summary():
    summary = data_loader.get_video_summary()
    if summary is None:
        raise HTTPException(status_code=404, detail="video_summary.json not found.")

    # Normalize the schema so the frontend never has to branch on the
    # plates/plate_registry naming difference found during inspection.
    registry = summary.get("plates", summary.get("plate_registry", []))
    detection = dict(summary.get("detection", {}))
    if "unique_plates" not in detection and "unique_plate_clusters" in detection:
        detection["unique_plates"] = detection["unique_plate_clusters"]

    video_file_info = data_loader.files_status()["annotated_video"]

    return {
        "video": summary.get("video", {}),
        "processing": summary.get("processing", {}),
        "detection": detection,
        "plate_count": len(registry),
        "annotated_video": {
            "available": video_file_info["exists"],
            "url": "/media/video/annotated_video.mp4" if video_file_info["exists"] else None,
        },
    }


@app.get("/api/video/detections")
def video_detections(
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=500),
):
    rows = data_loader.get_video_detection_rows()
    total = len(rows)
    start = (page - 1) * page_size
    page_rows = rows[start : start + page_size]
    return {"total": total, "page": page, "page_size": page_size, "results": page_rows}


# ============================================================
# PLATE REGISTRY
# ============================================================

def _plate_status(final_confidence) -> str:
    try:
        conf = float(final_confidence or 0)
    except (TypeError, ValueError):
        conf = 0.0
    if conf >= 0.8:
        return "HIGH_CONFIDENCE"
    if conf >= 0.5:
        return "REVIEW"
    return "LOW_CONFIDENCE"


@app.get("/api/plates")
def plates(
    search: Optional[str] = Query(None),
    sort: str = Query("confidence", pattern="^(confidence|detections|latest)$"),
):
    summary = data_loader.get_video_summary()
    if summary is None:
        return {"total": 0, "results": []}

    raw = summary.get("plates", summary.get("plate_registry", []))
    fps = (summary.get("video", {}) or {}).get("fps") or 1

    out = []
    for p in raw:
        plate_text = p.get("plate")
        if search and search.upper() not in (plate_text or "").upper():
            continue
        out.append(
            {
                "plate": plate_text,
                "confidence": p.get("final_confidence") or p.get("best_ocr_confidence"),
                "detection_count": p.get("detections"),
                "first_seen": p.get("first_timestamp"),
                "last_seen": p.get("last_timestamp"),
                "first_frame": p.get("first_frame"),
                "last_frame": p.get("last_frame"),
                "status": _plate_status(
                    p.get("final_confidence") or p.get("best_ocr_confidence")
                ),
            }
        )

    key_map = {
        "confidence": lambda r: r["confidence"] or 0,
        "detections": lambda r: r["detection_count"] or 0,
        "latest": lambda r: r["last_seen"] or 0,
    }
    out.sort(key=key_map[sort], reverse=True)

    return {"total": len(out), "results": out}


@app.get("/api/plates/{plate_text}")
def plate_detail(plate_text: str):
    summary = data_loader.get_video_summary()
    if summary is None:
        raise HTTPException(status_code=404, detail="No video data available.")

    raw = summary.get("plates", summary.get("plate_registry", []))
    match = next((p for p in raw if p.get("plate") == plate_text.upper()), None)
    if match is None:
        raise HTTPException(status_code=404, detail=f"Plate '{plate_text}' not found in registry.")

    detection_rows = [
        r for r in data_loader.get_video_detection_rows() if r.get("plate") == plate_text.upper()
    ]

    plate = dict(match)
    plate["status"] = _plate_status(
        plate.get("final_confidence") or plate.get("best_ocr_confidence")
    )

    return {"plate": plate, "detections": detection_rows}


# ============================================================
# MODEL / SYSTEM
# ============================================================

@app.get("/api/model")
def model_info():
    dash = data_loader.get_dashboard_data()
    tm = data_loader.get_training_metrics()
    if tm is None:
        training_metrics = {
            "mAP": "Not available",
            "mAP50": "Not available",
            "mAP50_95": "Not available",
            "precision": "Not available",
            "recall": "Not available",
            "epoch": "Not available",
            "note": "No training run artifacts (results.csv) were found. YOLO detection metrics only — not OCR accuracy.",
        }
    else:
        training_metrics = {
            "mAP": tm["mAP50"],
            "mAP50": tm["mAP50"],
            "mAP50_95": tm["mAP50_95"],
            "precision": tm["precision"],
            "recall": tm["recall"],
            "epoch": tm["epoch"],
            "note": (
                f"YOLO detection validation from results.csv, last epoch ({tm['epoch']}). "
                "Not OCR / plate-text accuracy."
            ),
        }

    return {
        "detector": {
            "name": "YOLO License Plate Detector",
            "framework": "Ultralytics YOLO",
            "weights_path": str(data_loader.MODEL_WEIGHTS_PATH),
            "weights_found": data_loader.MODEL_WEIGHTS_PATH.exists(),
        },
        "ocr": {
            "engine": "EasyOCR",
            "languages": ["en"],
        },
        "training_metrics": training_metrics,
        "observed_performance": {
            "mean_yolo_confidence": (dash or {}).get("detection", {}).get("mean_yolo_confidence"),
            "mean_ocr_confidence": (dash or {}).get("ocr", {}).get("mean_confidence"),
            "ocr_success_rate": (dash or {}).get("ocr", {}).get("success_rate"),
            "source": "Derived from batch_results (658 images), not a held-out validation set.",
        },
        "pipeline": [
            "Input image",
            "YOLO license plate detection",
            "Bounding box",
            "Plate cropping",
            "Adaptive OCR (CLAHE, then OTSU / denoise only if needed)",
            "EasyOCR",
            "Candidate scoring & Indian plate format validation",
            "Final confidence",
        ],
    }


@app.get("/api/system")
def system_info():
    return {
        "stack": {
            "detector": "YOLO (Ultralytics)",
            "ocr": "EasyOCR",
            "cv": "OpenCV",
            "backend": "FastAPI",
            "frontend": "React + Vite + Tailwind CSS",
        },
        "sources": data_loader.files_status(),
    }


# ============================================================
# DETECTION HISTORY (Detection page)
# ============================================================

@app.get("/api/history")
def detection_history(
    type: Optional[str] = Query(None, pattern="^(image|video)$"),
    page: int = Query(1, ge=1),
    page_size: int = Query(25, ge=1, le=200),
):
    return history_log.read_history(entry_type=type, page=page, page_size=page_size)


# ============================================================
# IMAGE DETECTION (real inference)
# ============================================================

MAX_IMAGE_BYTES = 10 * 1024 * 1024
MAX_VIDEO_BYTES = 50 * 1024 * 1024
MAX_DEMO_OCR_FRAMES = 120


@app.post("/api/detect/image")
async def detect_image(file: UploadFile = File(...), conf: float = Query(0.25, ge=0.05, le=0.95)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload an image file.")

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    if len(image_bytes) > MAX_IMAGE_BYTES:
        raise HTTPException(
            status_code=413,
            detail="Image is larger than the 10 MB demo limit.",
        )

    from services.detection_service import run_detection_on_image

    try:
        result = run_detection_on_image(image_bytes, conf_threshold=conf)
    except RuntimeError as exc:
        # Model failed to load (e.g. best.pt missing) — clean 503, not a stack trace.
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    best_plate = max(result["plates"], key=lambda p: p["final_confidence"], default=None)
    history_log.log_entry(
        "image",
        filename=file.filename,
        plates_detected=result["plates_detected"],
        best_plate_text=best_plate["ocr_text"] if best_plate else "",
        best_confidence=best_plate["final_confidence"] if best_plate else 0.0,
        status=best_plate["status"] if best_plate else "NO_PLATE",
        processing_time_seconds=result["processing_time_seconds"],
    )

    return result


# ============================================================
# VIDEO PROCESSING (explicit, background job — never automatic)
# Does not write to video_results/ (detections.csv, video_summary.json,
# annotated_video.mp4 stay as the saved 4K run).
# ============================================================

_video_jobs: dict[str, dict] = {}


def _process_video_job(job_id: str, video_path: str, frame_skip: int):
    import cv2
    from pathlib import Path

    from services.detection_service import _ensure_models_loaded, run_detection_on_image

    job = _video_jobs[job_id]
    cap = None
    try:
        _ensure_models_loaded()
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError("Could not open the uploaded video.")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        job["total_frames"] = total_frames
        job["status"] = "processing"
        job["frame_errors"] = 0

        frame_idx = 0
        detections = 0
        sampled = 0
        sampled_failures = 0
        decoded_any = False

        while True:
            ok, frame = cap.read()
            if not ok:
                break
            decoded_any = True
            if frame_idx % frame_skip == 0:
                if sampled >= MAX_DEMO_OCR_FRAMES:
                    job["status"] = "failed"
                    job["error"] = (
                        f"Demo cap: stopped after {MAX_DEMO_OCR_FRAMES} sampled frames. "
                        "Saved video_results files were not changed."
                    )
                    job["processed_frames"] = frame_idx
                    job["detections_so_far"] = detections
                    history_log.log_entry(
                        "video",
                        filename=job.get("filename"),
                        processed_frames=job.get("processed_frames", 0),
                        status="FAILED",
                        error=job["error"],
                    )
                    return

                sampled += 1
                ok2, buf = cv2.imencode(".jpg", frame)
                if not ok2:
                    sampled_failures += 1
                    job["frame_errors"] = sampled_failures
                else:
                    try:
                        result = run_detection_on_image(buf.tobytes())
                        detections += result["plates_detected"]
                    except Exception:
                        sampled_failures += 1
                        job["frame_errors"] = sampled_failures
                        if sampled <= 10 and sampled_failures == sampled:
                            raise RuntimeError(
                                "Video processing failed on the first sampled frames."
                            )

            frame_idx += 1
            job["processed_frames"] = frame_idx
            job["detections_so_far"] = detections

        if not decoded_any:
            raise RuntimeError("No frames could be decoded from the uploaded video.")

        job["status"] = "completed"
        job["finished_at"] = time.time()
        history_log.log_entry(
            "video",
            filename=job.get("filename"),
            processed_frames=job["processed_frames"],
            total_frames=job.get("total_frames"),
            detections_found=job["detections_so_far"],
            status="COMPLETED",
            processing_time_seconds=round(job["finished_at"] - job["started_at"], 2),
        )
    except Exception as exc:  # noqa: BLE001
        job["status"] = "failed"
        job["error"] = str(exc)
        history_log.log_entry(
            "video",
            filename=job.get("filename"),
            processed_frames=job.get("processed_frames", 0),
            status="FAILED",
            error=str(exc),
        )
    finally:
        if cap is not None:
            cap.release()
        Path(video_path).unlink(missing_ok=True)


@app.post("/api/process/video")
async def process_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    frame_skip: int = Query(15, ge=1, le=120),
):
    """Explicit, user-triggered demo video job. Never called automatically
    on dashboard load. Never writes to video_results/ (the saved 4K CSV,
    summary, and annotated mp4 are left untouched). Poll
    GET /api/process/video/{job_id} for progress."""

    if not file.content_type or not file.content_type.startswith("video/"):
        raise HTTPException(status_code=400, detail="Please upload a video file.")

    import tempfile

    payload = await file.read()
    if not payload:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    if len(payload) > MAX_VIDEO_BYTES:
        raise HTTPException(
            status_code=413,
            detail="Video is larger than the 50 MB demo limit.",
        )

    job_id = str(uuid.uuid4())
    suffix = "." + (file.filename.rsplit(".", 1)[-1] if "." in (file.filename or "") else "mp4")
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(payload)
    tmp.close()

    _video_jobs[job_id] = {
        "job_id": job_id,
        "status": "queued",
        "filename": file.filename,
        "processed_frames": 0,
        "total_frames": None,
        "detections_so_far": 0,
        "started_at": time.time(),
    }

    background_tasks.add_task(_process_video_job, job_id, tmp.name, frame_skip)

    return {"job_id": job_id, "status": "queued"}


@app.get("/api/process/video/{job_id}")
def process_video_status(job_id: str):
    job = _video_jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Unknown job id.")
    return job
