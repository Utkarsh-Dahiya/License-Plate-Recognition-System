"""
LICENSE PLATE DETECTION & OCR SYSTEM — FastAPI backend.

Run locally:
    uvicorn main:app --reload --port 8000

Production:
    uvicorn main:app --host 0.0.0.0 --port $PORT

API:
    GET  /api/health
    GET  /api/dashboard
    GET  /api/batch/results
    GET  /api/batch/analytics
    GET  /api/video/summary
    GET  /api/video/detections
    GET  /api/plates
    GET  /api/model
    GET  /api/system
    GET  /api/history
    POST /api/detect/image
    POST /api/process/video
    GET  /api/process/video/{job_id}

The API reads the project's real generated data files.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import (
    BackgroundTasks,
    FastAPI,
    File,
    HTTPException,
    Query,
    Request,
    UploadFile,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

from services import data_loader, history_log


# ============================================================
# APP
# ============================================================

# Declared before the lifespan so startup logging can use it. Uvicorn only
# configures its own loggers, so a module-level logger would otherwise fall
# back to the "no handler" behaviour and print nothing (or emit through the
# root logger's lastResort handler at WARNING with no formatting). Attach a
# plain stream handler once, matching the detection service's approach.
logger = logging.getLogger("license_vision.api")

if not logger.handlers:
    _api_handler = logging.StreamHandler()
    _api_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(_api_handler)
    logger.propagate = False

# One knob for "noisy diagnostics". Access lines for successful requests and
# the detection service's per-stage OCR timings share this switch. Level is
# lowered to WARNING in quiet mode so 5xx responses and unhandled exceptions
# are STILL logged; only 2xx/4xx access lines are suppressed.
_TIMING_ENABLED = os.environ.get("LVA_DEBUG_TIMING", "1") != "0"

logger.setLevel(logging.INFO if _TIMING_ENABLED else logging.WARNING)

# Model-heavy work (YOLO + OCR, one video job) runs on a DEDICATED bounded
# executor — NOT the process-wide AnyIO pool that FastAPI/Starlette use for
# every sync endpoint. Capping the shared pool instead (a previous attempt)
# made /api/health, dashboard and video-status polling queue behind running
# detections for a free worker thread, and Starlette runs sync background
# tasks (the minutes-long video job) on that same pool, which would pin one
# of only two workers for the job's whole duration.
#
# Light endpoints (health/readiness/dashboard/...) keep the uncapped default
# pool (40 threads) and stay responsive no matter how many detections run.
# Heavy work is bounded to DETECTION_WORKERS concurrent executions; further
# requests simply queue inside this executor (holding only their already-
# uploaded bytes, capped at 10 MB per request) instead of pinning threads.
DETECTION_WORKERS = 2

_detection_pool = ThreadPoolExecutor(
    max_workers=DETECTION_WORKERS,
    thread_name_prefix="detection",
)


# ============================================================
# LIFESPAN — one-time model warmup at startup
# ============================================================
#
# Models load lazily on the first detection request if warmup has not
# finished (or failed); nothing here changes that contract. Warmup runs
# on a worker thread so a slow load cannot block the event loop, and the
# load lock inside detection_service guarantees exactly one YOLO / one
# EasyOCR instance no matter who calls first.
# ============================================================

_startup_warmup_done = False


async def _warmup_models() -> None:
    global _startup_warmup_done

    try:
        from services.detection_service import _ensure_models_loaded

        await run_in_threadpool(_ensure_models_loaded)

        logger.info("Model warmup complete (YOLO + OCR ready).")

    except Exception:
        # Never crash the app because warmup failed (missing weights,
        # OOM on a cold free-tier dyno). /api/readiness surfaces the
        # load error; the next detection request retries the load via
        # the normal lazy path.
        logger.warning(
            "Model warmup failed at startup; detection requests will retry "
            "the load lazily.",
            exc_info=True,
        )

    finally:
        _startup_warmup_done = True


@asynccontextmanager
async def lifespan(app: FastAPI):
    # One-time model warmup off the event loop. The load lock inside
    # detection_service guarantees a single YOLO / EasyOCR instance even
    # if a user request triggers the load in parallel.
    warmup_task = asyncio.create_task(_warmup_models())

    yield

    # Shutdown: stop the in-flight warmup cleanly (models are process
    # state; nothing else needs tearing down).
    warmup_task.cancel()

    try:
        await warmup_task

    except (asyncio.CancelledError, Exception):
        pass

    _detection_pool.shutdown(wait=False, cancel_futures=True)


app = FastAPI(
    title="License Plate Detection & OCR System API",
    version="1.0.0",
    lifespan=lifespan,
)


# ============================================================
# CORS
# ============================================================
#
# Local development:
#   http://localhost:5173
#   http://127.0.0.1:5173
#
# Production:
# Set:
#
#   LVA_FRONTEND_ORIGIN=https://your-frontend-domain.com
#
# Multiple origins can be supplied as comma-separated values:
#
#   LVA_FRONTEND_ORIGIN=https://example.com,https://www.example.com
#
# A trailing slash on any origin is ignored (browsers always send
# scheme://host with no trailing slash, so "https://foo.com/" would
# otherwise silently fail CORSMiddleware's exact-origin match).
#
# ============================================================

DEFAULT_FRONTEND_ORIGINS = (
    "http://localhost:5173,"
    "http://127.0.0.1:5173"
)

FRONTEND_ORIGINS = [
    origin.strip().rstrip("/")
    for origin in os.environ.get(
        "LVA_FRONTEND_ORIGIN",
        DEFAULT_FRONTEND_ORIGINS,
    ).split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=FRONTEND_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# REQUEST LOGGING / TIMING
#
# One concise line per request: method, path, status, duration. This is the
# minimum needed to tell "the model is slow" apart from "the endpoint is
# broken" in a deployed log, without dumping payloads.
#
# It follows the SAME LVA_DEBUG_TIMING switch the detection service already
# uses (defined at the top of this module), so operators have one knob:
# LVA_DEBUG_TIMING=0 silences both the per-stage OCR timings and these access
# lines.
#
# Severity is chosen so that switching timing off can never hide a real
# incident: 5xx responses and unhandled exceptions are logged at WARNING and
# always emitted. 4xx responses (bad upload, unknown job id) are routine
# client mistakes and are logged at INFO, so they disappear in quiet mode.
#
# Deliberately excluded: query strings and bodies (they can carry filenames
# and plate text), and any header that might hold credentials.
# ============================================================

# Endpoints polled in tight loops by the frontend; a per-poll line would bury
# everything else without adding information.
_QUIET_PATHS = frozenset(
    {
        "/api/health",
        "/api/readiness",
    }
)


@app.middleware("http")
async def log_request(request: Request, call_next):
    """Access log with duration; logs every failure regardless of verbosity."""

    started = time.perf_counter()

    try:
        response = await call_next(request)

    except Exception:
        # Let the global handler render the 500; just record that it happened.
        elapsed_ms = (time.perf_counter() - started) * 1000
        logger.warning(
            "%s %s -> unhandled error in %.1f ms",
            request.method,
            request.url.path,
            elapsed_ms,
            exc_info=True,
        )
        raise

    elapsed_ms = (time.perf_counter() - started) * 1000
    path = request.url.path

    if response.status_code >= 500:
        logger.warning(
            "%s %s -> %d in %.1f ms",
            request.method,
            path,
            response.status_code,
            elapsed_ms,
        )

    elif response.status_code >= 400:
        # 4xx is usually the caller's mistake (bad file, unknown job id), so
        # it is worth seeing but is not an incident.
        logger.info(
            "%s %s -> %d in %.1f ms",
            request.method,
            path,
            response.status_code,
            elapsed_ms,
        )

    elif _TIMING_ENABLED and path not in _QUIET_PATHS:
        logger.info(
            "%s %s -> %d in %.1f ms",
            request.method,
            path,
            response.status_code,
            elapsed_ms,
        )

    return response


# ============================================================
# STATIC VIDEO FILES
# ============================================================

if data_loader.VIDEO_RESULTS_DIR.exists():
    app.mount(
        "/media/video",
        StaticFiles(
            directory=str(data_loader.VIDEO_RESULTS_DIR)
        ),
        name="video_media",
    )


# ============================================================
# GLOBAL ERROR HANDLING
# ============================================================

@app.exception_handler(Exception)
async def unhandled_exception_handler(request, exc):  # noqa: ANN001
    # Full traceback to the server log (never to the client); the response
    # body stays generic unless app.debug is explicitly enabled.
    logger.warning(
        "Unhandled error on %s %s",
        request.method,
        request.url.path,
        exc_info=True,
    )

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
# READINESS
# ============================================================
# ADDITIVE endpoint — /api/health above is unchanged.
# ============================================================

@app.get("/api/readiness")
def readiness():
    """Startup state for orchestration/UX: warming, ready, or failed."""

    from services.detection_service import load_error, models_ready

    warmup_attempted = _startup_warmup_done
    ready = models_ready()
    error = load_error()

    if ready:
        state = "ready"
    elif error:
        state = "error"
    else:
        state = "warming_up"

    return {
        "status": state,
        "models_loaded": ready,
        "warmup_attempted": warmup_attempted,
        "detail": error,
    }


# ============================================================
# CURATED SAMPLES
# ============================================================

@app.get("/api/samples")
def list_samples():
    """List curated test sample images for 1-click evaluation."""
    return {
        "samples": data_loader.get_curated_samples()
    }


@app.get("/api/samples/{sample_id}")
def sample_detail(sample_id: str):
    """Retrieve metadata for a specific curated sample."""
    sample = data_loader.get_sample_metadata(sample_id)
    if not sample:
        raise HTTPException(
            status_code=404,
            detail="Sample not found.",
        )
    return sample


@app.get("/api/samples/{sample_id}/image")
def sample_image(sample_id: str):
    """Stream binary JPEG for a curated sample image."""
    path = data_loader.get_sample_path(sample_id)
    if not path or not path.exists():
        raise HTTPException(
            status_code=404,
            detail="Sample image not found on disk.",
        )
    return FileResponse(
        str(path),
        media_type="image/jpeg",
        headers={"Cache-Control": "public, max-age=3600"},
    )


# ============================================================
# DASHBOARD
# ============================================================

@app.get("/api/dashboard")
def dashboard():
    dash = data_loader.get_dashboard_data()
    video = data_loader.get_video_summary()

    if dash is None:
        raise HTTPException(
            status_code=404,
            detail=(
                "dashboard_data.json not found. "
                "Run the batch pipeline first."
            ),
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
        "mean_yolo_confidence": dash.get("detection", {}).get(
            "mean_yolo_confidence"
        ),
        "mean_ocr_confidence": dash.get("ocr", {}).get(
            "mean_confidence"
        ),
        "unique_plates_in_video": unique_plates,
        "multi_plate_images": dash.get("detection", {}).get(
            "multi_plate_images"
        ),
        "no_plate_images": dash.get("no_plate", {}).get("count"),
        "status_distribution": dash.get("status_distribution", []),
        "ocr_confidence_distribution": dash.get(
            "ocr_confidence_distribution",
            [],
        ),
        "plate_distribution": dash.get(
            "plate_distribution",
            [],
        ),
        "video_available": video is not None,
    }


# ============================================================
# BATCH INTELLIGENCE
# ============================================================

@app.get("/api/batch/results")
def batch_results(
    status: Optional[str] = Query(
        None,
        description="Filter: SUCCESS, OCR_FAILED, NO_PLATE, ...",
    ),
    min_confidence: float = Query(
        0.0,
        ge=0.0,
        le=1.0,
    ),
    search: Optional[str] = Query(
        None,
        description="Search image name or plate text",
    ),
    page: int = Query(1, ge=1),
    page_size: int = Query(
        25,
        ge=1,
        le=200,
    ),
):
    rows = data_loader.get_batch_rows()

    def keep(row: dict) -> bool:
        if status and row.get("status") != status:
            return False

        try:
            conf = float(
                row.get("best_yolo_confidence") or 0
            )
        except (ValueError, TypeError):
            conf = 0.0

        if conf < min_confidence:
            return False

        if search:
            needle = search.lower()

            image_name = (
                row.get("image") or ""
            ).lower()

            plate_text = (
                row.get("plate_text") or ""
            ).lower()

            if needle not in image_name and needle not in plate_text:
                return False

        return True

    filtered = [
        row
        for row in rows
        if keep(row)
    ]

    total = len(filtered)

    start = (
        (page - 1)
        * page_size
    )

    page_rows = filtered[
        start:start + page_size
    ]

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
        raise HTTPException(
            status_code=404,
            detail="dashboard_data.json not found.",
        )

    return dash


# ============================================================
# VIDEO ANALYTICS
# ============================================================

@app.get("/api/video/summary")
def video_summary():
    summary = data_loader.get_video_summary()

    if summary is None:
        raise HTTPException(
            status_code=404,
            detail="video_summary.json not found.",
        )

    registry = summary.get(
        "plates",
        summary.get(
            "plate_registry",
            [],
        ),
    )

    detection = dict(
        summary.get(
            "detection",
            {},
        )
    )

    if (
        "unique_plates" not in detection
        and "unique_plate_clusters" in detection
    ):
        detection["unique_plates"] = detection[
            "unique_plate_clusters"
        ]

    video_file_info = (
        data_loader.files_status()["annotated_video"]
    )

    return {
        "video": summary.get("video", {}),
        "processing": summary.get("processing", {}),
        "detection": detection,
        "plate_count": len(registry),
        "annotated_video": {
            "available": video_file_info["exists"],
            "url": (
                "/media/video/annotated_video.mp4"
                if video_file_info["exists"]
                else None
            ),
        },
    }


@app.get("/api/video/detections")
def video_detections(
    page: int = Query(1, ge=1),
    page_size: int = Query(
        50,
        ge=1,
        le=500,
    ),
):
    rows = data_loader.get_video_detection_rows()

    total = len(rows)

    start = (
        (page - 1)
        * page_size
    )

    page_rows = rows[
        start:start + page_size
    ]

    return {
        "total": total,
        "page": page,
        "page_size": page_size,
        "results": page_rows,
    }


# ============================================================
# PLATE REGISTRY
# ============================================================

def _plate_status(final_confidence) -> str:
    try:
        conf = float(
            final_confidence or 0
        )
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
    sort: str = Query(
        "confidence",
        pattern="^(confidence|detections|latest)$",
    ),
):
    summary = data_loader.get_video_summary()

    if summary is None:
        return {
            "total": 0,
            "results": [],
        }

    raw = summary.get(
        "plates",
        summary.get(
            "plate_registry",
            [],
        ),
    )

    out = []

    for plate in raw:
        plate_text = plate.get("plate")

        if (
            search
            and search.upper()
            not in (plate_text or "").upper()
        ):
            continue

        confidence = (
            plate.get("final_confidence")
            or plate.get("best_ocr_confidence")
        )

        out.append(
            {
                "plate": plate_text,
                "confidence": confidence,
                "detection_count": plate.get(
                    "detections"
                ),
                "first_seen": plate.get(
                    "first_timestamp"
                ),
                "last_seen": plate.get(
                    "last_timestamp"
                ),
                "first_frame": plate.get(
                    "first_frame"
                ),
                "last_frame": plate.get(
                    "last_frame"
                ),
                "status": _plate_status(
                    confidence
                ),
            }
        )

    key_map = {
        "confidence": lambda row: (
            row["confidence"] or 0
        ),
        "detections": lambda row: (
            row["detection_count"] or 0
        ),
        "latest": lambda row: (
            row["last_seen"] or 0
        ),
    }

    out.sort(
        key=key_map[sort],
        reverse=True,
    )

    return {
        "total": len(out),
        "results": out,
    }


@app.get("/api/plates/{plate_text}")
def plate_detail(plate_text: str):
    summary = data_loader.get_video_summary()

    if summary is None:
        raise HTTPException(
            status_code=404,
            detail="No video data available.",
        )

    raw = summary.get(
        "plates",
        summary.get(
            "plate_registry",
            [],
        ),
    )

    match = next(
        (
            plate
            for plate in raw
            if plate.get("plate")
            == plate_text.upper()
        ),
        None,
    )

    if match is None:
        raise HTTPException(
            status_code=404,
            detail=(
                f"Plate '{plate_text}' "
                "not found in registry."
            ),
        )

    detection_rows = [
        row
        for row in data_loader.get_video_detection_rows()
        if row.get("plate")
        == plate_text.upper()
    ]

    plate = dict(match)

    plate["status"] = _plate_status(
        plate.get("final_confidence")
        or plate.get("best_ocr_confidence")
    )

    return {
        "plate": plate,
        "detections": detection_rows,
    }


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
            "note": (
                "No training run artifacts "
                "(results.csv) were found. "
                "YOLO detection metrics only — "
                "not OCR accuracy."
            ),
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
                f"YOLO detection validation from "
                f"results.csv, last epoch ({tm['epoch']}). "
                "Not OCR / plate-text accuracy."
            ),
        }

    return {
        "detector": {
            "name": "YOLO License Plate Detector",
            "framework": "Ultralytics YOLO",
            "weights_path": str(
                data_loader.MODEL_WEIGHTS_PATH
            ),
            "weights_found": (
                data_loader.MODEL_WEIGHTS_PATH.exists()
            ),
        },
        "ocr": {
            "engine": "EasyOCR",
            "languages": ["en"],
        },
        "training_metrics": training_metrics,
        "observed_performance": {
            "mean_yolo_confidence": (
                (dash or {})
                .get("detection", {})
                .get("mean_yolo_confidence")
            ),
            "mean_ocr_confidence": (
                (dash or {})
                .get("ocr", {})
                .get("mean_confidence")
            ),
            "ocr_success_rate": (
                (dash or {})
                .get("ocr", {})
                .get("success_rate")
            ),
            "source": (
                "Derived from batch_results "
                "(658 images), not a held-out "
                "validation set."
            ),
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
# DETECTION HISTORY
# ============================================================

@app.get("/api/history")
def detection_history(
    type: Optional[str] = Query(
        None,
        pattern="^(image|video)$",
    ),
    search: Optional[str] = Query(
        None,
        description=(
            "Case-insensitive substring match on filename or "
            "plate text."
        ),
    ),
    page: int = Query(1, ge=1),
    page_size: int = Query(
        25,
        ge=1,
        le=200,
    ),
):
    return history_log.read_history(
        entry_type=type,
        search=search,
        page=page,
        page_size=page_size,
    )


@app.get("/api/history/stats")
def detection_history_stats():
    """Live system statistics from ACTUALLY recorded runs only.

    Computed from the append-only history log; every field is None or
    empty when no runs have been recorded yet, so the frontend can
    render honest zero-states. No fabricated numbers.
    """
    return history_log.history_stats()


@app.get("/api/history/export")
def detection_history_export(
    format: str = Query(
        "csv",
        pattern="^(csv|json)$",
        description="Download format: csv or json.",
    ),
    type: Optional[str] = Query(
        None,
        pattern="^(image|video)$",
    ),
    search: Optional[str] = Query(
        None,
        description="Same filter semantics as /api/history.",
    ),
):
    """Download the (optionally filtered) history as CSV or JSON."""
    content, media_type = history_log.export_entries(
        fmt=format,
        entry_type=type,
        search=search,
    )

    stamp = time.strftime("%Y%m%d-%H%M%S")
    filename = f"license-plate-detection-history-{stamp}.{format}"

    return Response(
        content=content,
        media_type=media_type,
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"'
        },
    )


# ============================================================
# IMAGE DETECTION
# ============================================================

MAX_IMAGE_BYTES = 10 * 1024 * 1024
MAX_VIDEO_BYTES = 50 * 1024 * 1024
MAX_DEMO_OCR_FRAMES = 120

# Container extensions OpenCV is expected to open. The temp file that a video
# upload is staged through needs a suffix the decoder recognises, but that
# suffix must never be taken verbatim from the client-supplied filename: a name
# like "clip.mp4/../../x" yields the suffix "./../../x", which
# tempfile.NamedTemporaryFile turns into a path OUTSIDE the temp directory (or
# an unhandled FileNotFoundError when the intermediate directory is absent).
# Restricting the suffix to this allowlist removes that class of problem and
# keeps the staged filename predictable.
_ALLOWED_VIDEO_SUFFIXES = (
    ".mp4",
    ".mov",
    ".avi",
    ".mkv",
    ".webm",
    ".m4v",
    ".mpg",
    ".mpeg",
    ".wmv",
    ".flv",
    ".3gp",
)


def _safe_video_suffix(filename: str | None) -> str:
    """Return an allowlisted container suffix for the staging temp file.

    Anything unrecognised (no extension, a traversal-shaped name, an exotic
    container) falls back to ".mp4", which is the format the demo dataset and
    the documented upload path use. Never raises.
    """
    name = (filename or "").replace("\\", "/").rsplit("/", 1)[-1]
    _, dot, ext = name.rpartition(".")
    candidate = f".{ext.lower()}" if dot else ""
    if candidate in _ALLOWED_VIDEO_SUFFIXES:
        return candidate
    return ".mp4"


@app.post("/api/detect/image")
async def detect_image(
    file: UploadFile = File(...),
    conf: float = Query(
        0.20,
        ge=0.05,
        le=0.95,
    ),
    debug: bool = Query(
        False,
        description=(
            "Include debug payload (raw boxes, OCR candidates, "
            "timings). Requires LVA_DEBUG_MODE=1 on the backend; "
            "otherwise the flag is silently ignored."
        ),
    ),
):
    if (
        not file.content_type
        or not file.content_type.startswith("image/")
    ):
        raise HTTPException(
            status_code=400,
            detail="Please upload an image file.",
        )

    image_bytes = await file.read()

    if not image_bytes:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file is empty.",
        )

    if len(image_bytes) > MAX_IMAGE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=(
                "Image is larger than "
                "the 10 MB demo limit."
            ),
        )

    from services.detection_service import (
        run_detection_on_image,
    )

    try:
        # CPU-bound YOLO + OCR must NOT run on the event loop: an async
        # endpoint blocks the loop for the whole inference, which is what
        # left /api/health unresponsive (frontend "Failed to fetch" on
        # every concurrent call) while one detection ran. Offload to the
        # dedicated detection executor (bounded at DETECTION_WORKERS) so
        # light endpoints on the default pool never queue behind model work.
        loop = asyncio.get_running_loop()

        result = await loop.run_in_executor(
            _detection_pool,
            run_detection_on_image,
            image_bytes,
            conf,
            debug,
        )

    except RuntimeError as exc:
        raise HTTPException(
            status_code=503,
            detail=str(exc),
        ) from exc

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    best_plate = max(
        result["plates"],
        key=lambda plate: plate["final_confidence"],
        default=None,
    )

    # Strip debug-only artifacts before anything is logged (history is
    # read by the frontend; debug payloads never belong there).
    for plate in result.get("plates", []):
        plate.pop("debug_ocr_crop_b64", None)

    history_log.log_entry(
        "image",
        filename=file.filename,
        plates_detected=result["plates_detected"],
        best_plate_text=(
            best_plate["ocr_text"]
            if best_plate
            else ""
        ),
        best_confidence=(
            best_plate["final_confidence"]
            if best_plate
            else 0.0
        ),
        status=(
            best_plate["status"]
            if best_plate
            else "NO_PLATE"
        ),
        processing_time_seconds=(
            result["processing_time_seconds"]
        ),
    )

    return result


# ============================================================
# DEBUG ARTIFACTS (ADDITIVE)
# ============================================================
#
# When LVA_DEBUG_MODE=1, POST /api/debug/plate-crop returns the exact
# OCR input canvas for the best detection (the crop after box
# expansion and OCR preprocessing). This is the image to look at when
# OCR misreads: everything the recognizer saw is in this PNG.
#
# Disabled by default: in production the endpoint 404s (or 403s) and
# stores nothing. Captures are capped and rotated; nothing is written
# to disk. Debug payloads are also stripped from image-history logs.

_DEBUG_STORE_MAX = 20
_debug_store: dict[str, dict] = {}


def _debug_enabled() -> bool:
    # Mirrors detection_service.run_detection_on_image's own gate:
    # debug payloads only flow when the deployment opted in.
    import os

    return os.environ.get("LVA_DEBUG_MODE", "0") != "0"


@app.post("/api/debug/plate-crop")
async def debug_plate_crop(
    file: UploadFile = File(...),
    conf: float = Query(0.20, ge=0.05, le=0.95),
):
    if not _debug_enabled():
        raise HTTPException(
            status_code=404,
            detail="Debug mode is disabled (set LVA_DEBUG_MODE=1).",
        )

    if (
        not file.content_type
        or not file.content_type.startswith("image/")
    ):
        raise HTTPException(
            status_code=400,
            detail="Please upload an image file.",
        )

    image_bytes = await file.read()

    if not image_bytes:
        raise HTTPException(status_code=400, detail="Empty file.")

    if len(image_bytes) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail="Image too large.")

    from services.detection_service import run_detection_on_image

    loop = asyncio.get_running_loop()

    result = await loop.run_in_executor(
        _detection_pool,
        run_detection_on_image,
        image_bytes,
        conf,
        True,  # debug_mode
    )

    plates = result.get("plates") or []

    best = max(
        plates,
        key=lambda p: p["final_confidence"],
        default=None,
    )

    if best is None:
        raise HTTPException(
            status_code=404,
            detail="No plate detected in this image.",
        )

    crop_b64 = best.get("debug_ocr_crop_b64")

    if not crop_b64:
        raise HTTPException(
            status_code=404,
            detail="No OCR crop captured (debug capture unavailable).",
        )

    capture_id = uuid.uuid4().hex[:12]

    _debug_store[capture_id] = {
        "created": time.time(),
        "filename": file.filename,
        "ocr_text": best.get("ocr_text"),
        "crop_b64": crop_b64,
    }

    # Rotate the store so it cannot grow without bound.
    while len(_debug_store) > _DEBUG_STORE_MAX:
        oldest = min(
            _debug_store,
            key=lambda k: _debug_store[k]["created"],
        )
        del _debug_store[oldest]

    return {
        "capture_id": capture_id,
        "bbox": best["bbox"],
        "yolo_confidence": best["yolo_confidence"],
        "ocr_text": best.get("ocr_text"),
        "ocr_confidence": best.get("ocr_confidence"),
        "final_confidence": best.get("final_confidence"),
        "status": best.get("status"),
        "ocr_input_crop_base64": crop_b64,
        "debug": result.get("debug"),
    }


# ============================================================
# VIDEO PROCESSING
# ============================================================

# Registry of async video-processing jobs. Bounded: the oldest jobs are
# dropped once the cap is hit so a client hammering /api/process/video
# cannot grow server memory without limit. Video jobs run on the SAME
# dedicated detection executor as image detections (one long job holds
# one worker slot — consistent with _MAX_CONCURRENT_VIDEO_JOBS and the
# free-tier's single model); requests beyond that get 503 instead of a
# silent memory race.
_MAX_VIDEO_JOBS = 12
_MAX_CONCURRENT_VIDEO_JOBS = 1

_video_jobs: dict[str, dict] = {}


def _process_video_job(
    job_id: str,
    video_path: str,
    frame_skip: int,
):
    import cv2
    from pathlib import Path

    from services.detection_service import (
        _ensure_models_loaded,
        run_detection_on_image,
    )

    job = _video_jobs[job_id]

    cap = None

    try:
        logger.info("Video job %s starting.", job_id[:8])

        _ensure_models_loaded()

        cap = cv2.VideoCapture(
            video_path
        )

        if not cap.isOpened():
            raise RuntimeError(
                "Could not open the uploaded video."
            )

        total_frames = int(
            cap.get(
                cv2.CAP_PROP_FRAME_COUNT
            )
        )

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
                        f"Demo cap: stopped after "
                        f"{MAX_DEMO_OCR_FRAMES} "
                        "sampled frames. "
                        "Saved video_results files "
                        "were not changed."
                    )

                    job["processed_frames"] = frame_idx
                    job["detections_so_far"] = detections

                    history_log.log_entry(
                        "video",
                        filename=job.get(
                            "filename"
                        ),
                        processed_frames=job.get(
                            "processed_frames",
                            0,
                        ),
                        status="FAILED",
                        error=job["error"],
                    )

                    return

                sampled += 1

                ok2, buf = cv2.imencode(
                    ".jpg",
                    frame,
                )

                if not ok2:
                    sampled_failures += 1
                    job["frame_errors"] = (
                        sampled_failures
                    )

                else:
                    try:
                        result = run_detection_on_image(
                            buf.tobytes()
                        )

                        detections += result[
                            "plates_detected"
                        ]

                    except Exception:
                        sampled_failures += 1

                        job["frame_errors"] = (
                            sampled_failures
                        )

                        if (
                            sampled <= 10
                            and sampled_failures == sampled
                        ):
                            raise RuntimeError(
                                "Video processing failed "
                                "on the first sampled frames."
                            )

            frame_idx += 1

            job["processed_frames"] = frame_idx
            job["detections_so_far"] = detections

        if not decoded_any:
            raise RuntimeError(
                "No frames could be decoded "
                "from the uploaded video."
            )

        job["status"] = "completed"
        job["finished_at"] = time.time()

        logger.info(
            "Video job %s completed: %d frames, %d detections in %.1fs.",
            job_id[:8],
            job["processed_frames"],
            job["detections_so_far"],
            job["finished_at"] - job["started_at"],
        )

        history_log.log_entry(
            "video",
            filename=job.get("filename"),
            processed_frames=job[
                "processed_frames"
            ],
            total_frames=job.get(
                "total_frames"
            ),
            detections_found=job[
                "detections_so_far"
            ],
            status="COMPLETED",
            processing_time_seconds=round(
                job["finished_at"]
                - job["started_at"],
                2,
            ),
        )

    except Exception as exc:
        job["status"] = "failed"
        job["error"] = str(exc)

        logger.warning("Video job %s failed: %s", job_id[:8], exc)

        history_log.log_entry(
            "video",
            filename=job.get("filename"),
            processed_frames=job.get(
                "processed_frames",
                0,
            ),
            status="FAILED",
            error=str(exc),
        )

    finally:
        if cap is not None:
            cap.release()

        Path(video_path).unlink(
            missing_ok=True
        )


@app.post("/api/process/video")
async def process_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    frame_skip: int = Query(
        15,
        ge=1,
        le=120,
    ),
):
    if (
        not file.content_type
        or not file.content_type.startswith("video/")
    ):
        raise HTTPException(
            status_code=400,
            detail="Please upload a video file.",
        )

    import tempfile

    payload = await file.read()

    if not payload:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file is empty.",
        )

    if len(payload) > MAX_VIDEO_BYTES:
        raise HTTPException(
            status_code=413,
            detail=(
                "Video is larger than "
                "the 50 MB demo limit."
            ),
        )

    job_id = str(
        uuid.uuid4()
    )

    suffix = _safe_video_suffix(file.filename)

    tmp = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=suffix,
    )

    tmp.write(payload)
    tmp.close()

    # Drop the oldest finished jobs once the registry cap is reached
    # (insertion order = creation order).
    while len(_video_jobs) >= _MAX_VIDEO_JOBS:
        for old_id in list(_video_jobs):
            if _video_jobs[old_id].get("status") in (
                "completed",
                "failed",
            ):
                del _video_jobs[old_id]
                break
        else:
            # All jobs still active (cannot happen under the concurrency
            # cap below, but never evict a running job):
            del _video_jobs[next(iter(_video_jobs))]

    # Register FIRST, then check concurrency. "queued" counts as active:
    # the earlier check-before-register let a second submission race past
    # while job 1 was still queued (status not yet flipped to
    # "processing" by its worker). This whole block contains no `await`,
    # so the event loop serializes concurrent submissions — the counter
    # cannot be observed stale.
    _video_jobs[job_id] = {
        "job_id": job_id,
        "status": "queued",
        "filename": file.filename,
        "processed_frames": 0,
        "total_frames": None,
        "detections_so_far": 0,
        "started_at": time.time(),
    }

    active = sum(
        1
        for job in _video_jobs.values()
        if job.get("status") in ("queued", "processing")
    )

    if active > _MAX_CONCURRENT_VIDEO_JOBS:
        # Roll back: remove the just-registered job and its temp file.
        del _video_jobs[job_id]
        Path(tmp.name).unlink(missing_ok=True)

        raise HTTPException(
            status_code=503,
            detail=(
                "A video is already being processed. "
                "Please wait for it to finish."
            ),
        )

    # Video processing is heavy model work: it runs on the SAME dedicated
    # bounded executor as image detections (one long job occupies one
    # worker slot; the concurrency gate above keeps it to one job). The
    # default AnyIO pool that Starlette would otherwise use stays free for
    # health/dashboard/status endpoints for the whole duration.
    loop = asyncio.get_running_loop()

    try:
        loop.run_in_executor(
            _detection_pool,
            _process_video_job,
            job_id,
            tmp.name,
            frame_skip,
        )

    except Exception:
        # The worker never started, so nothing will ever clean this job up:
        # _process_video_job owns the temp-file unlink in its `finally`. Roll
        # the registration back here instead of leaving a permanently "queued"
        # job and an orphaned temp file on disk.
        _video_jobs.pop(job_id, None)
        Path(tmp.name).unlink(missing_ok=True)
        raise

    return {
        "job_id": job_id,
        "status": "queued",
    }


@app.get("/api/process/video/{job_id}")
def process_video_status(
    job_id: str,
):
    job = _video_jobs.get(
        job_id
    )

    if job is None:
        raise HTTPException(
            status_code=404,
            detail="Unknown job id.",
        )

    return job