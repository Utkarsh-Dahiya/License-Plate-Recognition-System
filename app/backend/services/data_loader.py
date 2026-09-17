"""
LICENSE VISION AI — Backend
Data access layer.

Reads the REAL existing output files from the actual project root
(C:\\Users\\Lenovo\\OneDrive\\Desktop\\License Plate Detection on the
user's machine), not a bundled copy:
  batch_results\\batch_results.csv
  batch_results\\analytics\\dashboard_data.json
  video_results\\detections.csv
  video_results\\video_summary.json
  video_results\\annotated_video.mp4
  runs\\detect\\models\\license_plate_detector\\weights\\best.pt

No metric here is invented. If a file is missing, endpoints return
a clear "not available" response instead of fabricating data.

Everything is cached in memory after first read (mtime-checked) so the
658-row batch CSV and 1000+-row detection CSV aren't re-parsed on every
request, and the YOLO/EasyOCR models (loaded elsewhere, in
detection_service.py) are never touched by this module.
"""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Any

# ============================================================
# PATHS
# ============================================================
# This file lives at:
#   <project root>/app/backend/services/data_loader.py
# so three parents up from this file is the project root itself
# (services -> backend -> app -> project root). That's what lets
# PROJECT_ROOT resolve to the real project without any env var —
# override with LVA_PROJECT_ROOT if this backend is ever moved
# somewhere that doesn't sit inside the project root at app/backend.

import os

BACKEND_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = Path(
    os.environ.get(
        "LVA_PROJECT_ROOT",
        Path(__file__).resolve().parents[3],
    )
)

BATCH_CSV_PATH = Path(
    os.environ.get(
        "LVA_BATCH_CSV",
        PROJECT_ROOT / "batch_results" / "batch_results.csv",
    )
)
DASHBOARD_JSON_PATH = Path(
    os.environ.get(
        "LVA_DASHBOARD_JSON",
        PROJECT_ROOT / "batch_results" / "analytics" / "dashboard_data.json",
    )
)
VIDEO_DETECTIONS_CSV_PATH = Path(
    os.environ.get(
        "LVA_VIDEO_DETECTIONS_CSV",
        PROJECT_ROOT / "video_results" / "detections.csv",
    )
)
VIDEO_SUMMARY_JSON_PATH = Path(
    os.environ.get(
        "LVA_VIDEO_SUMMARY_JSON",
        PROJECT_ROOT / "video_results" / "video_summary.json",
    )
)
MODEL_WEIGHTS_PATH = Path(
    os.environ.get(
        "LVA_MODEL_PATH",
        PROJECT_ROOT
        / "runs"
        / "detect"
        / "models"
        / "license_plate_detector"
        / "weights"
        / "best.pt",
    )
)
TRAINING_RESULTS_CSV = Path(
    os.environ.get(
        "LVA_TRAINING_RESULTS_CSV",
        PROJECT_ROOT
        / "runs"
        / "detect"
        / "models"
        / "license_plate_detector"
        / "results.csv",
    )
)

# ============================================================
# TINY MTIME-BASED CACHE
# ============================================================

_cache: dict[str, Any] = {}
_cache_mtime: dict[str, float] = {}


def _load_cached(path: Path, loader):
    """Return loader(path) result, cached until the file's mtime changes."""
    key = str(path)
    try:
        mtime = path.stat().st_mtime
    except FileNotFoundError:
        return None

    if _cache.get(key) is not None and _cache_mtime.get(key) == mtime:
        return _cache[key]

    value = loader(path)
    _cache[key] = value
    _cache_mtime[key] = mtime
    return value


def _read_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _read_csv_rows(path: Path):
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


# ============================================================
# PUBLIC ACCESSORS
# ============================================================

def get_dashboard_data() -> dict | None:
    """batch_results/analytics/dashboard_data.json, as-is."""
    return _load_cached(DASHBOARD_JSON_PATH, _read_json)


def get_batch_rows() -> list[dict]:
    """batch_results/batch_results.csv rows.

    Columns (verified from file): image, plates_detected,
    best_yolo_confidence, plate_text, ocr_confidence, status
    """
    rows = _load_cached(BATCH_CSV_PATH, _read_csv_rows)
    return rows or []


def get_video_summary() -> dict | None:
    """video_results/video_summary.json, as-is.

    Verified top-level keys: video, processing, detection, plates
    (NOT plate_registry — the file on disk predates that key name).
    """
    return _load_cached(VIDEO_SUMMARY_JSON_PATH, _read_json)


def get_video_detection_rows() -> list[dict]:
    """video_results/detections.csv rows.

    Columns (verified from file): frame, timestamp, plate,
    yolo_confidence, ocr_confidence, final_confidence, x1, y1, x2, y2
    """
    rows = _load_cached(VIDEO_DETECTIONS_CSV_PATH, _read_csv_rows)
    return rows or []


def get_training_metrics() -> dict | None:
    """Last epoch from Ultralytics results.csv, or None if the file is missing.

    Values are taken from the CSV as-is (YOLO box detection validation).
    """
    rows = _load_cached(TRAINING_RESULTS_CSV, _read_csv_rows)
    if not rows:
        return None

    last = {str(k).strip(): v for k, v in rows[-1].items()}

    def _float(key: str):
        raw = last.get(key)
        if raw is None or raw == "":
            return None
        try:
            return float(raw)
        except (TypeError, ValueError):
            return None

    epoch_raw = last.get("epoch")
    try:
        epoch = int(float(epoch_raw)) if epoch_raw not in (None, "") else None
    except (TypeError, ValueError):
        epoch = epoch_raw

    return {
        "epoch": epoch,
        "precision": _float("metrics/precision(B)"),
        "recall": _float("metrics/recall(B)"),
        "mAP50": _float("metrics/mAP50(B)"),
        "mAP50_95": _float("metrics/mAP50-95(B)"),
        "source": str(TRAINING_RESULTS_CSV),
    }


VIDEO_RESULTS_DIR = VIDEO_SUMMARY_JSON_PATH.parent
ANNOTATED_VIDEO_PATH = Path(
    os.environ.get("LVA_ANNOTATED_VIDEO", VIDEO_RESULTS_DIR / "annotated_video.mp4")
)

HISTORY_DIR = BACKEND_ROOT / "data" / "history"
HISTORY_LOG_PATH = HISTORY_DIR / "detections.jsonl"


def _info(path: Path) -> dict:
    exists = path.exists()
    return {
        "path": str(path),
        "exists": exists,
        "size_bytes": path.stat().st_size if exists else None,
        "modified": (
            time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(path.stat().st_mtime))
            if exists
            else None
        ),
    }


def files_status() -> dict:
    """Existence + freshness info for every source file, for /api/health and /api/system."""
    return {
        "batch_csv": _info(BATCH_CSV_PATH),
        "dashboard_json": _info(DASHBOARD_JSON_PATH),
        "video_detections_csv": _info(VIDEO_DETECTIONS_CSV_PATH),
        "video_summary_json": _info(VIDEO_SUMMARY_JSON_PATH),
        "model_weights": _info(MODEL_WEIGHTS_PATH),
        "annotated_video": _info(ANNOTATED_VIDEO_PATH),
    }
