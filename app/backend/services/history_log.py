"""
LICENSE PLATE DETECTION & OCR SYSTEM — Backend
Detection history log.

Every real call to /api/detect/image or /api/process/video appends one
entry here. This is what backs the "Detection" (history) page — it is
NOT a dataset generator; it only ever records runs the user actually
triggered, append-only, in the order they happened.

Storage is a flat JSONL file rather than a database: this is a
single-operator portfolio project, and a JSONL file is trivial to
inspect, diff, and back up by hand.
"""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Any

from services.data_loader import HISTORY_DIR, HISTORY_LOG_PATH


def _ensure_dir() -> None:
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)


def log_entry(entry_type: str, **fields: Any) -> dict:
    """Append one history entry and return it (with id/timestamp filled in)."""
    _ensure_dir()

    record = {
        "id": str(uuid.uuid4()),
        "type": entry_type,  # "image" | "video"
        "timestamp": time.time(),
        **fields,
    }

    with open(HISTORY_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")

    return record


def _load_entries(
    entry_type: str | None = None,
    search: str | None = None,
) -> list[dict]:
    """Load history entries, optionally filtered.

    Shared loader so read_history (paginated UI), history_stats()
    (analytics) and export_entries() (CSV/JSON export) always see the
    same data through the same filtering rules.
    """
    if not HISTORY_LOG_PATH.exists():
        return []

    entries = []
    with open(HISTORY_LOG_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    if entry_type:
        entries = [e for e in entries if e.get("type") == entry_type]

    # Search matches what the operator can actually remember: the file
    # name they uploaded or the plate text that came back.
    if search:
        needle = search.lower()
        entries = [
            e
            for e in entries
            if needle in str(e.get("filename", "")).lower()
            or needle in str(e.get("best_plate_text", "")).lower()
        ]

    return entries


def read_history(
    entry_type: str | None = None,
    search: str | None = None,
    page: int = 1,
    page_size: int = 25,
) -> dict:
    entries = _load_entries(entry_type, search)

    entries.sort(key=lambda e: e.get("timestamp", 0), reverse=True)

    total = len(entries)
    start = (page - 1) * page_size
    page_entries = entries[start : start + page_size]

    return {"total": total, "page": page, "page_size": page_size, "results": page_entries}


def history_stats() -> dict:
    """Live system statistics computed ONLY from actually recorded runs.

    Every number is derived from detections.jsonl (real user-triggered
    runs logged by /api/detect/image and /api/process/video). When no
    runs exist the zero-state return is explicit (None / empty map) —
    the frontend shows empty states, never fabricated numbers.
    """
    entries = _load_entries()

    image_entries = [e for e in entries if e.get("type") == "image"]
    video_entries = [e for e in entries if e.get("type") == "video"]

    plates_detected = sum(
        int(e.get("plates_detected") or 0) for e in image_entries
    )
    runs_with_text = sum(
        1
        for e in image_entries
        if str(e.get("best_plate_text") or "").strip()
    )

    # Average confidence is taken over runs that actually PRODUCED a
    # read: a NO_PLATE / OCR_FAILED run records best_confidence 0.0,
    # which is the absence of a measurement, not a zero-confidence
    # observation — including it would understate real read quality.
    confidences = [
        float(e["best_confidence"])
        for e in image_entries
        if e.get("best_confidence") is not None
        and str(e.get("best_plate_text") or "").strip()
    ]

    # Latency reflects the IMAGE recognition path only: video jobs run
    # minutes-long on a different executor and would distort the p95 the
    # dashboard reports for interactive detections.
    latencies = sorted(
        float(e["processing_time_seconds"])
        for e in image_entries
        if e.get("processing_time_seconds") is not None
    )

    status_counts: dict[str, int] = {}
    for e in image_entries:
        status = str(e.get("status") or "UNKNOWN")
        status_counts[status] = status_counts.get(status, 0) + 1

    def _percentile(sorted_vals: list[float], pct: float) -> float | None:
        if not sorted_vals:
            return None
        idx = min(
            len(sorted_vals) - 1,
            max(0, int(round(pct * (len(sorted_vals) - 1)))),
        )
        return sorted_vals[idx]

    return {
        "total_runs": len(entries),
        "image_runs": len(image_entries),
        "video_runs": len(video_entries),
        "plates_detected": plates_detected,
        "image_runs_with_text": runs_with_text,
        "ocr_text_rate": (
            round(runs_with_text / len(image_entries), 4)
            if image_entries
            else None
        ),
        "avg_best_confidence": (
            round(sum(confidences) / len(confidences), 4)
            if confidences
            else None
        ),
        "latency": {
            "mean_seconds": (
                round(sum(latencies) / len(latencies), 4)
                if latencies
                else None
            ),
            "p95_seconds": _percentile(latencies, 0.95),
            "samples": len(latencies),
        },
        "status_distribution": status_counts,
        "first_run_timestamp": (
            min(e.get("timestamp", 0) for e in entries) if entries else None
        ),
        "last_run_timestamp": (
            max(e.get("timestamp", 0) for e in entries) if entries else None
        ),
    }


# Flat column set for exports. Video entries carry different fields
# than image entries; missing fields export as empty strings.
EXPORT_FIELDS = [
    "id",
    "type",
    "timestamp",
    "filename",
    "plates_detected",
    "best_plate_text",
    "best_confidence",
    "status",
    "processing_time_seconds",
]


def export_entries(
    fmt: str,
    entry_type: str | None = None,
    search: str | None = None,
) -> tuple[str, str]:
    """Serialize history entries for download.

    Returns (content, media_type). CSV rows are flattened onto
    EXPORT_FIELDS so the output stays tabular regardless of entry type.
    """
    entries = _load_entries(entry_type, search)
    entries.sort(key=lambda e: e.get("timestamp", 0), reverse=True)

    if fmt == "json":
        payload = {
            "exported_at": time.time(),
            "count": len(entries),
            "entries": entries,
        }
        return (json.dumps(payload, indent=2), "application/json")

    def _csv_escape(value) -> str:
        s = "" if value is None else str(value)
        if any(ch in s for ch in '",\n'):
            s = '"' + s.replace('"', '""') + '"'
        return s

    lines = [",".join(EXPORT_FIELDS)]
    for e in entries:
        lines.append(
            ",".join(_csv_escape(e.get(field)) for field in EXPORT_FIELDS)
        )

    return ("\n".join(lines), "text/csv")
