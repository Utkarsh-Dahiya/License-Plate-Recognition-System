"""
LICENSE VISION AI — Backend
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


def read_history(entry_type: str | None = None, page: int = 1, page_size: int = 25) -> dict:
    if not HISTORY_LOG_PATH.exists():
        return {"total": 0, "page": page, "page_size": page_size, "results": []}

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

    entries.sort(key=lambda e: e.get("timestamp", 0), reverse=True)

    total = len(entries)
    start = (page - 1) * page_size
    page_entries = entries[start : start + page_size]

    return {"total": total, "page": page, "page_size": page_size, "results": page_entries}
