"""
LICENSE VISION AI — Backend
Image detection service.

This reuses the SAME logic as plate_pipeline.py (Indian plate scoring /
validation, candidate scoring philosophy) rather than reimplementing a
separate, weaker pipeline.

The YOLO model and EasyOCR reader are loaded ONCE, lazily, on first
request, and reused for every subsequent call — never reloaded per
request (see brief section 25, Performance).

PERFORMANCE NOTE (interactive path only):
Real-world testing showed 45–70s per image with the original 4-variant,
1800x600 OCR pass. Profiling (see _log_timing below) confirmed the EasyOCR
readtext() calls — not YOLO, not preprocessing — dominate that time on
CPU, and cost scales ~linearly with (a) number of variants run and (b)
pixel count per variant. Two changes address both:

  1. Primary + conditional fallback instead of always running 4 variants.
     The fallback OCR pass only runs when the primary result is missing,
     low-confidence, or format-inconsistent — see run_detection_on_image.
  2. OCR crop capped at ~1100px max dimension (was a fixed, aspect-ratio-
     distorting 1800x600) instead of a much larger canvas.

The offline batch pipeline (plate_pipeline.py) and video pipeline
(video_pipeline.py) are untouched — this file only backs the interactive
POST /api/detect/image endpoint.
"""

from __future__ import annotations

import logging
import os
import re
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from services.data_loader import MODEL_WEIGHTS_PATH

# ============================================================
# DEV TIMING (log-only — never part of the API response schema)
# ============================================================

logger = logging.getLogger("license_vision.detection")
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("[timing] %(message)s"))
    logger.addHandler(_handler)
logger.setLevel(
    logging.INFO if os.environ.get("LVA_DEBUG_TIMING", "1") != "0" else logging.WARNING
)


class _Stopwatch:
    """Tiny helper to log named stage durations without cluttering the
    main control flow. Purely diagnostic — has no effect on the response."""

    def __init__(self):
        self._t0 = time.perf_counter()
        self._last = self._t0
        self.stages: list[tuple[str, float]] = []

    def lap(self, label: str):
        now = time.perf_counter()
        elapsed = now - self._last
        self._last = now
        self.stages.append((label, elapsed))
        logger.info("%-22s %6.1f ms", label, elapsed * 1000)
        return elapsed

    def total(self) -> float:
        return time.perf_counter() - self._t0


# ============================================================
# LAZY, SINGLETON MODEL STATE
# ============================================================

_yolo_model = None
_ocr_reader = None
_load_error: str | None = None


def _ensure_models_loaded():
    global _yolo_model, _ocr_reader, _load_error

    if _yolo_model is not None and _ocr_reader is not None:
        return

    if _load_error is not None:
        # Don't retry a broken load on every request.
        raise RuntimeError(_load_error)

    try:
        from ultralytics import YOLO
        import easyocr

        if not MODEL_WEIGHTS_PATH.exists():
            raise FileNotFoundError(
                f"YOLO weights not found at {MODEL_WEIGHTS_PATH}. "
                "Point LVA_MODEL_PATH at your best.pt."
            )

        _yolo_model = YOLO(str(MODEL_WEIGHTS_PATH))
        _ocr_reader = easyocr.Reader(["en"], gpu=False)

    except Exception as exc:  # noqa: BLE001 - surface as a clean API error
        _load_error = str(exc)
        raise RuntimeError(_load_error) from exc


def models_ready() -> bool:
    return _yolo_model is not None and _ocr_reader is not None


# ============================================================
# INDIAN PLATE VALIDATION (unchanged — flexible format, not the old
# fixed 3-letter + 3-digit rule)
# ============================================================

INDIAN_PLATE_PATTERNS = [
    re.compile(r"^[A-Z]{2}\d{1,2}[A-Z]{1,3}\d{1,4}$"),
    re.compile(r"^[A-Z]{2,3}\d{1,4}[A-Z]{0,3}\d{0,4}$"),
]


def clean_text(text: str) -> str:
    text = str(text).upper()
    return re.sub(r"[^A-Z0-9]", "", text)


def indian_plate_score(text: str) -> int:
    s = clean_text(text)
    if not s:
        return 0

    score = 0
    length = len(s)

    if 8 <= length <= 12:
        score += 35
    elif 6 <= length <= 13:
        score += 15

    if len(s) >= 2 and s[:2].isalpha():
        score += 25

    if sum(c.isdigit() for c in s) >= 2:
        score += 20

    if len(s) >= 5 and sum(c.isalpha() for c in s[2:]) >= 1:
        score += 10

    for pattern in INDIAN_PLATE_PATTERNS:
        if pattern.match(s):
            score += 35
            break

    return min(score, 100)


def classify_validation(score: int) -> str:
    if score >= 75:
        return "INDIAN_PLATE"
    if score >= 45:
        return "POSSIBLE_INDIAN_PLATE"
    return "UNVERIFIED"


def status_from_confidence(final_confidence: float, ocr_text: str) -> str:
    if not ocr_text:
        return "OCR_FAILED"
    if final_confidence >= 0.80:
        return "HIGH_CONFIDENCE"
    if final_confidence >= 0.50:
        return "REVIEW"
    return "LOW_CONFIDENCE"


# ============================================================
# OCR PREPROCESSING — primary variant always runs, fallback is lazy
# ============================================================
# Max OCR canvas dimension. Was a fixed, aspect-distorting 1800x600
# (1,080,000 px). Capping the longer side at ~1100px while preserving
# aspect ratio keeps characters legible for EasyOCR's recognizer while
# cutting the pixel count — and therefore CPU inference time — roughly
# 60-70% versus the old canvas.
_OCR_MAX_DIM = 1100
_OCR_MAX_UPSCALE = 6.0  # don't blow up a tiny crop past the point of adding real detail

# Fallback only fires when the primary read is missing or looks weak.
_FALLBACK_OCR_CONF_THRESHOLD = 0.55
_FALLBACK_FORMAT_SCORE_THRESHOLD = 45  # matches classify_validation's POSSIBLE_INDIAN_PLATE cut


def _pad_and_scale(crop: np.ndarray) -> np.ndarray:
    """Replicate-pad a plate crop slightly (helps EasyOCR see full glyphs
    near the edge) and scale it to a bounded canvas, preserving aspect
    ratio — no stretching."""
    h, w = crop.shape[:2]
    border_x = max(6, int(w * 0.03))
    border_y = max(6, int(h * 0.10))
    padded = cv2.copyMakeBorder(
        crop, border_y, border_y, border_x, border_x, cv2.BORDER_REPLICATE
    )

    ph, pw = padded.shape[:2]
    longer_side = max(ph, pw)
    if longer_side < _OCR_MAX_DIM:
        # Small crop — upscale toward the target canvas, capped so we
        # don't blow up a tiny plate past the point of adding real detail.
        scale = min(_OCR_MAX_DIM / longer_side, _OCR_MAX_UPSCALE)
    else:
        # Already large — downscale to the cap, never distort.
        scale = _OCR_MAX_DIM / longer_side
    target_w, target_h = max(1, int(pw * scale)), max(1, int(ph * scale))
    interp = cv2.INTER_CUBIC if scale >= 1 else cv2.INTER_AREA
    return cv2.resize(padded, (target_w, target_h), interpolation=interp)


def _make_primary_variant(base_gray: np.ndarray) -> np.ndarray:
    """CLAHE-enhanced grayscale — the single highest-value variant from
    the original 4-variant sweep, kept as the default first (and often
    only) OCR pass."""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(base_gray)


def _make_fallback_variant(enhanced_gray: np.ndarray) -> np.ndarray:
    """OTSU binarization — the second highest-value variant, run only
    when the primary read is weak (see run_detection_on_image)."""
    _, otsu = cv2.threshold(
        enhanced_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    return otsu


def _candidate_score(text: str, ocr_conf: float) -> float:
    text = clean_text(text)
    if not text:
        return 0.0
    format_score = indian_plate_score(text)
    length_score = 20 if 8 <= len(text) <= 12 else (8 if 6 <= len(text) <= 13 else 0)
    return float(ocr_conf) * 45 + format_score * 0.65 + length_score


def _run_ocr(image: np.ndarray) -> list[tuple[str, float]]:
    """One EasyOCR pass -> list of (cleaned_text, confidence) candidates."""
    try:
        results = _ocr_reader.readtext(
            image,
            detail=1,
            paragraph=False,
            allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
        )
    except Exception:
        return []

    candidates = []
    for _, text, conf in results:
        cleaned = clean_text(text)
        if cleaned:
            candidates.append((cleaned, float(conf)))
    return candidates


def _best_candidate(candidates: list[tuple[str, float]]):
    if not candidates:
        return None
    return max(candidates, key=lambda c: _candidate_score(c[0], c[1]))


def _is_strong_enough(candidate) -> bool:
    """Decide whether the primary OCR pass is good enough to skip the
    fallback variant entirely — this is the main latency lever."""
    if candidate is None:
        return False
    text, conf = candidate
    if conf < _FALLBACK_OCR_CONF_THRESHOLD:
        return False
    if indian_plate_score(text) < _FALLBACK_FORMAT_SCORE_THRESHOLD:
        return False
    return True


# ============================================================
# MAIN ENTRY POINT
# ============================================================

def run_detection_on_image(image_bytes: bytes, conf_threshold: float = 0.25) -> dict[str, Any]:
    """Run YOLO -> crop -> primary (+ conditional fallback) OCR -> scoring
    on a single uploaded image. Returns a JSON-serializable result dict
    with the exact same schema as before this optimization pass."""
    sw = _Stopwatch()

    _ensure_models_loaded()
    sw.lap("ensure_models_loaded")

    np_arr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Could not decode image. Is it a valid image file?")
    sw.lap("image_decode")

    h, w = image.shape[:2]

    results = _yolo_model.predict(source=image, conf=conf_threshold, verbose=False)
    result = results[0]
    sw.lap("yolo_inference")

    detections = []
    if result.boxes is not None:
        for box in result.boxes:
            confidence = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            detections.append((confidence, x1, y1, x2, y2))

    detections.sort(key=lambda d: d[0], reverse=True)

    plates = []
    annotated = image.copy()
    ocr_pass_count = 0

    for idx, (yolo_conf, x1, y1, x2, y2) in enumerate(detections, start=1):
        x1c, y1c = max(0, x1), max(0, y1)
        x2c, y2c = min(w, x2), min(h, y2)
        crop = image[y1c:y2c, x1c:x2c]

        plate_entry = {
            "plate_id": idx,
            "bbox": [x1c, y1c, x2c, y2c],
            "yolo_confidence": round(yolo_conf, 4),
            "ocr_text": "",
            "ocr_confidence": 0.0,
            "validation_score": 0,
            "validation": "UNVERIFIED",
            "final_confidence": 0.0,
            "status": "OCR_FAILED",
        }

        if crop.size > 0:
            base = _pad_and_scale(crop)
            base_gray = cv2.cvtColor(base, cv2.COLOR_BGR2GRAY)
            sw.lap(f"plate{idx}_crop_preprocess")

            enhanced = _make_primary_variant(base_gray)
            primary_candidates = _run_ocr(enhanced)
            ocr_pass_count += 1
            sw.lap(f"plate{idx}_ocr_primary")

            candidates = list(primary_candidates)
            best = _best_candidate(candidates)

            if not _is_strong_enough(best):
                # Primary read was missing, low-confidence, or format-
                # inconsistent — spend the extra OCR pass on a fallback
                # variant instead of accepting a weak read.
                fallback_img = _make_fallback_variant(enhanced)
                fallback_candidates = _run_ocr(fallback_img)
                ocr_pass_count += 1
                sw.lap(f"plate{idx}_ocr_fallback")

                candidates.extend(fallback_candidates)
                best = _best_candidate(candidates)

            if best is not None:
                best_text, best_ocr_conf = best
                validation_score = indian_plate_score(best_text)
                validation = classify_validation(validation_score)
                final_confidence = min(
                    1.0, best_ocr_conf * 0.6 + (validation_score / 100) * 0.4
                )

                plate_entry.update(
                    {
                        "ocr_text": best_text,
                        "ocr_confidence": round(best_ocr_conf, 4),
                        "validation_score": validation_score,
                        "validation": validation,
                        "final_confidence": round(final_confidence, 4),
                        "status": status_from_confidence(final_confidence, best_text),
                    }
                )

        plates.append(plate_entry)

        color = (0, 200, 100) if plate_entry["ocr_text"] else (0, 140, 255)
        cv2.rectangle(annotated, (x1c, y1c), (x2c, y2c), color, 3)
        label = f"{plate_entry['ocr_text'] or 'UNKNOWN'} | {plate_entry['final_confidence']*100:.0f}%"
        cv2.putText(
            annotated, label, (x1c, max(25, y1c - 10)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA,
        )

    ok, buf = cv2.imencode(".jpg", annotated)
    annotated_b64 = None
    if ok:
        import base64
        annotated_b64 = base64.b64encode(buf.tobytes()).decode("ascii")
    sw.lap("annotation_encode")

    elapsed = sw.total()
    logger.info(
        "TOTAL %6.1f ms | %d plate(s) | %d OCR pass(es)",
        elapsed * 1000, len(plates), ocr_pass_count,
    )

    return {
        "image": {"width": w, "height": h},
        "plates_detected": len(plates),
        "plates": plates,
        "annotated_image_base64": annotated_b64,
        "processing_time_seconds": round(elapsed, 3),
    }
