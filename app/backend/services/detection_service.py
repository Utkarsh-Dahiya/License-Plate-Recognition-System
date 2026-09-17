"""
LICENSE VISION AI — Backend
Image detection service (the ONE canonical interactive detection path).

This backs POST /api/detect/image only. The offline batch pipeline
(plate_pipeline.py) and video pipeline (video_pipeline.py) are separate,
untouched files — nothing here changes their behavior or output format.

============================================================
HISTORY / WHY THIS FILE LOOKS THE WAY IT DOES
============================================================

The interactive OCR path uses an adaptive cascade:

    Tier 1: CLAHE + EasyOCR
    Tier 2: OTSU + EasyOCR fallback
    Tier 3: denoise + OTSU + EasyOCR last resort

The important optimization is to avoid running all three OCR passes
when the first result is already sufficiently supported.

IMPORTANT:
    EasyOCR .recognize() is intentionally NOT used.

Every OCR pass goes through readtext(), preserving EasyOCR's detection
stage for real-world plate crops.
"""

from __future__ import annotations

import logging
import os
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from services.data_loader import MODEL_WEIGHTS_PATH


# ============================================================
# DEV TIMING
# ============================================================

logger = logging.getLogger("license_vision.detection")

if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(_handler)

logger.setLevel(
    logging.INFO
    if os.environ.get("LVA_DEBUG_TIMING", "1") != "0"
    else logging.WARNING
)


class _TimingBucket:
    """Accumulates timing information for one request."""

    def __init__(self):
        self._t0 = time.perf_counter()
        self.buckets: dict[str, float] = defaultdict(float)
        self.detail: list[str] = []

    def add(self, bucket: str, seconds: float, note: str = ""):
        self.buckets[bucket] += seconds

        if note:
            self.detail.append(
                f"  {note}: {seconds * 1000:.1f} ms"
            )

    def total(self) -> float:
        return time.perf_counter() - self._t0

    def report(self, plate_count: int, ocr_pass_count: int):
        total_ms = self.total() * 1000

        lines = ["[Timing]"]

        for bucket in (
            "YOLO",
            "Crop",
            "Preprocess",
            "OCR",
            "Scoring",
        ):
            lines.append(
                f"{bucket}: "
                f"{self.buckets.get(bucket, 0.0) * 1000:.1f} ms"
            )

        lines.append(f"Total: {total_ms:.1f} ms")
        lines.append(
            f"(plates={plate_count}, ocr_passes={ocr_pass_count})"
        )

        if self.detail:
            lines.extend(self.detail)

        logger.info("\n".join(lines))


# ============================================================
# LAZY SINGLETON MODEL STATE
# ============================================================

_yolo_model = None
_ocr_reader = None
_load_error: str | None = None


def _ensure_models_loaded():
    global _yolo_model, _ocr_reader, _load_error

    if _yolo_model is not None and _ocr_reader is not None:
        return

    if _load_error is not None:
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

        _ocr_reader = easyocr.Reader(
            ["en"],
            gpu=False,
        )

    except Exception as exc:
        _load_error = str(exc)
        raise RuntimeError(_load_error) from exc


def models_ready() -> bool:
    return (
        _yolo_model is not None
        and _ocr_reader is not None
    )


# ============================================================
# INDIAN PLATE VALIDATION
# ============================================================

# Flexible patterns.
#
# Examples:
#   MH20EE7597
#   TN07BU5427
#   FAG643
#
# Do NOT use a fixed 3-letter + 3-digit assumption.

INDIAN_PLATE_PATTERNS = [
    re.compile(
        r"^[A-Z]{2}\d{1,2}[A-Z]{1,3}\d{1,4}$"
    ),
    re.compile(
        r"^[A-Z]{2,3}\d{1,4}[A-Z]{0,3}\d{0,4}$"
    ),
]


def clean_text(text: str) -> str:
    text = str(text).upper()

    return re.sub(
        r"[^A-Z0-9]",
        "",
        text,
    )


def indian_plate_score(text: str) -> int:
    s = clean_text(text)

    if not s:
        return 0

    score = 0
    length = len(s)

    # Typical plate length.
    if 8 <= length <= 12:
        score += 35

    elif 6 <= length <= 13:
        score += 15

    # State / region prefix.
    if len(s) >= 2 and s[:2].isalpha():
        score += 25

    # At least two digits.
    if sum(c.isdigit() for c in s) >= 2:
        score += 20

    # Alphabetic characters after prefix.
    if (
        len(s) >= 5
        and sum(c.isalpha() for c in s[2:]) >= 1
    ):
        score += 10

    # Full regex match.
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


def status_from_confidence(
    final_confidence: float,
    ocr_text: str,
) -> str:

    if not ocr_text:
        return "OCR_FAILED"

    if final_confidence >= 0.80:
        return "HIGH_CONFIDENCE"

    if final_confidence >= 0.50:
        return "REVIEW"

    return "LOW_CONFIDENCE"


# ============================================================
# OCR PREPROCESSING
# ============================================================

# Keep the bounded OCR canvas.
#
# The old implementation used a distorted 1800x600 canvas.
# This preserves aspect ratio and limits CPU work.

_OCR_MAX_DIM = 1100

# Don't excessively enlarge tiny crops.
_OCR_MAX_UPSCALE = 6.0


# ============================================================
# ADAPTIVE OCR THRESHOLDS
# ============================================================

# Strong result:
#
#   OCR confidence >= 0.55
#   AND
#   Indian plate format >= 45
#
# stops the cascade immediately.

_STRONG_OCR_CONF = 0.55
_STRONG_FORMAT_SCORE = 45


# A second condition is intentionally more conservative:
#
# If EasyOCR returns the same text multiple times during one pass,
# and that text has a strong plate structure, we don't necessarily
# need to spend another 2-3 seconds on another OCR pass.
#
# This is especially useful on clean plates where EasyOCR may return
# duplicate text regions.

_CONSENSUS_OCR_CONF = 0.40
_CONSENSUS_FORMAT_SCORE = 60
_MIN_CONSENSUS_VOTES = 2


def _pad_and_scale(crop: np.ndarray) -> np.ndarray:
    """
    Add small replicate padding and resize while preserving aspect ratio.
    """

    h, w = crop.shape[:2]

    border_x = max(
        6,
        int(w * 0.03),
    )

    border_y = max(
        6,
        int(h * 0.10),
    )

    padded = cv2.copyMakeBorder(
        crop,
        border_y,
        border_y,
        border_x,
        border_x,
        cv2.BORDER_REPLICATE,
    )

    ph, pw = padded.shape[:2]

    longer_side = max(
        ph,
        pw,
    )

    if longer_side < _OCR_MAX_DIM:
        scale = min(
            _OCR_MAX_DIM / longer_side,
            _OCR_MAX_UPSCALE,
        )
    else:
        scale = (
            _OCR_MAX_DIM
            / longer_side
        )

    target_w = max(
        1,
        int(pw * scale),
    )

    target_h = max(
        1,
        int(ph * scale),
    )

    interp = (
        cv2.INTER_CUBIC
        if scale >= 1
        else cv2.INTER_AREA
    )

    return cv2.resize(
        padded,
        (target_w, target_h),
        interpolation=interp,
    )


def _variant_primary(
    base_gray: np.ndarray,
) -> np.ndarray:
    """
    Tier 1 — CLAHE-enhanced grayscale.
    """

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8),
    )

    return clahe.apply(base_gray)


def _variant_fallback(
    enhanced_gray: np.ndarray,
) -> np.ndarray:
    """
    Tier 2 — OTSU binarization.
    """

    _, otsu = cv2.threshold(
        enhanced_gray,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )

    return otsu


def _variant_tertiary(
    enhanced_gray: np.ndarray,
) -> np.ndarray:
    """
    Tier 3 — bilateral filtering + OTSU.

    Last resort for genuinely difficult crops.
    """

    denoised = cv2.bilateralFilter(
        enhanced_gray,
        7,
        45,
        45,
    )

    _, otsu = cv2.threshold(
        denoised,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )

    return otsu


# ============================================================
# OCR SCORING
# ============================================================

def _candidate_score(
    text: str,
    ocr_conf: float,
) -> float:
    """Plate-aware OCR score; structure must outweigh confidence alone."""
    text = clean_text(text)
    if not text:
        return 0.0

    format_score = indian_plate_score(text)
    length = len(text)

    if 8 <= length <= 12:
        length_score = 20.0
    elif 6 <= length <= 13:
        length_score = 8.0
    else:
        length_score = 0.0

    return (
        float(ocr_conf) * 35.0
        + format_score * 0.90
        + length_score
    )


# ============================================================
# OCR
# ============================================================

def _run_ocr(
    image: np.ndarray,
) -> list[tuple[str, float]]:
    """
    Run exactly one EasyOCR readtext() pass.

    IMPORTANT:
    We intentionally use readtext() rather than recognize().
    """

    try:
        results = _ocr_reader.readtext(
            image,
            detail=1,
            paragraph=False,
            allowlist=(
                "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                "0123456789"
            ),
        )

    except Exception:
        return []

    candidates = []

    for _, text, conf in results:

        cleaned = clean_text(text)

        if cleaned:
            candidates.append(
                (
                    cleaned,
                    float(conf),
                )
            )

    return candidates


# ============================================================
# CONSENSUS
# ============================================================

def _sequence_similarity(a: str, b: str) -> float:
    """Simple normalized character-position similarity."""
    a = clean_text(a)
    b = clean_text(b)

    if not a or not b:
        return 0.0

    common = min(len(a), len(b))
    positional = sum(
        1
        for i in range(common)
        if a[i] == b[i]
    )

    length_penalty = abs(len(a) - len(b))
    return max(
        0.0,
        (positional - length_penalty * 0.5)
        / max(len(a), len(b)),
    )


def _log_ocr_candidates(
    plate_index: int,
    tier_name: str,
    candidates: list[tuple[str, float]],
) -> None:
    """Log OCR candidates for debugging without affecting inference."""
    if not candidates:
        logger.info(
            "plate%d %s: no candidates",
            plate_index,
            tier_name,
        )
        return

    formatted = ", ".join(
        f"{text} ({conf:.3f})"
        for text, conf in candidates
    )

    logger.info(
        "plate%d %s candidates: %s",
        plate_index,
        tier_name,
        formatted,
    )


def _evidence_select(
    tier_candidates: list[tuple[int, list[tuple[str, float]]]],
):
    """Select the most plausible plate using structure, confidence and agreement."""
    all_candidates: list[tuple[str, float, int]] = []

    for tier_index, candidates in tier_candidates:
        for text, conf in candidates:
            text = clean_text(text)
            if text and len(text) >= 6:
                all_candidates.append((text, float(conf), tier_index))

    if not all_candidates:
        return None

    groups: dict[str, list[tuple[float, int]]] = defaultdict(list)
    for text, conf, tier in all_candidates:
        groups[text].append((conf, tier))

    scored = []
    for text, observations in groups.items():
        confs = [c for c, _ in observations]
        avg_conf = sum(confs) / len(confs)
        max_conf = max(confs)
        votes = len(observations)
        format_score = indian_plate_score(text)

        primary_conf = max(
            (c for c, tier in observations if tier == 1),
            default=0.0,
        )
        agreement_bonus = min(votes - 1, 2) * 7.0
        prefix_bonus = 25.0 if len(text) >= 2 and text[:2].isalpha() else 0.0
        length_bonus = (
            20.0 if 8 <= len(text) <= 12
            else 8.0 if 6 <= len(text) <= 13
            else 0.0
        )

        score = (
            max_conf * 35.0
            + avg_conf * 15.0
            + format_score * 0.90
            + prefix_bonus
            + length_bonus
            + agreement_bonus
            + primary_conf * 8.0
        )

        scored.append((text, avg_conf, votes, score, format_score))

    scored.sort(key=lambda item: item[3], reverse=True)
    best_text, best_conf, best_votes, best_score, _ = scored[0]

    for text, avg_conf, votes, score, _ in scored[1:]:
        if _sequence_similarity(best_text, text) >= 0.70 and score > best_score + 5.0:
            best_text, best_conf, best_votes, best_score = text, avg_conf, votes, score

    return best_text, best_conf, best_votes


def _consensus_select(
    candidates: list[tuple[str, float]],
):
    """Select the best candidate from one OCR pass using plate structure."""
    if not candidates:
        return None

    groups: dict[str, list[float]] = defaultdict(list)

    for text, conf in candidates:
        text = clean_text(text)
        if not text or len(text) < 6:
            continue
        groups[text].append(float(conf))

    if not groups:
        return None

    scored = []
    for text, confs in groups.items():
        votes = len(confs)
        avg_conf = sum(confs) / votes
        format_score = indian_plate_score(text)
        consensus_bonus = min(votes - 1, 2) * 7.0

        score = (
            avg_conf * 35.0
            + format_score * 0.90
            + (20.0 if 8 <= len(text) <= 12 else 8.0 if 6 <= len(text) <= 13 else 0.0)
            + consensus_bonus
        )

        scored.append((text, avg_conf, votes, score))

    return max(scored, key=lambda item: item[3])[:3]


# ============================================================
# ADAPTIVE ESCALATION
# ============================================================

def _is_strong_enough(candidate) -> bool:
    """Stop OCR when the candidate is both readable and plate-shaped."""
    if candidate is None:
        return False

    text, conf, votes = candidate
    text = clean_text(text)
    if not text or len(text) < 6:
        return False

    format_score = indian_plate_score(text)

    if conf >= 0.55 and format_score >= 45:
        return True

    if 0.45 <= conf < 0.55 and format_score >= 60 and len(text) <= 13:
        return True

    if votes >= 2 and conf >= 0.40 and format_score >= 60:
        return True

    return False


# ============================================================
# MAIN ENTRY POINT
# ============================================================

def run_detection_on_image(
    image_bytes: bytes,
    conf_threshold: float = 0.25,
) -> dict[str, Any]:
    """
    Run:

        image
          ↓
        YOLO
          ↓
        crop
          ↓
        CLAHE OCR
          ↓
        optional OTSU OCR
          ↓
        optional denoise OCR
          ↓
        scoring
          ↓
        annotation

    API response schema remains unchanged.
    """

    # --------------------------------------------------------
    # Lazy model initialization
    # --------------------------------------------------------
    # Keep one-time model startup out of per-image processing latency.

    _ensure_models_loaded()

    timing = _TimingBucket()

    # --------------------------------------------------------
    # Decode image
    # --------------------------------------------------------

    np_arr = np.frombuffer(
        image_bytes,
        np.uint8,
    )

    image = cv2.imdecode(
        np_arr,
        cv2.IMREAD_COLOR,
    )

    if image is None:
        raise ValueError(
            "Could not decode image. "
            "Is it a valid image file?"
        )

    h, w = image.shape[:2]

    # --------------------------------------------------------
    # YOLO detection
    # --------------------------------------------------------

    t0 = time.perf_counter()

    results = _yolo_model.predict(
        source=image,
        conf=conf_threshold,
        verbose=False,
    )

    timing.add(
        "YOLO",
        time.perf_counter() - t0,
    )

    result = results[0]

    detections = []

    if result.boxes is not None:

        for box in result.boxes:

            confidence = float(
                box.conf[0]
            )

            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0].tolist(),
            )

            detections.append(
                (
                    confidence,
                    x1,
                    y1,
                    x2,
                    y2,
                )
            )

    # Highest confidence first.
    detections.sort(
        key=lambda d: d[0],
        reverse=True,
    )

    plates = []

    annotated = image.copy()

    ocr_pass_count = 0

    # ========================================================
    # PROCESS EVERY DETECTED PLATE
    # ========================================================

    for idx, (
        yolo_conf,
        x1,
        y1,
        x2,
        y2,
    ) in enumerate(
        detections,
        start=1,
    ):

        # ----------------------------------------------------
        # Clamp bounding box
        # ----------------------------------------------------

        x1c = max(
            0,
            x1,
        )

        y1c = max(
            0,
            y1,
        )

        x2c = min(
            w,
            x2,
        )

        y2c = min(
            h,
            y2,
        )

        crop = image[
            y1c:y2c,
            x1c:x2c,
        ]

        # ----------------------------------------------------
        # Default plate result
        # ----------------------------------------------------

        plate_entry = {
            "plate_id": idx,
            "bbox": [
                x1c,
                y1c,
                x2c,
                y2c,
            ],
            "yolo_confidence": round(
                yolo_conf,
                4,
            ),
            "ocr_text": "",
            "ocr_confidence": 0.0,
            "validation_score": 0,
            "validation": "UNVERIFIED",
            "final_confidence": 0.0,
            "status": "OCR_FAILED",
        }

        # ====================================================
        # OCR
        # ====================================================

        if crop.size > 0:

            # ------------------------------------------------
            # Crop / resize / grayscale
            # ------------------------------------------------

            t0 = time.perf_counter()

            base = _pad_and_scale(
                crop
            )

            base_gray = cv2.cvtColor(
                base,
                cv2.COLOR_BGR2GRAY,
            )

            timing.add(
                "Crop",
                time.perf_counter() - t0,
            )

            # =================================================
            # TIER 1 — CLAHE
            # =================================================

            tier_evidence: list[tuple[int, list[tuple[str, float]]]] = []

            t0 = time.perf_counter()

            enhanced = _variant_primary(
                base_gray
            )

            timing.add(
                "Preprocess",
                time.perf_counter() - t0,
            )

            t0 = time.perf_counter()

            tier1_candidates = _run_ocr(
                enhanced
            )

            ocr_pass_count += 1

            timing.add(
                "OCR",
                time.perf_counter() - t0,
                note=f"plate{idx} tier1(CLAHE)",
            )

            _log_ocr_candidates(
                idx,
                "tier1(CLAHE)",
                tier1_candidates,
            )

            tier_evidence.append(
                (1, tier1_candidates)
            )

            # Use the current tier for the stopping decision.
            t0 = time.perf_counter()

            best = _consensus_select(
                tier1_candidates
            )

            timing.add(
                "Scoring",
                time.perf_counter() - t0,
            )

            # =================================================
            # TIER 2 — OTSU FALLBACK
            # =================================================

            if not _is_strong_enough(best):

                t0 = time.perf_counter()

                fallback_img = _variant_fallback(
                    enhanced
                )

                timing.add(
                    "Preprocess",
                    time.perf_counter() - t0,
                )

                t0 = time.perf_counter()

                tier2_candidates = _run_ocr(
                    fallback_img
                )

                ocr_pass_count += 1

                timing.add(
                    "OCR",
                    time.perf_counter() - t0,
                    note=f"plate{idx} tier2(OTSU)",
                )

                _log_ocr_candidates(
                    idx,
                    "tier2(OTSU)",
                    tier2_candidates,
                )

                tier_evidence.append(
                    (2, tier2_candidates)
                )

                t0 = time.perf_counter()

                best = _evidence_select(
                    tier_evidence
                )

                timing.add(
                    "Scoring",
                    time.perf_counter() - t0,
                )

                # =================================================
                # TIER 3 — DENOISE LAST RESORT
                # =================================================

                # Only run Tier 3 if the combined evidence is still weak.
                if best is None or not _is_strong_enough(best):

                    t0 = time.perf_counter()

                    tertiary_img = _variant_tertiary(
                        enhanced
                    )

                    timing.add(
                        "Preprocess",
                        time.perf_counter() - t0,
                    )

                    t0 = time.perf_counter()

                    tier3_candidates = _run_ocr(
                        tertiary_img
                    )

                    ocr_pass_count += 1

                    timing.add(
                        "OCR",
                        time.perf_counter() - t0,
                        note=f"plate{idx} tier3(denoise)",
                    )

                    _log_ocr_candidates(
                        idx,
                        "tier3(denoise)",
                        tier3_candidates,
                    )

                    tier_evidence.append(
                        (3, tier3_candidates)
                    )

                    t0 = time.perf_counter()

                    best = _evidence_select(
                        tier_evidence
                    )

                    timing.add(
                        "Scoring",
                        time.perf_counter() - t0,
                    )

            # Final evidence-based selection.
            if tier_evidence:
                t0 = time.perf_counter()

                evidence_best = _evidence_select(
                    tier_evidence
                )

                if evidence_best is not None:
                    best = evidence_best

                timing.add(
                    "Scoring",
                    time.perf_counter() - t0,
                )

            # =================================================
            # FINAL PLATE SCORING
            # =================================================

            if best is not None:

                (
                    best_text,
                    best_ocr_conf,
                    best_votes,
                ) = best

                validation_score = indian_plate_score(
                    best_text
                )

                validation = classify_validation(
                    validation_score
                )

                final_confidence = min(
                    1.0,
                    best_ocr_conf * 0.6
                    + (
                        validation_score
                        / 100
                    ) * 0.4,
                )

                plate_entry.update(
                    {
                        "ocr_text": best_text,
                        "ocr_confidence": round(
                            best_ocr_conf,
                            4,
                        ),
                        "validation_score": validation_score,
                        "validation": validation,
                        "final_confidence": round(
                            final_confidence,
                            4,
                        ),
                        "status": status_from_confidence(
                            final_confidence,
                            best_text,
                        ),
                    }
                )

                # -------------------------------------------------
                # Debug logging
                # -------------------------------------------------

                if os.environ.get(
                    "LVA_DEBUG_TIMING",
                    "1",
                ) != "0":

                    logger.info(
                        "  plate%d consensus: "
                        "text=%r votes=%d "
                        "avg_conf=%.3f format=%d",
                        idx,
                        best_text,
                        best_votes,
                        best_ocr_conf,
                        validation_score,
                    )

        # ====================================================
        # SAVE PLATE
        # ====================================================

        plates.append(
            plate_entry
        )

        # ====================================================
        # ANNOTATION
        # ====================================================

        if plate_entry["ocr_text"]:

            color = (
                0,
                200,
                100,
            )

        else:

            color = (
                0,
                140,
                255,
            )

        cv2.rectangle(
            annotated,
            (
                x1c,
                y1c,
            ),
            (
                x2c,
                y2c,
            ),
            color,
            3,
        )

        label = (
            f"{plate_entry['ocr_text'] or 'UNKNOWN'}"
            f" | "
            f"{plate_entry['final_confidence'] * 100:.0f}%"
        )

        cv2.putText(
            annotated,
            label,
            (
                x1c,
                max(
                    25,
                    y1c - 10,
                ),
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
            cv2.LINE_AA,
        )

    # ========================================================
    # ENCODE ANNOTATED IMAGE
    # ========================================================

    ok, buf = cv2.imencode(
        ".jpg",
        annotated,
    )

    annotated_b64 = None

    if ok:

        import base64

        annotated_b64 = base64.b64encode(
            buf.tobytes()
        ).decode(
            "ascii"
        )

    # ========================================================
    # TIMING
    # ========================================================

    elapsed = timing.total()

    timing.report(
        plate_count=len(plates),
        ocr_pass_count=ocr_pass_count,
    )

    # ========================================================
    # API RESPONSE
    # ========================================================

    return {
        "image": {
            "width": w,
            "height": h,
        },
        "plates_detected": len(plates),
        "plates": plates,
        "annotated_image_base64": annotated_b64,
        "processing_time_seconds": round(
            elapsed,
            3,
        ),
    }