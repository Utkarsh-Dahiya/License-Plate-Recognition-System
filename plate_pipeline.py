"""
LICENSE VISION AI
YOLO + EasyOCR
Indian License Plate Detection + OCR Pipeline

Version: 2.0
"""

from pathlib import Path
from collections import Counter
import json
import re

import cv2
import numpy as np
from ultralytics import YOLO
import easyocr


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent

MODEL_PATH = (
    ROOT
    / "runs"
    / "detect"
    / "models"
    / "license_plate_detector"
    / "weights"
    / "best.pt"
)

INPUT_IMAGE = (
    ROOT
    / "runs"
    / "detect"
    / "predict-2"
    / "bbcac63e32bd8137_jpg.rf.ef4704b0ada4fbbf613143abf52f6f86.jpg"
)

OUTPUT_DIR = ROOT / "final_pipeline_easyocr"
OUTPUT_DIR.mkdir(exist_ok=True)

YOLO_CONF = 0.25

# ============================================================
# INDIAN LICENSE PLATE PATTERNS
# ============================================================

INDIAN_PLATE_PATTERNS = [
    # MH20EE7597
    re.compile(r"^[A-Z]{2}\d{1,2}[A-Z]{1,3}\d{1,4}$"),

    # DL3CAY9324
    re.compile(r"^[A-Z]{2}\d{1,2}[A-Z]{1,3}\d{1,4}$"),

    # KA03AB1234
    re.compile(r"^[A-Z]{2}\d{1,2}[A-Z]{1,3}\d{1,4}$"),

    # Generic Indian-style fallback
    re.compile(r"^[A-Z]{2,3}\d{1,4}[A-Z]{0,3}\d{0,4}$"),
]


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(text):
    """
    Normalize OCR output.

    Removes spaces, punctuation and special characters.
    """

    text = str(text).upper()

    text = text.replace(" ", "")
    text = text.replace("-", "")
    text = text.replace("_", "")
    text = text.replace(".", "")

    text = re.sub(r"[^A-Z0-9]", "", text)

    return text


# ============================================================
# OCR CHARACTER NORMALIZATION
# ============================================================

def normalize_ocr_characters(text):
    """
    Apply conservative OCR corrections.

    We only make substitutions when the character appears
    suspicious and when the resulting plate structure improves.
    """

    s = clean_text(text)

    if not s:
        return ""

    candidates = [s]

    # Common OCR confusions.
    substitutions = {
        "O": "0",
        "Q": "0",
        "I": "1",
        "L": "1",
        "Z": "2",
        "S": "5",
        "G": "6",
        "B": "8",
    }

    # Create a numeric-biased candidate.
    numeric_candidate = ""

    for c in s:
        numeric_candidate += substitutions.get(c, c)

    candidates.append(numeric_candidate)

    # Also create an alpha-biased candidate.
    alpha_map = {
        "0": "O",
        "1": "I",
        "2": "Z",
        "5": "S",
        "6": "G",
        "8": "B",
    }

    alpha_candidate = ""

    for c in s:
        alpha_candidate += alpha_map.get(c, c)

    candidates.append(alpha_candidate)

    return candidates[0]


# ============================================================
# INDIAN PLATE VALIDATION
# ============================================================

def indian_plate_score(text):
    """
    Score OCR result based on common Indian registration structure.

    This does NOT assume one exact plate length.
    """

    s = clean_text(text)

    if not s:
        return 0

    score = 0

    length = len(s)

    # Typical Indian registration strings are generally 8-12 chars.
    if 8 <= length <= 12:
        score += 35

    elif 6 <= length <= 13:
        score += 15

    # Starts with state/UT code.
    if len(s) >= 2 and s[:2].isalpha():
        score += 25

    # Registration number contains digits.
    digit_count = sum(c.isdigit() for c in s)

    if digit_count >= 2:
        score += 20

    # Contains letters after state code.
    if len(s) >= 5:
        middle_letters = sum(c.isalpha() for c in s[2:])

        if middle_letters >= 1:
            score += 10

    # Test actual patterns.
    for pattern in INDIAN_PLATE_PATTERNS:
        if pattern.match(s):
            score += 35
            break

    return min(score, 100)


# ============================================================
# PLATE FORMAT CLASSIFICATION
# ============================================================

def classify_plate(text):
    """
    Human-readable plate classification.
    """

    s = clean_text(text)

    if not s:
        return "UNKNOWN"

    score = indian_plate_score(s)

    if score >= 75:
        return "INDIAN_PLATE"

    if score >= 45:
        return "POSSIBLE_INDIAN_PLATE"

    if len(s) >= 4:
        return "UNVERIFIED"

    return "LOW_INFORMATION"


# ============================================================
# CROP
# ============================================================

def crop_from_bbox(image, box, pad_x=0.02, pad_y=0.05):

    x1, y1, x2, y2 = box

    h, w = image.shape[:2]

    bw = x2 - x1
    bh = y2 - y1

    x1 = max(0, int(x1 - bw * pad_x))
    y1 = max(0, int(y1 - bh * pad_y))

    x2 = min(w, int(x2 + bw * pad_x))
    y2 = min(h, int(y2 + bh * pad_y))

    return image[y1:y2, x1:x2].copy()


# ============================================================
# OCR VARIANTS
# ============================================================

def make_variants(crop):

    variants = {}

    h, w = crop.shape[:2]

    border_x = max(6, int(w * 0.03))
    border_y = max(6, int(h * 0.10))

    padded = cv2.copyMakeBorder(
        crop,
        border_y,
        border_y,
        border_x,
        border_x,
        cv2.BORDER_REPLICATE,
    )

    target_w = 1800
    target_h = 600

    base = cv2.resize(
        padded,
        (target_w, target_h),
        interpolation=cv2.INTER_CUBIC,
    )

    gray = cv2.cvtColor(base, cv2.COLOR_BGR2GRAY)

    variants["color"] = base

    variants["gray"] = cv2.cvtColor(
        gray,
        cv2.COLOR_GRAY2BGR,
    )

    # CLAHE
    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8),
    )

    enhanced = clahe.apply(gray)

    variants["clahe"] = cv2.cvtColor(
        enhanced,
        cv2.COLOR_GRAY2BGR,
    )

    # Bilateral denoise
    denoised = cv2.bilateralFilter(
        enhanced,
        7,
        45,
        45,
    )

    variants["denoised"] = cv2.cvtColor(
        denoised,
        cv2.COLOR_GRAY2BGR,
    )

    # Sharpen
    kernel = np.array(
        [
            [0, -1, 0],
            [-1, 5, -1],
            [0, -1, 0],
        ],
        dtype=np.float32,
    )

    sharpened = cv2.filter2D(
        denoised,
        -1,
        kernel,
    )

    variants["sharpened"] = cv2.cvtColor(
        sharpened,
        cv2.COLOR_GRAY2BGR,
    )

    # OTSU
    _, otsu = cv2.threshold(
        enhanced,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )

    variants["otsu"] = cv2.cvtColor(
        otsu,
        cv2.COLOR_GRAY2BGR,
    )

    # Adaptive threshold
    adaptive = cv2.adaptiveThreshold(
        enhanced,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        7,
    )

    variants["adaptive"] = cv2.cvtColor(
        adaptive,
        cv2.COLOR_GRAY2BGR,
    )

    return variants


# ============================================================
# OCR CANDIDATE SCORING
# ============================================================

def candidate_score(text, ocr_conf):

    text = clean_text(text)

    if not text:
        return 0

    format_score = indian_plate_score(text)

    length_score = 0

    if 8 <= len(text) <= 12:
        length_score = 20

    elif 6 <= len(text) <= 13:
        length_score = 8

    confidence_score = float(ocr_conf) * 45

    score = (
        confidence_score
        + format_score * 0.65
        + length_score
    )

    return score


# ============================================================
# MAIN
# ============================================================

print("=" * 78)
print("LICENSE VISION AI — YOLO + EASYOCR")
print("INDIAN LICENSE PLATE OCR PIPELINE")
print("=" * 78)


# ============================================================
# VALIDATION
# ============================================================

if not MODEL_PATH.exists():

    raise FileNotFoundError(
        f"""
YOLO model not found:

{MODEL_PATH}

Expected:

runs\\detect\\models\\license_plate_detector\\weights\\best.pt
"""
    )


if not INPUT_IMAGE.exists():

    raise FileNotFoundError(
        f"""
Input image not found:

{INPUT_IMAGE}
"""
    )


# ============================================================
# LOAD IMAGE
# ============================================================

image = cv2.imread(str(INPUT_IMAGE))

if image is None:

    raise RuntimeError(
        "OpenCV could not read the input image."
    )


h, w = image.shape[:2]

print()
print(f"Input image : {INPUT_IMAGE.name}")
print(f"Resolution  : {w} x {h}")


# ============================================================
# YOLO
# ============================================================

print("\n" + "=" * 78)
print("YOLO LICENSE PLATE DETECTION")
print("=" * 78)

model = YOLO(str(MODEL_PATH))

result = model.predict(
    source=image,
    conf=YOLO_CONF,
    verbose=False,
)[0]


if result.boxes is None or len(result.boxes) == 0:

    raise RuntimeError(
        "YOLO did not detect any license plate."
    )


boxes = result.boxes.xyxy.cpu().numpy()

confs = result.boxes.conf.cpu().numpy()


print(f"Detected plates : {len(boxes)}")


# ============================================================
# OCR READER
# ============================================================

print("\n" + "=" * 78)
print("INITIALIZING EASYOCR")
print("=" * 78)

reader = easyocr.Reader(
    ["en"],
    gpu=False,
    verbose=False,
)


# ============================================================
# PROCESS EVERY PLATE
# ============================================================

plate_results = []

detection_image = image.copy()


for plate_index, (box, yolo_conf) in enumerate(
    zip(boxes, confs),
    start=1,
):

    x1, y1, x2, y2 = map(int, box)

    print()
    print("-" * 78)
    print(
        f"PLATE #{plate_index} | "
        f"YOLO confidence: {yolo_conf * 100:.2f}%"
    )

    cv2.rectangle(
        detection_image,
        (x1, y1),
        (x2, y2),
        (255, 0, 0),
        3,
    )

    # --------------------------------------------------------
    # MULTIPLE CROPS
    # --------------------------------------------------------

    crop_specs = [
        ("tight", 0.00, 0.00),
        ("small", 0.015, 0.035),
        ("medium", 0.03, 0.06),
        ("wide", 0.05, 0.08),
    ]

    all_candidates = []

    for crop_name, px, py in crop_specs:

        crop = crop_from_bbox(
            image,
            (x1, y1, x2, y2),
            px,
            py,
        )

        if crop.size == 0:
            continue

        crop_dir = (
            OUTPUT_DIR
            / f"plate_{plate_index}"
        )

        crop_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        cv2.imwrite(
            str(
                crop_dir
                / f"{crop_name}.jpg"
            ),
            crop,
        )

        variants = make_variants(crop)

        for variant_name, variant in variants.items():

            detections = reader.readtext(
                variant,
                detail=1,
                paragraph=False,
                allowlist=(
                    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                    "0123456789"
                ),
                width_ths=0.25,
                height_ths=0.25,
                mag_ratio=1.0,
                text_threshold=0.25,
                low_text=0.15,
                link_threshold=0.15,
            )

            for bbox_ocr, text, conf in detections:

                raw = clean_text(text)

                if len(raw) < 3:
                    continue

                normalized = normalize_ocr_characters(raw)

                score = candidate_score(
                    normalized,
                    conf,
                )

                all_candidates.append(
                    {
                        "text": normalized,
                        "raw": raw,
                        "confidence": float(conf),
                        "score": float(score),
                        "crop": crop_name,
                        "variant": variant_name,
                        "bbox": bbox_ocr,
                    }
                )


    # ========================================================
    # CONSENSUS
    # ========================================================

    if not all_candidates:

        print("OCR: No text detected.")

        plate_results.append(
            {
                "plate_id": plate_index,
                "bbox": [
                    int(x1),
                    int(y1),
                    int(x2),
                    int(y2),
                ],
                "yolo_confidence": float(yolo_conf),
                "ocr_text": "",
                "ocr_confidence": 0.0,
                "validation_score": 0,
                "validation": "OCR_FAILED",
                "status": "OCR_FAILED",
                "candidate_count": 0,
            }
        )

        continue


    # Group OCR candidates by text
    groups = {}

    for candidate in all_candidates:

        text = candidate["text"]

        if text not in groups:

            groups[text] = []

        groups[text].append(candidate)


    ranked = []


    for text, items in groups.items():

        votes = len(items)

        avg_conf = np.mean(
            [
                item["confidence"]
                for item in items
            ]
        )

        avg_score = np.mean(
            [
                item["score"]
                for item in items
            ]
        )

        validation_score = indian_plate_score(text)

        # Consensus bonus
        consensus_bonus = min(
            votes,
            10
        ) * 8

        final_score = (
            avg_score
            + consensus_bonus
            + validation_score * 0.50
        )

        ranked.append(
            {
                "text": text,
                "votes": votes,
                "avg_confidence": float(
                    avg_conf
                ),
                "validation_score": validation_score,
                "score": float(final_score),
                "items": items,
            }
        )


    ranked.sort(
        key=lambda x: x["score"],
        reverse=True,
    )


    winner = ranked[0]

    final_text = winner["text"]

    ocr_conf = winner[
        "avg_confidence"
    ]

    validation_score = winner[
        "validation_score"
    ]

    votes = winner["votes"]

    validation = classify_plate(
        final_text
    )


    # ========================================================
    # FINAL CONFIDENCE
    # ========================================================

    consensus = min(
        votes / 5.0,
        1.0
    )

    final_confidence = (
        0.45 * float(yolo_conf)
        + 0.35 * float(ocr_conf)
        + 0.20 * consensus
    )

    # Validation adjustment
    if validation_score >= 75:

        final_confidence += 0.08

    elif validation_score < 30:

        final_confidence -= 0.08


    final_confidence = max(
        0.0,
        min(
            final_confidence,
            0.99,
        ),
    )


    if final_confidence >= 0.75:

        status = "HIGH_CONFIDENCE"

    elif final_confidence >= 0.50:

        status = "MEDIUM_CONFIDENCE"

    else:

        status = "LOW_CONFIDENCE"


    # ========================================================
    # PRINT RESULT
    # ========================================================

    print(
        f"OCR result      : {final_text}"
    )

    print(
        f"OCR confidence  : "
        f"{ocr_conf * 100:.2f}%"
    )

    print(
        f"Consensus votes : {votes}"
    )

    print(
        f"Indian format   : "
        f"{validation_score}/100"
    )

    print(
        f"Validation      : "
        f"{validation}"
    )

    print(
        f"Final confidence: "
        f"{final_confidence * 100:.2f}%"
    )

    print(
        f"Status          : "
        f"{status}"
    )


    # ========================================================
    # SAVE RESULT
    # ========================================================

    plate_results.append(
        {
            "plate_id": plate_index,

            "bbox": [
                int(x1),
                int(y1),
                int(x2),
                int(y2),
            ],

            "yolo_confidence": float(
                yolo_conf
            ),

            "ocr_text": final_text,

            "ocr_confidence": float(
                ocr_conf
            ),

            "ocr_votes": int(
                votes
            ),

            "validation_score": int(
                validation_score
            ),

            "validation": validation,

            "final_confidence": float(
                final_confidence
            ),

            "status": status,

            "candidate_count": len(
                all_candidates
            ),

            "top_candidates": [
                {
                    "text": r["text"],
                    "votes": r["votes"],
                    "confidence": r[
                        "avg_confidence"
                    ],
                    "validation_score": r[
                        "validation_score"
                    ],
                    "score": r["score"],
                }

                for r in ranked[:5]
            ],
        }
    )


    # ========================================================
    # DRAW FINAL RESULT
    # ========================================================

    color = (
        0,
        255,
        0,
    )

    cv2.rectangle(
        detection_image,
        (x1, y1),
        (x2, y2),
        color,
        3,
    )

    label = (
        f"{final_text} "
        f"| {final_confidence * 100:.0f}%"
    )

    label_y = max(
        25,
        y1 - 10,
    )

    cv2.putText(
        detection_image,
        label,
        (x1, label_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        color,
        2,
        cv2.LINE_AA,
    )


# ============================================================
# SAVE DETECTION IMAGE
# ============================================================

cv2.imwrite(
    str(
        OUTPUT_DIR
        / "final_detection.jpg"
    ),
    detection_image,
)


# ============================================================
# OVERALL SUMMARY
# ============================================================

successful = [
    p
    for p in plate_results
    if p["ocr_text"]
]

failed = [
    p
    for p in plate_results
    if not p["ocr_text"]
]


if successful:

    mean_ocr_conf = float(
        np.mean(
            [
                p["ocr_confidence"]
                for p in successful
            ]
        )
    )

    mean_final_conf = float(
        np.mean(
            [
                p["final_confidence"]
                for p in successful
            ]
        )
    )

else:

    mean_ocr_conf = 0.0
    mean_final_conf = 0.0


mean_yolo_conf = float(
    np.mean(confs)
)


# ============================================================
# JSON FOR DASHBOARD
# ============================================================

dashboard_data = {

    "project": "License Vision AI",

    "version": "2.0",

    "input": {
        "filename": INPUT_IMAGE.name,
        "width": int(w),
        "height": int(h),
    },

    "model": {
        "name": "YOLO License Plate Detector",
        "path": str(MODEL_PATH),
    },

    "summary": {

        "plates_detected": len(
            plate_results
        ),

        "ocr_success": len(
            successful
        ),

        "ocr_failed": len(
            failed
        ),

        "mean_yolo_confidence": mean_yolo_conf,

        "mean_ocr_confidence": mean_ocr_conf,

        "mean_final_confidence": mean_final_conf,

        "ocr_success_rate": (
            len(successful)
            / len(plate_results)
            if plate_results
            else 0
        ),
    },

    "plates": plate_results,

}


with open(
    OUTPUT_DIR / "result.json",
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        dashboard_data,
        f,
        indent=4,
    )


# ============================================================
# TEXT REPORT
# ============================================================

report = []

report.append(
    "LICENSE VISION AI"
)

report.append(
    "YOLO + EASYOCR INDIAN LICENSE PLATE PIPELINE"
)

report.append(
    "=" * 60
)

report.append(
    f"Input image: {INPUT_IMAGE.name}"
)

report.append(
    f"Resolution: {w} x {h}"
)

report.append(
    f"Plates detected: {len(plate_results)}"
)

report.append(
    f"Successful OCR: {len(successful)}"
)

report.append(
    f"OCR failed: {len(failed)}"
)

report.append(
    f"Mean YOLO confidence: "
    f"{mean_yolo_conf * 100:.2f}%"
)

report.append(
    f"Mean OCR confidence: "
    f"{mean_ocr_conf * 100:.2f}%"
)

report.append(
    f"Mean final confidence: "
    f"{mean_final_conf * 100:.2f}%"
)

report.append(
    f"OCR success rate: "
    f"{dashboard_data['summary']['ocr_success_rate'] * 100:.2f}%"
)

report.append("")
report.append(
    "PLATE RESULTS"
)
report.append(
    "-" * 60
)


for p in plate_results:

    report.append(
        f"Plate #{p['plate_id']} | "
        f"{p['ocr_text'] or 'NO TEXT'} | "
        f"YOLO={p['yolo_confidence'] * 100:.1f}% | "
        f"OCR={p['ocr_confidence'] * 100:.1f}% | "
        f"FINAL={p.get('final_confidence', 0) * 100:.1f}% | "
        f"{p['status']}"
    )


save_text = "\n".join(report)


with open(
    OUTPUT_DIR / "final_result.txt",
    "w",
    encoding="utf-8",
) as f:

    f.write(save_text)


# ============================================================
# FINAL CONSOLE
# ============================================================

print()
print("=" * 78)
print("LICENSE VISION AI — FINAL RESULT")
print("=" * 78)

for p in plate_results:

    print(
        f"\nPlate #{p['plate_id']}"
    )

    print(
        f"  Text       : "
        f"{p['ocr_text'] or 'NO TEXT'}"
    )

    print(
        f"  YOLO       : "
        f"{p['yolo_confidence'] * 100:.2f}%"
    )

    print(
        f"  OCR        : "
        f"{p['ocr_confidence'] * 100:.2f}%"
    )

    print(
        f"  Validation : "
        f"{p['validation']}"
    )

    print(
        f"  Final      : "
        f"{p['final_confidence'] * 100:.2f}%"
    )

    print(
        f"  Status     : "
        f"{p['status']}"
    )


print()
print("=" * 78)
print("PIPELINE COMPLETED")
print("=" * 78)

print(
    f"\nResults saved to:\n"
    f"{OUTPUT_DIR}"
)

print(
    "\nImportant files:"
)

print(
    "  final_detection.jpg"
)

print(
    "  result.json"
)

print(
    "  final_result.txt"
)

print("=" * 78)