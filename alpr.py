import cv2
import pytesseract
import re
import time
from pathlib import Path
from collections import Counter
from ultralytics import YOLO


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = "runs/detect/models/license_plate_detector/weights/best.pt"

INPUT_IMAGE = (
    "test_images/test/images/"
    "b193070a9c45b5ab_jpg.rf.57e5987eb896a7bf9fc7a1a96a660c7e.jpg"
)

OUTPUT_DIR = Path("alpr_results")
OUTPUT_DIR.mkdir(exist_ok=True)

PREPROCESS_DIR = OUTPUT_DIR / "preprocessed"
PREPROCESS_DIR.mkdir(exist_ok=True)

TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH


# ============================================================
# OCR CONFIG
# ============================================================

OCR_CONFIGS = [
    "--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
    "--psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
    "--psm 13 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
]


# ============================================================
# OCR CLEANING
# ============================================================

def clean_text(text):

    text = text.upper()

    text = re.sub(
        r"[^A-Z0-9]",
        "",
        text
    )

    return text


# ============================================================
# PLATE FORMAT SCORING
# ============================================================

def plate_format_score(text):

    score = 0

    length = len(text)

    # Indian plates are commonly 7–10 characters
    if 7 <= length <= 10:
        score += 5

    elif 6 <= length <= 11:
        score += 2

    else:
        score -= 5

    has_letter = bool(re.search(r"[A-Z]", text))
    has_digit = bool(re.search(r"[0-9]", text))

    if has_letter:
        score += 2

    if has_digit:
        score += 2

    # Strong structural pattern:
    # XX00XX0000
    if re.search(r"[A-Z]{2}[0-9]{1,2}[A-Z]{1,3}[0-9]{1,4}", text):
        score += 8

    # Common ending with digits
    if re.search(r"[0-9]{3,4}$", text):
        score += 4

    return score


# ============================================================
# OCR PROCESSING
# ============================================================

def run_ocr(plate):

    # --------------------------------------------------------
    # UPSCALE
    # --------------------------------------------------------

    upscaled = cv2.resize(
        plate,
        None,
        fx=6,
        fy=6,
        interpolation=cv2.INTER_CUBIC
    )

    # --------------------------------------------------------
    # GRAYSCALE
    # --------------------------------------------------------

    gray = cv2.cvtColor(
        upscaled,
        cv2.COLOR_BGR2GRAY
    )

    # --------------------------------------------------------
    # CLAHE
    # --------------------------------------------------------

    clahe = cv2.createCLAHE(
        clipLimit=3.0,
        tileGridSize=(8, 8)
    )

    enhanced = clahe.apply(gray)

    # --------------------------------------------------------
    # DENOISE
    # --------------------------------------------------------

    denoised = cv2.fastNlMeansDenoising(
        enhanced,
        None,
        10,
        7,
        21
    )

    # --------------------------------------------------------
    # OTSU
    # --------------------------------------------------------

    blur = cv2.GaussianBlur(
        denoised,
        (3, 3),
        0
    )

    _, otsu = cv2.threshold(
        blur,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # --------------------------------------------------------
    # ADAPTIVE
    # --------------------------------------------------------

    adaptive = cv2.adaptiveThreshold(
        denoised,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        7
    )

    # --------------------------------------------------------
    # SHARPEN
    # --------------------------------------------------------

    sharpened = cv2.addWeighted(
        enhanced,
        2.0,
        cv2.GaussianBlur(
            enhanced,
            (0, 0),
            3
        ),
        -1.0,
        0
    )

    # --------------------------------------------------------
    # MORPHOLOGICAL CLOSE
    # --------------------------------------------------------

    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (3, 3)
    )

    closed = cv2.morphologyEx(
        otsu,
        cv2.MORPH_CLOSE,
        kernel
    )

    variants = {
        "upscaled": upscaled,
        "gray": gray,
        "enhanced": enhanced,
        "denoised": denoised,
        "otsu": otsu,
        "adaptive": adaptive,
        "sharpened": sharpened,
        "closed": closed,
    }

    candidates = []

    print()
    print("OCR PROCESSING")
    print("-" * 70)

    for name, image in variants.items():

        save_path = PREPROCESS_DIR / f"{name}.jpg"

        cv2.imwrite(
            str(save_path),
            image
        )

        for config in OCR_CONFIGS:

            raw = pytesseract.image_to_string(
                image,
                config=config
            )

            cleaned = clean_text(raw)

            if cleaned:

                candidates.append(cleaned)

                print(
                    f"{name:12} -> {cleaned}"
                )

    return candidates


# ============================================================
# SMART OCR SELECTION
# ============================================================

def select_best_candidate(candidates):

    if not candidates:
        return "", 0, 0

    counter = Counter(candidates)

    scored = []

    for text, votes in counter.items():

        format_score = plate_format_score(text)

        total_score = (
            format_score +
            votes * 2
        )

        scored.append(
            (
                total_score,
                votes,
                format_score,
                text
            )
        )

    scored.sort(
        reverse=True
    )

    print()
    print("-" * 70)
    print("OCR CANDIDATES")
    print("-" * 70)

    for score, votes, fmt, text in scored:

        print(
            f"{text:15} "
            f"votes={votes:<3} "
            f"format={fmt:<3} "
            f"total={score}"
        )

    best_score, best_votes, best_format, best_text = scored[0]

    total_votes = sum(counter.values())

    agreement = (
        best_votes / total_votes
        if total_votes
        else 0
    )

    # --------------------------------------------------------
    # Confidence protection
    # --------------------------------------------------------

    # Require a reasonable plate-like candidate.
    if best_format < 5:

        return "", best_score, agreement

    # Require minimum OCR agreement.
    if agreement < 0.15:

        return "", best_score, agreement

    return best_text, best_score, agreement


# ============================================================
# START
# ============================================================

start_time = time.time()

print("=" * 70)
print("AI LICENSE PLATE RECOGNITION SYSTEM v5")
print("=" * 70)

print()
print("Loading YOLO model...")

model = YOLO(
    MODEL_PATH
)

print("YOLO model loaded successfully.")


# ============================================================
# LOAD INPUT
# ============================================================

image = cv2.imread(
    INPUT_IMAGE
)

if image is None:

    raise FileNotFoundError(
        INPUT_IMAGE
    )

print()
print(
    f"Input resolution: "
    f"{image.shape[1]} x {image.shape[0]}"
)


# ============================================================
# HIGH-RESOLUTION DETECTION
# ============================================================

print()
print("Running high-resolution license plate detection...")

results = model.predict(
    source=image,
    imgsz=1280,
    conf=0.15,
    iou=0.5,
    verbose=False
)

result = results[0]

boxes = result.boxes

print(
    f"Detected plates: {len(boxes)}"
)


# ============================================================
# NO DETECTION
# ============================================================

if len(boxes) == 0:

    print()
    print("❌ No license plate detected.")

    annotated = image.copy()

    cv2.imwrite(
        str(OUTPUT_DIR / "final_result.jpg"),
        annotated
    )

    raise SystemExit


# ============================================================
# PROCESS DETECTED PLATES
# ============================================================

final_results = []

h, w = image.shape[:2]

for i, box in enumerate(boxes):

    confidence = float(
        box.conf[0]
    )

    x1, y1, x2, y2 = map(
        int,
        box.xyxy[0].tolist()
    )

    width = x2 - x1
    height = y2 - y1

    print()
    print("=" * 70)
    print(
        f"PLATE {i + 1}"
    )
    print("=" * 70)

    print(
        f"YOLO confidence: "
        f"{confidence * 100:.2f}%"
    )

    print(
        f"Bounding box: "
        f"{width} x {height}"
    )

    print(
        f"Coordinates: "
        f"({x1}, {y1}) -> ({x2}, {y2})"
    )


    # ========================================================
    # SMART PADDING
    # ========================================================

    pad_x = max(
        int(width * 0.25),
        10
    )

    pad_y = max(
        int(height * 0.40),
        8
    )

    x1p = max(
        0,
        x1 - pad_x
    )

    y1p = max(
        0,
        y1 - pad_y
    )

    x2p = min(
        w,
        x2 + pad_x
    )

    y2p = min(
        h,
        y2 + pad_y
    )

    plate = image[
        y1p:y2p,
        x1p:x2p
    ]


    # ========================================================
    # SAVE HIGH-RES CROP
    # ========================================================

    crop_path = (
        OUTPUT_DIR /
        f"plate_{i + 1}_crop.jpg"
    )

    cv2.imwrite(
        str(crop_path),
        plate
    )

    print(
        f"High-res crop: "
        f"{plate.shape[1]} x {plate.shape[0]}"
    )

    print(
        f"Crop saved: {crop_path}"
    )


    # ========================================================
    # OCR
    # ========================================================

    candidates = run_ocr(
        plate
    )

    best_text, ocr_score, agreement = (
        select_best_candidate(
            candidates
        )
    )


    # ========================================================
    # STORE RESULT
    # ========================================================

    final_results.append(
        {
            "text": best_text,
            "yolo": confidence,
            "ocr_score": ocr_score,
            "agreement": agreement,
            "box": (x1, y1, x2, y2),
        }
    )


# ============================================================
# ANNOTATION
# ============================================================

annotated = image.copy()

for item in final_results:

    x1, y1, x2, y2 = item["box"]

    text = item["text"]

    if not text:

        display_text = "PLATE: UNKNOWN"

    else:

        display_text = (
            f"PLATE: {text}"
        )

    # Bounding box
    cv2.rectangle(
        annotated,
        (x1, y1),
        (x2, y2),
        (0, 255, 0),
        2
    )

    # Label background
    cv2.rectangle(
        annotated,
        (x1, max(0, y1 - 35)),
        (x2, y1),
        (0, 255, 0),
        -1
    )

    cv2.putText(
        annotated,
        display_text,
        (x1 + 5, max(20, y1 - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 0, 0),
        2,
        cv2.LINE_AA
    )


# ============================================================
# SAVE FINAL
# ============================================================

final_path = (
    OUTPUT_DIR /
    "final_result.jpg"
)

cv2.imwrite(
    str(final_path),
    annotated
)


# ============================================================
# FINAL REPORT
# ============================================================

elapsed = time.time() - start_time

print()
print("=" * 70)
print("FINAL LICENSE PLATE RESULTS")
print("=" * 70)

for i, item in enumerate(final_results):

    print()
    print(
        f"Plate {i + 1}"
    )

    print(
        f"Result        : "
        f"{item['text'] or 'UNKNOWN'}"
    )

    print(
        f"YOLO Score    : "
        f"{item['yolo'] * 100:.2f}%"
    )

    print(
        f"OCR Score     : "
        f"{item['ocr_score']}"
    )

    print(
        f"OCR Agreement : "
        f"{item['agreement'] * 100:.2f}%"
    )


print()
print("=" * 70)
print("PIPELINE COMPLETED")
print("=" * 70)

print(
    f"Processing time : "
    f"{elapsed:.2f} seconds"
)

print(
    f"Final image     : "
    f"{final_path}"
)

print(
    f"Preprocessed    : "
    f"{PREPROCESS_DIR}"
)

print("=" * 70)