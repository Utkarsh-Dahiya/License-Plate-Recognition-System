from pathlib import Path
import cv2
import easyocr
from ultralytics import YOLO
import csv
import time

# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = (
    BASE_DIR
    / "runs"
    / "detect"
    / "models"
    / "license_plate_detector"
    / "weights"
    / "best.pt"
)

# Your actual Roboflow dataset
IMAGE_DIR = BASE_DIR / "YOLO_dataset" / "images"

OUTPUT_DIR = BASE_DIR / "batch_results"
OUTPUT_DIR.mkdir(exist_ok=True)

CONF_THRESHOLD = 0.25

# ============================================================
# HEADER
# ============================================================

print("=" * 78)
print("AI LICENSE PLATE BATCH TESTING PIPELINE")
print("=" * 78)

print(f"\nDataset : {IMAGE_DIR}")
print(f"Model   : {MODEL_PATH}")
print(f"Output  : {OUTPUT_DIR}")

# ============================================================
# CHECK FILES
# ============================================================

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"\nYOLO model not found:\n{MODEL_PATH}"
    )

if not IMAGE_DIR.exists():
    raise FileNotFoundError(
        f"\nImage directory not found:\n{IMAGE_DIR}"
    )

# ============================================================
# FIND IMAGES
# ============================================================

extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

images = [
    p for p in IMAGE_DIR.rglob("*")
    if p.is_file() and p.suffix.lower() in extensions
]

images.sort()

print(f"\nImages found: {len(images)}")

if not images:
    raise RuntimeError("No images found in the dataset.")

# ============================================================
# LOAD MODELS
# ============================================================

print("\nLoading YOLO model...")
model = YOLO(str(MODEL_PATH))
print("YOLO loaded successfully.")

print("\nLoading EasyOCR...")
reader = easyocr.Reader(["en"], gpu=False)
print("EasyOCR loaded successfully.")

# ============================================================
# CSV
# ============================================================

csv_path = OUTPUT_DIR / "batch_results.csv"

csv_file = open(
    csv_path,
    "w",
    newline="",
    encoding="utf-8"
)

writer = csv.writer(csv_file)

writer.writerow([
    "image",
    "plates_detected",
    "best_yolo_confidence",
    "plate_text",
    "ocr_confidence",
    "status"
])

# ============================================================
# PROCESS DATASET
# ============================================================

total = len(images)

successful = 0
failed = 0
plates_total = 0

start_time = time.time()

print("\n" + "=" * 78)
print("STARTING DATASET PROCESSING")
print("=" * 78)

for index, image_path in enumerate(images, start=1):

    print("\n" + "-" * 78)
    print(f"[{index}/{total}] {image_path.name}")

    image = cv2.imread(str(image_path))

    if image is None:
        print("ERROR: Could not read image.")
        failed += 1

        writer.writerow([
            image_path.name,
            0,
            0,
            "",
            0,
            "IMAGE_READ_ERROR"
        ])

        continue

    # --------------------------------------------------------
    # YOLO DETECTION
    # --------------------------------------------------------

    results = model.predict(
        source=image,
        conf=CONF_THRESHOLD,
        verbose=False
    )

    result = results[0]

    detections = []

    if result.boxes is not None:

        for box in result.boxes:

            confidence = float(box.conf[0])

            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0].tolist()
            )

            detections.append(
                (confidence, x1, y1, x2, y2)
            )

    plates_total += len(detections)

    # --------------------------------------------------------
    # NO PLATE
    # --------------------------------------------------------

    if not detections:

        print("YOLO: No plate detected.")

        writer.writerow([
            image_path.name,
            0,
            0,
            "",
            0,
            "NO_PLATE"
        ])

        failed += 1
        continue

    # Best detection
    detections.sort(
        key=lambda x: x[0],
        reverse=True
    )

    best_conf, x1, y1, x2, y2 = detections[0]

    print(
        f"YOLO: {len(detections)} plate(s)"
        f" | Best confidence: {best_conf * 100:.2f}%"
    )

    # --------------------------------------------------------
    # CROP PLATE
    # --------------------------------------------------------

    h, w = image.shape[:2]

    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w, x2)
    y2 = min(h, y2)

    plate = image[y1:y2, x1:x2]

    if plate.size == 0:

        writer.writerow([
            image_path.name,
            len(detections),
            best_conf,
            "",
            0,
            "INVALID_CROP"
        ])

        failed += 1
        continue

    # --------------------------------------------------------
    # UPSCALE FOR OCR
    # --------------------------------------------------------

    plate = cv2.resize(
        plate,
        None,
        fx=4,
        fy=4,
        interpolation=cv2.INTER_CUBIC
    )

    # --------------------------------------------------------
    # EASY OCR
    # --------------------------------------------------------

    try:

        ocr_results = reader.readtext(
            plate,
            detail=1,
            paragraph=False,
            allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        )

    except Exception as e:

        print(f"OCR ERROR: {e}")

        writer.writerow([
            image_path.name,
            len(detections),
            best_conf,
            "",
            0,
            "OCR_ERROR"
        ])

        failed += 1
        continue

    # --------------------------------------------------------
    # OCR RESULT
    # --------------------------------------------------------

    if ocr_results:

        # Select highest-confidence OCR result
        ocr_results.sort(
            key=lambda x: x[2],
            reverse=True
        )

        text = ocr_results[0][1]

        ocr_conf = float(
            ocr_results[0][2]
        )

        # Clean OCR text
        text = "".join(
            c for c in text.upper()
            if c.isalnum()
        )

        print(
            f"OCR: {text}"
            f" | Confidence: {ocr_conf * 100:.2f}%"
        )

        status = "SUCCESS"
        successful += 1

    else:

        text = ""
        ocr_conf = 0

        print("OCR: No text detected.")

        status = "OCR_FAILED"
        failed += 1

    # --------------------------------------------------------
    # SAVE DEBUG IMAGE
    # --------------------------------------------------------

    debug = image.copy()

    cv2.rectangle(
        debug,
        (x1, y1),
        (x2, y2),
        (0, 255, 0),
        2
    )

    label = (
        f"{text if text else 'UNKNOWN'} "
        f"| YOLO {best_conf * 100:.1f}%"
    )

    cv2.putText(
        debug,
        label,
        (x1, max(25, y1 - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2
    )

    output_name = (
        f"{index:04d}_{image_path.stem}.jpg"
    )

    cv2.imwrite(
        str(OUTPUT_DIR / output_name),
        debug
    )

    # --------------------------------------------------------
    # CSV
    # --------------------------------------------------------

    writer.writerow([
        image_path.name,
        len(detections),
        round(best_conf, 4),
        text,
        round(ocr_conf, 4),
        status
    ])

# ============================================================
# FINISH
# ============================================================

csv_file.close()

elapsed = time.time() - start_time

print("\n")
print("=" * 78)
print("BATCH PROCESSING COMPLETED")
print("=" * 78)

print(f"\nTotal images processed : {total}")
print(f"Successful OCR        : {successful}")
print(f"Failed/No result      : {failed}")
print(f"Total plates detected : {plates_total}")

print(f"\nProcessing time       : {elapsed:.2f} seconds")

if total > 0:
    print(
        f"Average per image     : "
        f"{elapsed / total:.2f} seconds"
    )

print("\nResults saved to:")
print(OUTPUT_DIR)

print("\nCSV report:")
print(csv_path)

print("\n" + "=" * 78)