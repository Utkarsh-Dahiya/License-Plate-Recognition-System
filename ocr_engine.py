import cv2
import pytesseract
import re
from pathlib import Path
from collections import defaultdict
import numpy as np


# ============================================================
# TESSERACT
# ============================================================

pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)


# ============================================================
# CONFIG
# ============================================================

INPUT_IMAGE = "plate_extraction/03_upscaled.jpg"

OUTPUT_DIR = Path("ocr_engine_results")
OUTPUT_DIR.mkdir(exist_ok=True)

OCR_SCALE = 2


# ============================================================
# OCR CONFIGURATIONS
# ============================================================

OCR_CONFIGS = [
    "--oem 3 --psm 6",
    "--oem 3 --psm 7",
    "--oem 3 --psm 8",
    "--oem 3 --psm 11",
    "--oem 3 --psm 13",
]


# ============================================================
# CLEAN OCR TEXT
# ============================================================

def clean_text(text):

    if not text:
        return ""

    text = text.upper().strip()

    # Common OCR garbage
    text = text.replace(" ", "")
    text = text.replace("\n", "")
    text = text.replace("\r", "")

    # Keep only alphanumeric
    text = re.sub(r"[^A-Z0-9]", "", text)

    return text


# ============================================================
# CHARACTER NORMALIZATION
# ============================================================

def normalize_ocr(text):

    text = clean_text(text)

    if not text:
        return ""

    # Remove obvious one-character noise
    if len(text) == 1:
        return text

    return text


# ============================================================
# PLATE FORMAT SCORE
# ============================================================

def plate_format_score(text):

    if not text:
        return -100

    n = len(text)

    score = 0

    # Indian plates are commonly around 8-10 characters,
    # but don't force a specific format.
    if 8 <= n <= 10:
        score += 12

    elif n == 7:
        score += 8

    elif n == 6:
        score += 4

    elif 5 <= n <= 11:
        score += 1

    else:
        score -= 12

    letters = sum(c.isalpha() for c in text)
    digits = sum(c.isdigit() for c in text)

    if letters >= 2:
        score += 5

    elif letters == 1:
        score += 1

    if digits >= 2:
        score += 5

    elif digits == 1:
        score += 2

    # Real plates usually contain both
    if letters > 0 and digits > 0:
        score += 5

    # Penalize excessive repetition
    if len(set(text)) <= 2 and n >= 5:
        score -= 8

    # Penalize obvious OCR garbage
    garbage_patterns = [
        "AAAAAA",
        "FFFFFF",
        "GGGGGG",
        "IIIIII",
        "LLLLLL",
        "SSSSSS",
        "OOOOOO",
        "111111",
        "000000",
    ]

    for pattern in garbage_patterns:
        if pattern in text:
            score -= 10

    return score


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

def create_variants(image):

    variants = {}

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    variants["gray"] = gray

    # --------------------------------------------------------
    # CLAHE
    # --------------------------------------------------------

    clahe = cv2.createCLAHE(
        clipLimit=2.5,
        tileGridSize=(8, 8)
    )

    enhanced = clahe.apply(gray)

    variants["enhanced"] = enhanced

    # --------------------------------------------------------
    # DENOISE
    # --------------------------------------------------------

    denoised = cv2.bilateralFilter(
        enhanced,
        7,
        45,
        45
    )

    variants["denoised"] = denoised

    # --------------------------------------------------------
    # SHARPEN
    # --------------------------------------------------------

    blur = cv2.GaussianBlur(
        denoised,
        (0, 0),
        2
    )

    sharpened = cv2.addWeighted(
        denoised,
        1.8,
        blur,
        -0.8,
        0
    )

    variants["sharpened"] = sharpened

    # --------------------------------------------------------
    # OTSU
    # --------------------------------------------------------

    _, otsu = cv2.threshold(
        sharpened,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    variants["otsu"] = otsu

    variants["otsu_inv"] = cv2.bitwise_not(otsu)

    # --------------------------------------------------------
    # ADAPTIVE
    # --------------------------------------------------------

    adaptive = cv2.adaptiveThreshold(
        sharpened,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        9
    )

    variants["adaptive"] = adaptive

    variants["adaptive_inv"] = cv2.bitwise_not(
        adaptive
    )

    # --------------------------------------------------------
    # MORPHOLOGICAL CLOSE
    # --------------------------------------------------------

    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (3, 3)
    )

    closed = cv2.morphologyEx(
        sharpened,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=1
    )

    variants["closed"] = closed

    return variants


# ============================================================
# CREATE MULTIPLE PLATE CROPS
# ============================================================

def create_crops(image):

    h, w = image.shape[:2]

    crops = {}

    # Full image
    crops["full"] = image

    # Remove small outer padding
    x1 = int(w * 0.05)
    x2 = int(w * 0.95)

    y1 = int(h * 0.08)
    y2 = int(h * 0.92)

    crops["inner"] = image[y1:y2, x1:x2]

    # Central horizontal band
    y1 = int(h * 0.18)
    y2 = int(h * 0.82)

    crops["center_band"] = image[y1:y2, :]

    # Slightly wider center band
    y1 = int(h * 0.12)
    y2 = int(h * 0.88)

    crops["wide_band"] = image[y1:y2, :]

    return crops


# ============================================================
# UPSCALE
# ============================================================

def upscale(image):

    return cv2.resize(
        image,
        None,
        fx=OCR_SCALE,
        fy=OCR_SCALE,
        interpolation=cv2.INTER_CUBIC
    )


# ============================================================
# OCR USING TESSERACT
# ============================================================

def run_ocr(image, config):

    whitelist = (
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        "0123456789"
    )

    final_config = (
        config +
        " -c "
        "tessedit_char_whitelist=" +
        whitelist
    )

    # --------------------------------------------------------
    # image_to_string
    # --------------------------------------------------------

    text = pytesseract.image_to_string(
        image,
        config=final_config
    )

    text = normalize_ocr(text)

    # --------------------------------------------------------
    # image_to_data
    # --------------------------------------------------------

    try:

        data = pytesseract.image_to_data(
            image,
            config=final_config,
            output_type=pytesseract.Output.DICT
        )

        confidences = []

        for conf in data["conf"]:

            try:

                value = float(conf)

                if value >= 0:
                    confidences.append(value)

            except:
                pass

        if confidences:

            confidence = float(
                np.mean(confidences)
            )

        else:

            confidence = 0.0

    except:

        confidence = 0.0

    return text, confidence


# ============================================================
# LOAD
# ============================================================

print("=" * 75)
print("FINAL ROBUST LICENSE PLATE OCR ENGINE")
print("=" * 75)

image = cv2.imread(INPUT_IMAGE)

if image is None:

    raise FileNotFoundError(
        f"Could not read: {INPUT_IMAGE}"
    )

print(
    f"Input resolution: "
    f"{image.shape[1]} x {image.shape[0]}"
)


# ============================================================
# OCR MASTER UPSCALE
# ============================================================

ocr_image = upscale(image)

print(
    f"OCR resolution: "
    f"{ocr_image.shape[1]} x {ocr_image.shape[0]}"
)


# ============================================================
# CREATE CROPS
# ============================================================

crops = create_crops(ocr_image)


# ============================================================
# SAVE CROPS
# ============================================================

for name, crop in crops.items():

    cv2.imwrite(
        str(
            OUTPUT_DIR /
            f"crop_{name}.jpg"
        ),
        crop
    )


# ============================================================
# OCR
# ============================================================

results = []

print()
print("=" * 75)
print("OCR PROCESSING")
print("=" * 75)


for crop_name, crop in crops.items():

    variants = create_variants(crop)

    for variant_name, variant in variants.items():

        # Save variant
        cv2.imwrite(
            str(
                OUTPUT_DIR /
                f"{crop_name}_{variant_name}.jpg"
            ),
            variant
        )

        for config in OCR_CONFIGS:

            try:

                text, confidence = run_ocr(
                    variant,
                    config
                )

            except Exception as e:

                print(
                    f"OCR ERROR: "
                    f"{crop_name} "
                    f"{variant_name} "
                    f"{config}: {e}"
                )

                continue

            if not text:
                continue

            format_score = plate_format_score(
                text
            )

            results.append(
                {
                    "text": text,
                    "confidence": confidence,
                    "format": format_score,
                    "crop": crop_name,
                    "variant": variant_name,
                    "config": config
                }
            )

            print(
                f"{crop_name:12} "
                f"{variant_name:12} "
                f"{config:18} -> "
                f"{text:25} "
                f"conf={confidence:5.1f}% "
                f"format={format_score:3}"
            )


# ============================================================
# NO RESULTS
# ============================================================

if not results:

    print()
    print("=" * 75)
    print("FINAL LICENSE PLATE")
    print("=" * 75)

    print("Plate       : UNKNOWN")
    print("Confidence  : 0.0%")

    raise SystemExit


# ============================================================
# AGGREGATE VOTES
# ============================================================

groups = defaultdict(
    lambda: {
        "votes": 0,
        "confidences": [],
        "format": 0
    }
)


for result in results:

    text = result["text"]

    groups[text]["votes"] += 1

    groups[text]["confidences"].append(
        result["confidence"]
    )

    groups[text]["format"] = (
        result["format"]
    )


# ============================================================
# RANK UNIQUE CANDIDATES
# ============================================================

ranked = []

for text, info in groups.items():

    votes = info["votes"]

    average_confidence = (
        np.mean(
            info["confidences"]
        )
    )

    format_score = info["format"]

    # --------------------------------------------------------
    # Consensus score
    #
    # Confidence is useful when available,
    # but repeated OCR agreement is more important
    # for this low-resolution plate.
    # --------------------------------------------------------

    consensus_score = (
        votes * 8
        +
        format_score * 2
        +
        min(average_confidence, 50) * 0.20
    )

    # Reward mixed letters + digits
    if (
        any(c.isalpha() for c in text)
        and
        any(c.isdigit() for c in text)
    ):

        consensus_score += 6

    # Penalize extremely short results
    if len(text) < 5:

        consensus_score -= 10

    ranked.append(
        {
            "text": text,
            "votes": votes,
            "confidence": average_confidence,
            "format": format_score,
            "score": consensus_score
        }
    )


ranked.sort(
    key=lambda x: x["score"],
    reverse=True
)


# ============================================================
# RANKING
# ============================================================

print()
print("=" * 75)
print("OCR CANDIDATE RANKING")
print("=" * 75)

for candidate in ranked[:15]:

    print(
        f"{candidate['text']:15} "
        f"votes={candidate['votes']:2} "
        f"conf={candidate['confidence']:5.1f}% "
        f"format={candidate['format']:3} "
        f"score={candidate['score']:6.2f}"
    )


# ============================================================
# FINAL DECISION
# ============================================================

best = ranked[0]

final_text = best["text"]

# ------------------------------------------------------------
# Confidence calculation
# ------------------------------------------------------------

vote_confidence = min(
    best["votes"] / max(1, len(results)) * 100,
    100
)

ocr_confidence = min(
    best["confidence"],
    100
)

# Combined confidence
#
# Since Tesseract can report 0% for useful OCR on
# low-resolution images, consensus gets higher weight.
#

final_confidence = (
    vote_confidence * 0.65
    +
    ocr_confidence * 0.35
)


# ============================================================
# SAFETY CHECK
# ============================================================

# We only return UNKNOWN when the candidate is clearly
# too weak. We do NOT reject simply because Tesseract
# reports 0% confidence.

valid_length = (
    5 <= len(final_text) <= 12
)

has_letter = any(
    c.isalpha()
    for c in final_text
)

has_digit = any(
    c.isdigit()
    for c in final_text
)

has_mixed_content = (
    has_letter and has_digit
)


# If candidate has reasonable length and mixed content,
# accept it as the best OCR hypothesis.

if (
    valid_length
    and has_mixed_content
):

    final_result = final_text

else:

    # Look for next strongest candidate
    final_result = "UNKNOWN"

    for candidate in ranked:

        text = candidate["text"]

        if (
            5 <= len(text) <= 12
            and
            any(c.isalpha() for c in text)
            and
            any(c.isdigit() for c in text)
        ):

            final_result = text
            break


# ============================================================
# FINAL OUTPUT
# ============================================================

print()
print("=" * 75)
print("FINAL LICENSE PLATE")
print("=" * 75)

print(
    f"Plate          : {final_result}"
)

print(
    f"Confidence     : {final_confidence:.1f}%"
)

print(
    f"Best candidate : {best['text']}"
)

print(
    f"Votes          : {best['votes']}"
)

print(
    f"OCR confidence : {best['confidence']:.1f}%"
)

print(
    f"Format score   : {best['format']}"
)


# ============================================================
# SAVE FINAL RESULT
# ============================================================

with open(
    OUTPUT_DIR / "final_result.txt",
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "FINAL LICENSE PLATE OCR RESULT\n"
    )

    f.write(
        "================================\n"
    )

    f.write(
        f"Plate: {final_result}\n"
    )

    f.write(
        f"Confidence: "
        f"{final_confidence:.1f}%\n"
    )

    f.write(
        f"Best candidate: "
        f"{best['text']}\n"
    )

    f.write(
        f"Votes: {best['votes']}\n"
    )


# ============================================================
# COMPLETED
# ============================================================

print()
print("=" * 75)
print("OCR ENGINE COMPLETED")
print("=" * 75)

print(
    f"Results saved to: "
    f"{OUTPUT_DIR}"
)

print("=" * 75)