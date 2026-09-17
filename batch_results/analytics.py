from pathlib import Path
import json
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================
BASE_DIR = Path(__file__).resolve().parent.parent
CSV_PATH = BASE_DIR / "batch_results" / "batch_results.csv"
OUTPUT_DIR = BASE_DIR / "batch_results" / "analytics"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# LOAD DATA
# ============================================================

if not CSV_PATH.exists():
    raise FileNotFoundError(
        f"\nCSV file not found:\n{CSV_PATH}\n"
        "\nRun batch_test.py first."
    )

df = pd.read_csv(CSV_PATH)


# ============================================================
# BASIC CLEANING
# ============================================================

df["plates_detected"] = pd.to_numeric(
    df["plates_detected"], errors="coerce"
).fillna(0)

df["best_yolo_confidence"] = pd.to_numeric(
    df["best_yolo_confidence"], errors="coerce"
).fillna(0)

df["ocr_confidence"] = pd.to_numeric(
    df["ocr_confidence"], errors="coerce"
).fillna(0)

df["plate_text"] = df["plate_text"].fillna("").astype(str)
df["status"] = df["status"].fillna("UNKNOWN").astype(str)


# ============================================================
# CORE METRICS
# ============================================================

total_images = len(df)

total_plates = int(df["plates_detected"].sum())

successful_ocr = int(
    (df["status"] == "SUCCESS").sum()
)

ocr_failed = int(
    (df["status"] == "OCR_FAILED").sum()
)

no_plate = int(
    (df["status"] == "NO_PLATE").sum()
)

ocr_success_rate = (
    successful_ocr / total_images * 100
    if total_images else 0
)

ocr_failure_rate = (
    ocr_failed / total_images * 100
    if total_images else 0
)

no_plate_rate = (
    no_plate / total_images * 100
    if total_images else 0
)

mean_yolo_confidence = (
    df["best_yolo_confidence"].mean() * 100
)

mean_ocr_confidence = (
    df["ocr_confidence"].mean() * 100
)

yolo_80 = int(
    (df["best_yolo_confidence"] >= 0.80).sum()
)

ocr_80 = int(
    (df["ocr_confidence"] >= 0.80).sum()
)

ocr_50 = int(
    (df["ocr_confidence"] >= 0.50).sum()
)

multi_plate_images = int(
    (df["plates_detected"] > 1).sum()
)


# ============================================================
# PLATE COUNT DISTRIBUTION
# ============================================================

plate_distribution = (
    df["plates_detected"]
    .value_counts()
    .sort_index()
)

plate_distribution_data = [
    {
        "plates": int(k),
        "images": int(v)
    }
    for k, v in plate_distribution.items()
]


# ============================================================
# STATUS DISTRIBUTION
# ============================================================

status_distribution = (
    df["status"]
    .value_counts()
)

status_data = [
    {
        "status": str(k),
        "count": int(v)
    }
    for k, v in status_distribution.items()
]


# ============================================================
# CONFIDENCE DISTRIBUTION
# ============================================================

confidence_bins = [
    0,
    0.20,
    0.40,
    0.60,
    0.80,
    1.01
]

confidence_labels = [
    "0–20%",
    "20–40%",
    "40–60%",
    "60–80%",
    "80–100%"
]

df["ocr_confidence_band"] = pd.cut(
    df["ocr_confidence"],
    bins=confidence_bins,
    labels=confidence_labels,
    right=False
)

ocr_confidence_distribution = (
    df["ocr_confidence_band"]
    .value_counts()
    .reindex(confidence_labels)
    .fillna(0)
)

ocr_confidence_data = [
    {
        "range": str(k),
        "count": int(v)
    }
    for k, v in ocr_confidence_distribution.items()
]


# ============================================================
# TOP OCR RESULTS
# ============================================================

successful_df = df[
    (df["status"] == "SUCCESS") &
    (df["plate_text"].str.strip() != "")
].copy()

successful_df = successful_df.sort_values(
    "ocr_confidence",
    ascending=False
)

top_results = []

for _, row in successful_df.head(20).iterrows():
    top_results.append({
        "image": row["image"],
        "plate_text": row["plate_text"],
        "yolo_confidence": round(
            float(row["best_yolo_confidence"]) * 100, 2
        ),
        "ocr_confidence": round(
            float(row["ocr_confidence"]) * 100, 2
        ),
        "plates_detected": int(row["plates_detected"])
    })


# ============================================================
# LOW CONFIDENCE RESULTS
# ============================================================

low_confidence_df = df[
    (df["status"] == "SUCCESS") &
    (df["ocr_confidence"] < 0.50)
].copy()

low_confidence_df = low_confidence_df.sort_values(
    "ocr_confidence"
)

low_confidence_results = []

for _, row in low_confidence_df.head(50).iterrows():
    low_confidence_results.append({
        "image": row["image"],
        "plate_text": row["plate_text"],
        "yolo_confidence": round(
            float(row["best_yolo_confidence"]) * 100, 2
        ),
        "ocr_confidence": round(
            float(row["ocr_confidence"]) * 100, 2
        )
    })


# ============================================================
# MULTI-PLATE IMAGES
# ============================================================

multi_plate_df = df[
    df["plates_detected"] > 1
].sort_values(
    "plates_detected",
    ascending=False
)

multi_plate_results = []

for _, row in multi_plate_df.head(50).iterrows():
    multi_plate_results.append({
        "image": row["image"],
        "plates_detected": int(row["plates_detected"]),
        "yolo_confidence": round(
            float(row["best_yolo_confidence"]) * 100, 2
        ),
        "plate_text": row["plate_text"]
    })


# ============================================================
# DASHBOARD SUMMARY
# ============================================================

dashboard_summary = {
    "dataset": {
        "total_images": total_images,
        "total_plates": total_plates
    },

    "detection": {
        "mean_yolo_confidence": round(
            mean_yolo_confidence, 2
        ),
        "yolo_confidence_80_plus": yolo_80,
        "multi_plate_images": multi_plate_images
    },

    "ocr": {
        "successful": successful_ocr,
        "failed": ocr_failed,
        "mean_confidence": round(
            mean_ocr_confidence, 2
        ),
        "success_rate": round(
            ocr_success_rate, 2
        ),
        "failure_rate": round(
            ocr_failure_rate, 2
        ),
        "confidence_80_plus": ocr_80,
        "confidence_50_plus": ocr_50
    },

    "no_plate": {
        "count": no_plate,
        "rate": round(
            no_plate_rate, 2
        )
    },

    "status_distribution": status_data,

    "plate_distribution": plate_distribution_data,

    "ocr_confidence_distribution": ocr_confidence_data,

    "top_results": top_results,

    "low_confidence_results": low_confidence_results,

    "multi_plate_results": multi_plate_results
}


# ============================================================
# SAVE JSON
# ============================================================

json_path = OUTPUT_DIR / "dashboard_data.json"

with open(
    json_path,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        dashboard_summary,
        f,
        indent=4
    )


# ============================================================
# CONSOLE REPORT
# ============================================================

print("\n")
print("=" * 70)
print("LICENSE VISION AI — ANALYTICS ENGINE")
print("=" * 70)

print("\nDATASET")
print("-" * 70)

print(f"Images processed       : {total_images}")
print(f"Total plates detected  : {total_plates}")

print("\nDETECTION")
print("-" * 70)

print(
    f"Mean YOLO confidence  : "
    f"{mean_yolo_confidence:.2f}%"
)

print(
    f"YOLO >= 80%           : "
    f"{yolo_80}"
)

print(
    f"Multi-plate images    : "
    f"{multi_plate_images}"
)

print("\nOCR")
print("-" * 70)

print(
    f"Successful OCR        : "
    f"{successful_ocr}"
)

print(
    f"OCR failed            : "
    f"{ocr_failed}"
)

print(
    f"No plate detected     : "
    f"{no_plate}"
)

print(
    f"OCR success rate      : "
    f"{ocr_success_rate:.2f}%"
)

print(
    f"Mean OCR confidence   : "
    f"{mean_ocr_confidence:.2f}%"
)

print(
    f"OCR >= 80%            : "
    f"{ocr_80}"
)

print(
    f"OCR >= 50%            : "
    f"{ocr_50}"
)

print("\nSTATUS")
print("-" * 70)

for item in status_data:
    print(
        f"{item['status']:15} : "
        f"{item['count']}"
    )

print("\nOUTPUT")
print("-" * 70)

print(f"Dashboard data:")
print(json_path)

print("\n")
print("=" * 70)
print("ANALYTICS ENGINE COMPLETED")
print("=" * 70)