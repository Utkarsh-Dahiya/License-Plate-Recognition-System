"""
GROUND-TRUTH CROP PREPARATION — one-off, read-only diagnostic.

The YOLO labels in this repo carry bounding boxes only (no plate text), and
batch_results/batch_results.csv carries the SYSTEM's OCR output, not ground
truth. benchmark_ocr.py documents this explicitly: "no text GT available".

So this helper upscales each sampled plate crop so the plate text can be
transcribed by eye for the accuracy report.

It reads images and writes PNGs into diagnostics/gt_crops/. It never
modifies a production file.

Usage (from project root):
    venv/Scripts/python.exe diagnostics/make_gt_crops.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app" / "backend"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent / "gt_crops"
SAMPLE = Path(__file__).resolve().parent / "sample.txt"

# Upscale target: plates are ~100-300 px wide in the source crops; a fixed
# large width keeps every transcription image the same, comfortable size.
TARGET_W = 1400


def upscale(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    if w >= TARGET_W:
        return img
    scale = TARGET_W / float(w)
    return cv2.resize(
        img,
        (TARGET_W, int(round(h * scale))),
        interpolation=cv2.INTER_CUBIC,
    )


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    basenames = [
        line.strip()
        for line in SAMPLE.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    made = 0
    for name in basenames:
        stem = name.rsplit(".", 1)[0]
        # Crops are keyed by the numeric prefix only:
        # "0001_Cars0.jpg" -> "0001_plate.jpg"
        prefix = stem.split("_", 1)[0]
        crop_path = ROOT / "batch_results" / f"{prefix}_plate.jpg"

        if not crop_path.exists():
            print(f"MISSING crop: {crop_path.name}")
            continue

        img = cv2.imread(str(crop_path))
        if img is None:
            print(f"UNREADABLE: {crop_path.name}")
            continue

        big = upscale(img)
        out = OUT_DIR / f"{stem}.png"
        cv2.imwrite(str(out), big, [cv2.IMWRITE_PNG_COMPRESSION, 1])
        made += 1
        print(
            f"{stem}: {img.shape[1]}x{img.shape[0]} -> {big.shape[1]}x{big.shape[0]}"
        )

    print(f"\nWrote {made} crops to {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
