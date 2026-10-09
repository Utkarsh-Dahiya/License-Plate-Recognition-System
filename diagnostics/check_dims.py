"""Read-only helper: compare source-scene vs crop dimensions for the three
hard-to-read sample crops (0034, 0067, 0333), and upscale them so the
original scene resolution can be judged.

Usage (from project root): venv/Scripts/python.exe diagnostics/check_dims.py
"""

from __future__ import annotations

from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
BR = ROOT / "batch_results"

PAIRS = [
    ("0034_Cars135.png", "0034_plate.jpg"),
    ("0067_Cars174.png", "0067_plate.jpg"),
    ("0333_Cars85.png", "0333_plate.jpg"),
]

for scene, crop in PAIRS:
    for name in (scene, crop):
        img = cv2.imread(str(BR / name))
        shape = None if img is None else img.shape
        print(f"{name}: {shape}")

print("--- YOLO train originals ---")
for name in ("Cars135.png", "Cars174.png", "Cars85.png"):
    path = ROOT / "YOLO_dataset" / "images" / "train" / name
    img = cv2.imread(str(path))
    print(f"{path.name}: {None if img is None else img.shape}")
