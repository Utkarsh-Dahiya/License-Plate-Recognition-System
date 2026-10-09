"""Read-only probe: confirm production import, GT boxes, char-conf hook."""
from __future__ import annotations
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
BACKEND = ROOT / "app" / "backend"
sys.path.insert(0, str(BACKEND))

import services.detection_service as ds  # noqa: E402

GT = json.loads((ROOT / "diagnostics" / "audit_gt.json").read_text())["plates"]
LABELS = ROOT / "YOLO_dataset" / "labels" / "train"
IMGS = ROOT / "YOLO_dataset" / "images" / "train"


def gt_pixel_boxes(scene: str):
    """Return list of (cls, cx, cy, w, h) normalized from label file."""
    p = LABELS / f"{scene}.txt"
    out = []
    if p.exists():
        for line in p.read_text().strip().splitlines():
            parts = line.split()
            out.append(tuple(float(x) for x in parts))
    return out


print("=== scene sizes + GT boxes ===")
for pid, meta in GT.items():
    scene = meta["scene"]
    img_path = IMGS / f"{scene}.png"
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"{pid} {scene}: IMAGE MISSING")
        continue
    H, W = img.shape[:2]
    print(f"\n{pid} scene={scene} img={W}x{H} text={meta.get('text')!r} "
          f"readable={meta.get('readable')} layout={meta.get('layout')}")
    for (cls, cx, cy, w, h) in gt_pixel_boxes(scene):
        x1 = int((cx - w / 2) * W)
        y1 = int((cy - h / 2) * H)
        x2 = int((cx + w / 2) * W)
        y2 = int((cy + h / 2) * H)
        print(f"   box px=({x1},{y1},{x2},{y2}) size={x2-x1}x{y2-y1}")

print("\n=== loading production models ===")
t = time.perf_counter()
ds._ensure_models_loaded()
print(f"models loaded in {time.perf_counter()-t:.1f}s; "
      f"reader={ds._ocr_reader is not None}")

print("\n=== char-conf hook + _run_ocr on one GT crop ===")
scene = "Cars135"
img = cv2.imread(str(IMGS / f"{scene}.png"))
H, W = img.shape[:2]
cls, cx, cy, w, h = gt_pixel_boxes(scene)[0]
x1 = int((cx - w / 2) * W); y1 = int((cy - h / 2) * H)
x2 = int((cx + w / 2) * W); y2 = int((cy + h / 2) * H)
crop = img[y1:y2, x1:x2]
base_gray = ds._canvas_gray(crop)
enhanced = ds._variant_primary(base_gray)
t = time.perf_counter()
cands = ds._run_ocr(enhanced)
dt = time.perf_counter() - t
print(f"crop={crop.shape[1]}x{crop.shape[0]} canvas={base_gray.shape[1]}x{base_gray.shape[0]}")
print(f"_run_ocr -> {cands}  ({dt*1000:.0f} ms)")
print("char_probs present:", any(len(c) > 2 and c[2] for c in cands))
