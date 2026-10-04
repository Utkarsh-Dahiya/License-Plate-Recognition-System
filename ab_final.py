"""
FINAL BEFORE/AFTER A/B — one process, one harness, same GT mapping.

Measures the shipped configuration against the pre-session configuration
by toggling the two things that actually changed:

  detection:  retry policy   960 / gate 0.60 / min-conf 0.30
                           ->  704 / gate 0.80 / min-conf 0.20
  OCR:        ambiguity table O/0, I/1, S/5, B/8
                           ->  + Z/2, G/6, Q/0

Everything else is identical, and BOTH halves are measured through the
real service entry point (run_detection_on_image) with GT mapped into
the same pixel space the service reports, so the comparison isolates the
changes rather than the harness.

Usage:
    LVA_DEBUG_TIMING=0 venv/Scripts/python.exe ab_final.py
"""

from __future__ import annotations

import io
import statistics
import sys
import time
from contextlib import redirect_stderr
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "app" / "backend"))

import cv2  # noqa: E402

import services.detection_service as ds  # noqa: E402

OLD_AMBIGUOUS = {
    "O": "0", "0": "O", "I": "1", "1": "I",
    "S": "5", "5": "S", "B": "8", "8": "B",
}


def load_eval_set():
    base = ROOT / "YOLO_dataset"
    images_dir = base / "images" / "val"
    labels_dir = base / "labels" / "val"

    items = []
    for img_path in sorted(
        list(images_dir.glob("*.jpg"))
        + list(images_dir.glob("*.png"))
        + list(images_dir.glob("*.jpeg"))
    ):
        label = labels_dir / (img_path.stem + ".txt")
        if not label.exists():
            continue
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h0, w0 = img.shape[:2]
        lines = label.read_text().strip().splitlines()
        gt = []
        s = (
            ds._MAX_INFER_SIDE / max(h0, w0)
            if max(h0, w0) > ds._MAX_INFER_SIDE
            else 1.0
        )
        for line in lines:
            parts = line.split()
            if len(parts) != 5:
                continue
            _, cx, cy, bw, bh = map(float, parts)
            gt.append(
                (
                    int((cx - bw / 2) * w0 * s),
                    int((cy - bh / 2) * h0 * s),
                    int((cx + bw / 2) * w0 * s),
                    int((cy + bh / 2) * h0 * s),
                )
            )
        if gt:
            items.append((img_path, gt, img))
    return items


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    aa = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    ab = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    return inter / max(1e-6, aa + ab - inter)


def run(items, tag):
    hits = hits75 = total = fps = 0
    st = sh = 0
    ious = []
    ocr_ok = strict_ok = 0
    times = []

    for img_path, gt, img in items:
        ok, buf = cv2.imencode(".jpg", img)
        t0 = time.perf_counter()
        with redirect_stderr(io.StringIO()):
            out = ds.run_detection_on_image(buf.tobytes())
        times.append(time.perf_counter() - t0)

        dets = [
            (p["bbox"][0], p["bbox"][1], p["bbox"][2], p["bbox"][3])
            for p in out["plates"]
        ]
        total += len(gt)

        for g in gt:
            b = max((iou(g, d) for d in dets), default=0.0)
            if b >= 0.5:
                hits += 1
                ious.append(b)
            if b >= 0.75:
                hits75 += 1
            if (g[3] - g[1]) < 28:
                st += 1
                if b >= 0.5:
                    sh += 1

        for d in dets:
            if all(iou(d, g) < 0.5 for g in gt):
                fps += 1

        best = max(
            out["plates"], key=lambda p: p["final_confidence"], default=None
        )
        if best and best["ocr_text"]:
            ocr_ok += 1
            if best.get("strict_format"):
                strict_ok += 1

    t = sorted(times)
    n = len(items)
    res = {
        "tag": tag,
        "images": n,
        "gt_plates": total,
        "recall_iou50": round(hits / total, 4),
        "recall_iou75": round(hits75 / total, 4),
        "mean_matched_iou": round(statistics.mean(ious), 4) if ious else 0.0,
        "fp_boxes": fps,
        "fp_per_image": round(fps / n, 3),
        "small_plate_recall": round(sh / st, 4) if st else None,
        "small_plates": st,
        "ocr_text_rate": round(ocr_ok / total, 4),
        "images_with_text_rate": round(ocr_ok / n, 4),
        "strict_format_rate": round(strict_ok / max(1, ocr_ok), 4),
        "mean_ms": round(1000 * statistics.mean(t), 1),
        "p50_ms": round(1000 * t[n // 2], 1),
        "p95_ms": round(1000 * t[int(n * 0.95) - 1], 1),
    }
    print(
        f"  {tag:8s} recall@.50={res['recall_iou50']:.4f} "
        f"recall@.75={res['recall_iou75']:.4f} "
        f"mIoU={res['mean_matched_iou']:.4f} fp={res['fp_boxes']:3d} "
        f"({res['fp_per_image']:.2f}/img) small={res['small_plate_recall']} "
        f"text={res['ocr_text_rate']:.4f} strict={res['strict_format_rate']:.4f} "
        f"mean={res['mean_ms']:.0f}ms p95={res['p95_ms']:.0f}ms"
    )
    return res


def main():
    ds._ensure_models_loaded()

    # Capture the shipped table from the module BEFORE the first toggle
    # overwrites the module global with the historical one.
    shipped_ambiguous = dict(ds._AMBIGUOUS_CHARS)

    items = load_eval_set()
    print(
        f"eval set: {len(items)} images, "
        f"{sum(len(g) for _, g, _ in items)} GT plates\n"
    )

    print("=== SERVICE END-TO-END, SAME HARNESS ===")

    ds._RETRY_IMGSZ, ds._WEAK_PASS_CONF, ds._RETRY_MIN_CONF = 960, 0.60, 0.30
    ds._AMBIGUOUS_CHARS = dict(OLD_AMBIGUOUS)
    before = run(items, "BEFORE")

    ds._AMBIGUOUS_CHARS = shipped_ambiguous
    ds._RETRY_IMGSZ, ds._WEAK_PASS_CONF, ds._RETRY_MIN_CONF = 704, 0.80, 0.20
    after = run(items, "AFTER")

    print("\n  delta :")
    for key in (
        "recall_iou50", "recall_iou75", "mean_matched_iou", "fp_boxes",
        "fp_per_image", "small_plate_recall", "ocr_text_rate",
        "strict_format_rate", "mean_ms", "p95_ms",
    ):
        b, a = before[key], after[key]
        if b is None or a is None:
            continue
        arrow = "+" if a >= b else ""
        print(f"    {key:22s} {b} -> {a}  ({arrow}{round(a - b, 4)})")


if __name__ == "__main__":
    main()
