"""
DETECTION EVALUATION — measured, not assumed.

Answers, with evidence (NO ground-truth OCR text is invented):
  1. Detection recall @ IoU 0.5 per confidence threshold (0.15..0.50)
  2. Detection recall / box quality / false positives per inference size
     (640 / 960 / 1280)
  3. Small-plate recall (GT height < 28 px) — does higher inference
     resolution materially help?
  4. End-to-end service metrics after the fix (warm).

Ground truth: the repository's labeled val split (test_images/valid,
Roboflow YOLO format — the same labeled data the model's data.yaml uses
for validation). IMPORTANT CAVEAT: best.pt was checkpoint-selected on
this split during training, so absolute numbers are slightly optimistic.
The numbers are still valid for A/B comparisons of inference settings,
which is what this harness exists for.

Usage:
    LVA_DEBUG_TIMING=0 venv/Scripts/python.exe eval_detection.py --out before
    LVA_DEBUG_TIMING=0 venv/Scripts/python.exe eval_detection.py --out after
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BACKEND = ROOT / "app" / "backend"
sys.path.insert(0, str(BACKEND))

import cv2
import numpy as np

import services.detection_service as ds


# ------------------------------------------------------------
# Ground truth loading
# ------------------------------------------------------------

def load_gt(label_path: Path) -> list[tuple[int, int, int, int]]:
    """YOLO txt (class cx cy w h, normalized) -> [(x1, y1, x2, y2)]."""
    img_path = next(
        (
            p
            for p in label_path.parent.parent.glob(
                f"images/*/img/**/{label_path.stem}.*"
            )
        ),
        None,
    )
    # Roboflow layout: test_images/valid/labels/*.txt next to
    # test_images/valid/images/... actually images sit in
    # test_images/valid/images/ (flat) — handled below.
    boxes = []
    return boxes, img_path


def parse_boxes(label_path: Path, img_w: int, img_h: int):
    boxes = []
    try:
        lines = label_path.read_text().strip().splitlines()
    except FileNotFoundError:
        return boxes
    for line in lines:
        parts = line.split()
        if len(parts) != 5:
            continue
        _, cx, cy, bw, bh = map(float, parts)
        x1 = (cx - bw / 2) * img_w
        y1 = (cy - bh / 2) * img_h
        x2 = (cx + bw / 2) * img_w
        y2 = (cy + bh / 2) * img_h
        boxes.append((int(x1), int(y1), int(x2), int(y2)))
    return boxes


def iou_xyxy(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    return inter / max(1e-6, area_a + area_b - inter)


# ------------------------------------------------------------
# Eval set
# ------------------------------------------------------------

def load_eval_set():
    """(image_path, gt_boxes, image) for every labeled val image.

    Uses YOLO_dataset/images/val (the model's own val split): its labels
    follow the SAME convention as training data (actual plate boxes).
    The test_images/ Roboflow split was audited and REJECTED as ground
    truth: 128/218 of its boxes are whole-image regions, near-square
    collage cells, or degenerate slivers (e.g. 8x5 px) — labeling junk
    that no honest detector can match.
    """
    base = ROOT / "YOLO_dataset"
    images_dir = base / "images" / "val"
    labels_dir = base / "labels" / "val"

    items = []
    for img_path in sorted(images_dir.glob("*.jpg")) + sorted(
        images_dir.glob("*.png")
    ) + sorted(images_dir.glob("*.jpeg")):
        label = labels_dir / (img_path.stem + ".txt")
        if not label.exists():
            continue
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        gt = parse_boxes(label, w, h)
        if not gt:
            continue
        items.append((img_path, gt, img))
    return items


# ------------------------------------------------------------
# Metrics
# ------------------------------------------------------------

def evaluate_detections(items, conf, imgsz, iou_nms=0.7):
    """Run raw YOLO (not the service) and score against GT."""
    yolo = ds._yolo_model
    import torch

    recall_hits = 0
    total_gt = 0
    matched_ious = []
    fp_boxes = 0
    total_boxes = 0
    times = []
    small_total = 0
    small_hits = 0
    per_image = []

    with torch.inference_mode():
        for img_path, gt, img in items:
            h, w = img.shape[:2]
            t0 = time.perf_counter()
            res = yolo.predict(
                source=img, imgsz=imgsz, conf=conf, iou=iou_nms,
                verbose=False,
            )
            times.append(time.perf_counter() - t0)
            boxes = res[0].boxes
            dets = []
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    dets.append((x1, y1, x2, y2, float(box.conf[0])))
            del res

            total_gt += len(gt)
            total_boxes += len(dets)

            # Best IoU of each GT against detections
            for g in gt:
                best = max(
                    (iou_xyxy(g, d[:4]) for d in dets), default=0.0
                )
                if best >= 0.5:
                    recall_hits += 1
                    matched_ious.append(best)
                gh = g[3] - g[1]
                if gh < 28:
                    small_total += 1
                    if best >= 0.5:
                        small_hits += 1

            # False positives: detections with < 0.5 IoU to every GT
            for d in dets:
                if all(iou_xyxy(d[:4], g) < 0.5 for g in gt):
                    fp_boxes += 1

            per_image.append(
                {
                    "image": img_path.name,
                    "gt": len(gt),
                    "dets": len(dets),
                    "dets_conf": [round(d[4], 3) for d in dets],
                }
            )

    n = max(1, total_gt)
    n_img = max(1, len(items))
    return {
        "conf": conf,
        "imgsz": imgsz,
        "images": len(items),
        "gt_plates": total_gt,
        "recall_iou50": round(recall_hits / n, 4),
        "mean_matched_iou": round(float(np.mean(matched_ious)), 4)
        if matched_ious
        else 0.0,
        "fp_boxes": fp_boxes,
        "boxes_per_image": round(total_boxes / n_img, 3),
        "small_plate_recall": round(small_hits / small_total, 4)
        if small_total
        else None,
        "small_plate_count": small_total,
        "mean_yolo_ms": round(1000 * statistics.mean(times), 1),
        "per_image": per_image,
    }


# ------------------------------------------------------------
# End-to-end service eval (uses the service's own entry point)
# ------------------------------------------------------------

def scale_boxes_to_service_space(gt, img_w, img_h):
    """Map GT boxes into the pixel space the SERVICE reports.

    run_detection_on_image() downscales any upload whose longest side
    exceeds _MAX_INFER_SIDE and returns boxes in the PROCESSED frame's
    coordinates (the same space as its `image.width/height`). Feeding it
    the original image while scoring against original-space GT silently
    mismatches every large image: the boxes look translated/misscaled,
    so real plates score as misses and correct detections score as
    false positives. This mirrors the service's own downscale so both
    sides live in the same space.
    """

    longest = max(img_w, img_h)

    if longest <= ds._MAX_INFER_SIDE:
        return [tuple(g[:4]) for g in gt]

    s = ds._MAX_INFER_SIDE / longest

    return [
        (
            int(g[0] * s),
            int(g[1] * s),
            int(g[2] * s),
            int(g[3] * s),
        )
        for g in gt
    ]


def service_eval(items, tag):
    import io
    from contextlib import redirect_stderr

    recall_hits = 0
    recall75_hits = 0
    total_gt = 0
    matched_ious = []
    fp_boxes = 0
    times = []
    ocr_ok = 0
    strict_ok = 0
    small_total = 0
    small_hits = 0
    rows = []

    for img_path, gt_raw, img in items:
        img_h, img_w = img.shape[:2]
        gt = scale_boxes_to_service_space(gt_raw, img_w, img_h)
        ok, buf = cv2.imencode(".jpg", img)
        data = buf.tobytes()

        t0 = time.perf_counter()
        with redirect_stderr(io.StringIO()):
            out = ds.run_detection_on_image(data)
        times.append(time.perf_counter() - t0)

        dets = [
            (p["bbox"][0], p["bbox"][1], p["bbox"][2], p["bbox"][3])
            for p in out["plates"]
        ]
        total_gt += len(gt)
        for g in gt:
            best = max((iou_xyxy(g, d) for d in dets), default=0.0)
            if best >= 0.5:
                recall_hits += 1
                matched_ious.append(best)
            if best >= 0.75:
                recall75_hits += 1
            gh = g[3] - g[1]
            if gh < 28:
                small_total += 1
                if best >= 0.5:
                    small_hits += 1
        for d in dets:
            if all(iou_xyxy(d, g) < 0.5 for g in gt):
                fp_boxes += 1

        best_plate = max(
            out["plates"],
            key=lambda p: p["final_confidence"],
            default=None,
        )
        if best_plate and best_plate["ocr_text"]:
            ocr_ok += 1
            if best_plate.get("strict_format"):
                strict_ok += 1

        rows.append(
            {
                "image": img_path.name,
                "dets": len(dets),
                "gt": len(gt),
                "best_iou": round(
                    max(
                        (
                            max((iou_xyxy(g, d) for d in dets), default=0.0)
                            for g in gt
                        ),
                        default=0.0,
                    ),
                    3,
                ),
                "best_text": (
                    best_plate["ocr_text"] if best_plate else ""
                ),
                "best_conf": (
                    best_plate["final_confidence"] if best_plate else 0.0
                ),
            }
        )

    n = max(1, total_gt)
    n_img = max(1, len(items))
    summary = {
        "tag": tag,
        "images": len(items),
        "gt_plates": total_gt,
        "detection_recall_iou50": round(recall_hits / n, 4),
        "detection_recall_iou75": round(recall75_hits / n, 4),
        "fp_per_image": round(fp_boxes / n_img, 3),
        "images_with_text_rate": round(ocr_ok / n_img, 4),
        "mean_matched_iou": round(float(np.mean(matched_ious)), 4)
        if matched_ious
        else 0.0,
        "fp_boxes": fp_boxes,
        "ocr_text_rate": round(ocr_ok / n, 4),
        "strict_format_rate_of_detected": round(strict_ok / max(1, ocr_ok), 4),
        "small_plate_recall": round(small_hits / small_total, 4)
        if small_total
        else None,
        "mean_total_ms": round(1000 * statistics.mean(times), 1),
        "p95_total_ms": round(
            1000 * statistics.quantiles(times, n=20)[18], 1
        )
        if len(times) >= 20
        else round(1000 * max(times), 1),
        "rows": rows,
    }
    return summary


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="before", help="result tag")
    ap.add_argument("--skip-sweep", action="store_true")
    args = ap.parse_args()

    ds._ensure_models_loaded()
    items = load_eval_set()
    print(f"eval set: {len(items)} labeled images "
          f"({sum(len(g) for _, g, _ in items)} GT plates)")

    results = {"eval_images": len(items)}

    if not args.skip_sweep:
        print("\n=== CONF THRESHOLD SWEEP (imgsz=640) ===")
        sweep = []
        for conf in (0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50):
            m = evaluate_detections(items, conf, 640)
            sweep.append(m)
            small = m["small_plate_recall"]
            print(
                f"  conf={conf:.2f} recall@.5={m['recall_iou50']:.3f} "
                f"meanIoU={m['mean_matched_iou']:.3f} "
                f"fp/img={m['fp_boxes'] / len(items):.2f} "
                f"boxes/img={m['boxes_per_image']:.2f} "
                f"small_recall={small if small is None else round(small, 3)} "
                f"({m['mean_yolo_ms']:.0f} ms)"
            )
        results["conf_sweep_640"] = sweep

        print("\n=== INFERENCE SIZE SWEEP (conf=0.20) ===")
        size_sweep = []
        for imgsz in (640, 960, 1280):
            m = evaluate_detections(items, 0.20, imgsz)
            size_sweep.append(m)
            print(
                f"  imgsz={imgsz} recall@.5={m['recall_iou50']:.3f} "
                f"meanIoU={m['mean_matched_iou']:.3f} "
                f"fp/img={m['fp_boxes'] / len(items):.2f} "
                f"small_recall={m['small_plate_recall']} "
                f"({m['mean_yolo_ms']:.0f} ms)"
            )
        results["size_sweep_conf020"] = size_sweep

    print(f"\n=== SERVICE END-TO-END ({args.out}) ===")
    results["service"] = service_eval(items, args.out)
    s = results["service"]
    print(
        f"  recall@.50={s['detection_recall_iou50']:.3f} "
        f"recall@.75={s['detection_recall_iou75']:.3f} "
        f"meanIoU={s['mean_matched_iou']:.3f} "
        f"fp={s['fp_boxes']} (fp/img {s['fp_per_image']:.2f}) "
        f"ocr_text_rate={s['ocr_text_rate']:.3f} "
        f"strict_rate={s['strict_format_rate_of_detected']:.3f}"
    )
    print(
        f"  mean={s['mean_total_ms']:.0f} ms  "
        f"p95={s['p95_total_ms']:.0f} ms  (incl. OCR)"
    )

    out_dir = ROOT / "batch_results" / "ocr_benchmark"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"detection_eval_{args.out}.json"
    out_file.write_text(json.dumps(results, indent=2))
    print(f"\nsaved -> {out_file.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
