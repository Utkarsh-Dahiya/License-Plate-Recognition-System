"""
DETECTION FAILURE ANALYSIS + POLICY A/B (diagnostic, not app runtime).

Answers with evidence, on the repo's own labeled val split
(YOLO_dataset/images/val: 133 images, 145 GT plates):

  1. Per-image failure breakdown: missed plates, loose boxes, false
     positives, small plates, edge plates, near-square plates.
  2. Which GT plates are missed by the PRIMARY pass (640) and why
     (size band, distance from image edge, GT aspect).
  3. A/B of bounded retry policies for the missed population:
        p640          : primary only
        p640_960      : current production policy (960 retry)
        p640_tile     : overlapping 2x2 tiles re-run at 640
        p640_960_tile : 960 retry, then tiled 640 retry
  4. Sizing of the retry gate: how often does the current gate fire?

Usage:
    LVA_DEBUG_TIMING=0 venv/Scripts/python.exe analyze_detection.py
    LVA_DEBUG_TIMING=0 venv/Scripts/python.exe analyze_detection.py --policies
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BACKEND = ROOT / "app" / "backend"
sys.path.insert(0, str(BACKEND))

import cv2
import numpy as np
import torch

import services.detection_service as ds


# ------------------------------------------------------------
# Eval set (same source as eval_detection.py)
# ------------------------------------------------------------

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
        boxes.append(
            (
                int((cx - bw / 2) * img_w),
                int((cy - bh / 2) * img_h),
                int((cx + bw / 2) * img_w),
                int((cy + bh / 2) * img_h),
            )
        )
    return boxes


def load_eval_set():
    """(img_path, gt_boxes, processed_image, scale) with GT scaled the
    same way the service scales the image (_MAX_INFER_SIDE=1280)."""
    base = ROOT / "YOLO_dataset"
    images_dir = base / "images" / "val"
    labels_dir = base / "labels" / "val"

    items = []

    for img_path in sorted(
        list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
        + list(images_dir.glob("*.jpeg"))
    ):
        label = labels_dir / (img_path.stem + ".txt")
        if not label.exists():
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            continue

        h, w = img.shape[:2]
        longest = max(h, w)

        if longest > ds._MAX_INFER_SIDE:
            s = ds._MAX_INFER_SIDE / longest
            img = cv2.resize(
                img,
                (max(1, int(w * s)), max(1, int(h * s))),
                interpolation=cv2.INTER_AREA,
            )

        h, w = img.shape[:2]
        gt = parse_boxes(label, w, h)

        if not gt:
            continue

        items.append((img_path, gt, img))

    return items


def iou(a, b) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    aa = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    ab = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    return inter / max(1e-6, aa + ab - inter)


# ------------------------------------------------------------
# Detection policies (all return [(conf, x1, y1, x2, y2)])
# ------------------------------------------------------------

def raw_pass(image, imgsz, conf):
    with torch.inference_mode():
        res = ds._yolo_model.predict(
            source=image, imgsz=imgsz, conf=conf, verbose=False
        )
    dets = []
    if res and res[0].boxes is not None:
        for box in res[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            dets.append((float(box.conf[0]), x1, y1, x2, y2))
    del res
    return dets


def tiled_pass(image, imgsz, conf, grid=2, overlap=0.25):
    """Overlapping grid tiles re-run at `imgsz`, boxes mapped back."""
    h, w = image.shape[:2]
    tw = int(w / grid * (1 + overlap))
    th = int(h / grid * (1 + overlap))
    step_x = max(1, (w - tw) // max(1, grid - 1) if grid > 1 else w)
    step_y = max(1, (h - th) // max(1, grid - 1) if grid > 1 else h)

    out = []
    seen_origins = set()

    for gy in range(grid):
        for gx in range(grid):
            x0 = min(max(0, gx * step_x), max(0, w - tw))
            y0 = min(max(0, gy * step_y), max(0, h - th))
            if (x0, y0) in seen_origins:
                continue
            seen_origins.add((x0, y0))
            x1t, y1t = x0 + tw, y0 + th
            tile = image[y0:y1t, x0:x1t]
            if tile.size == 0:
                continue
            # Skip tiles smaller than a meaningful fraction of the model
            # input: YOLO would have to pad them up and add nothing the
            # full-frame pass did not already cover.
            if min(tile.shape[:2]) < 64:
                continue
            for conf_v, bx1, by1, bx2, by2 in raw_pass(tile, imgsz, conf):
                # Clamp to the tile's own extent before mapping back, so a
                # box touching a shared border is not duplicated wrongly.
                bx1 = max(0, min(bx1, tw))
                by1 = max(0, min(by1, th))
                bx2 = max(0, min(bx2, tw))
                by2 = max(0, min(by2, th))
                if bx2 - bx1 < 2 or by2 - by1 < 2:
                    continue
                out.append(
                    (conf_v, bx1 + x0, by1 + y0, bx2 + x0, by2 + y0)
                )
    return out


def merge_primary_pref(primary, extra):
    merged = sorted(primary, key=lambda d: d[0], reverse=True)
    for det in sorted(extra, key=lambda d: d[0], reverse=True):
        if not any(
            ds._box_iou(det[1:5], e[1:5]) >= 0.60 for e in merged
        ):
            merged.append(det)
    return merged


def gate_fires(primary, w, h):
    """The production retry gate, evaluated on the primary pass."""
    plausible = [
        d for d in primary if ds._looks_like_plate_box(d[1:5], w, h)
    ]
    best = max((d[0] for d in plausible), default=0.0)
    return (not plausible) or best < ds._WEAK_PASS_CONF


def policy_detect(image, policy, conf):
    h, w = image.shape[:2]
    primary = raw_pass(image, ds._PRIMARY_IMGSZ, conf)

    if policy == "p640":
        return primary, gate_fires(primary, w, h)

    if policy == "p640_960_always":
        extra = raw_pass(
            image, ds._RETRY_IMGSZ, max(ds._RETRY_MIN_CONF, conf)
        )
        return merge_primary_pref(primary, extra), True

    if policy == "p640_960_gate85":
        plausible = [
            d for d in primary if ds._looks_like_plate_box(d[1:5], w, h)
        ]
        best = max((d[0] for d in plausible), default=0.0)
        fired = (not plausible) or best < 0.85
        if fired:
            extra = raw_pass(
                image, ds._RETRY_IMGSZ, max(ds._RETRY_MIN_CONF, conf)
            )
            return merge_primary_pref(primary, extra), fired
        return primary, fired

    if policy == "p640_960_plaus":
        """Always-retry, but only PLATE-SHAPED retry boxes merge.

        Removes the collage-image flood (3-6 overlapping frame-sized
        boxes) while keeping genuinely plate-shaped recoveries.
        """
        extra = raw_pass(
            image, ds._RETRY_IMGSZ, max(ds._RETRY_MIN_CONF, conf)
        )
        extra = [
            d for d in extra if ds._looks_like_plate_box(d[1:5], w, h)
        ]
        return merge_primary_pref(primary, extra), True

    if policy == "p640_960":
        fired = gate_fires(primary, w, h)
        if fired:
            extra = raw_pass(
                image, ds._RETRY_IMGSZ, max(ds._RETRY_MIN_CONF, conf)
            )
            return merge_primary_pref(primary, extra), fired
        return primary, fired

    if policy == "p640_tile":
        fired = gate_fires(primary, w, h)
        if fired:
            extra = tiled_pass(image, ds._PRIMARY_IMGSZ, conf)
            return merge_primary_pref(primary, extra), fired
        return primary, fired

    if policy == "p640_960_tile":
        fired = gate_fires(primary, w, h)
        if not fired:
            return primary, fired
        extra = raw_pass(
            image, ds._RETRY_IMGSZ, max(ds._RETRY_MIN_CONF, conf)
        )
        merged = merge_primary_pref(primary, extra)
        extra2 = tiled_pass(image, ds._PRIMARY_IMGSZ, conf)
        return merge_primary_pref(merged, extra2), fired

    raise ValueError(policy)


def score(items, policy, conf, collect_failures=False):
    hits = 0
    total = 0
    ious = []
    fps = 0
    boxes = 0
    small_total = small_hits = 0
    t = []
    fired = 0
    failures = []
    fail_kind = Counter()

    for path, gt, img in items:
        h, w = img.shape[:2]
        t0 = time.perf_counter()
        dets, did_fire = policy_detect(img, policy, conf)
        t.append(time.perf_counter() - t0)
        fired += int(did_fire)

        total += len(gt)
        boxes += len(dets)

        for g in gt:
            if len(g) >= 5:
                g = g[:4]
            best = max((iou(g, d[1:5]) for d in dets), default=0.0)
            if best >= 0.5:
                hits += 1
                ious.append(best)
            elif collect_failures:
                gw, gh = g[2] - g[0], g[3] - g[1]
                edge = (
                    g[0] < 0.02 * w
                    or g[1] < 0.02 * h
                    or g[2] > 0.98 * w
                    or g[3] > 0.98 * h
                )
                kind = "small" if gh < 28 else "normal"
                if edge:
                    kind += "+edge"
                fail_kind[kind] += 1
                failures.append(
                    {
                        "image": path.name,
                        "gt": list(g),
                        "gt_w_h": [gw, gh],
                        "best_iou": round(best, 3),
                        "kind": kind,
                        "n_dets": len(dets),
                    }
                )
            if (g[3] - g[1]) < 28:
                small_total += 1
                if best >= 0.5:
                    small_hits += 1

        for d in dets:
            if all(iou(d[1:5], (g[0], g[1], g[2], g[3])) < 0.5 for g in gt):
                fps += 1

    n = max(1, total)
    res = {
        "policy": policy,
        "conf": conf,
        "images": len(items),
        "gt": total,
        "recall_iou50": round(hits / n, 4),
        "mean_iou": round(float(np.mean(ious)), 4) if ious else 0.0,
        "fp": fps,
        "fp_per_image": round(fps / max(1, len(items)), 3),
        "boxes_per_image": round(boxes / max(1, len(items)), 3),
        "small_recall": round(small_hits / small_total, 4)
        if small_total
        else None,
        "small_n": small_total,
        "retry_fired": fired,
        "mean_ms": round(1000 * statistics.mean(t), 1),
        "misses": n - hits,
    }
    if collect_failures:
        res["failures"] = failures
        res["fail_kinds"] = dict(fail_kind)
    return res


def gate_sweep(items, conf):
    """Sweep (retry imgsz, gate threshold) in one pass per imgsz.

    Caches the two passes per image so each (imgsz, gate) cell only
    costs the merge, not another inference run.
    """
    print("\n=== RETRY GATE x RETRY-IMGSZ SWEEP (conf 0.20) ===")

    retry_sizes = (768, 960)
    gates = (0.60, 0.70, 0.75, 0.80, 0.85)

    passes = {sz: [] for sz in retry_sizes}
    primaries = []

    for path, gt, img in items:
        h, w = img.shape[:2]
        primaries.append((raw_pass(img, ds._PRIMARY_IMGSZ, conf), w, h))
        for sz in retry_sizes:
            passes[sz].append(
                raw_pass(img, sz, max(ds._RETRY_MIN_CONF, conf))
            )

    rows = []

    for sz in retry_sizes:
        for gate in gates:
            hits = total = fps = 0
            st = sh = 0
            ious = []
            fired = 0

            for (path, gt, img), (primary, w, h), retry in zip(
                items, primaries, passes[sz]
            ):
                plausible = [
                    d for d in primary
                    if ds._looks_like_plate_box(d[1:5], w, h)
                ]
                best = max((d[0] for d in plausible), default=0.0)
                fire = (not plausible) or best < gate
                dets = (
                    merge_primary_pref(primary, retry)
                    if fire
                    else primary
                )
                fired += int(fire)

                total += len(gt)
                for g in gt:
                    b = max((iou(g, d[1:5]) for d in dets), default=0.0)
                    if b >= 0.5:
                        hits += 1
                        ious.append(b)
                    if (g[3] - g[1]) < 28:
                        st += 1
                        if b >= 0.5:
                            sh += 1
                for d in dets:
                    if all(iou(d[1:5], g[:4]) < 0.5 for g in gt):
                        fps += 1

            rows.append(
                {
                    "retry_imgsz": sz,
                    "gate": gate,
                    "recall": round(hits / max(1, total), 4),
                    "mean_iou": round(float(np.mean(ious)), 4)
                    if ious
                    else 0.0,
                    "fp": fps,
                    "small_recall": round(sh / st, 4) if st else None,
                    "retry_fired": fired,
                }
            )
            r = rows[-1]
            print(
                f"  retry={sz} gate={gate:.2f} recall={r['recall']:.4f} "
                f"meanIoU={r['mean_iou']:.4f} fp={r['fp']:3d} "
                f"small={r['small_recall']} fired={r['retry_fired']:3d}"
            )

    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policies", action="store_true")
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--out", default="detection_analysis.json")
    args = ap.parse_args()

    ds._ensure_models_loaded()
    items = load_eval_set()
    print(
        f"eval set: {len(items)} images, "
        f"{sum(len(g) for _, g, _ in items)} GT plates"
    )

    print("\n=== PER-IMAGE FAILURE ANALYSIS (production policy, conf 0.20) ===")
    base = score(
        items, "p640_960", ds._DEFAULT_CONF, collect_failures=True
    )
    print(
        f"  recall@.5={base['recall_iou50']:.4f} meanIoU={base['mean_iou']:.4f} "
        f"fp={base['fp']} small_recall={base['small_recall']} "
        f"retry_fired={base['retry_fired']}/{base['images']} "
        f"({base['mean_ms']:.0f} ms/img)"
    )
    print(f"  missed plates: {base['misses']}  kinds: {base['fail_kinds']}")

    print("\n  every missed GT plate:")
    for f in sorted(base["failures"], key=lambda x: x["gt_w_h"][0]):
        print(
            f"    {f['image'][:44]:44s} gt={f['gt']} "
            f"{f['gt_w_h'][0]}x{f['gt_w_h'][1]} bestIoU={f['best_iou']} "
            f"kind={f['kind']} dets={f['n_dets']}"
        )

    # FP attribution: which detections are unmatched
    print("\n=== FALSE POSITIVE ATTRIBUTION ===")
    fp_rows = []
    for path, gt, img in items:
        dets, _ = policy_detect(img, "p640_960", ds._DEFAULT_CONF)
        for d in dets:
            if all(iou(d[1:5], g[:4]) < 0.5 for g in gt):
                h, w = img.shape[:2]
                fp_rows.append(
                    {
                        "image": path.name,
                        "box": [d[1], d[2], d[3], d[4]],
                        "conf": round(d[0], 3),
                        "w_h": [d[3] - d[1], d[4] - d[2]],
                        "plausible": ds._looks_like_plate_box(
                            d[1:5], w, h
                        ),
                    }
                )
    print(f"  total FP boxes: {len(fp_rows)}")
    print(
        "  FP conf bands: "
        f"<0.30={sum(1 for r in fp_rows if r['conf'] < 0.30)} "
        f"0.30-0.45={sum(1 for r in fp_rows if 0.30 <= r['conf'] < 0.45)} "
        f">=0.45={sum(1 for r in fp_rows if r['conf'] >= 0.45)}"
    )
    print(
        f"  FP failing plate geometry: "
        f"{sum(1 for r in fp_rows if not r['plausible'])}/{len(fp_rows)}"
    )
    for r in sorted(fp_rows, key=lambda x: -x["conf"])[:20]:
        print(
            f"    {r['image'][:40]:40s} conf={r['conf']:.3f} "
            f"{r['w_h'][0]}x{r['w_h'][1]} plausible={r['plausible']}"
        )

    results = {
        "eval_images": len(items),
        "gt_plates": sum(len(g) for _, g, _ in items),
        "production_policy": base,
        "fp_rows": fp_rows,
    }

    if args.policies:
        print("\n=== POLICY A/B (conf 0.20) ===")
        results["policy_ab"] = []
        for policy in (
            "p640",
            "p640_960",
            "p640_960_always",
            "p640_960_gate85",
            "p640_960_plaus",
            "p640_tile",
            "p640_960_tile",
        ):
            r = score(items, policy, ds._DEFAULT_CONF)
            results["policy_ab"].append(r)
            print(
                f"  {policy:14s} recall@.5={r['recall_iou50']:.4f} "
                f"meanIoU={r['mean_iou']:.4f} fp={r['fp']:3d} "
                f"small={r['small_recall']} missed={r['misses']:3d} "
                f"fired={r['retry_fired']:3d} ({r['mean_ms']:.0f} ms)"
            )

    if args.sweep:
        results["gate_sweep"] = gate_sweep(items, ds._DEFAULT_CONF)

    out = ROOT / "batch_results" / "ocr_benchmark" / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\nsaved -> {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
