"""
AUDIT PASS 1 (collection) — one-off, READ-ONLY diagnostic.

Runs the UNMODIFIED production entry point (detection_service.
run_detection_on_image, conf threshold 0.20, debug mode on) over the
YOLO_dataset val split (Indian plates, trustworthy GT boxes; the
Roboflow test_images split was rejected as GT by eval_detection.py).

Per image it records:
  - GT plate boxes (label file) mapped into service space
  - every detection (raw bbox via debug), matched to GT by IoU >= 0.5
  - every plate entry: yolo conf, ocr_text, ocr_confidence,
    validation_score / validation, final_confidence, status,
    strict_format, per-tier OCR candidates + selected (debug data)
  - LITERAL raw EasyOCR strings (before clean_text) captured with a
    read-only monkey-patch of the reader's recognize() method; the
    production code on disk is never modified
  - a PNG crop per GT plate (production det crop when matched, else
    the GT box region) for later visual transcription

Writes only to diagnostics/audit/ (collection.json + crops/*.png).

Usage (from project root):
    venv/Scripts/python.exe diagnostics/audit_collect.py
    venv/Scripts/python.exe diagnostics/audit_collect.py --pool val
    venv/Scripts/python.exe diagnostics/audit_collect.py --limit 10
"""

from __future__ import annotations

import argparse
import io
import json
import os
import re
import sys
import time
from contextlib import redirect_stderr
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
BACKEND = ROOT / "app" / "backend"
sys.path.insert(0, str(BACKEND))
sys.path.insert(0, str(ROOT))

os.environ.setdefault("LVA_DEBUG_MODE", "1")
os.environ.setdefault("LVA_DEBUG_TIMING", "0")

import cv2
import numpy as np

import services.detection_service as ds

AUDIT_DIR = HERE / "audit"
CROPS_DIR = AUDIT_DIR / "crops"
OUT_PATH = AUDIT_DIR / "collection.json"

CONF = 0.20          # production default (_DEFAULT_CONF); kept explicit
IOU_MATCH = 0.50     # detection <-> GT match threshold

# Raw (pre-clean_text) EasyOCR strings captured per image run.
_raw_log: list[dict] = []


def norm(s: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", (str(s) or "").upper())


def parse_boxes(label_path: Path, img_w: int, img_h: int) -> list[list[int]]:
    """YOLO txt (class cx cy w h, normalized) -> [[x1,y1,x2,y2]]."""
    boxes: list[list[int]] = []
    try:
        lines = label_path.read_text(encoding="utf-8").strip().splitlines()
    except FileNotFoundError:
        return boxes
    for line in lines:
        parts = line.split()
        if len(parts) != 5:
            continue
        cls, cx, cy, bw, bh = parts
        if int(float(cls)) != 0:      # class 0 = license-plate
            continue
        cx, cy, bw, bh = map(float, parts[1:])
        boxes.append([
            int((cx - bw / 2) * img_w),
            int((cy - bh / 2) * img_h),
            int((cx + bw / 2) * img_w),
            int((cy + bh / 2) * img_h),
        ])
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


def scale_to_service_space(boxes, img_w: int, img_h: int):
    """Mirror eval_detection.scale_boxes_to_service_space()."""
    longest = max(img_w, img_h)
    if longest <= ds._MAX_INFER_SIDE:
        return [list(b) for b in boxes]
    s = ds._MAX_INFER_SIDE / longest
    return [[int(b[0] * s), int(b[1] * s), int(b[2] * s), int(b[3] * s)]
            for b in boxes]


def load_pool(pool: str):
    """[(img_path, gt_boxes_orig)] from YOLO_dataset labels."""
    base = ROOT / "YOLO_dataset"
    images_dir = base / "images" / pool
    labels_dir = base / "labels" / pool
    items = []
    for pat in ("*.jpg", "*.png", "*.jpeg"):
        for img_path in sorted(images_dir.glob(pat)):
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
            items.append((img_path, gt, img, w, h))
    return items


def install_raw_capture():
    """Read-only monkey-patch: record literal recognize() outputs."""
    ds._ensure_models_loaded()
    reader = ds._ocr_reader
    orig = reader.recognize

    def wrapped(img, *args, **kwargs):
        res = orig(img, *args, **kwargs)
        try:
            for item in res:
                _raw_log.append({
                    "text": str(item[1]),
                    "conf": round(float(item[2]), 4),
                    "shape": [int(img.shape[0]), int(img.shape[1])],
                })
        except Exception:
            pass
        return res

    reader.recognize = wrapped


# ------------------------------------------------------------

def attribute_raw(raw_entries: list[dict], plate_texts: dict[int, set[str]],
                  n_plates: int) -> None:
    """Tag each raw pass with the plate(s) whose candidates it matches."""
    for r in raw_entries:
        rn = norm(r["text"])
        owners = [pid for pid, texts in plate_texts.items() if rn in texts]
        if not owners and n_plates == 1 and rn:
            owners = list(plate_texts.keys())
        r["attributed_to"] = owners


def save_crop(img_service, bbox, gt_index: int, source: str,
              plate_id, stem: str) -> dict | None:
    x1, y1, x2, y2 = [int(v) for v in bbox]
    h, w = img_service.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 - x1 < 2 or y2 - y1 < 2:
        return None
    crop = img_service[y1:y2, x1:x2]
    suffix = f"p{plate_id}" if plate_id is not None else "gt"
    fname = f"{stem}__gt{gt_index}_{source}{suffix}.png"
    cv2.imwrite(str(CROPS_DIR / fname), crop)
    cw, ch = crop.shape[1], crop.shape[0]
    return {
        "file": fname,
        "gt_index": gt_index,
        "source": source,
        "plate_id": plate_id,
        "bbox": [x1, y1, x2, y2],
        "w": cw,
        "h": ch,
        "aspect": round(cw / max(ch, 1), 3),
    }


def process_image(img_path: Path, gt_orig, img, w: int, h: int) -> dict:
    _raw_log.clear()

    gt_service = scale_to_service_space(gt_orig, w, h)
    ok, buf = cv2.imencode(".jpg", img)
    data = buf.tobytes()

    t0 = time.perf_counter()
    with redirect_stderr(io.StringIO()):
        out = ds.run_detection_on_image(
            data, conf_threshold=CONF, debug_mode=True,
        )
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    # Service-space decoded frame (same downscale rule as the service).
    frame = img
    longest = max(h, w)
    if longest > ds._MAX_INFER_SIDE:
        s = ds._MAX_INFER_SIDE / longest
        frame = cv2.resize(img, (max(1, int(w * s)), max(1, int(h * s))),
                           interpolation=cv2.INTER_AREA)

    debug = out.get("debug") or {}
    dbg_by_id = {}
    for d in debug.get("detections", []):
        if "plate_id" in d and "raw_bbox" in d:
            dbg_by_id[d["plate_id"]] = d

    plates = []
    plate_texts: dict[int, set[str]] = {}
    for p in out.get("plates", []):
        dbg = dbg_by_id.get(p["plate_id"], {})
        cands = dbg.get("ocr_candidates") or {}
        cand_texts = {norm(t) for tier in cands.values()
                      for t in (c.get("text") for c in tier) if t}
        plate_texts[p["plate_id"]] = cand_texts
        plates.append({
            "plate_id": p["plate_id"],
            "bbox": list(p["bbox"]),
            "raw_bbox": list(dbg.get("raw_bbox") or []),
            "yolo_confidence": p.get("yolo_confidence"),
            "ocr_text": p.get("ocr_text") or "",
            "ocr_confidence": p.get("ocr_confidence"),
            "validation_score": p.get("validation_score"),
            "validation": p.get("validation"),
            "final_confidence": p.get("final_confidence"),
            "status": p.get("status"),
            "strict_format": p.get("strict_format"),
            "state_code": p.get("state_code"),
            "candidates": cands,
            "selected": dbg.get("selected"),
        })

    raw_entries = [dict(r) for r in _raw_log]
    attribute_raw(raw_entries, plate_texts, len(plates))
    raw_entries = [dict(r) for r in _raw_log]
    attribute_raw(raw_entries, plate_texts, len(plates))

    # Match GT boxes <-> detections (IoU >= 0.5).
    det_bboxes = [list(p["bbox"]) for p in plates]
    gt_match: list[dict] = []
    crops = []
    for gi, gtb in enumerate(gt_service):
        ious = [iou_xyxy(gtb, d) for d in det_bboxes]
        best_i = max(range(len(ious)), key=lambda i: ious[i]) if ious else -1
        best_iou = ious[best_i] if ious else 0.0
        matched = best_iou >= IOU_MATCH
        pid = plates[best_i]["plate_id"] if matched else None
        if matched:
            crop_bbox, source = det_bboxes[best_i], "det"
        else:
            crop_bbox, source = gtb, "gtbox"
        c = save_crop(frame, crop_bbox, gi, source, pid, img_path.stem)
        if c:
            crops.append(c)
        gt_match.append({
            "gt_index": gi,
            "gt_box": list(gtb),
            "matched": matched,
            "matched_plate_id": pid,
            "best_iou": round(best_iou, 4),
        })

    for p in plates:
        p["matched_gt_index"] = next(
            (m["gt_index"] for m in gt_match
             if m["matched"] and m["matched_plate_id"] == p["plate_id"]),
            None,
        )
        p["is_fp"] = p["matched_gt_index"] is None

    return {
        "image": img_path.name,
        "split_pool": img_path.parent.name,
        "orig_size": [w, h],
        "service_size": [int(frame.shape[1]), int(frame.shape[0])],
        "gt_boxes_service": gt_service,
        "gt_match": gt_match,
        "plates": plates,
        "raw_passes": raw_entries,
        "crops": crops,
        "plates_detected": out.get("plates_detected"),
        "elapsed_ms": round(elapsed_ms, 1),
    }




def main() -> int:
    ap = argparse.ArgumentParser(description="Read-only audit collection pass.")
    ap.add_argument("--pool", default="val", choices=["val", "train"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=str(OUT_PATH))
    args = ap.parse_args()

    items = load_pool(args.pool)
    if args.limit:
        items = items[: args.limit]

    AUDIT_DIR.mkdir(exist_ok=True)
    CROPS_DIR.mkdir(exist_ok=True)

    print(f"pool={args.pool} images={len(items)} conf={CONF} "
          f"max_infer_side={ds._MAX_INFER_SIDE}", flush=True)
    t_load = time.perf_counter()
    install_raw_capture()
    print(f"models ready in {time.perf_counter() - t_load:.1f}s", flush=True)

    images = []
    for i, (img_path, gt, img, w, h) in enumerate(items, 1):
        rec = process_image(img_path, gt, img, w, h)
        images.append(rec)
        n_match = sum(1 for m in rec["gt_match"] if m["matched"])
        print(f"[{i}/{len(items)}] {img_path.name}: gt={len(gt)} "
              f"det={rec['plates_detected']} matched={n_match} "
              f"raws={len(rec['raw_passes'])} {rec['elapsed_ms']:.0f}ms",
              flush=True)

    out_path = Path(args.out)
    payload = {
        "meta": {
            "generated": datetime.now(timezone.utc).isoformat(),
            "pool": args.pool,
            "conf_threshold": CONF,
            "iou_match": IOU_MATCH,
            "max_infer_side": ds._MAX_INFER_SIDE,
            "images": len(images),
            "note": "production run_detection_on_image, unmodified; "
                    "raw EasyOCR strings captured via read-only "
                    "recognize() wrapper",
        },
        "images": images,
    }
    out_path.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    print(f"\ncollection written -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

