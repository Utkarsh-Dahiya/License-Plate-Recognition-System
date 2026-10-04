"""Diagnostic: which detected plates produce NO OCR text? (one-off)

Runs the shipped configuration once over the val split and records, per
detected plate, whether it matched GT (IoU >= .5), its crop size in
service space, its YOLO confidence, and its selected OCR text. Buckets
matched-but-empty plates by crop height and YOLO confidence band to show
where the text-rate ceiling actually lives.

Usage:
    LVA_DEBUG_TIMING=0 venv/Scripts/python.exe diag_ocr_empties.py
"""

from __future__ import annotations

import io
import sys
from collections import Counter
from contextlib import redirect_stderr
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "app" / "backend"))

import cv2  # noqa: E402

import services.detection_service as ds  # noqa: E402
from ab_final import iou, load_eval_set  # one harness, one GT mapping


def main() -> None:
    ds._ensure_models_loaded()
    items = load_eval_set()
    print(f"eval set: {len(items)} images\n")

    rows = []
    for img_path, gt, img in items:
        ok, buf = cv2.imencode(".jpg", img)
        with redirect_stderr(io.StringIO()):
            out = ds.run_detection_on_image(buf.tobytes())
        for pl in out["plates"]:
            x1, y1, x2, y2 = pl["bbox"]
            w, h = x2 - x1, y2 - y1
            matched = any(
                iou((x1, y1, x2, y2), g) >= 0.5 for g in gt
            )
            rows.append(
                {
                    "name": img_path.name,
                    "w": w,
                    "h": h,
                    "yolo": pl.get("yolo_confidence"),
                    "text": pl.get("ocr_text") or "",
                    "final": pl.get("final_confidence"),
                    "matched": matched,
                }
            )

    matched = [r for r in rows if r["matched"]]
    empty_m = [r for r in matched if not r["text"]]
    fp = [r for r in rows if not r["matched"]]
    fp_empty = [r for r in fp if not r["text"]]

    print(f"detected boxes: {len(rows)}")
    print(f"  matched to GT: {len(matched)}  empty text: {len(empty_m)}")
    print(f"  unmatched (FP): {len(fp)}  empty text: {len(fp_empty)}")

    print("\nmatched-EMPTY by crop height (service space):")
    bands = Counter()
    for r in empty_m:
        bands["h<=18" if r["h"] <= 18 else
              "h 19-27" if r["h"] <= 27 else
              "h 28-39" if r["h"] <= 39 else "h>39"] += 1
    for k in ("h<=18", "h 19-27", "h 28-39", "h>39"):
        print(f"  {k:8s} {bands[k]}")

    print("\nmatched-EMPTY by YOLO conf:")
    cbands = Counter()
    for r in empty_m:
        cbands["y<0.30" if r["yolo"] < 0.30 else
               "y 0.30-0.44" if r["yolo"] < 0.45 else
               "y 0.45-0.69" if r["yolo"] < 0.70 else "y>=0.70"] += 1
    for k in ("y<0.30", "y 0.30-0.44", "y 0.45-0.69", "y>=0.70"):
        print(f"  {k:12s} {cbands[k]}")

    print("\nmatched-EMPTY plates (detail):")
    for r in sorted(empty_m, key=lambda r: r["h"]):
        print(
            f"  {r['name']:14s} {r['w']}x{r['h']} yolo={r['yolo']:.3f} "
            f"final={r['final']}"
        )

    # Reference: matched plates WITH text, same buckets, for contrast.
    ok_m = [r for r in matched if r["text"]]
    print("\nreference — matched WITH text, height dist:")
    hb = Counter()
    for r in ok_m:
        hb["h<=18" if r["h"] <= 18 else
           "h 19-27" if r["h"] <= 27 else
           "h 28-39" if r["h"] <= 39 else "h>39"] += 1
    for k in ("h<=18", "h 19-27", "h 28-39", "h>39"):
        print(f"  {k:8s} {hb[k]}")


if __name__ == "__main__":
    main()
