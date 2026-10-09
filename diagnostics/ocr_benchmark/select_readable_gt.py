# Readable-plate GT selection + review workflow (READ-ONLY to production).
#
# Purpose
#   Build a trustworthy, human-verified ground-truth evaluation set of
#   ~50-100 READABLE license plates BEFORE selecting a new OCR recognizer.
#
# Hard rules enforced here
#   * OCR output is NEVER used as ground truth.
#   * Only text already MANUALLY transcribed in diagnostics/audit_gt.json
#     is written as ground_truth_text. Everything else is proposed for
#     human transcription with verified=false.
#   * Readability is ranked by objective resolution proxies (crop pixels,
#     YOLO confidence). It is a RANKING heuristic to help a human review the
#     best candidates first -- it is NOT a substitute for human verification.
#
# Usage
#   python diagnostics/ocr_benchmark/select_readable_gt.py          # propose
#   python diagnostics/ocr_benchmark/select_readable_gt.py --fill   # review mode
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent

AUDIT = ROOT / "diagnostics" / "audit"
GT_VERIFIED = ROOT / "diagnostics" / "audit_gt.json"
COLLECTION = AUDIT / "collection.json"
CROPS = AUDIT / "crops"
OUT_JSON = ROOT / "diagnostics" / "ocr_benchmark" / "ground_truth_readable.json"
REVIEW_DIR = ROOT / "diagnostics" / "ocr_benchmark" / "review_crops"

# Readability proxy thresholds (documented, not tuned to any sample).
MIN_W = 100          # px -- below this a plate is generally hard to read
MIN_H = 25           # px
MIN_ASPECT = 1.4     # width/height -- real single-line plates are wide
MIN_YOLO = 0.40      # detection must be reasonably confident
TARGET_N = 90        # aim for ~50-100; we over-propose to allow rejects


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def build_candidates():
    """Return readable-plate proposals from the audit collection."""
    coll = load(COLLECTION)
    verified = load(GT_VERIFIED)["plates"]

    # Scene names are unique per GT plate here.
    scene_gt = {meta["scene"]: meta for meta in verified.values()}

    proposals = {}
    for img in coll["images"]:
        scene = Path(img["image"]).stem
        for crop in img.get("crops", []):
            if crop.get("source") != "det":
                continue
            w, h = crop["w"], crop["h"]
            if w < MIN_W or h < MIN_H:
                continue
            aspect = w / max(h, 1)
            if aspect < MIN_ASPECT:
                continue
            pid = crop.get("plate_id")
            plate = next((p for p in img["plates"] if p["plate_id"] == pid), None)
            yolo = plate["yolo_confidence"] if plate else None
            if yolo is None or yolo < MIN_YOLO:
                continue
            # readability score: resolution x detection confidence
            score = (w * h) ** 0.5 * yolo
            key = crop["file"]
            proposals[key] = {
                "id": key,
                "image": img["image"],
                "source_scene": scene,
                "crop_path": f"diagnostics/audit/crops/{key}",
                "plate_id": pid,
                "crop_width": w,
                "crop_height": h,
                "aspect_ratio": round(aspect, 3),
                "yolo_confidence": yolo,
                "layout": "single-line" if aspect >= 2.0 else "two-line-or-tall",
                "_readability_score": round(score, 1),
                # informational only -- NEVER used as ground truth
                "_system_ocr_hint": (plate or {}).get("ocr_text", ""),
                "ground_truth_text": None,
                "ground_truth_source": "manual_transcription",
                "verified": False,
            }
    return proposals, verified, scene_gt



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fill", action="store_true",
                    help="interactive review mode to enter verified text")
    ap.add_argument("--target", type=int, default=TARGET_N)
    args = ap.parse_args()

    proposals, verified, scene_gt = build_candidates()
    ranked = sorted(proposals.values(),
                    key=lambda c: c["_readability_score"], reverse=True)
    selected = ranked[:args.target]

    # Pre-fill any crop whose scene has an already-verified GT plate.
    prefill = 0
    for c in selected:
        v = scene_gt.get(c["source_scene"])
        if v and v.get("text") and v.get("readable"):
            c["ground_truth_text"] = v["text"]
            c["ground_truth_source"] = "audit_gt.json (manual, verified)"
            c["verified"] = True
            c["gt_confidence"] = v.get("confidence", {})
            prefill += 1

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    REVIEW_DIR.mkdir(parents=True, exist_ok=True)

    # Copy review crops for fast human review.
    for c in selected:
        src = ROOT / c["crop_path"]
        if src.exists():
            shutil.copy(src, REVIEW_DIR / c["id"])

    payload = {
        "created_note": (
            "Proposals ranked by resolution x YOLO confidence. ground_truth_text "
            "is ONLY populated from manual transcriptions (audit_gt.json). "
            "system_ocr_hint is informational and is NOT ground truth."
        ),
        "selection_thresholds": {
            "min_w": MIN_W, "min_h": MIN_H, "min_aspect": MIN_ASPECT,
            "min_yolo": MIN_YOLO, "target": args.target,
        },
        "counts": {
            "proposed": len(ranked),
            "selected": len(selected),
            "prefilled_verified": prefill,
            "needs_transcription": len(selected) - prefill,
        },
        "plates": selected,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"Proposed readable candidates (passed filters): {len(ranked)}")
    print(f"Selected for review: {len(selected)}")
    print(f"Prefilled from verified audit_gt.json: {prefill}")
    print(f"Need manual transcription: {len(selected) - prefill}")
    print(f"Wrote: {OUT_JSON.relative_to(ROOT)}")
    print(f"Review crops copied to: {REVIEW_DIR.relative_to(ROOT)}")

    if args.fill:
        interactive_fill(selected)


def interactive_fill(selected):
    """Terminal review loop: show each crop, enter plate text or skip."""
    print("\n=== MANUAL TRANSCRIPTION REVIEW ===")
    print("For each plate: type the EXACT text you read, press Enter to")
    print("skip, 'u' to mark unreadable, 'q' to save and quit.\n")
    changed = 0
    for c in selected:
        if c["verified"]:
            continue
        print(f"[{c['id']}] {c['crop_width']}x{c['crop_height']} "
              f"aspect={c['aspect_ratio']} yolo={c['yolo_confidence']}")
        print(f"   system OCR hint (NOT truth): {c['_system_ocr_hint']!r}")
        print(f"   open: {ROOT / c['crop_path']}")
        try:
            val = input("   plate text > ").strip()
        except EOFError:
            break
        if val.lower() == "q":
            break
        if val.lower() == "u":
            c["ground_truth_text"] = None
            c["verified"] = True
            c["readable"] = False
            changed += 1
            continue
        if not val:
            continue
        c["ground_truth_text"] = val.upper()
        c["verified"] = True
        c["readable"] = True
        c["ground_truth_source"] = "manual_transcription"
        changed += 1
    OUT_JSON.write_text(json.dumps({
        "created_note": "Human-reviewed. verified=true entries are trusted GT.",
        "selection_thresholds": {"min_w": MIN_W, "min_h": MIN_H},
        "counts": {
            "selected": len(selected),
            "verified": sum(1 for c in selected if c.get("verified")),
        },
        "plates": selected,
    }, indent=2), encoding="utf-8")
    print(f"\nSaved {changed} manual entries to {OUT_JSON.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
