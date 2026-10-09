# READ-ONLY: build diagnostics/ocr_benchmark/ground_truth_readable.json
#
# Selects READABLE-plate candidates from diagnostics/audit/collection.json by
# GEOMETRY (large pixel area + plausible Indian-plate aspect band). Ground
# truth text is NOT invented and NOT taken from OCR. Every selected plate is
# marked PENDING_TRANSCRIPTION with text=null until a human transcribes it via
# the HTML tool (transcribe.html). The 3 already human-verified plates from
# audit_gt.json are merged in and flagged verified=true.
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
COLL = ROOT / "diagnostics" / "audit" / "collection.json"
AUDIT_GT = ROOT / "diagnostics" / "audit_gt.json"
OUT = ROOT / "diagnostics" / "ocr_benchmark" / "ground_truth_readable.json"
CROPS_DIR_REL = "../audit/crops"

TARGET = 60  # user asked for ~50-100; we surface the 60 largest readable ones

coll = json.loads(COLL.read_text(encoding="utf-8"))
imgs = coll["images"]

rows = []
for im in imgs:
    image = im["image"]
    plate_by_id = {p["plate_id"]: p for p in im.get("plates", [])}
    for c in im.get("crops", []):
        pid = c.get("plate_id")
        plate = plate_by_id.get(pid, {})
        w, h = c["w"], c["h"]
        aspect = c.get("aspect", 0.0)
        # Plausible readable Indian plate: wide-ish, tall enough for glyphs.
        if not (1.8 <= aspect <= 6.0 and w >= 60 and h >= 12):
            continue
        rows.append({
            "image": image,
            "crop_file": c["file"],
            "gt_index": c.get("gt_index"),
            "plate_id": pid,
            "width": w,
            "height": h,
            "aspect_ratio": round(aspect, 3),
            "area_px": w * h,
            "yolo_confidence": plate.get("yolo_confidence"),
            "system_ocr_text": plate.get("ocr_text"),   # NOT ground truth
            "system_ocr_confidence": plate.get("ocr_confidence"),
            "bbox_service": plate.get("bbox"),
        })

# Largest first => most likely genuinely readable.
rows.sort(key=lambda r: r["area_px"], reverse=True)
selected = rows[:TARGET]


def layout_guess(aspect: float) -> str:
    if aspect >= 3.4:
        return "single-line (heuristic)"
    if aspect >= 2.2:
        return "single-line-or-two-line (heuristic)"
    return "two-line (heuristic)"


plates = {}
for i, r in enumerate(selected, 1):
    key = f"R{i:03d}"
    plates[key] = {
        "id": key,
        "source_image": r["image"],
        "crop_path": f"{CROPS_DIR_REL}/{r['crop_file']}",
        "crop_file": r["crop_file"],
        "ground_truth_text": None,
        "status": "PENDING_TRANSCRIPTION",
        "verified": False,
        "layout": layout_guess(r["aspect_ratio"]),
        "crop_width": r["width"],
        "crop_height": r["height"],
        "crop_aspect_ratio": r["aspect_ratio"],
        "crop_area_px": r["area_px"],
        "yolo_confidence": r["yolo_confidence"],
        "system_ocr_text": r["system_ocr_text"],
        "system_ocr_confidence": r["system_ocr_confidence"],
        "bbox_service": r["bbox_service"],
        "human_confidence": None,
        "notes": None,
    }

# merge the 3 already human-verified plates from audit_gt.json
verified_gt = json.loads(AUDIT_GT.read_text(encoding="utf-8"))["plates"]
n_verified = 0
for vid, meta in verified_gt.items():
    if not meta.get("text"):
        continue  # 0067 has null text (unreadable) - not part of readable set
    n_verified += 1
    cp = meta.get("crop_px") or [0, 0]
    plates[f"V{vid}"] = {
        "id": f"V{vid}",
        "source_image": meta["scene"],
        "crop_path": meta.get("crop"),
        "crop_file": Path(meta.get("crop", "")).name,
        "ground_truth_text": meta["text"],
        "status": "VERIFIED",
        "verified": True,
        "layout": meta.get("layout"),
        "crop_width": cp[0],
        "crop_height": cp[1],
        "crop_aspect_ratio": round(cp[0] / max(1, cp[1]), 3),
        "crop_area_px": cp[0] * cp[1],
        "yolo_confidence": None,
        "system_ocr_text": None,
        "system_ocr_confidence": None,
        "bbox_service": None,
        "human_confidence": meta.get("confidence"),
        "notes": meta.get("notes"),
    }

doc = {
    "purpose": "Human-verified READABLE-plate ground-truth set for offline OCR "
               "recognizer evaluation. Ground truth is transcribed by a human; "
               "it is NEVER auto-filled from OCR output.",
    "created": "2026-10-08",
    "source": "diagnostics/audit/collection.json (crops in diagnostics/audit/crops)",
    "selection_rule": "Indian-plate plausible aspect 1.8-6.0, width>=60, height>=12; "
                      "largest pixel-area first. Geometry is a proxy for legibility; "
                      "the human reviewer confirms readability during transcription.",
    "transcription_workflow": "Open transcribe.html, view each crop, type the exact "
                              "characters you can read, mark unreadable as null, then "
                              "Export JSON to overwrite this file.",
    "counts": {
        "candidates_selected": len(selected),
        "already_verified": n_verified,
        "pending_transcription": len(selected),
        "total_in_file": len(plates),
    },
    "plates": plates,
}

OUT.write_text(json.dumps(doc, indent=2), encoding="utf-8")
print("Selected readable candidates:", len(selected))
print("Already human-verified (merged):", n_verified)
print("Pending transcription:", len(selected))
print("Total entries in file:", len(plates))
print("Wrote:", OUT.name)
print("\nTop 12 by size (need transcription):")
for i in range(1, 13):
    p = plates[f"R{i:03d}"]
    print(f"  R{i:03d}: {p['crop_width']}x{p['crop_height']} a={p['crop_aspect_ratio']} "
          f"ocr_guess={p['system_ocr_text']!r}  {p['crop_file']}")

