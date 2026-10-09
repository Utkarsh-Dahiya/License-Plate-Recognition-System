# READ-ONLY candidate selection for the readable-plate ground-truth set.
#
# Uses ONLY diagnostics/audit/collection.json (already contains per-crop
# dimensions, aspect, YOLO confidence and the PRODUCTION ocr_text). We do NOT
# re-read PNGs (OneDrive cloud files are slow to hydrate). We do NOT use OCR
# output as ground truth — it is only a cross-reference so the human reviewer
# knows what the system guessed. Ground truth is filled in by MANUAL review.
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
COLL = ROOT / "diagnostics" / "audit" / "collection.json"

coll = json.loads(COLL.read_text(encoding="utf-8"))
imgs = coll["images"]

rows = []
for im in imgs:
    image = im["image"]
    pool = im.get("split_pool")
    # index plates by plate_id for confidence/OCR cross-reference
    plate_by_id = {p["plate_id"]: p for p in im.get("plates", [])}
    for c in im.get("crops", []):
        pid = c.get("plate_id")
        plate = plate_by_id.get(pid, {})
        rows.append({
            "image": image,
            "pool": pool,
            "crop_file": c["file"],
            "gt_index": c.get("gt_index"),
            "plate_id": pid,
            "w": c["w"],
            "h": c["h"],
            "aspect": round(c.get("aspect", 0.0), 2),
            "area": c["w"] * c["h"],
            "yolo_conf": plate.get("yolo_confidence"),
            "ocr_text": plate.get("ocr_text"),
            "ocr_conf": plate.get("ocr_confidence"),
            "validation": plate.get("validation"),
            "status": plate.get("status"),
        })

print("total crops:", len(rows))

# Plausible-plate aspect band (Indian plates ~2:1 to ~5:1 single line).
def plausible(r):
    return 1.8 <= r["aspect"] <= 6.0 and r["w"] >= 60 and r["h"] >= 12

# Rank readable candidates: bigger area + tighter aspect + higher YOLO conf.
cand = [r for r in rows if plausible(r)]
cand.sort(key=lambda r: (r["area"], r["yolo_conf"] or 0), reverse=True)

print("plausible-shape candidates (w>=60,h>=12,1.8<=aspect<=6):", len(cand))
print("\n=== top 40 by pixel area ===")
for r in cand[:40]:
    ocr = (r["ocr_text"] or "")[:14]
    print(f"{r['w']:4d}x{r['h']:3d} a={r['aspect']:<5} y={r['yolo_conf']} "
          f"ocr={ocr:<14} {r['crop_file']}")

print("\n=== size distribution (width) ===")
ws = sorted(r["w"] for r in rows)
import statistics
for thr in (200, 150, 120, 100, 80, 60):
    print(f"crops with w>={thr}: {sum(1 for w in ws if w>=thr)}")
print("median w:", statistics.median(ws), "median h:",
      statistics.median([r["h"] for r in rows]))

# persist the ranked candidate list for the selector tool
out = ROOT / "diagnostics" / "ocr_benchmark" / "_candidates.json"
out.write_text(json.dumps(cand, indent=2), encoding="utf-8")
print("\nwrote", out.name, "with", len(cand), "candidates")
