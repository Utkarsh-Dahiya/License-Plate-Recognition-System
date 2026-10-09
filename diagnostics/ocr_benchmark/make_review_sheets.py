# Build human-review CONTACT SHEETS for the readable-plate GT candidates.
#
# Renders the proposed crops (diagnostics/ocr_benchmark/ground_truth_readable.json)
# into labeled grid images so a human can read each plate and transcribe it.
# Each cell shows: index number, crop size, and the system OCR HINT (in grey,
# clearly marked NON-AUTHORITATIVE) to speed review -- the hint is never GT.
#
# Usage:
#   python diagnostics/ocr_benchmark/make_review_sheets.py
#   -> writes diagnostics/ocr_benchmark/review_sheets/sheet_1.png ... _N.png
#   -> writes a matching transcription template:
#      diagnostics/ocr_benchmark/review_sheets/transcript_answers.json
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
HERE = Path(__file__).resolve().parent  # diagnostics/ocr_benchmark/
GT_FILE = ROOT / "diagnostics" / "ocr_benchmark" / "ground_truth_readable.json"
CROPS = ROOT / "diagnostics" / "audit" / "crops"
OUT_DIR = ROOT / "diagnostics" / "ocr_benchmark" / "review_sheets"

COLS = 4
CELL_W = 460
CELL_H = 220
PAD = 8
HDR = 30  # header band per cell for text


def load():
    data = json.loads(GT_FILE.read_text(encoding="utf-8"))
    plates = data["plates"]
    # ground_truth_readable.json stores plates as an ordered dict of
    # {id: {...}}; normalise to an ordered list preserving display order.
    if isinstance(plates, dict):
        return list(plates.values())
    return list(plates)


def make_cell(crop, label, hint):
    h, w = crop.shape[:2]
    scale = min((CELL_W - 2 * PAD) / w, (CELL_H - HDR - 2 * PAD) / h, 6.0)
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    interp = cv2.INTER_CUBIC if scale >= 1 else cv2.INTER_AREA
    resized = cv2.resize(crop, (nw, nh), interpolation=interp)
    if resized.ndim == 2:
        resized = cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)
    cell = np.full((CELL_H, CELL_W, 3), 245, np.uint8)
    x0 = (CELL_W - nw) // 2
    y0 = HDR + (CELL_H - HDR - nh) // 2
    cell[y0:y0 + nh, x0:x0 + nw] = resized
    # header: index + size (black), hint (grey, marked non-authoritative)
    cv2.putText(cell, label, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (0, 0, 0), 1, cv2.LINE_AA)
    hint_txt = f"hint(NOT GT): {hint}" if hint else "hint: -"
    cv2.putText(cell, hint_txt[:52], (170, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                (110, 110, 110), 1, cv2.LINE_AA)
    return cell



def main():
    plates = load()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    answers = {
        "instructions": (
            "For each index below, set 'text' to the EXACT plate text you read "
            "from review_sheets/sheet_*.png, or 'unreadable': true if not "
            "readable. The 'hint' is the system OCR output and is NOT truth. "
            "When done, run: python diagnostics/ocr_benchmark/apply_transcript.py"
        ),
        "answers": {},
    }

    per_sheet = COLS * 3  # 3 rows of cells per sheet
    n = len(plates)
    sheet_i = 0
    idx = 0
    while idx < n:
        chunk = plates[idx:idx + per_sheet]
        rows = (len(chunk) + COLS - 1) // COLS
        sheet = np.full((rows * CELL_H + 40, COLS * CELL_W + 8, 3), 255, np.uint8)
        cv2.putText(sheet,
                    f"Readable-plate GT review  (indices {idx}..{idx+len(chunk)-1})",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
        for j, p in enumerate(chunk):
            r, c = divmod(j, COLS)
            crop_path = (HERE / p["crop_path"]).resolve()
            crop = cv2.imread(str(crop_path))
            if crop is None:
                crop = np.full((40, 120, 3), 200, np.uint8)
            gi = idx + j
            label = f"#{gi} {p['crop_width']}x{p['crop_height']} a{p.get('crop_aspect_ratio')}"
            hint = p.get("system_ocr_text") or ""
            cell = make_cell(crop, label, hint)
            y = 40 + r * CELL_H
            x = 4 + c * CELL_W
            sheet[y:y + CELL_H, x:x + CELL_W] = cell
            answers["answers"][str(gi)] = {
                "id": p.get("id"), "crop_path": p["crop_path"],
                "hint": hint,
                "text": "", "unreadable": False,
            }
        sheet_i += 1
        out = OUT_DIR / f"sheet_{sheet_i}.png"
        cv2.imwrite(str(out), sheet)
        print(f"wrote {out.relative_to(ROOT)}  ({len(chunk)} cells)")
        idx += per_sheet

    ans_path = OUT_DIR / "transcript_answers.json"
    ans_path.write_text(json.dumps(answers, indent=2), encoding="utf-8")
    print(f"wrote {ans_path.relative_to(ROOT)}  ({n} entries)")


if __name__ == "__main__":
    main()
