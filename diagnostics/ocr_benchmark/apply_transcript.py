# Merge human transcriptions (transcript_answers.json) into the verified
# ground-truth file. Only entries you actually filled in are marked verified;
# OCR hints are ignored. Produces the FINAL evaluation GT set.
#
# Usage:
#   1. python diagnostics/ocr_benchmark/make_review_sheets.py
#   2. edit diagnostics/ocr_benchmark/review_sheets/transcript_answers.json
#      (fill 'text' or set 'unreadable': true for each index)
#   3. python diagnostics/ocr_benchmark/apply_transcript.py
#   -> updates diagnostics/ocr_benchmark/ground_truth_readable.json
#      (verified entries only are usable for evaluation)
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
GT_FILE = ROOT / "diagnostics" / "ocr_benchmark" / "ground_truth_readable.json"
ANSWERS = ROOT / "diagnostics" / "ocr_benchmark" / "review_sheets" / "transcript_answers.json"


def clean_text(s: str) -> str:
    """Uppercase alphanumeric only -- matches how plates are stored."""
    return re.sub(r"[^A-Z0-9]", "", str(s or "").upper())


def main():
    gt = json.loads(GT_FILE.read_text(encoding="utf-8"))
    plates_dict = gt["plates"]
    # Preserve the same display order the review sheets were built with
    # (dict insertion order), so numeric answer indices map positionally.
    ordered = list(plates_dict.values())
    if not ANSWERS.exists():
        print("No transcript_answers.json yet. Run make_review_sheets.py first.")
        return
    ans = json.loads(ANSWERS.read_text(encoding="utf-8"))["answers"]

    filled = unreadable = skipped = 0
    for idx_str, a in ans.items():
        idx = int(idx_str)
        if idx < 0 or idx >= len(ordered):
            print(f"  ! answer index {idx} out of range ({len(ordered)} plates); skipped")
            continue
        p = ordered[idx]
        if a.get("unreadable"):
            p["ground_truth_text"] = None
            p["readable"] = False
            p["verified"] = True
            p["ground_truth_source"] = "manual_transcription"
            unreadable += 1
            continue
        txt = clean_text(a.get("text", ""))
        if not txt:
            skipped += 1
            continue
        p["ground_truth_text"] = txt
        p["readable"] = True
        p["verified"] = True
        p["ground_truth_source"] = "manual_transcription"
        filled += 1

    verified = [p for p in plates if p.get("verified") and p.get("ground_truth_text")]
    gt["counts"]["verified_readable"] = len(verified)
    gt["counts"]["marked_unreadable"] = unreadable
    gt["counts"]["still_pending"] = skipped
    GT_FILE.write_text(json.dumps(gt, indent=2), encoding="utf-8")

    print(f"Filled & verified readable: {filled}")
    print(f"Marked unreadable (excluded): {unreadable}")
    print(f"Still pending (skipped): {skipped}")
    print(f"TOTAL verified readable GT now: {len(verified)}")
    print(f"Updated: {GT_FILE.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
