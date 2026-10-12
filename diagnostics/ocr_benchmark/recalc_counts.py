# READ-ONLY dataset integrity recalculation.
# Recomputes every count from the ACTUAL files (never trusts summary counters).
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
GT = HERE / "ground_truth_readable.json"
ANSWERS = HERE / "review_sheets" / "transcript_answers.json"
CROPS = ROOT / "diagnostics" / "audit" / "crops"


def alnum(s):
    return "".join(ch for ch in str(s or "").upper() if ch.isalnum())


def main():
    data = json.loads(GT.read_text(encoding="utf-8"))
    plates = data["plates"]
    print("declared counts:", data.get("counts"))
    print("plates type:", type(plates).__name__, " n=", len(plates))

    ids = list(plates.keys())
    dup = len(ids) - len(set(ids))
    print("duplicate ids:", dup)

    total = verified = pending = unreadable = missing_crop = invalid_id = 0
    missing_list = []
    verified_ids = []
    for pid, p in plates.items():
        total += 1
        txt = p.get("ground_truth_text")
        ver = bool(p.get("verified"))
        rd = p.get("readable")
        status = (p.get("status") or "").upper()
        # crop existence (crop_path is relative to THIS dir)
        cp = p.get("crop_path") or ""
        crop_abs = (HERE / cp).resolve()
        if not crop_abs.exists():
            missing_crop += 1
            missing_list.append((pid, cp))
        if rd is False or status == "UNREADABLE" or txt is None and ver:
            unreadable += 1
        if ver and txt and alnum(txt):
            verified += 1
            verified_ids.append((pid, txt))
        elif status in ("PENDING_TRANSCRIPTION", "PENDING", "") or not ver:
            if not (rd is False):
                pending += 1

    print("\n--- RECOMPUTED ---")
    print("total entries        :", total)
    print("verified readable GT :", verified)
    print("  ->", verified_ids)
    print("pending              :", pending)
    print("unreadable           :", unreadable)
    print("missing crops        :", missing_crop)
    for pid, cp in missing_list[:10]:
        print("   missing:", pid, cp)
    print("duplicate/invalid ids:", dup)

    # answers file
    if ANSWERS.exists():
        ans = json.loads(ANSWERS.read_text(encoding="utf-8"))["answers"]
        filled = sum(1 for a in ans.values() if alnum(a.get("text")))
        unread = sum(1 for a in ans.values() if a.get("unreadable"))
        print("\n--- transcript_answers.json ---")
        print("slots:", len(ans), " filled-text:", filled, " marked-unreadable:", unread)

    # threshold gate
    print("\n=== READY FOR >=50 VERIFIED BENCHMARK:", "YES" if verified >= 50 else "NO",
          f"(have {verified}) ===")


if __name__ == "__main__":
    main()
