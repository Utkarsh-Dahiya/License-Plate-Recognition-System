"""
GROUND-TRUTH vs OCR COMPARISON — one-off, read-only diagnostic
(writes only to diagnostics/gt_comparison.json).

Runs the production OCR tier cascade (CLAHE / OTSU / denoise, mirroring
diagnose_plates.diagnose_crop) on the hard plate crops named in
diagnostics/ground_truth.json, then scores every OCR candidate against the
manual transcriptions.

Verdicts per plate:
    MATCH         a candidate matches GT (or one of its recorded
                  alternatives) after alphanumeric normalisation
    FUZZY_MATCH   best difflib ratio >= --fuzzy-threshold (default 0.80)
    MISMATCH      readable GT but no candidate close enough
    GT_UNREADABLE ground truth is null (e.g. 0067) - candidates are
                  reported for information only, no verdict

This mirrors the tier cascade only; it does not reproduce the production
band-split tier 4 or the candidate-selection scoring, because the question
answered here is "can ANY tier read what a human could transcribe?".

Usage (from project root):
    venv/Scripts/python.exe diagnostics/compare_gt_ocr.py
    venv/Scripts/python.exe diagnostics/compare_gt_ocr.py --plates 0034 0333
"""

from __future__ import annotations

import argparse
import csv
import difflib
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
BACKEND = ROOT / "app" / "backend"
sys.path.insert(0, str(BACKEND))
sys.path.insert(0, str(ROOT))

import cv2

import services.detection_service as ds

GT_PATH = HERE / "ground_truth.json"
OUT_PATH = HERE / "gt_comparison.json"
BATCH_CSV = ROOT / "batch_results" / "batch_results.csv"


def load_batch_rows() -> dict[str, dict]:
    """batch_results.csv indexed by image stem (e.g. 'Cars135')."""
    rows: dict[str, dict] = {}
    if not BATCH_CSV.exists():
        return rows
    with BATCH_CSV.open(newline="", encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            rows[Path(row.get("image", "")).stem] = row
    return rows



def norm(s: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", (s or "").upper())


def gt_strings(entry: dict) -> list[str]:
    """GT text + per-line values + recorded alternatives, de-duplicated."""
    out: list[str] = []
    if entry.get("text"):
        out.append(entry["text"])
    for v in (entry.get("lines") or {}).values():
        if v:
            out.append(v)
    out.extend(entry.get("alternatives") or [])
    seen: set[str] = set()
    uniq: list[str] = []
    for s in out:
        n = norm(s)
        if n and n not in seen:
            seen.add(n)
            uniq.append(s)
    return uniq


def strict_valid(text: str) -> bool | None:
    """Indian-format validity via benchmark_ocr, if importable."""
    try:
        from benchmark_ocr import strict_indian_parse
    except Exception:
        return None
    ok, _ = strict_indian_parse(norm(text))
    return bool(ok)


def tier_candidates(img) -> list[tuple[int, str, list[tuple[str, float, object]]]]:
    """Same three tiers as diagnose_plates.diagnose_crop (no band split)."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    enhanced = ds._variant_primary(gray)
    return [
        (1, "CLAHE", ds._run_ocr(enhanced)),
        (2, "OTSU", ds._run_ocr(ds._variant_fallback(enhanced))),
        (3, "denoise", ds._run_ocr(ds._variant_tertiary(enhanced))),
    ]


def main() -> int:
    ap = argparse.ArgumentParser(description="Compare OCR tiers against manual GT.")
    ap.add_argument("--plates", nargs="*", help="plate ids (default: all in ground_truth.json)")
    ap.add_argument("--fuzzy-threshold", type=float, default=0.80)
    args = ap.parse_args()

    gt = json.loads(GT_PATH.read_text(encoding="utf-8"))
    ids = args.plates or sorted(gt["plates"].keys())
    unknown = [p for p in ids if p not in gt["plates"]]
    if unknown:
        print(f"UNKNOWN plate ids: {', '.join(unknown)}")
        return 1

    print(f"loading models ({len(ids)} plates: {', '.join(ids)}) ...")
    batch_rows = load_batch_rows()
    ds._ensure_models_loaded()

    report: dict = {
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "fuzzy_threshold": args.fuzzy_threshold,
        "tiers": "1=CLAHE 2=OTSU 3=denoise (diagnose_plates cascade, no band split)",
        "plates": {},
    }

    for pid in ids:
        entry = gt["plates"][pid]
        crop_path = ROOT / str(entry["crop"])
        print(f"\n{'=' * 78}\nPLATE {pid}  scene={entry['scene']}  crop={entry['crop']}\n{'=' * 78}")

        gts = gt_strings(entry)
        gtn = [norm(s) for s in gts]
        if not entry.get("readable"):
            print(f"  GT: UNREADABLE ({str(entry.get('notes', '')).strip()})")

        img = cv2.imread(str(crop_path))
        if img is None:
            print(f"  MISSING crop {crop_path}")
            report["plates"][pid] = {"error": f"missing crop {crop_path}"}
            continue

        t0 = time.perf_counter()
        tiers = tier_candidates(img)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        cands: list[dict] = []
        best_ratio = 0.0
        best_text = ""
        exact = False
        for tier_no, tier_name, results in tiers:
            for text, conf, _chars in results:
                n = norm(text)
                ratio = max(
                    (difflib.SequenceMatcher(None, n, g).ratio() for g in gtn),
                    default=0.0,
                )
                if n in gtn:
                    exact = True
                if ratio > best_ratio:
                    best_ratio, best_text = ratio, text
                cands.append(
                    {
                        "tier": tier_no,
                        "tier_name": tier_name,
                        "text": text,
                        "conf": round(float(conf), 4),
                        "gt_ratio": round(ratio, 4),
                        "strict_valid": strict_valid(text),
                    }
                )
                print(f"  T{tier_no} {tier_name:<7} {text!r} conf={conf:.4f} gt_ratio={ratio:.3f}")

        if not cands:
            print("  NO OCR CANDIDATES from any tier")

        if not entry.get("readable"):
            verdict = "GT_UNREADABLE"
        elif exact:
            verdict = "MATCH"
        elif best_ratio >= args.fuzzy_threshold:
            verdict = "FUZZY_MATCH"
        else:
            verdict = "MISMATCH"

        print(f"  -> verdict={verdict}  best={best_text!r} ratio={best_ratio:.3f}  wall={elapsed_ms:.0f} ms")

        report["plates"][pid] = {
            "gt_text": entry.get("text"),
            "gt_lines": entry.get("lines"),
            "gt_alternatives": entry.get("alternatives") or [],
            "gt_readable": bool(entry.get("readable")),
            "gt_confidence": entry.get("confidence"),
            "candidates": cands,
            "best_text": best_text,
            "best_ratio": round(best_ratio, 4),
            "verdict": verdict,
            "cascade_ms": round(elapsed_ms, 1),
        }

        # recorded production batch output for the same scene, scored vs GT
        brow = batch_rows.get(str(entry["scene"]))
        if brow is None:
            print(f"  BATCH pipeline: no row for scene {entry['scene']}")
            report["plates"][pid]["batch_pipeline"] = None
        else:
            btext = str(brow.get("plate_text") or "").strip()
            bn = norm(btext)
            bratio = max(
                (difflib.SequenceMatcher(None, bn, g).ratio() for g in gtn),
                default=0.0,
            )
            if not entry.get("readable"):
                bverdict = "GT_UNREADABLE"
            elif not bn:
                bverdict = "NO_OUTPUT"
            elif bn in gtn:
                bverdict = "MATCH"
            elif bratio >= args.fuzzy_threshold:
                bverdict = "FUZZY_MATCH"
            else:
                bverdict = "MISMATCH"
            print(
                f"  BATCH pipeline: {btext!r} conf={brow.get('ocr_confidence')} "
                f"status={brow.get('status')} -> {bverdict} ratio={bratio:.3f}"
            )
            report["plates"][pid]["batch_pipeline"] = {
                "image": brow.get("image"),
                "plate_text": btext,
                "ocr_confidence": brow.get("ocr_confidence"),
                "status": brow.get("status"),
                "best_ratio": round(bratio, 4),
                "verdict": bverdict,
            }

    OUT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nreport written to {OUT_PATH}")
    for pid, r in report["plates"].items():
        if "error" not in r:
            b = r.get("batch_pipeline") or {}
            print(
                f"  {pid}: cascade={r['verdict']:<14} best={r['best_text']!r} "
                f"ratio={r['best_ratio']}   batch={b.get('verdict', 'n/a')}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
