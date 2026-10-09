"""
AUDIT REPORT — one-off, READ-ONLY diagnostic
(writes only to diagnostics/audit_report.json + stdout summary).

Joins the collection pass (audit_collect.py) with the manual
transcriptions (audit_gt.json) and, for every selected sample:

  source image / GT / crop WxH / aspect / YOLO conf / raw OCR /
  normalized OCR / repaired OCR / final OCR / OCR conf / strict
  validation / exact match / character accuracy / diagnosis (A-H)

Then aggregates detection success, exact-match, character accuracy,
OCR failure rate and crop dimensions for ALL / READABLE / LOW-RES
subsets, lists the 10 most important failures, and investigates the
GJ08DJ3136 -> 76J080J3136 example.

Failure categories (exactly one per sample):
    A DETECTION FAILURE   no det IoU>=0.5 against the GT plate box
    B BAD CROP            matched det covers <70% of the GT box
    C LOW RESOLUTION      human could not read the crop reliably
    D OCR RECOGNITION FAILURE  no tier candidate close to GT (or the
                          pipeline kept a wrong read nothing fixed)
    E NORMALIZATION FAILURE    raw EasyOCR text was GT, cleaning lost it
    F REPAIR FAILURE      a tier-5 format repair won and made it worse
    G VALIDATION FAILURE  format/validation scoring steered the choice
                          of a wrong (or distrust of a correct) read
    H CORRECT             normalized final == normalized GT

Usage (from project root):
    venv/Scripts/python.exe diagnostics/audit_report.py
    venv/Scripts/python.exe diagnostics/audit_report.py --fuzzy 0.80
"""

from __future__ import annotations

import argparse
import difflib
import json
import re
import statistics
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
AUDIT = HERE / "audit"
COLLECTION = AUDIT / "collection.json"
GT_PATH = AUDIT / "audit_gt.json"
OUT_PATH = HERE / "audit_report.json"


def norm(s: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", str(s or "").upper())


def lev(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1,
                           prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def char_accuracy(final: str, gt: str) -> float:
    g = norm(gt)
    if not g:
        return 0.0
    f = norm(final)
    return max(0.0, 1.0 - lev(f, g) / len(g))


def sim(a: str, b: str) -> float:
    a, b = norm(a), norm(b)
    if not a or not b:
        return 0.0
    return difflib.SequenceMatcher(None, a, b).ratio()


def load_json(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8"))


def index_collection(data: dict) -> tuple[dict, dict]:
    """(crops_by_file, image_by_name) lookups."""
    crops, images = {}, {}
    for im in data["images"]:
        images[im["image"]] = im
        for c in im["crops"]:
            crops[c["file"]] = (im, c)
    return crops, images


def all_candidate_texts(plate: dict) -> dict[str, list[tuple[str, float]]]:
    """tier -> [(text, conf)] from debug candidates."""
    return {
        tier: [(c["text"], float(c["conf"])) for c in cands]
        for tier, cands in (plate.get("candidates") or {}).items()
    }


def best_plate_for_gt(im: dict, gt_index: int) -> dict | None:
    m = next((m for m in im["gt_match"]
              if m["gt_index"] == gt_index and m["matched"]), None)
    if not m:
        return None
    return next((p for p in im["plates"]
                 if p["plate_id"] == m["matched_plate_id"]), None)


def inter_area(a, b) -> int:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    return max(0, x2 - x1) * max(0, y2 - y1)


def area(b) -> int:
    return max(0, b[2] - b[0]) * max(0, b[3] - b[1])


def pick_raw(im: dict, plate: dict | None) -> list[dict]:
    """Literal raw EasyOCR strings belonging to this plate."""
    if plate is None:
        return []
    pid = plate["plate_id"]
    mine = [r for r in im["raw_passes"] if pid in (r.get("attributed_to") or [])]
    if not mine and len(im["plates"]) == 1:
        mine = list(im["raw_passes"])
    return mine


def classify(sample: dict, fuzzy: float) -> tuple[str, str]:
    """Return (category, one-sentence explanation) — exactly one of A-H."""
    if sample["exact_match"]:
        return "H", (
            f"final OCR {sample['final_ocr']!r} equals GT "
            f"{sample['gt_text']!r} after normalization."
        )

    if not sample["detected"]:
        return "A", (
            f"YOLO produced no box with IoU>=0.5 on the GT plate "
            f"(best IoU {sample['best_iou']}); OCR never saw this plate."
        )

    if sample["crop_coverage"] is not None and sample["crop_coverage"] < 0.70:
        return "B", (
            f"matched detection covers only {sample['crop_coverage']:.0%} "
            f"of the GT plate box - the OCR crop clips the text region."
        )

    if not sample["gt_readable"] or sample["gt_confidence"] == "low":
        return "C", (
            f"crop {sample['crop_w']}x{sample['crop_h']} px is not reliably "
            f"readable by a human at native resolution - information is "
            f"absent before OCR runs."
        )

    if sample["best_candidate_sim"] < fuzzy:
        src = ("no detection ran on this plate" if not sample["detected"]
               else f"best tier candidate {sample['best_candidate_text']!r} "
                    f"reaches only {sample['best_candidate_sim']:.2f} similarity")
        return "D", (
            f"human-readable crop but EasyOCR never produced a string close "
            f"to GT ({src})."
        )

    if (sample["repair_winner"]
            and sample["final_sim"] < sample["best_candidate_sim"]):
        return "F", (
            f"tier-5 format repair produced {sample['final_ocr']!r} which "
            f"scored worse against GT than the pre-repair candidate "
            f"{sample['best_candidate_text']!r}."
        )

    if sample["final_strict_valid"] and sample["best_candidate_sim"] >= 0.90:
        return "G", (
            f"format/validation scoring promoted the wrong strict-valid "
            f"string {sample['final_ocr']!r} over the near-correct candidate "
            f"{sample['best_candidate_text']!r}."
        )

    if sample["correct_candidate_found"] and not sample["final_strict_valid"]:
        return "G", (
            f"a correct read was available but final {sample['final_ocr']!r} "
            f"was kept and strict validation rejects it "
            f"(strict_format=False)."
        )

    return "D", (
        f"selection kept {sample['final_ocr']!r} while best available "
        f"candidate was {sample['best_candidate_text']!r} "
        f"(sim {sample['best_candidate_sim']:.2f}) - OCR evidence too weak "
        f"or the wrong candidate outranked it."
    )


def build_sample(file_key: str, gt_entry: dict, crops_idx: dict) -> dict:
    im, crop = crops_idx[file_key]
    gt_text = gt_entry.get("text")
    gt_n = norm(gt_text)
    gt_index = crop["gt_index"]

    match = next((m for m in im["gt_match"] if m["gt_index"] == gt_index), {})
    plate = best_plate_for_gt(im, gt_index)
    detected = bool(match.get("matched"))

    gt_box = im["gt_boxes_service"][gt_index]
    if plate:
        coverage = inter_area(plate["bbox"], gt_box) / max(area(gt_box), 1)
    else:
        coverage = None
    best_iou = match.get("best_iou", 0.0)

    raws = pick_raw(im, plate)
    raw_first = raws[0]["text"] if raws else None

    final = (plate or {}).get("ocr_text") or ""
    final_n = norm(final)

    cands = all_candidate_texts(plate) if plate else {}
    flat = [(t, c) for lst in cands.values() for t, c in lst]
    best_text, best_s = "", 0.0
    for t, _c in flat:
        s = sim(t, gt_text)
        if s > best_s:
            best_text, best_s = t, s
    non_repair = [(t, c) for tier, lst in cands.items()
                  if tier != "tier5" for t, c in lst]
    best_nr_text, best_nr_s = "", 0.0
    for t, _c in non_repair:
        s = sim(t, gt_text)
        if s > best_nr_s:
            best_nr_text, best_nr_s = t, s

    tier5 = cands.get("tier5") or []
    tier5_texts = {norm(t) for t, _ in tier5}
    non_repair_texts = {norm(t) for t, _ in non_repair}
    repair_winner = (
        bool(final_n) and final_n in tier5_texts
        and final_n not in non_repair_texts
    )

    raw_norm = norm(raw_first) if raw_first else ""

    sample = {
        "sample_id": file_key.rsplit(".", 1)[0],
        "crop_file": file_key,
        "source_image": im["image"],
        "gt_text": gt_text,
        "gt_readable": bool(gt_entry.get("readable")),
        "gt_confidence": gt_entry.get("confidence"),
        "gt_notes": gt_entry.get("notes"),
        "crop_w": crop["w"],
        "crop_h": crop["h"],
        "crop_aspect": crop["aspect"],
        "crop_source": crop["source"],
        "yolo_confidence": (plate or {}).get("yolo_confidence"),
        "detected": detected,
        "best_iou": best_iou,
        "crop_coverage": round(coverage, 4) if coverage is not None else None,
        "raw_ocr": raw_first,
        "raw_ocr_all": [r["text"] for r in raws],
        "normalized_ocr": raw_norm or None,
        "repaired_ocr": final if repair_winner else None,
        "final_ocr": final,
        "ocr_confidence": (plate or {}).get("ocr_confidence"),
        "final_confidence": (plate or {}).get("final_confidence"),
        "validation": (plate or {}).get("validation"),
        "validation_score": (plate or {}).get("validation_score"),
        "strict_format": (plate or {}).get("strict_format"),
        "status": (plate or {}).get("status"),
        "tier5_repairs": tier5,
        "candidates": cands,
        "repair_winner": repair_winner,
        "exact_match": bool(final_n) and final_n == gt_n,
        "char_accuracy": round(char_accuracy(final, gt_text), 4),
        "final_sim": round(sim(final, gt_text), 4),
        "best_candidate_text": best_text or None,
        "best_candidate_sim": round(best_s, 4),
        "best_nonrepair_text": best_nr_text or None,
        "best_nonrepair_sim": round(best_nr_s, 4),
        "correct_candidate_found": best_nr_s >= 0.90,
        "final_strict_valid": bool((plate or {}).get("strict_format")),
        "final_empty": not bool(final),
        "raw_equals_gt": bool(raw_norm) and raw_norm == gt_n,
    }
    cat, why = classify(sample, fuzzy=FUZZY)
    sample["category"] = cat
    sample["diagnosis"] = why
    return sample

    if sample["correct_candidate_found"] and not sample["final_strict_valid"]:
        return "G", (
            f"a correct read was available but final {sample['final_ocr']!r} "
            f"was kept and strict validation rejects it (strict_format=False)."
        )

    return "D", (
        f"selection kept {sample['final_ocr']!r} while best available "
        f"candidate was {sample['best_candidate_text']!r} "
        f"(sim {sample['best_candidate_sim']:.2f}) - OCR evidence too weak "
        f"or the wrong candidate outranked it."
    )


FUZZY = 0.80


def subset_metrics(samples: list[dict]) -> dict:
    n = len(samples)
    if not n:
        return {"n": 0}
    widths = [s["crop_w"] for s in samples]
    heights = [s["crop_h"] for s in samples]
    cats: dict[str, int] = {}
    for s in samples:
        cats[s["category"]] = cats.get(s["category"], 0) + 1
    detected = sum(1 for s in samples if s["detected"])
    exact = sum(1 for s in samples if s["exact_match"])
    empty = sum(1 for s in samples if s["final_empty"])
    return {
        "n": n,
        "detection_success_rate": round(detected / n, 4),
        "ocr_exact_match_rate": round(exact / n, 4),
        "char_accuracy_mean": round(
            sum(s["char_accuracy"] for s in samples) / n, 4),
        "ocr_failure_rate": round(empty / n, 4),
        "ocr_failure_note": "final OCR text empty / status OCR_FAILED",
        "exact_match_given_detection": (
            round(exact / detected, 4) if detected else None),
        "avg_crop_wh": [round(sum(widths) / n, 1),
                        round(sum(heights) / n, 1)],
        "median_crop_wh": [statistics.median(widths),
                           statistics.median(heights)],
        "categories": cats,
    }


def pool_detection_stats(images: list[dict]) -> dict:
    gt_total = matched = 0
    for im in images:
        for m in im["gt_match"]:
            gt_total += 1
            if m["matched"]:
                matched += 1
    return {"gt_boxes": gt_total, "matched_iou50": matched,
            "recall_iou50": round(matched / gt_total, 4) if gt_total else None}


def gj08_investigation(images: list[dict], samples: list[dict]) -> dict:
    """Locate GJ08DJ3136 / 76J080J3136 in the collected pool."""
    target_gt = "GJ08DJ3136"
    target_out = "76J080J3136"

    out: dict = {
        "target_gt": target_gt,
        "target_system_output": target_out,
        "found_in_gt_transcriptions": False,
        "found_in_pool_raw_or_candidates": [],
    }

    for s in samples:
        if norm(s.get("gt_text")) == norm(target_gt):
            out["found_in_gt_transcriptions"] = True
            out["gt_sample"] = s["sample_id"]

    for im in images:
        hits = []
        for r in im["raw_passes"]:
            rn = norm(r["text"])
            if rn == norm(target_out) or (
                    "J08" in rn and "3136" in rn) or rn == norm(target_gt):
                hits.append({"where": "raw", "text": r["text"],
                             "conf": r["conf"]})
        for p in im["plates"]:
            for tier, lst in (p.get("candidates") or {}).items():
                for c in lst:
                    cn = norm(c["text"])
                    if cn == norm(target_out) or (
                            "J08" in cn and "3136" in cn) or cn == norm(target_gt):
                        hits.append({"where": f"plate{p['plate_id']}/{tier}",
                                     "text": c["text"], "conf": c["conf"]})
            if norm(p.get("ocr_text") or "") == norm(target_out):
                hits.append({"where": f"plate{p['plate_id']}/final",
                             "text": p["ocr_text"],
                             "conf": p["ocr_confidence"]})
        if hits:
            out["found_in_pool_raw_or_candidates"].append(
                {"image": im["image"], "hits": hits})
    return out
