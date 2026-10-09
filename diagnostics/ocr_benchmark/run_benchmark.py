# OCR BENCHMARK RUNNER — executes all experiments and writes results.json.
#
# Read-only. Imports benchmark_core which imports the production service.
# Never mutates production files.
from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

import cv2
import numpy as np

import benchmark_core as bc
import services.detection_service as ds


def pct(x):
    return round(100.0 * x, 2)


def summarize_lat(lat):
    if not lat:
        return {"avg": 0.0, "p50": 0.0, "p95": 0.0}
    s = sorted(lat)
    return {
        "avg": round(sum(s) / len(s) * 1000, 1),
        "p50": round(s[len(s) // 2] * 1000, 1),
        "p95": round(s[min(len(s) - 1, int(0.95 * len(s)))] * 1000, 1),
    }


def load_gt_samples():
    """Build the real ground-truth sample list (only 3 human-verified)."""
    plates = bc.gt_plates()
    samples = []
    for pid, meta in sorted(plates.items()):
        scene = meta["scene"]
        boxes = bc.gt_pixel_boxes(scene)
        if not boxes:
            continue
        x1, y1, x2, y2 = boxes[0]
        img = cv2.imread(str(bc.IMAGES_DIR / f"{scene}.png"))
        if img is None:
            continue
        crop = img[y1:y2, x1:x2]
        samples.append({
            "plate_id": pid,
            "scene": scene,
            "gt_text": meta.get("text"),
            "readable": bool(meta.get("readable")),
            "layout": meta.get("layout"),
            "gt_confidence": meta.get("confidence"),
            "crop_w": int(crop.shape[1]),
            "crop_h": int(crop.shape[0]),
            "crop": crop,
        })
    return samples


def production_baseline(crop):
    """Full production cascade on a crop -> final OCR text + confidence.
    Mirrors run_detection_on_image's per-plate OCR block (tier1 CLAHE, the
    production selector/repairs). Returns dict."""
    base_gray = ds._canvas_gray(crop)
    enhanced = ds._variant_primary(base_gray)
    t1, lat1 = bc.run_ocr(enhanced)
    tier_evidence = [(1, t1)]
    best, repairs = ds._finalize_selection([list(te) for te in tier_evidence])
    final_text = ds.clean_text(best[0]) if best else ""
    final_conf = float(best[1]) if best else 0.0
    return {
        "final_text": final_text,
        "final_conf": round(final_conf, 4),
        "tier1_raw": [(ds.clean_text(c[0]), round(float(c[1]), 4))
                      for c in t1],
        "latency": lat1,
    }


def eval_variants(crop, gt_text):
    """Experiment 2: every preprocessing variant, single production pass."""
    out = {}
    for name, fn in bc.VARIANTS:
        try:
            img = fn(crop)
        except Exception:
            img = None
        if img is None:
            out[name] = {"text": "", "conf": 0.0, "exact": False,
                         "char_acc": 0.0, "fail": True, "lat": 0.0}
            continue
        cands, lat = bc.run_ocr(img)
        b = bc.best_candidate(cands)
        text = b[0] if b else ""
        conf = b[1] if b else 0.0
        exact = bool(gt_text) and bc.norm(text) == bc.norm(gt_text)
        out[name] = {
            "text": text,
            "conf": round(conf, 4),
            "exact": exact,
            "char_acc": round(bc.char_acc(text, gt_text or ""), 4),
            "fail": text == "",
            "lat": round(lat, 4),
        }
    return out


def multipass(crop, gt_text):
    """Experiment 3: run every variant, keep ALL raw candidates."""
    all_cands = []          # (variant, text, conf, probs)
    per_variant = {}
    total_lat = 0.0
    for name, fn in bc.VARIANTS:
        try:
            img = fn(crop)
        except Exception:
            img = None
        if img is None:
            per_variant[name] = []
            continue
        cands, lat = bc.run_ocr(img)
        total_lat += lat
        rows = []
        for c in cands:
            text = ds.clean_text(c[0])
            if not text:
                continue
            probs = c[2] if len(c) > 2 else None
            all_cands.append((name, text, float(c[1]), probs))
            rows.append((text, round(float(c[1]), 4)))
        per_variant[name] = rows
    return {
        "all_cands": all_cands,
        "per_variant": per_variant,
        "latency": total_lat,
        "gt_produced": any(bc.norm(t) == bc.norm(gt_text or "")
                           for _, t, _, _ in all_cands),
        "gt_sim_best": max([bc.sim(t, gt_text or "")
                            for _, t, _, _ in all_cands] or [0.0]),
    }


def consensus_and_charconf(mp, gt_text):
    """Experiments 4 & 6 (consensus + ranking) from a multi-pass result."""
    all_cands = mp["all_cands"]
    # Candidate list shaped like production _run_ocr output for the rankers.
    cand_like = [(t, conf, probs) for _, t, conf, probs in all_cands]
    obs = [(t, conf) for _, t, conf, _ in all_cands]

    cons = bc.consensus_vote(obs) or ""
    rc = bc.rank_charconf(cand_like) or ""
    rcons = bc.rank_consensus(cand_like) or ""

    return {
        "consensus": {
            "text": cons,
            "exact": bool(gt_text) and bc.norm(cons) == bc.norm(gt_text),
            "char_acc": round(bc.char_acc(cons, gt_text or ""), 4),
        },
        "rank_charconf": {
            "text": rc,
            "exact": bool(gt_text) and bc.norm(rc) == bc.norm(gt_text),
            "char_acc": round(bc.char_acc(rc, gt_text or ""), 4),
        },
        "rank_consensus": {
            "text": rcons,
            "exact": bool(gt_text) and bc.norm(rcons) == bc.norm(gt_text),
            "char_acc": round(bc.char_acc(rcons, gt_text or ""), 4),
        },
        "n_distinct": len({t for _, t, _, _ in all_cands}),
        "n_raw": len(all_cands),
    }


def char_conf_probe(crop):
    """Experiment 5: is per-character CTC confidence available + usable?"""
    enhanced = ds._variant_primary(ds._canvas_gray(crop))
    cands, _ = bc.run_ocr(enhanced)
    rows = []
    for c in cands:
        probs = c[2] if len(c) > 2 else None
        text = ds.clean_text(c[0])
        if probs:
            rows.append({
                "text": text,
                "conf": round(float(c[1]), 4),
                "chars": list(text),
                "char_conf": [round(float(p), 4) for p in probs],
                "min_char": round(min(float(p) for p in probs), 4),
            })
    return {"available": bool(rows), "rows": rows}


# ── val-pool behavioural metrics (no text GT needed) ───────────────────────
def val_pool_behavioral(limit=None):
    """Measure OCR-failure rate, candidate diversity, char-conf availability
    and latency across the 133 val images using saved crops. Returns dict."""
    coll = bc.load_json(bc.COLLECTION_PATH)["images"]
    lat = []
    fail = 0
    total = 0
    char_conf_avail = 0
    distinct_counts = []
    empty_ocr = 0
    for im in coll:
        for cr in im.get("crops", []):
            if limit is not None and total >= limit:
                break
            f = bc.CROPS_DIR / cr["file"]
            img = cv2.imread(str(f))
            if img is None:
                continue
            total += 1
            enhanced = ds._variant_primary(ds._canvas_gray(img))
            cands, l = bc.run_ocr(enhanced)
            lat.append(l)
            texts = [ds.clean_text(c[0]) for c in cands if ds.clean_text(c[0])]
            distinct_counts.append(len(set(texts)))
            if not texts:
                empty_ocr += 1
            if any(len(c) > 2 and c[2] for c in cands):
                char_conf_avail += 1
        if limit is not None and total >= limit:
            break
    return {
        "n_crops": total,
        "empty_ocr_rate": round(empty_ocr / total, 4) if total else 0.0,
        "char_conf_available_rate": round(char_conf_avail / total, 4) if total else 0.0,
        "avg_distinct_candidates": round(sum(distinct_counts) / len(distinct_counts), 3) if distinct_counts else 0.0,
        "latency": summarize_lat(lat),
    }


def main():
    print("[bench] loading GT samples ...")
    samples = load_gt_samples()
    print(f"[bench] {len(samples)} GT plates: "
          f"{[(s['plate_id'], s['gt_text'], s['readable']) for s in samples]}")

    print("[bench] ensuring production models loaded ...")
    ds._ensure_models_loaded()

    per_sample = {}
    for s in samples:
        pid = s["plate_id"]
        print(f"[bench] === {pid} GT={s['gt_text']!r} "
              f"{s['crop_w']}x{s['crop_h']} ===")
        base = production_baseline(s["crop"])
        variants = eval_variants(s["crop"], s["gt_text"])
        mp = multipass(s["crop"], s["gt_text"])
        cc = consensus_and_charconf(mp, s["gt_text"])
        chc = char_conf_probe(s["crop"])
        gt = s["gt_text"] or ""
        per_sample[pid] = {
            "scene": s["scene"], "gt_text": gt,
            "readable": s["readable"], "layout": s["layout"],
            "crop_w": s["crop_w"], "crop_h": s["crop_h"],
            "aspect": round(s["crop_w"] / max(1, s["crop_h"]), 3),
            "baseline": {
                "final_text": base["final_text"],
                "final_conf": base["final_conf"],
                "exact": bc.norm(base["final_text"]) == bc.norm(gt),
                "char_acc": round(bc.char_acc(base["final_text"], gt), 4),
                "latency": round(base["latency"], 4),
            },
            "variants": variants,
            "multipass": {
                "gt_produced": mp["gt_produced"],
                "gt_sim_best": round(mp["gt_sim_best"], 4),
                "n_distinct": len({t for _, t, _, _ in mp["all_cands"]}),
                "n_raw": len(mp["all_cands"]),
                "latency": round(mp["latency"], 4),
                "raw_list": [(vn, t, round(c, 4))
                             for vn, t, c, _ in mp["all_cands"]][:40],
            },
            "consensus_and_ranking": cc,
            "char_confidence": chc,
        }

    print("[bench] val-pool behavioural pass ...")
    vpool = val_pool_behavioral()

    results = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "note": ("Only 3 human-verified text-GT plates exist; exact-match "
                 "metrics are computed on those. GJ08DJ3136 is not in data."),
        "per_sample": per_sample,
        "val_pool_behavioral": vpool,
    }
    out = bc.OUT_DIR / "results.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"[bench] wrote {out}")


if __name__ == "__main__":
    main()

