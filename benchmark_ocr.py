"""
OCR BENCHMARK — candidate generation / selection diagnostics.

One-off diagnostic harness (NOT part of the app runtime).

Usage (from project root):
    venv/Scripts/python.exe benchmark_ocr.py --sample 40 --seed 7
    venv/Scripts/python.exe benchmark_ocr.py --full          # + end-to-end latency
    venv/Scripts/python.exe benchmark_ocr.py --cases only    # FAG643 + KL01AP8921 repro only

Ground truths available:
    FAG643  -> runs/detect/predict-2/bbcac63e...jpg bbox [168,170,306,268]
               (verified: final_pipeline_easyocr/result.json, 6/6 OCR votes, 95.3%)
    KL01AP8921 -> original upload not persisted by the backend; reproduced
               synthetically (rendered plate + replicate-padded edges, the
               same failure source flagged in the code history).
Everything else comes from batch_results/*_plate.jpg (no text GT available;
those are scored with strict-format validity + consensus proxies).
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BACKEND = ROOT / "app" / "backend"
sys.path.insert(0, str(BACKEND))

import cv2
import numpy as np

import services.detection_service as ds


# ============================================================
# STRICT INDIAN PLATE FORMAT (current-series + legacy)
# ============================================================

INDIAN_STATES = {
    "AN", "AP", "AR", "AS", "BR", "CG", "CH", "DD", "DL", "DN",
    "GA", "GJ", "HR", "HP", "JH", "JK", "KA", "KL", "LD", "MH",
    "ML", "MN", "MP", "MZ", "NL", "OD", "PB", "PY", "RJ", "SK",
    "TN", "TR", "TS", "TG", "UA", "UK", "UP", "WB",
}

# Current series:  STATE(2) + district(01-99) + series letters(0-3) + 1-4 digits
# Legacy series:   STATE(2) + up to 2 letters + 1-4 digits  (pre-2005 style)
RE_CURRENT = r"^(?P<st>[A-Z]{2})(?P<dt>0[1-9]|[1-9][0-9])(?P<se>[A-Z]{0,3})(?P<no>[0-9]{1,4})$"
RE_LEGACY = r"^(?P<st>[A-Z]{2})(?P<se>[A-Z]{1,2})(?P<no>[0-9]{1,4})$"

RE_CURRENT_C = re.compile(RE_CURRENT)
RE_LEGACY_C = re.compile(RE_LEGACY)


def strict_indian_parse(s: str):
    """Return (ok, groups) for a strictly-valid Indian plate string."""
    m = RE_CURRENT_C.match(s)
    if m and m.group("st") in INDIAN_STATES:
        return True, m.groupdict()
    m = RE_LEGACY_C.match(s)
    if m and m.group("st") in INDIAN_STATES:
        return True, m.groupdict()
    return False, None


# ============================================================
# CHAR-LEVEL CONFIDENCE (standalone, mirrors _run_ocr settings)
# ============================================================

def get_char_conf(reader, img, allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-"):
    """EasyOCR recognize(detail=2) -> (text, mean_conf, [(char, conf), ...])."""
    import torch
    with torch.inference_mode():
        result = reader.recognize(
            img,
            detail=2,
            paragraph=False,
            allowlist=allowlist,
        )
    if not result:
        return "", 0.0, []
    # one image -> one entry
    _, text, conf = result[0]
    text_c = ds.clean_text(text)
    if not text_c:
        return "", 0.0, []
    # detail=2 gives no per-char info; per-char comes via the CTC probe below
    return text_c, float(conf), None


def char_conf_profile(reader, img):
    """
    Per-character confidence via the recognizer's own CTC distribution:
    greedy max-prob per emitted timestep, collapsed like decode_greedy.
    Read-only use of installed EasyOCR internals (no monkeypatching here).
    """
    import torch
    import torch.nn.functional as F

    recog = reader.recognizer
    converter = reader.converter  # converter.character[0] == CTC blank

    imgH = getattr(reader, "imgH", 64)
    y_max, x_max = img.shape[:2]
    from easyocr.recognition import AlignCollate, ListDataset

    # replicate what recognize() does for a full-frame read
    from easyocr.utils import get_image_list

    allowlist = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-"
    # converter.character[0] is the CTC blank; dict chars start at 1.
    # ignore_idx uses the dict-char index + 1 == model class index
    # (mirrors easyocr.recognition.get_text's construction).
    dict_chars = converter.character[1:]
    ignore_idx = [0] + [
        dict_chars.index(c) + 1
        for c in set(dict_chars) - set(allowlist)
    ]

    image_list, max_width = get_image_list([[0, x_max, 0, y_max]], [], img, model_height=imgH)
    imgW = int(max_width)
    collate = AlignCollate(imgH=imgH, imgW=imgW, keep_ratio_with_pad=True)
    loader = torch.utils.data.DataLoader(
        ListDataset([im for _, im in image_list]), batch_size=1, shuffle=False,
        collate_fn=collate, pin_memory=False,
    )

    out = []
    for image_tensors in loader:
        batch_max_length = int(imgW / 10)
        length_for_pred = torch.IntTensor([batch_max_length])
        text_for_pred = torch.LongTensor(1, batch_max_length + 1).fill_(0)

        preds = recog(image_tensors, text_for_pred)
        preds_prob = F.softmax(preds, dim=2)
        preds_prob = preds_prob.cpu().detach().numpy()
        # zero blank + every char outside the allowlist (mirrors
        # recognizer_predict's ignore_idx filtering)
        for idx_ig in ignore_idx:
            preds_prob[:, :, idx_ig] = 0.0
        pred_norm = preds_prob.sum(axis=2)
        preds_prob = preds_prob / np.expand_dims(pred_norm, axis=-1)

        values = preds_prob.max(axis=2)[0]
        indices = preds_prob.argmax(axis=2)[0]

        # collapse CTC repeats (same rule as decode_greedy)
        t = indices
        a = np.insert(~((t[1:] == t[:-1])), 0, True)
        keep = a & (t != 0)
        chars = [converter.character[i] for i in t[keep.nonzero()]]
        probs = [float(v) for v, k in zip(values, keep) if k]
        out.append(("".join(chars), probs))
    if not out:
        return "", []
    text, probs = out[0]
    text_c = ds.clean_text(text)
    if len(text_c) != len(probs):
        # '-' or filtered chars shift alignment; trim conservatively
        probs = probs[: len(text_c)]
    return text_c, probs


# ============================================================
# PIPELINE PROBES (through the service's own functions)
# ============================================================

def run_tiers_on_crop(gray):
    """Run the service's tier variants, collecting candidates + timings."""
    evidence = []
    times = []

    enhanced = ds._variant_primary(gray)
    t0 = time.perf_counter()
    c1 = ds._run_ocr(enhanced)
    times.append(("tier1_CLAHE", time.perf_counter() - t0))
    evidence.append((1, c1))

    t0 = time.perf_counter()
    c2 = ds._run_ocr(ds._variant_fallback(enhanced))
    times.append(("tier2_OTSU", time.perf_counter() - t0))
    evidence.append((2, c2))

    t0 = time.perf_counter()
    c3 = ds._run_ocr(ds._variant_tertiary(enhanced))
    times.append(("tier3_denoise", time.perf_counter() - t0))
    evidence.append((3, c3))

    return evidence, times


# ============================================================
# CASE REPORTS
# ============================================================

def fmt_conf_profile(text, probs):
    if not probs:
        return "n/a"
    return " ".join(f"{c}:{p:.2f}" for c, p in zip(text, probs))


def case_report(name, crop_bgr, gt, reader, explain_scores=True, baseline=False):
    print(f"\n{'='*78}\nCASE: {name}   ground_truth={gt!r}\n{'='*78}")

    if baseline:
        # ORIGINAL committed path: resize+gray, NO text-region trim,
        # selection without strict-format/repairs/edge evidence.
        base = ds._prepare_ocr_canvas(crop_bgr)
        gray = cv2.cvtColor(base, cv2.COLOR_BGR2GRAY)
    else:
        # PRODUCTION canvas path (resize + grayscale + text-region trim)
        gray = ds._canvas_gray(crop_bgr)

    evidence, times = run_tiers_on_crop(gray)

    # raw EasyOCR (unfiltered detail=2 view, what EasyOCR literally returns)
    raw_view = []
    for tier, cands in evidence:
        for item in cands:
            raw_view.append((tier, item[0], item[1]))

    print("\n-- Raw EasyOCR candidates per tier (cleaned) --")
    for tier, text, conf in raw_view:
        ok, _ = strict_indian_parse(text)
        print(f"  tier{tier}: {text!r:<14} conf={conf:.3f}  strict_valid={ok}  len={len(text)}")

    # char-level confidence on the CLAHE variant (skip in baseline mode:
    # the original pipeline has no char-evidence hook)
    reader = reader or getattr(ds, "_ocr_reader", None)
    if reader is not None:
        enhanced = ds._variant_primary(gray)
        text, probs = char_conf_profile(reader, enhanced)
        print("\n-- Char-level confidence (CLAHE, CTC greedy max-prob) --")
        print(f"  text={text!r}")
        print(f"  profile: {fmt_conf_profile(text, probs)}")
        if probs:
            weakest = min(range(len(probs)), key=lambda i: probs[i]) if probs else None
            if weakest is not None and len(text) == len(probs):
                print(f"  weakest char: {text[weakest]!r} @ {probs[weakest]:.3f} (position {weakest+1}/{len(text)})")

    # selection through the service's own full pipeline
    if baseline:
        best = ds._evidence_select(evidence)
        repairs = []
    else:
        best, repairs = ds._finalize_selection(evidence)

    if repairs:
        print("\n-- Evidence-gated format repairs (tier 5) --")
        for r in repairs:
            src_conf = r[1] / 0.90 if r[1] else 0.0
            print(f"  {r[0]!r:<14} conf={r[1]:.3f} (source {src_conf:.3f})")

    if best is not None:
        sel_text, sel_conf, sel_votes = best
        ok, groups = strict_indian_parse(sel_text)
        exact = sel_text == gt
        print("\n-- FINAL SELECTION (service pipeline) --")
        print(f"  selected={sel_text!r}  conf={sel_conf:.3f}  votes={sel_votes}")
        print(f"  strict_valid={ok}  EXACT_MATCH={exact}")
        if explain_scores and not exact:
            explain_why(sel_text, evidence, gt)
    else:
        print("\n-- FINAL SELECTION: none --")

    total_ocr = sum(t for _, t in times)
    print(f"\n  ocr_passes=3  total_ocr={total_ocr*1000:.0f} ms  "
          f"(CLAHE {times[0][1]*1000:.0f} / OTSU {times[1][1]*1000:.0f} / denoise {times[2][1]*1000:.0f})")
    return best


def explain_why(sel_text, evidence, gt):
    """Score decomposition of the winner vs the GT-shaped alternative."""
    groups = {}
    for _, cands in evidence:
        for item in cands:
            groups.setdefault(item[0], []).append(item[1])
    print("\n  -- why this candidate won (score components) --")
    rows = []
    for text, confs in groups.items():
        if len(text) < 6:
            continue
        confs_ = [float(c) for c in confs]
        avg = sum(confs_) / len(confs_)
        fmt = ds.indian_plate_score(text)
        votes = len(confs_)
        score = (
            max(confs_) * 35.0 + avg * 15.0 + fmt * 0.90
            + (25.0 if text[:2].isalpha() else 0.0)
            + (20.0 if 8 <= len(text) <= 12 else 8.0 if 6 <= len(text) <= 13 else 0.0)
            + (min(votes - 1, 2) * 7.0)
        )
        rows.append((score, text, avg, max(confs_), fmt, votes))
    rows.sort(reverse=True)
    for score, text, avg, mx, fmt, votes in rows[:4]:
        mark = "  <-- SELECTED" if text == sel_text else ""
        gtm = "  (== GT)" if text == gt else ""
        print(f"    {text!r:<14} score={score:6.1f} avg_conf={avg:.3f} max_conf={mx:.3f} "
              f"format={fmt:3d} votes={votes}{mark}{gtm}")


# ============================================================
# POPULATION BENCHMARK
# ============================================================

def population(sample_n, seed, tag="after", baseline=False):
    crops = sorted(ROOT.glob("batch_results/*_plate.jpg"))
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(crops), size=min(sample_n, len(crops)), replace=False)
    chosen = [crops[i] for i in idx]

    print(f"\n{'='*78}\nPOPULATION: {len(chosen)} real plate crops "
          f"(seed={seed}, of {len(crops)})\n{'='*78}")
    print(f"{'file':<18} {'raw_max_conf':<14} {'len':>4} {'strict':>7} "
          f"{'winner':<14} {'conf':>6} {'votes':>5}")

    strict_count = 0
    pass_times = []
    cand_counts = []
    len_hist = Counter()
    rows_detail = []

    for path in chosen:
        bgr = cv2.imread(str(path))
        if bgr is None:
            continue
        if baseline:
            base = ds._prepare_ocr_canvas(bgr)
            gray = cv2.cvtColor(base, cv2.COLOR_BGR2GRAY)
        else:
            gray = ds._canvas_gray(bgr)
        evidence, times = run_tiers_on_crop(gray)
        if baseline:
            best = ds._evidence_select(evidence)
        else:
            best, _ = ds._finalize_selection(evidence)

        flat = [item for _, cc in evidence for item in cc]
        if not flat:
            continue
        top = max(flat, key=lambda x: x[1])
        len_hist[len(top[0])] += 1

        cand_counts.append(len(flat))
        pass_times.extend(t for _, t in times)

        if best:
            sel_text, sel_conf, votes = best
            ok, _ = strict_indian_parse(sel_text)
            strict_count += int(ok)
            print(f"{path.name:<18} {top[0]!r:<14} {len(top[0]):>4} {str(ok):>7} "
                  f"{sel_text!r:<14} {sel_conf:>6.3f} {votes:>5}")
            rows_detail.append({
                "file": path.name, "raw_top": top[0], "raw_conf": round(top[1], 4),
                "selected": sel_text, "sel_conf": round(sel_conf, 4),
                "votes": votes, "strict_valid": ok,
            })
        else:
            print(f"{path.name:<18} {top[0]!r:<14} {len(top[0]):>4} {'-':>7} "
                  f"{'(none)':<14} {'-':>6} {'-':>5}")

    n = max(1, len(rows_detail))
    print("\n-- aggregates --")
    print(f"  strict-format-valid selections : {strict_count}/{n} ({100*strict_count/n:.1f}%)")
    print(f"  mean OCR pass time             : {1000*statistics.mean(pass_times):.0f} ms"
          f"  (n={len(pass_times)} passes)")
    print(f"  mean candidates per plate      : {statistics.mean(cand_counts):.1f}")
    print(f"  raw read length histogram      : {dict(sorted(len_hist.items()))}")

    return rows_detail, {
        "strict_valid": strict_count, "n": n,
        "mean_pass_ms": 1000 * statistics.mean(pass_times) if pass_times else 0,
        "mean_candidates": statistics.mean(cand_counts) if cand_counts else 0,
        "len_hist": dict(len_hist),
    }


# ============================================================
# SYNTHETIC KL CASE
# ============================================================

def _load_font(px=72):
    """Prefer a real Windows TTF (much closer to plate fonts than the
    Hershey strokes); fall back to OpenCV's built-in face. Arial bold
    first: Consolas' slashed zero reads as 'O' even for humans."""
    for name in ("arialbd.ttf", "arial.ttf", "segoeui.ttf", "consolab.ttf"):
        p = Path("C:/Windows/Fonts") / name
        if p.exists():
            try:
                from PIL import ImageFont
                return ("ttf", ImageFont.truetype(str(p), px))
            except Exception:
                pass
    return ("hershey", None)


def make_kl_crop(text="KL01AP8921", corrupt_edges=True):
    """Render a single-line Indian plate; optionally add replicate-padded
    edge strips — the exact failure source the code comments blame for
    phantom leading/trailing characters (an over-wide YOLO box does the
    same thing to real crops)."""
    img = np.full((220, 900, 3), 245, dtype=np.uint8)

    kind, font = _load_font(64)
    if kind == "ttf":
        from PIL import Image, ImageDraw
        pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil)
        bbox = draw.textbbox((0, 0), text, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        draw.text(((900 - tw) // 2 - bbox[0], (220 - th) // 2 - bbox[1]),
                  text, font=font, fill=(20, 20, 20))
        img = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    else:
        cv2.putText(img, text, (40, 165), cv2.FONT_HERSHEY_DUPLEX, 3.2,
                    (15, 15, 15), 7, cv2.LINE_AA)

    cv2.rectangle(img, (8, 8), (891, 211), (15, 15, 15), 5)
    if corrupt_edges:
        pad = 40
        img = cv2.copyMakeBorder(img, 0, 0, pad, pad, cv2.BORDER_REPLICATE)
    return img


# ============================================================
# END-TO-END LATENCY
# ============================================================

def end_to_end(car_images):
    print(f"\n{'='*78}\nEND-TO-END (service entry point, incl. YOLO)\n{'='*78}")
    results = []
    for path in car_images:
        data = path.read_bytes()
        t0 = time.perf_counter()
        out = ds.run_detection_on_image(data)
        dt = time.perf_counter() - t0
        texts = [p["ocr_text"] for p in out["plates"]]
        print(f"  {path.name:<18} {dt*1000:7.0f} ms  plates={out['plates_detected']}  texts={texts}")
        results.append((path.name, dt, texts))
    return results


# ============================================================
# MAIN
# ============================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=40)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--full", action="store_true", help="+ end-to-end latency on car images")
    ap.add_argument("--cases", choices=["with", "only"], default="with")
    ap.add_argument("--tag", default="after")
    ap.add_argument("--baseline", action="store_true",
                    help="benchmark the ORIGINAL committed pipeline (old canvas/selector)")
    args = ap.parse_args()

    t_start = time.perf_counter()

    if not args.full:
        ds._ensure_models_loaded()
        print(f"models loaded in {time.perf_counter()-t_start:.1f}s "
              f"(YOLO + EasyOCR recognizer-only)")

    reader = getattr(ds, "_ocr_reader", None)

    if args.full:
        # FIRST REQUEST includes lazy model load — exactly what a fresh
        # Render container serves. Subsequent calls are warm.
        cars = sorted(ROOT.glob("batch_results/*_Cars*.jpg"))[:6]
        print(f"\n{'='*78}\nEND-TO-END (service entry, first request includes model load)\n{'='*78}")
        for i, path in enumerate(cars):
            data = path.read_bytes()
            t0 = time.perf_counter()
            out = ds.run_detection_on_image(data)
            dt = time.perf_counter() - t0
            texts = [p["ocr_text"] for p in out["plates"]]
            label = "FIRST (incl. load)" if i == 0 else "warm"
            print(f"  [{label:>17}] {path.name:<18} {dt*1000:8.0f} ms  "
                  f"plates={out['plates_detected']}  texts={texts}")

    # ---- known-GT case: FAG643 ----
    fag_src = ROOT / "runs/detect/predict-2/bbcac63e32bd8137_jpg.rf.ef4704b0ada4fbbf613143abf52f6f86.jpg"
    if fag_src.exists():
        img = cv2.imread(str(fag_src))
        crop = img[170:268, 168:306]
        case_report("FAG643 (known GT, real photo)", crop, "FAG643", reader,
                    baseline=args.baseline)

    # ---- synthetic KL01AP8921 repro ----
    kl = make_kl_crop(corrupt_edges=True)
    case_report("KL01AP8921 (synthetic repro: replicate-padded edges)",
                kl, "KL01AP8921", reader, baseline=args.baseline)
    kl_clean = make_kl_crop(corrupt_edges=False)
    case_report("KL01AP8921 (synthetic, clean edges)", kl_clean, "KL01AP8921", reader,
                baseline=args.baseline)

    if args.cases != "only":
        rows, agg = population(args.sample, args.seed, tag=args.tag,
                               baseline=args.baseline)

        out_dir = ROOT / "batch_results" / "ocr_benchmark"
        out_dir.mkdir(parents=True, exist_ok=True)
        tag = "baseline" if args.baseline else args.tag
        payload = {
            "mode": tag,
            "seed": args.seed,
            "sample": args.sample,
            "rows": rows,
            "aggregates": agg,
            "wall_seconds": round(time.perf_counter() - t_start, 1),
        }
        out_file = out_dir / f"benchmark_{tag}.json"
        out_file.write_text(json.dumps(payload, indent=2))
        print(f"\nsaved -> {out_file.relative_to(ROOT)}")

    if args.full and args.cases == "only":
        pass  # end-to-end already ran above

    if args.full and not args.baseline and args.cases != "only":
        # extra warm end-to-end confirmation for the new pipeline
        cars = sorted(ROOT.glob("batch_results/*_Cars*.jpg"))[:6]
        end_to_end(cars[3:])


if __name__ == "__main__":
    main()
