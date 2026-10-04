"""
DIAGNOSTIC HARNESS — per-image, per-candidate pipeline measurement (STEP 2).

NOT part of the app runtime. Answers, with evidence:
  1. What did YOLO detect (conf, bbox, crop size/aspect)?
  2. What OCR candidates did EVERY tier produce, BEFORE scoring?
  3. Which candidate was selected and why (score breakdown)?
  4. Stage timings (YOLO / preprocess / OCR / scoring) + peak RSS.

Usage:
    venv/Scripts/python.exe diagnose_plates.py            # all cases
    venv/Scripts/python.exe diagnose_plates.py --crops    # crop-level deep dive only
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BACKEND = ROOT / "app" / "backend"
sys.path.insert(0, str(BACKEND))

import cv2
import numpy as np

import services.detection_service as ds


# ============================================================
# MEMORY
# ============================================================

try:
    import psutil

    _PROC = psutil.Process()

    def rss_mb() -> float:
        return _PROC.memory_info().rss / (1024 * 1024)

    def peak_mb() -> float:
        return _PROC.memory_info().peak_wset / (1024 * 1024)

except Exception:  # pragma: no cover
    def rss_mb() -> float:
        return 0.0

    def peak_mb() -> float:
        return 0.0


# ============================================================
# SYNTHETIC TWO-LINE PLATES (realistic stand-ins for sceneA/sceneB)
# ============================================================

def _load_font(px=72):
    for name in ("arialbd.ttf", "arial.ttf", "segoeui.ttf"):
        p = Path("C:/Windows/Fonts") / name
        if p.exists():
            try:
                from PIL import ImageFont
                return ImageFont.truetype(str(p), px)
            except Exception:
                pass
    return None


def render_two_line_plate(top: str, bottom: str, width: int = 520) -> np.ndarray:
    """White motorcycle plate, two stacked black text lines, thin frame."""
    h = 340
    img = np.full((h, width, 3), 235, dtype=np.uint8)
    font = _load_font(96)

    if font is not None:
        from PIL import Image, ImageDraw
        pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil)
        for text, cy in ((top, h * 0.28), (bottom, h * 0.72)):
            bbox = draw.textbbox((0, 0), text, font=font)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
            draw.text(
                ((width - tw) // 2 - bbox[0], int(cy) - th // 2 - bbox[1]),
                text, font=font, fill=(25, 25, 25),
            )
        img = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    else:  # pragma: no cover
        cv2.putText(img, top, (60, 140), cv2.FONT_HERSHEY_DUPLEX, 3.0,
                    (25, 25, 25), 8, cv2.LINE_AA)
        cv2.putText(img, bottom, (90, 300), cv2.FONT_HERSHEY_DUPLEX, 3.0,
                    (25, 25, 25), 8, cv2.LINE_AA)

    cv2.rectangle(img, (6, 6), (width - 7, h - 7), (25, 25, 25), 4)
    return img


def make_two_plate_scene(plates: list[tuple[str, str]]) -> np.ndarray:
    """Two motorcycle plates mounted side by side on a textured backdrop."""
    rng = np.random.default_rng(42)
    scene = np.full((480, 1280, 3), 96, dtype=np.uint8)
    scene += rng.integers(-12, 12, scene.shape, dtype=np.int16).astype(np.uint8)

    xs = (100, 700)
    for (top, bottom), x in zip(plates, xs):
        plate = render_two_line_plate(top, bottom)
        plate = cv2.resize(plate, (420, 280), interpolation=cv2.INTER_AREA)
        plate = cv2.GaussianBlur(plate, (3, 3), 0)
        # slight perspective jitter so crops are not pixel-perfect
        hs, ws = plate.shape[:2]
        src = np.float32([[0, 0], [ws, 0], [ws, hs], [0, hs]])
        dst = np.float32([[3, 5], [ws - 2, 0], [ws - 4, hs - 3], [1, hs - 6]])
        plate = cv2.warpPerspective(
            plate, cv2.getPerspectiveTransform(src, dst), (ws, hs)
        )
        scene[100:100 + hs, x:x + ws] = plate

    return scene


# ============================================================
# CANDIDATE-VERBOSE CASCADE (mirrors run_detection_on_image exactly)
# ============================================================

def _fmt_cands(cands) -> str:
    if not cands:
        return "(none)"
    return ", ".join(
        f"'{t}' conf={c:.3f}" for t, c, *_ in cands
    )


def diagnose_crop(tag: str, crop: np.ndarray) -> None:
    """Run the exact service cascade on one crop, printing every candidate."""
    print(f"\n  --- crop diagnosis: {tag} "
          f"({crop.shape[1]}x{crop.shape[0]}, aspect="
          f"{crop.shape[0] / max(crop.shape[1], 1):.2f}) ---")

    t0 = time.perf_counter()
    base_gray = ds._canvas_gray(crop)
    t_canvas = time.perf_counter() - t0
    print(f"  canvas: {base_gray.shape[1]}x{base_gray.shape[0]} "
          f"({t_canvas * 1000:.0f} ms incl. tighten)")

    tier_evidence: list[tuple[int, list]] = []

    enhanced = ds._variant_primary(base_gray)

    t0 = time.perf_counter()
    tier1 = ds._run_ocr(enhanced)
    t_ocr1 = time.perf_counter() - t0
    print(f"  TIER1 (CLAHE)  [{t_ocr1 * 1000:.0f} ms]: {_fmt_cands(tier1)}")
    tier_evidence.append((1, tier1))

    best = ds._consensus_select(tier1)
    low_effort = best is not None  # yolo gate checked by caller
    strong = ds._is_strong_enough(best)
    print(f"  tier1 select: {best and best[0]!r} strong={strong} low_effort={low_effort}")

    if not low_effort or not strong:
        t0 = time.perf_counter()
        tier2 = ds._run_ocr(ds._variant_fallback(enhanced))
        t_ocr2 = time.perf_counter() - t0
        print(f"  TIER2 (OTSU)   [{t_ocr2 * 1000:.0f} ms]: {_fmt_cands(tier2)}")
        tier_evidence.append((2, tier2))
        best = ds._evidence_select(tier_evidence)

        tier2_new = any(
            ds.clean_text(i[0]) not in {ds.clean_text(p[0]) for p in tier1}
            for i in tier2
        )
        if (best is None or not ds._is_strong_enough(best)) and tier2_new:
            t0 = time.perf_counter()
            tier3 = ds._run_ocr(ds._variant_tertiary(enhanced))
            t_ocr3 = time.perf_counter() - t0
            print(f"  TIER3 (denoise)[{t_ocr3 * 1000:.0f} ms]: {_fmt_cands(tier3)}")
            tier_evidence.append((3, tier3))
            best = ds._evidence_select(tier_evidence)

    crop_aspect = float(base_gray.shape[0]) / max(float(base_gray.shape[1]), 1.0)
    whole_crop_weak = (
        best is None
        or not ds._is_strong_enough(best)
        or (best is not None and not ds._edge_supported_in(best[0], tier1))
    )
    print(f"  tier4 gate: aspect={crop_aspect:.2f} "
          f"(>= {ds._MAX_SPLIT_ASPECT}) weak={whole_crop_weak}")

    if crop_aspect >= ds._MAX_SPLIT_ASPECT and whole_crop_weak:
        bands = ds._split_two_line_bands(enhanced)
        print(f"  TIER4 split bands: {bands}")
        if bands:
            # per-band verbose (mirrors _recognize_bands internals)
            for bi, (y0, y1) in enumerate(bands):
                band = enhanced[max(0, y0):max(0, y1), :]
                band = ds._tight_text_region(band)
                if band.size == 0:
                    print(f"    band{bi} rows[{y0}:{y1}]: EMPTY after tighten")
                    continue
                pad = max(4, band.shape[0] // 5)
                border_value = int(np.percentile(band, 70))
                band = cv2.copyMakeBorder(
                    band, pad, pad, pad, pad,
                    cv2.BORDER_CONSTANT, value=border_value,
                )
                t0 = time.perf_counter()
                reads = ds._run_ocr(band)
                print(f"    band{bi} rows[{y0}:{y1}] "
                      f"({band.shape[1]}x{band.shape[0]}) "
                      f"[{time.perf_counter() - t0:.0f} s]: {_fmt_cands(reads)}")

            t0 = time.perf_counter()
            band_cands = ds._recognize_bands(enhanced, base_gray, bands)
            print(f"  TIER4 combined  : {_fmt_cands(band_cands)}")
            if band_cands:
                tier_evidence.append((4, band_cands))
                best = ds._evidence_select(tier_evidence)

    if tier_evidence:
        t0 = time.perf_counter()
        sel, repairs = ds._finalize_selection(tier_evidence)
        t_sel = time.perf_counter() - t0
        if repairs:
            print(f"  TIER5 repairs   : {_fmt_cands(repairs)}")
        print(f"  selection took {t_sel * 1000:.0f} ms")
        if sel is not None:
            text, conf, votes = sel
            fs = ds.indian_plate_score(text)
            strict = ds.strict_indian_plate(text)
            final = min(1.0, conf * 0.6 + fs / 100 * 0.4)
            print(f"  SELECTED: '{text}' ocr_conf={conf:.3f} votes={votes} "
                  f"format={fs} strict={strict} final={final:.3f} "
                  f"status={ds.status_from_confidence(final, text)}")
        else:
            print("  SELECTED: None -> OCR_FAILED / UNKNOWN")
    return


def geometry_report(yolo_conf: float, x1: int, y1: int, x2: int, y2: int,
                    img_w: int, img_h: int) -> None:
    w, h = x2 - x1, y2 - y1
    aspect = h / max(w, 1)
    area_frac = (w * h) / (img_w * img_h)
    print(f"    box conf={yolo_conf:.3f} bbox=({x1},{y1},{x2},{y2}) "
          f"size={w}x{h} aspect(h/w)={aspect:.2f} area={area_frac * 100:.1f}% of image")


# ============================================================
# FULL-IMAGE DIAGNOSIS (YOLO included, service path)
# ============================================================

def diagnose_image(name: str, data: bytes, deep_crop_indices: list[int]) -> None:
    print(f"\n{'=' * 78}\nIMAGE: {name}  ({len(data) // 1024} KB)\n{'=' * 78}")

    t0 = time.perf_counter()
    out = ds.run_detection_on_image(data)
    total = time.perf_counter() - t0

    for p in out["plates"]:
        geometry_report(p["yolo_confidence"], *p["bbox"],
                        out["image"]["width"], out["image"]["height"])
        print(f"    -> ocr='{p['ocr_text']}' ocr_conf={p['ocr_confidence']} "
              f"final={p['final_confidence']} status={p['status']} "
              f"strict={p.get('strict_format')} edge={p.get('edge_confidence')}")
    print(f"  TOTAL: {total:.2f}s (service reports "
          f"{out['processing_time_seconds']}s)  plates={out['plates_detected']}")
    print(f"  RSS now {rss_mb():.0f} MB, peak {peak_mb():.0f} MB")

    # deep dive on selected boxes (reruns cascade with full printing)
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    h, w = img.shape[:2]
    if max(h, w) > ds._MAX_INFER_SIDE:
        s = ds._MAX_INFER_SIDE / max(h, w)
        img = cv2.resize(img, (max(1, int(w * s)), max(1, int(h * s))))

    import torch
    with torch.inference_mode():
        results = ds._yolo_model.predict(source=img, imgsz=640, conf=0.25, verbose=False)
    detections = []
    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        detections.append((float(box.conf[0]), x1, y1, x2, y2))
    del results
    detections.sort(key=lambda d: d[0], reverse=True)

    for i, (yolo_conf, x1, y1, x2, y2) in enumerate(detections, start=1):
        if i not in deep_crop_indices:
            continue
        crop = img[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
        diagnose_crop(f"plate{i} (yolo_conf={yolo_conf:.3f})", crop)


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    ds._ensure_models_loaded()
    print(f"models loaded. RSS {rss_mb():.0f} MB\n")

    cases: list[tuple[str, bytes, list[int]]] = []

    # 1+2. the two reported plates, each as its own synthetic scene
    scene1 = make_two_plate_scene([("37-N1", "4635")])
    ok, buf = cv2.imencode(".jpg", scene1)
    cases.append(("synthetic scene: 37-N1/4635 (plate1 repro)", buf.tobytes(), [1]))

    scene2 = make_two_plate_scene([("37-EA", "6789")])
    ok, buf = cv2.imencode(".jpg", scene2)
    cases.append(("synthetic scene: 37-EA/6789 (plate2 repro)", buf.tobytes(), [1]))

    # both plates in one frame (matches the user's description)
    scene_both = make_two_plate_scene([("37-N1", "4635"), ("37-EA", "6789")])
    ok, buf = cv2.imencode(".jpg", scene_both)
    cases.append(("synthetic scene: BOTH plates one frame", buf.tobytes(), [1, 2]))

    # 3. known-GT real photo FAG643 (crop direct, no YOLO)
    fag_src = ROOT / "runs/detect/predict-2/bbcac63e32bd8137_jpg.rf.ef4704b0ada4fbbf613143abf52f6f86.jpg"
    if fag_src.exists():
        img = cv2.imread(str(fag_src))
        crop = img[170:268, 168:306]
        print(f"\n{'=' * 78}\nDIRECT CROP: FAG643 (known GT)\n{'=' * 78}")
        diagnose_crop("FAG643 manual bbox [168,170,306,268]", crop)

    # 4. Cars102 full image (historical text 68361136)
    cars102 = ROOT / "batch_results/0526_Cars102.jpg"
    if cars102.exists():
        cases.append(("Cars102 (historical read 68361136)", cars102.read_bytes(), [1]))

    # 5. Cars0 (KL01CA2555 from earlier rounds)
    cars0 = ROOT / "batch_results/0001_Cars0.jpg"
    if cars0.exists():
        cases.append(("Cars0 (prior rounds: KL01CA2555)", cars0.read_bytes(), [1]))

    # 6. KL01AP8921 regression (synthetic, corrupt edges)
    kl = ds.__dict__.get("_kl_cache")
    sys.path.insert(0, str(ROOT))
    from benchmark_ocr import make_kl_crop
    kl_img = make_kl_crop("KL01AP8921", corrupt_edges=True)
    ok, buf = cv2.imencode(".jpg", kl_img)
    cases.append(("KL01AP8921 synthetic (corrupt edges)", buf.tobytes(), [1]))

    for name, data, deep in cases:
        diagnose_image(name, data, deep)

    print(f"\nFINAL peak RSS: {peak_mb():.0f} MB")


if __name__ == "__main__":
    main()
