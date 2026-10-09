# OCR BENCHMARK - isolated, READ-ONLY experiment harness.
#
# Reproduces the CURRENT production OCR behaviour as BASELINE, then measures
# preprocessing A/B, multi-pass, consensus, character-confidence and an
# offline ranking experiment against the real production pipeline.
#
# It IMPORTS and CALLS the real production functions (services.detection_service)
# so the baseline is byte-for-byte production. It never mutates production.
#
# Ground-truth honesty:
#   * Human-verified TEXT ground truth exists for only 3 plates
#     (diagnostics/audit_gt.json): 0034, 0333 (readable) and 0067 (low-res).
#   * diagnostics/audit/collection.json (133 val images) contains PRODUCTION
#     OCR output but NO human text GT, so exact-match/character accuracy are
#     only computable on the 3 GT plates. Behavioural metrics that need no
#     text GT (OCR-failure rate, candidate diversity, char-confidence
#     availability, latency) are measured across the whole val pool.
#   * "GJ08DJ3136" is NOT present in any data file; it appears only as an
#     illustrative string in diagnostics/audit_report.py. A real per-sample
#     trace of it is therefore impossible and is reported as such.
from __future__ import annotations

import difflib
import json
import re
import statistics
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
BACKEND = ROOT / "app" / "backend"
sys.path.insert(0, str(BACKEND))

import services.detection_service as ds  # noqa: E402  (production, read-only)

GT_PATH = ROOT / "diagnostics" / "audit_gt.json"
COLLECTION_PATH = ROOT / "diagnostics" / "audit" / "collection.json"
CROPS_DIR = ROOT / "diagnostics" / "audit" / "crops"
LABELS_DIR = ROOT / "YOLO_dataset" / "labels" / "train"
IMAGES_DIR = ROOT / "YOLO_dataset" / "images" / "train"
OUT_DIR = ROOT / "diagnostics" / "ocr_benchmark"

FUZZY = 0.80  # similarity threshold for a "correct candidate"


def norm(s: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", str(s or "").upper())


def _lev(a: str, b: str) -> int:
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
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def char_acc(final: str, gt: str) -> float:
    g = norm(gt)
    if not g:
        return 0.0
    return max(0.0, 1.0 - _lev(norm(final), g) / len(g))


def sim(a: str, b: str) -> float:
    a, b = norm(a), norm(b)
    if not a or not b:
        return 0.0
    return difflib.SequenceMatcher(None, a, b).ratio()


def load_json(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))


def gt_plates() -> dict:
    return load_json(GT_PATH)["plates"]


def gt_pixel_boxes(scene: str):
    """All GT boxes (x1,y1,x2,y2) in pixels for a train scene."""
    img = cv2.imread(str(IMAGES_DIR / f"{scene}.png"))
    if img is None:
        return []
    H, W = img.shape[:2]
    p = LABELS_DIR / f"{scene}.txt"
    if not p.exists():
        return []
    boxes = []
    for line in p.read_text().strip().splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        _, cx, cy, w, h = (float(x) for x in parts[:5])
        boxes.append((
            int((cx - w / 2) * W), int((cy - h / 2) * H),
            int((cx + w / 2) * W), int((cy + h / 2) * H),
        ))
    return boxes


def iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / union if union > 0 else 0.0


# ── preprocessing variants (fed to production _run_ocr) ────────────────────
def _upscale(crop: np.ndarray, factor: float = 3.0) -> np.ndarray:
    h, w = crop.shape[:2]
    return cv2.resize(crop, (int(w * factor), int(h * factor)),
                      interpolation=cv2.INTER_CUBIC)


def v_original(crop):
    """A. Original crop (production OCR canvas, grayscale)."""
    return ds._canvas_gray(crop)


def v_clahe(crop):
    """B. CLAHE (production tier-1 primary)."""
    return ds._variant_primary(ds._canvas_gray(crop))


def v_gray_contrast(crop):
    """C. Grayscale + contrast normalization."""
    g = ds._canvas_gray(crop)
    return cv2.normalize(g, None, 0, 255, cv2.NORM_MINMAX)


def v_otsu(crop):
    """D. OTSU thresholding."""
    g = ds._canvas_gray(crop)
    _, t = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return t


def v_adaptive(crop):
    """E. Adaptive thresholding."""
    g = ds._canvas_gray(crop)
    block = max(11, (g.shape[1] // 12) | 1)
    return cv2.adaptiveThreshold(
        g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block, 10)


def v_denoise_otsu(crop):
    """F. Denoising (bilateral) + OTSU (production tier-3 style)."""
    g = ds._canvas_gray(crop)
    d = cv2.bilateralFilter(g, 7, 45, 45)
    _, t = cv2.threshold(d, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return t


def v_up_gray(crop):
    """G. Upscaling + grayscale."""
    return cv2.cvtColor(_upscale(crop, 3.0), cv2.COLOR_BGR2GRAY)


def v_up_clahe(crop):
    """H. Upscaling + CLAHE."""
    g = cv2.cvtColor(_upscale(crop, 3.0), cv2.COLOR_BGR2GRAY)
    return ds._variant_primary(g)


def v_up_adaptive(crop):
    """I. Upscaling + adaptive threshold."""
    g = cv2.cvtColor(_upscale(crop, 3.0), cv2.COLOR_BGR2GRAY)
    block = max(11, (g.shape[1] // 12) | 1)
    return cv2.adaptiveThreshold(
        g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block, 10)


VARIANTS = [
    ("A_original", v_original),
    ("B_clahe", v_clahe),
    ("C_gray_contrast", v_gray_contrast),
    ("D_otsu", v_otsu),
    ("E_adaptive", v_adaptive),
    ("F_denoise_otsu", v_denoise_otsu),
    ("G_up_gray", v_up_gray),
    ("H_up_clahe", v_up_clahe),
    ("I_up_adaptive", v_up_adaptive),
]


# ── OCR wrapper (production engine, timed) ─────────────────────────────────
def run_ocr(image: np.ndarray):
    """One production recognizer pass; returns (candidates, latency_s)."""
    import time
    t = time.perf_counter()
    try:
        cands = ds._run_ocr(image)
    except Exception:
        cands = []
    return cands, time.perf_counter() - t


def best_candidate(cands):
    """Highest-confidence cleaned candidate as (text, conf, char_probs)."""
    if not cands:
        return None
    c = max(cands, key=lambda x: float(x[1]))
    probs = c[2] if len(c) > 2 else None
    return (ds.clean_text(c[0]), float(c[1]), probs)



# ── confusion-aware consensus (Experiment 4) ───────────────────────────────
_CONFUSION_PAIRS = [
    ("G", "7"), ("J", "6"), ("D", "0"), ("B", "8"),
    ("I", "1"), ("L", "1"), ("O", "0"), ("S", "5"), ("Z", "2"),
]


def conf_class(ch: str) -> str:
    """Canonical class id so confusable characters vote together."""
    ch = ch.upper()
    for i, grp in enumerate(_CONFUSION_PAIRS):
        if ch in grp:
            return f"cls{i}"
    return ch


def consensus_vote(observations):
    """observations: list of (text, conf). Position-wise confusion-aware
    majority vote among the most common candidate length. Returns the voted
    string (highest-confidence literal per winning class) or None."""
    obs = [(norm(t), float(c)) for t, c in observations if norm(t)]
    if not obs:
        return None
    lengths = [len(t) for t, _ in obs]
    target_len = statistics.mode(lengths)
    same = [(t, c) for t, c in obs if len(t) == target_len]
    if not same or target_len == 0:
        return max(obs, key=lambda x: x[1])[0]
    out_chars = []
    for pos in range(target_len):
        class_weight = {}
        literal_weight = {}
        for t, c in same:
            ch = t[pos]
            cl = conf_class(ch)
            class_weight[cl] = class_weight.get(cl, 0.0) + c
            literal_weight[ch] = literal_weight.get(ch, 0.0) + c
        win_cls = max(class_weight, key=class_weight.get)
        lits = {l: w for l, w in literal_weight.items()
                if conf_class(l) == win_cls}
        out_chars.append(max(lits, key=lits.get))
    return "".join(out_chars)


# ── offline experimental rankers (Experiment 6) ────────────────────────────
def char_stats(probs):
    if not probs:
        return None
    arr = [float(p) for p in probs]
    return {
        "mean": sum(arr) / len(arr),
        "min": min(arr),
        "max": max(arr),
        "var": statistics.pvariance(arr) if len(arr) > 1 else 0.0,
    }


def _edge_pen(text, probs):
    edge = ds._first_last_conf(text, probs)
    if edge is not None and edge[0] < ds._EDGE_CONF_PENALTY_THRESHOLD:
        return (ds._EDGE_CONF_PENALTY
                * (ds._EDGE_CONF_PENALTY_THRESHOLD - edge[0])
                / ds._EDGE_CONF_PENALTY_THRESHOLD)
    return 0.0


def rank_charconf(cands):
    """Character-confidence-aware ranking (offline experiment). Uses OCR
    confidence + format + mean/min character confidence + edge penalty.
    Weights live on the same 0-140 scale as production; not tuned per-sample."""
    groups = {}
    for c in cands:
        text = ds.clean_text(c[0])
        if not text or len(text) < ds._MIN_CANDIDATE_LEN:
            continue
        probs = c[2] if len(c) > 2 else None
        groups.setdefault(text, []).append((float(c[1]), probs))
    if not groups:
        return None
    best = None
    for text, obs in groups.items():
        confs = [c for c, _ in obs]
        max_conf, avg_conf = max(confs), sum(confs) / len(confs)
        fmt = ds.indian_plate_score(text)
        best_obs = max(obs, key=lambda o: o[0])
        st = char_stats(best_obs[1])
        mean_c = st["mean"] if st else 0.0
        min_c = st["min"] if st else 0.0
        score = (max_conf * 30.0 + avg_conf * 10.0 + fmt * 0.90
                 + mean_c * 20.0 + min_c * 15.0
                 + ds._strict_format_adjustment(text)
                 - _edge_pen(text, best_obs[1]))
        if best is None or score > best[1]:
            best = (text, score)
    return best[0] if best else None


def rank_consensus(cands):
    """Multi-pass consensus ranking: score each candidate by how many passes
    (confidence-weighted) produced it, plus format validity."""
    groups = {}
    for c in cands:
        text = ds.clean_text(c[0])
        if not text or len(text) < ds._MIN_CANDIDATE_LEN:
            continue
        groups.setdefault(text, []).append(float(c[1]))
    if not groups:
        return None
    best = None
    for text, confs in groups.items():
        votes = len(confs)
        score = ((sum(confs) / votes) * 40.0
                 + min(votes - 1, 3) * 8.0
                 + ds.indian_plate_score(text) * 0.9
                 + ds._strict_format_adjustment(text))
        if best is None or score > best[1]:
            best = (text, score)
    return best[0] if best else None


def rank_production(tier_evidence):
    """The real production selector (read-only call)."""
    try:
        best, _ = ds._finalize_selection([list(te) for te in tier_evidence])
        return best[0] if best else None
    except Exception:
        return None

