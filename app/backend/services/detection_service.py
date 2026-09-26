"""
LICENSE VISION AI — Backend
Image detection service (the ONE canonical interactive detection path).

This backs POST /api/detect/image only. The offline batch pipeline
(plate_pipeline.py) and video pipeline (video_pipeline.py) are separate,
untouched files — nothing here changes their behavior or output format.

============================================================
HISTORY / WHY THIS FILE LOOKS THE WAY IT DOES
============================================================

The interactive OCR path uses an adaptive cascade:

    Tier 1: CLAHE + EasyOCR
    Tier 4: two-band split (tall crops with weak whole-crop reads)
    Tier 2: OTSU + EasyOCR fallback
    Tier 3: denoise + OTSU + EasyOCR last resort

Tiers 4 runs before 2/3: a stacked plate read as one line can produce
long structured junk that looks "strong" and would otherwise stop the
cascade before the band splitter ever sees it.

The important optimization is to avoid running all three OCR passes
when the first result is already sufficiently supported.

ACCURACY LAYER (format-aware candidate selection):

    1. The whole-crop OCR canvas is first tightened to the plate's
       actual text rows/columns (high-pass ink profile, plate-rim
       frame lines zeroed). Previously this trimming ran only in the
       multi-line tier, so single-line crops carried frame strokes
       and dark margins into the recognizer — the source of phantom
       LEADING characters (KL01AP8921 -> LKL0AP8921).
    2. EasyOCR reports only the MEAN of its per-step CTC max
       probabilities. A tiny read-only patch (see
       _install_char_conf_hook) additionally captures the per-step
       probabilities so the leading/trailing characters of every
       candidate can be scored on their own evidence.
    3. Candidates are validated against the REAL Indian registration
       formats (current 2005+ series AND legacy series, with actual
       state codes and district numbers 01-38). A candidate like
       KKL91APB924 (double state letter, 9-char serial) no longer
       scores a perfect format 100.
    4. Selection penalizes candidates whose first/last characters
       have low CTC confidence — the classic hallucinated-edge
       signature — and rewards strict-format matches. The correct
       KL01AP8921 beats LKL0AP8921 because its edge characters are
       actually supported by the recognizer; nothing is deleted
       blindly, weakly-evidenced candidates simply lose.

IMPORTANT (multi-line plates):
    Motorcycle plates stack state code ("37-N1") over the serial
    ("4635"). Feeding that crop to the recognizer as one horizontal
    line produces garbage or nothing. The cascade adds a memory-safe
    tier that splits such crops into horizontal text bands with OpenCV
    projection profiles and recognizes each band with the SAME
    recognizer-only reader — no CRAFT, no extra models.

    The whole-crop canvas must PRESERVE both bands: its row trim
    collapses to the single longest text run, which for stacked plates
    would silently turn the canvas back into a one-line image (this is
    exactly how 37-N1/4635 used to come out UNKNOWN).
"""

from __future__ import annotations

import gc
import logging
import os
import re
import threading
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from services.data_loader import MODEL_WEIGHTS_PATH


# ============================================================
# DEV TIMING
# ============================================================

logger = logging.getLogger("license_vision.detection")

if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(_handler)

logger.setLevel(
    logging.INFO
    if os.environ.get("LVA_DEBUG_TIMING", "1") != "0"
    else logging.WARNING
)


class _TimingBucket:
    """Accumulates timing information for one request."""

    def __init__(self):
        self._t0 = time.perf_counter()
        self.buckets: dict[str, float] = defaultdict(float)
        self.detail: list[str] = []

    def add(self, bucket: str, seconds: float, note: str = ""):
        self.buckets[bucket] += seconds

        if note:
            self.detail.append(
                f"  {note}: {seconds * 1000:.1f} ms"
            )

    def total(self) -> float:
        return time.perf_counter() - self._t0

    def report(self, plate_count: int, ocr_pass_count: int):
        total_ms = self.total() * 1000

        lines = ["[Timing]"]

        for bucket in (
            "YOLO",
            "Crop",
            "Preprocess",
            "OCR",
            "Scoring",
        ):
            lines.append(
                f"{bucket}: "
                f"{self.buckets.get(bucket, 0.0) * 1000:.1f} ms"
            )

        lines.append(f"Total: {total_ms:.1f} ms")
        lines.append(
            f"(plates={plate_count}, ocr_passes={ocr_pass_count})"
        )

        if self.detail:
            lines.extend(self.detail)

        logger.info("\n".join(lines))


# ============================================================
# LAZY SINGLETON MODEL STATE
# ============================================================
#
# Free-tier deployment notes (Render free tier has very little RAM):
#   - OMP/MKL/OPENBLAS thread counts are capped via env vars BEFORE
#     torch is imported: every spare intra-op thread reserves its own
#     stack and memory arena.
#   - A load lock guarantees exactly one YOLO instance and exactly one
#     EasyOCR reader, no matter how many requests arrive concurrently
#     while models are still loading.
#   - torch.inference_mode() around YOLO avoids building autograd state.
#   - A one-time tiny warmup predict triggers all lazy allocations
#     (model graph, workspaces) at load time instead of inside the
#     first user request, and gc.collect() trims the transient peak.

_yolo_model = None
_ocr_reader = None
_load_error: str | None = None
_load_lock = threading.Lock()

for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_var, "1")


def _apply_low_memory_runtime_caps():
    """Cap runtime thread pools.

    Env vars alone do not cover torch's own pool management: each intra-op
    / inter-op thread reserves a private memory arena, which on a big or
    shared CPU multiplies baseline RSS by the core count. For 1-plate-at-a-
    time CPU inference a single thread is the lowest-memory configuration.
    cv2 likewise keeps a parallel-for pool that small crop ops never need.
    """

    try:
        import torch

        torch.set_num_threads(1)
    except Exception:
        pass

    # Must run before any parallel work starts; harmless if it cannot.
    try:
        import torch

        torch.set_num_interop_threads(1)
    except Exception:
        pass

    try:
        cv2.setNumThreads(0)
    except Exception:
        pass


_low_memory_caps_applied = False


def _ensure_low_memory_runtime():
    global _low_memory_caps_applied

    if _low_memory_caps_applied:
        return

    _apply_low_memory_runtime_caps()

    _low_memory_caps_applied = True


def _release_memory_to_os():
    """glibc malloc_trim: hand freed heap back to the OS.

    CPython frees objects but glibc often keeps the pages resident, and
    container memory limits are enforced on RSS. No-op on non-glibc
    platforms (Render is Debian/Ubuntu Linux; Windows dev is unaffected).
    """

    if os.name != "posix":
        return

    try:
        import ctypes

        ctypes.CDLL("libc.so.6").malloc_trim(0)

    except Exception:
        pass


def _ensure_models_loaded():
    global _yolo_model, _ocr_reader, _load_error

    if _yolo_model is not None and _ocr_reader is not None:
        return

    with _load_lock:
        # Re-check: another thread may have finished loading while we waited.
        if _yolo_model is not None and _ocr_reader is not None:
            return

        if _load_error is not None:
            raise RuntimeError(_load_error)

        # Cap thread pools before torch's pools are created.
        _ensure_low_memory_runtime()

        try:
            import easyocr
            import torch
            from ultralytics import YOLO

            if not MODEL_WEIGHTS_PATH.exists():
                raise FileNotFoundError(
                    f"YOLO weights not found at {MODEL_WEIGHTS_PATH}. "
                    "Point LVA_MODEL_PATH at your best.pt."
                )

            _yolo_model = YOLO(str(MODEL_WEIGHTS_PATH))

            # One-time warmup: initializes every lazy allocation now,
            # not during the first real user request.
            with torch.inference_mode():
                _yolo_model.predict(
                    source=np.zeros((320, 320, 3), dtype=np.uint8),
                    imgsz=640,
                    conf=0.25,
                    verbose=False,
                )

            _ocr_reader = easyocr.Reader(
                ["en"],
                gpu=False,
                detector=False,
                verbose=False,
            )

            gc.collect()
            _release_memory_to_os()

        except Exception as exc:
            _load_error = str(exc)
            raise RuntimeError(_load_error) from exc


def models_ready() -> bool:
    return (
        _yolo_model is not None
        and _ocr_reader is not None
    )


# ============================================================
# INDIAN PLATE VALIDATION
# ============================================================

# Flexible patterns.
#
# Examples:
#   MH20EE7597
#   TN07BU5427
#   FAG643
#
# Do NOT use a fixed 3-letter + 3-digit assumption.

INDIAN_PLATE_PATTERNS = [
    re.compile(
        r"^[A-Z]{2}\d{1,2}[A-Z]{1,3}\d{1,4}$"
    ),
    re.compile(
        r"^[A-Z]{2,3}\d{1,4}[A-Z]{0,3}\d{0,4}$"
    ),
]


def clean_text(text: str) -> str:
    text = str(text).upper()

    return re.sub(
        r"[^A-Z0-9]",
        "",
        text,
    )


def indian_plate_score(text: str) -> int:
    s = clean_text(text)

    if not s:
        return 0

    score = 0
    length = len(s)

    # Typical plate length.
    if 8 <= length <= 12:
        score += 35

    elif 6 <= length <= 13:
        score += 15

    # State / region prefix.
    if len(s) >= 2 and s[:2].isalpha():
        score += 25

    # At least two digits.
    if sum(c.isdigit() for c in s) >= 2:
        score += 20

    # Alphabetic characters after prefix.
    if (
        len(s) >= 5
        and sum(c.isalpha() for c in s[2:]) >= 1
    ):
        score += 10

    # Full regex match.
    for pattern in INDIAN_PLATE_PATTERNS:
        if pattern.match(s):
            score += 35
            break

    return min(score, 100)


# ------------------------------------------------------------
# STRICT Indian plate validation (real registration formats).
#
# The legacy scorer above accepts near-miss junk like KKL91APB924
# (double leading letter, 5-char serial) at full marks. The strict
# layer below knows the actual issuance rules:
#
#   current series (2005+):
#       STATE(2, real code) + district(01-38) + series(0-3 letters)
#       + number(1-4 digits)          e.g. KL01AP8921, MH12DE1433
#   legacy series (pre-2005):
#       STATE(2, real code) + letters(1-2) + number(1-4 digits)
#                                       e.g. MHA4321, KLB1055
# ------------------------------------------------------------

INDIAN_STATE_CODES = frozenset({
    "AN", "AP", "AR", "AS", "BR", "CG", "CH", "DD", "DL", "DN",
    "GA", "GJ", "HR", "HP", "JH", "JK", "KA", "KL", "LD", "MH",
    "ML", "MN", "MP", "MZ", "NL", "OD", "PB", "PY", "RJ", "SK",
    "TN", "TR", "TS", "TG", "UA", "UK", "UP", "WB",
})

# District numbers actually seen in the wild top out in the high 30s
# (Delhi renumbered up to ~38); anything beyond is a misread.
# District/RTO numbers are always TWO digits on current plates
# (01-99, leading zero required; big states go well past 30: MH-48,
# KA-51, UP-78). A dropped leading zero (KL9AP8921 for KL09...) fails
# here. Exception: Delhi, whose heritage single-digit RTOs (DL 8 Car,
# DL 9 New Delhi) and letter-suffixed codes (DL 3C, DL 4C, DL 1S)
# are still on legions of registered vehicles.
_RTO_GROUP = r"(?:0[1-9]|[1-9][0-9])"
_RTO_GROUP_DL = r"(?:0[1-9]|[1-9][0-9]|[1-9][A-Z]|[1-9])"

_RE_CURRENT_PLATE = re.compile(
    r"^(?:DL" + _RTO_GROUP_DL + r"|[A-Z]{2}" + _RTO_GROUP + r")"
    r"[A-Z]{0,3}[0-9]{1,4}$"
)

_RE_LEGACY_PLATE = re.compile(
    r"^[A-Z]{2}[A-Z]{1,2}[0-9]{1,4}$"
)


def strict_indian_plate(text: str) -> bool:
    """True when the string is a valid Indian registration number
    under the current OR legacy series with a REAL state code."""

    s = clean_text(text)

    if not 5 <= len(s) <= 10:
        return False

    if s[:2] not in INDIAN_STATE_CODES:
        return False

    return bool(
        _RE_CURRENT_PLATE.match(s)
        or _RE_LEGACY_PLATE.match(s)
    )


# Format adjustments used by candidate selection (0-100 legacy score
# still feeds validation display; these adjust the SELECTION score).

_STRICT_FORMAT_BONUS = 22.0
_INVALID_FORMAT_PENALTY = 12.0
_UNKNOWN_STATE_PENALTY = 18.0


def _strict_format_adjustment(text: str) -> float:
    """Signed format adjustment for a candidate string."""

    s = clean_text(text)

    if strict_indian_plate(s):
        return _STRICT_FORMAT_BONUS

    penalty = 0.0

    if len(s) >= 2 and s[:2].isalpha() and s[:2] not in INDIAN_STATE_CODES:
        penalty += _UNKNOWN_STATE_PENALTY

    if s:
        penalty += _INVALID_FORMAT_PENALTY

    return -penalty


# ------------------------------------------------------------
# Per-character CTC confidence capture.
#
# EasyOCR's recognizer_predict() computes a max class probability for
# EVERY output timestep but only reports custom_mean() of them, and
# get_text()'s merge rebuilds rows as (box, text, conf) — dropping
# anything extra. So the ONLY way to surface per-character evidence
# is a behavior-preserving re-implementation of those two functions
# whose rows additionally carry the per-character probability list:
#
#     recognizer_predict -> [pred, conf, char_probs]
#     get_text           -> (box, text, conf, char_probs)
#
# Text and confidence stay BIT-IDENTICAL to stock EasyOCR (the same
# arrays, the same custom_mean). The copies are guarded: they are
# installed only when the installed EasyOCR's own source still
# contains the tokens this re-implementation relies on; on any
# mismatch the hook stays off, char evidence is simply None, and the
# pipeline degrades gracefully to mean-confidence-only selection.
# ------------------------------------------------------------

_char_conf_hook_installed = False
_char_conf_hook_failed = False

# Tokens that must appear in the installed EasyOCR source for the
# re-implementation to be considered safe.
_PREDICT_TOKENS = ("custom_mean", "preds_max_prob", "decode_greedy")
_GET_TEXT_TOKENS = ("contrast_ths", "AlignCollate", "low_confident_idx")


def _install_char_conf_hook() -> None:

    global _char_conf_hook_installed, _char_conf_hook_failed

    if _char_conf_hook_installed or _char_conf_hook_failed:
        return

    _char_conf_hook_installed = True

    try:
        import inspect

        import easyocr.recognition as _recog

        predict_src = inspect.getsource(_recog.recognizer_predict)
        get_text_src = inspect.getsource(_recog.get_text)

        if (
            not all(tok in predict_src for tok in _PREDICT_TOKENS)
            or not all(tok in get_text_src for tok in _GET_TEXT_TOKENS)
        ):
            raise RuntimeError("easyocr source drift")

        import numpy as _np
        import torch
        import torch.nn.functional as _F

        def _predict_with_chars(
            model,
            converter,
            test_loader,
            batch_max_length,
            ignore_idx,
            char_group_idx,
            decoder="greedy",
            beamWidth=5,
            device="cpu",
        ):
            model.eval()
            result = []

            with torch.no_grad():
                for image_tensors in test_loader:
                    batch_size = image_tensors.size(0)
                    image = image_tensors.to(device)

                    length_for_pred = torch.IntTensor(
                        [batch_max_length] * batch_size
                    ).to(device)
                    text_for_pred = torch.LongTensor(
                        batch_size, batch_max_length + 1
                    ).fill_(0).to(device)

                    preds = model(image, text_for_pred)
                    preds_size = torch.IntTensor([preds.size(1)] * batch_size)

                    # ---- identical to stock easyocr 1.7.2 ----
                    preds_prob = _F.softmax(preds, dim=2)
                    preds_prob = preds_prob.cpu().detach().numpy()
                    preds_prob[:, :, ignore_idx] = 0.
                    pred_norm = preds_prob.sum(axis=2)
                    preds_prob = preds_prob / _np.expand_dims(
                        pred_norm, axis=-1
                    )
                    preds_prob = torch.from_numpy(preds_prob).float().to(device)

                    if decoder == "greedy":
                        _, preds_index = preds_prob.max(2)
                        preds_index = preds_index.view(-1)
                        preds_str = converter.decode_greedy(
                            preds_index.data.cpu().detach().numpy(),
                            preds_size.data,
                        )
                    elif decoder == "beamsearch":
                        k = preds_prob.cpu().detach().numpy()
                        preds_str = converter.decode_beamsearch(
                            k, beamWidth=beamWidth
                        )
                    elif decoder == "wordbeamsearch":
                        k = preds_prob.cpu().detach().numpy()
                        preds_str = converter.decode_wordbeamsearch(
                            k, beamWidth=beamWidth
                        )
                    else:
                        preds_str = []

                    preds_prob_np = preds_prob.cpu().detach().numpy()
                    values = preds_prob_np.max(axis=2)
                    indices = preds_prob_np.argmax(axis=2)

                    preds_max_prob = []

                    for v, i in zip(values, indices):
                        max_probs = v[i != 0]
                        if len(max_probs) > 0:
                            preds_max_prob.append(max_probs)
                        else:
                            preds_max_prob.append(_np.array([0]))

                    for batch_elem, (pred, pred_max_prob) in enumerate(
                        zip(preds_str, preds_max_prob)
                    ):
                        confidence_score = _recog.custom_mean(pred_max_prob)

                        # ---- per-character alignment (greedy only) ----
                        # Collapse CTC repeats exactly like decode_greedy:
                        # skip blank steps and repeat continuations; the
                        # remaining step probabilities align 1:1 with the
                        # decoded characters.
                        char_probs = None

                        if decoder == "greedy":
                            step_indices = indices[batch_elem]
                            step_values = values[batch_elem]

                            collapsed: list[float] = []
                            prev = -1
                            aligned = True

                            for step in range(len(step_indices)):
                                cur = int(step_indices[step])

                                if cur in ignore_idx:
                                    aligned = False
                                    break

                                if cur == 0 or cur == prev:
                                    prev = cur
                                    continue

                                collapsed.append(float(step_values[step]))
                                prev = cur

                            if aligned and len(collapsed) == len(pred):
                                char_probs = collapsed

                        result.append([pred, confidence_score, char_probs])

            return result

        def _get_text_with_chars(
            character,
            imgH,
            imgW,
            recognizer,
            converter,
            image_list,
            ignore_char="",
            decoder="greedy",
            beamWidth=5,
            batch_size=1,
            contrast_ths=0.1,
            adjust_contrast=0.5,
            filter_ths=0.003,
            workers=1,
            device="cpu",
        ):
            batch_max_length = int(imgW / 10)

            char_group_idx: dict = {}
            ignore_idx: list[int] = []

            for char in ignore_char:
                try:
                    ignore_idx.append(character.index(char) + 1)
                except Exception:
                    pass

            coord = [item[0] for item in image_list]
            img_list = [item[1] for item in image_list]

            AlignCollate_normal = _recog.AlignCollate(
                imgH=imgH,
                imgW=imgW,
                keep_ratio_with_pad=True,
            )
            test_data = _recog.ListDataset(img_list)
            test_loader = torch.utils.data.DataLoader(
                test_data,
                batch_size=batch_size,
                shuffle=False,
                num_workers=int(workers),
                collate_fn=AlignCollate_normal,
                pin_memory=True,
            )

            # predict first round
            result1 = _predict_with_chars(
                recognizer,
                converter,
                test_loader,
                batch_max_length,
                ignore_idx,
                char_group_idx,
                decoder,
                beamWidth,
                device=device,
            )

            # predict second round
            low_confident_idx = [
                i for i, item in enumerate(result1)
                if (item[1] < contrast_ths)
            ]

            if len(low_confident_idx) > 0:
                img_list2 = [img_list[i] for i in low_confident_idx]

                AlignCollate_contrast = _recog.AlignCollate(
                    imgH=imgH,
                    imgW=imgW,
                    keep_ratio_with_pad=True,
                    adjust_contrast=adjust_contrast,
                )
                test_data2 = _recog.ListDataset(img_list2)
                test_loader2 = torch.utils.data.DataLoader(
                    test_data2,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=int(workers),
                    collate_fn=AlignCollate_contrast,
                    pin_memory=True,
                )

                result2 = _predict_with_chars(
                    recognizer,
                    converter,
                    test_loader2,
                    batch_max_length,
                    ignore_idx,
                    char_group_idx,
                    decoder,
                    beamWidth,
                    device=device,
                )

            result = []

            for i, zipped in enumerate(zip(coord, result1)):
                box, pred1 = zipped

                if i in low_confident_idx:
                    pred2 = result2[low_confident_idx.index(i)]

                    if pred1[1] > pred2[1]:
                        result.append(
                            (box, pred1[0], pred1[1], pred1[2])
                        )
                    else:
                        result.append(
                            (box, pred2[0], pred2[1], pred2[2])
                        )
                else:
                    result.append(
                        (box, pred1[0], pred1[1], pred1[2])
                    )

            return result

        # recognize() calls the get_text it imported into easyocr's own
        # module namespace at import time, so BOTH bindings must point
        # at the instrumented copy.
        import easyocr as _easyocr_pkg
        import easyocr.easyocr as _easyocr_mod

        _recog.recognizer_predict = _predict_with_chars
        _recog.get_text = _get_text_with_chars
        _easyocr_mod.get_text = _get_text_with_chars
        if hasattr(_easyocr_pkg, "get_text"):
            _easyocr_pkg.get_text = _get_text_with_chars

    except Exception:
        _char_conf_hook_failed = True


# ------------------------------------------------------------
# Edge-character evidence.
# ------------------------------------------------------------

# Ambiguous glyph pairs that genuinely swap on Indian plates (stamped
# zeros vs O, serif ones vs I, etc.). Used ONLY by the narrow repair
# rule below: a candidate one swap away from strict validity, where
# the recognizer itself was unsure about that glyph.
_AMBIGUOUS_CHARS = {
    "O": "0",
    "0": "O",
    "I": "1",
    "1": "I",
    "S": "5",
    "5": "S",
    "B": "8",
    "8": "B",
}

# The recognizer's per-character probability for the swapped glyph
# must be BELOW this for a repair to be generated — overriding a
# confident read would be guesswork, not evidence.
_REPAIR_UNSURE_BELOW = 0.75

# Repaired candidates carry a small confidence discount: the swap is
# format-driven, so they must not beat the raw read on confidence
# alone — only on combined structure + confidence evidence.
_REPAIR_CONF_DISCOUNT = 0.90

# Hard cap on generated repairs per plate (bounds selection cost).
_MAX_REPAIRS = 8


def _generate_format_repairs(
    tier_evidence: list[tuple[int, list]],
) -> list[tuple[str, float, list[float] | None]]:
    """Add format-repaired variants of near-miss candidates.

    A candidate one or two ambiguous glyphs (O/0, I/1, S/5, B/8) away
    from strict validity earns extra candidates with those glyphs
    swapped — but ONLY positions where the recognizer's own
    per-character probability shows it was UNSURE. Originals are never
    removed or edited; repaired strings compete on the same evidence
    scale with a small confidence discount.

    KL01AP8921 is preserved as-is; KLO1AP8921 gains KL01AP8921 as a
    rival only if the recognizer was unsure about the O. A confident
    misread is never overridden — that would be guesswork, not
    evidence.
    """

    repairs: list[tuple[str, float, list[float] | None]] = []

    seen_sources: set[str] = set()
    seen_fixed: set[str] = set()

    for _tier, candidates in tier_evidence:

        for item in candidates:

            text = clean_text(item[0])

            if not text or len(text) < 5 or text in seen_sources:
                continue

            seen_sources.add(text)

            if strict_indian_plate(text):
                continue

            probs = item[2] if len(item) > 2 else None
            conf = float(item[1])

            # Positions where an ambiguous glyph sits.
            swap_positions: list[tuple[int, bool]] = []
            # (position, glyph_was_unsure)

            for i, ch in enumerate(text):

                if ch not in _AMBIGUOUS_CHARS:
                    continue

                unsure = True

                if (
                    probs is not None
                    and i < len(probs)
                    and probs[i] >= _REPAIR_UNSURE_BELOW
                ):
                    unsure = False

                swap_positions.append((i, unsure))

            if not swap_positions:
                continue

            # Single swaps: the source string is ALREADY format-invalid,
            # so one ambiguous-glyph swap that lands on a strictly valid
            # registration is backed by strong format evidence even when
            # the recognizer was confident (stamped zeros read as solid
            # 'O's). Pairs are more speculative and require the
            # recognizer to have been unsure at BOTH positions.
            swap_sets: list[tuple[int, ...]] = [
                (i,) for i, _ in swap_positions
            ]

            unsure_only = [
                i for i, unsure in swap_positions if unsure
            ]

            swap_sets += [
                (a, b)
                for ai, a in enumerate(unsure_only)
                for b in unsure_only[ai + 1 :]
            ]

            for swaps in swap_sets:

                fixed = list(text)

                for i in swaps:
                    fixed[i] = _AMBIGUOUS_CHARS[fixed[i]]

                fixed_str = "".join(fixed)

                if (
                    not strict_indian_plate(fixed_str)
                    or fixed_str in seen_fixed
                ):
                    continue

                seen_fixed.add(fixed_str)

                repaired_probs: list[float] | None = None

                if probs is not None and len(probs) == len(text):
                    # Keep the ORIGINAL uncertainties on the swapped
                    # positions: the repair inherits the read's doubt.
                    repaired_probs = list(probs)

                repairs.append(
                    (
                        fixed_str,
                        conf * _REPAIR_CONF_DISCOUNT,
                        repaired_probs,
                    )
                )

                if len(repairs) >= _MAX_REPAIRS:
                    return repairs

    return repairs

# Fraction of a candidate's characters treated as its edges.
_EDGE_FRACTION = 0.20

# A hallucinated edge usually rests on 1-2 weak CTC steps.
_EDGE_MIN_STEPS = 2

# How strongly weak edges pull a candidate down (on the ~0-140
# selection scale: conf terms max ~50, format terms ~50).
_EDGE_CONF_PENALTY = 45.0

# Edge mean-probability at or above which no penalty applies. A real
# character the recognizer is sure of sits at 0.85-1.0; a hallucinated
# frame/margin character typically rests on 0.1-0.4 steps.
_EDGE_CONF_PENALTY_THRESHOLD = 0.60


def _first_last_conf(
    text: str,
    char_conf: list[float] | None,
) -> tuple[float, int] | None:
    """Mean CTC probability of the candidate's leading/trailing
    characters, and how many steps backed it.

    Returns None when no per-character evidence is available.
    """

    if not char_conf:
        return None

    text = clean_text(text)

    if not text:
        return None

    k = min(
        _EDGE_MIN_STEPS,
        max(1, int(len(text) * _EDGE_FRACTION)),
    )

    lead = char_conf[:k]
    tail = char_conf[-k:]

    steps = len(lead) + len(tail)

    if steps == 0:
        return None

    return sum(lead + tail) / steps, steps


# ------------------------------------------------------------
# Dominant structural run.
#
#    KL01AP8921   ->  9  (state + district + series + serial)
#    KKL91APB924  ->  4  (double letter + 9-series junk)
#    KL01AP892    ->  8  (truncated serial, still structured)
# ------------------------------------------------------------

_STRUCT_RUN_RE = re.compile(
    r"^[A-Z]{2}(?:0[1-9]|[12][0-9]|3[0-8])?[A-Z]{0,3}"
)


def _score_dominant_run(text: str) -> int:

    m = _STRUCT_RUN_RE.match(clean_text(text))

    return len(m.group(0)) if m else 0


def _edge_supported_in(text: str, candidates: list) -> bool:
    """True when the candidate's own per-character evidence supports its
    first/last characters.

    No char evidence (hook unavailable / candidate not found) counts as
    supported — absence of evidence never rejects, it only stops
    granting the extra trust.
    """

    wanted = clean_text(text)

    for item in candidates:
        if clean_text(item[0]) != wanted:
            continue

        probs = item[2] if len(item) > 2 else None
        edge = _first_last_conf(wanted, probs)

        if edge is None:
            return True

        return edge[0] >= _EDGE_CONF_PENALTY_THRESHOLD

    return True


def classify_validation(score: int) -> str:
    if score >= 75:
        return "INDIAN_PLATE"

    if score >= 45:
        return "POSSIBLE_INDIAN_PLATE"

    return "UNVERIFIED"


def status_from_confidence(
    final_confidence: float,
    ocr_text: str,
) -> str:

    if not ocr_text:
        return "OCR_FAILED"

    if final_confidence >= 0.80:
        return "HIGH_CONFIDENCE"

    if final_confidence >= 0.50:
        return "REVIEW"

    return "LOW_CONFIDENCE"


# ============================================================
# OCR PREPROCESSING
# ============================================================

# Cap full-image inference resolution. The offline video pipeline
# already processed at 1280px wide; huge camera/phone uploads would
# otherwise multiply YOLO and annotation memory several-fold for no
# accuracy gain.
_MAX_INFER_SIDE = 1280

# Bounded OCR canvas.
#
# The old implementation used a distorted 1800x600 canvas; 1100
# preserved aspect ratio. 900 was A/B-tested on real crops: identical
# selection outcomes at ~11% less recognizer time per pass (the
# recognizer resizes to a fixed 64px height anyway; extra width mostly
# costs CPU on blank steps).
_OCR_MAX_DIM = 900

# Don't excessively enlarge tiny crops.
_OCR_MAX_UPSCALE = 6.0

# ------------------------------------------------------------
# Effort policy (evidence-based latency bounding).
#
# The historical batch run (batch_results/batch_results.csv, 658
# images) recorded EVERY box below 0.45 YOLO confidence as junk:
# 9/9 produced empty or 2-character OCR output. Meanwhile the strong
# population sits at 0.71-0.94. A sub-0.45 box is therefore almost
# always a wheel arch / headlight / shadow — and running the full
# multi-variant cascade on such a crop burns CPU seconds to produce
# the 27%-confidence reads seen in production. Effort is spent where
# the evidence says plates actually are.
# ------------------------------------------------------------

# Boxes at/above this get the full cascade; below it, a bounded
# primary (+ one fallback only when the primary read nothing).
_MIN_FULL_EFFORT_YOLO_CONF = 0.45

# Hard cap on boxes that receive OCR per image. Real images carry
# 1-2 plates; the cap only bites when NMS misfires badly, bounding
# worst-case latency multiplicatively.
_MAX_OCR_BOXES = 3


# ============================================================
# ADAPTIVE OCR THRESHOLDS
# ============================================================

# Strong result:
#
#   OCR confidence >= 0.55
#   AND
#   Indian plate format >= 45
#
# stops the cascade immediately.

_STRONG_OCR_CONF = 0.55
_STRONG_FORMAT_SCORE = 45


# A second condition is intentionally more conservative:
#
# If EasyOCR returns the same text multiple times during one pass,
# and that text has a strong plate structure, we don't necessarily
# need to spend another 2-3 seconds on another OCR pass.
#
# This is especially useful on clean plates where EasyOCR may return
# duplicate text regions.

_CONSENSUS_OCR_CONF = 0.40
_CONSENSUS_FORMAT_SCORE = 60
_MIN_CONSENSUS_VOTES = 2


def _pad_and_scale(crop: np.ndarray) -> np.ndarray:
    """
    Add small replicate padding and resize while preserving aspect ratio.

    Kept for the offline scripts that import it; the live API path uses
    _prepare_ocr_canvas() below (replicate borders leak edge pixels into
    the recognizer's sequence when no text-detection stage filters them).
    """

    h, w = crop.shape[:2]

    border_x = max(
        6,
        int(w * 0.03),
    )

    border_y = max(
        6,
        int(h * 0.10),
    )

    padded = cv2.copyMakeBorder(
        crop,
        border_y,
        border_y,
        border_x,
        border_x,
        cv2.BORDER_REPLICATE,
    )

    ph, pw = padded.shape[:2]

    longer_side = max(
        ph,
        pw,
    )

    if longer_side < _OCR_MAX_DIM:
        scale = min(
            _OCR_MAX_DIM / longer_side,
            _OCR_MAX_UPSCALE,
        )
    else:
        scale = (
            _OCR_MAX_DIM
            / longer_side
        )

    target_w = max(
        1,
        int(pw * scale),
    )

    target_h = max(
        1,
        int(ph * scale),
    )

    interp = (
        cv2.INTER_CUBIC
        if scale >= 1
        else cv2.INTER_AREA
    )

    return cv2.resize(
        padded,
        (target_w, target_h),
        interpolation=interp,
    )


def _prepare_ocr_canvas(crop: np.ndarray) -> np.ndarray:
    """
    Aspect-preserving resize of a YOLO plate crop for the recognizer.

    Unlike _pad_and_scale, no replicate border is added: the recognizer
    reads the whole crop as one text line, and replicated edge pixels
    produce phantom leading/trailing characters.
    """

    h, w = crop.shape[:2]

    longer_side = max(h, w)

    if longer_side < _OCR_MAX_DIM:
        scale = min(
            _OCR_MAX_DIM / longer_side,
            _OCR_MAX_UPSCALE,
        )
    else:
        scale = _OCR_MAX_DIM / longer_side

    target_w = max(1, int(w * scale))
    target_h = max(1, int(h * scale))

    interp = (
        cv2.INTER_CUBIC
        if scale >= 1
        else cv2.INTER_AREA
    )

    return cv2.resize(
        crop,
        (target_w, target_h),
        interpolation=interp,
    )


def _canvas_gray(crop: np.ndarray) -> np.ndarray:
    """Shared OCR canvas for the whole-crop tiers: aspect-preserving
    resize, grayscale, then a conservative trim to the crop's actual
    text region.

    The trim (frame lines + margins removed) is what keeps the
    recognizer from seeing plate-rim strokes and empty edges as phantom
    leading/trailing characters (KL01AP8921 -> LKL0AP8921). It only
    runs when the canvas is large enough to survive it; small crops
    pass through untouched.
    """

    base = _prepare_ocr_canvas(crop)

    gray = cv2.cvtColor(
        base,
        cv2.COLOR_BGR2GRAY,
    )

    if (
        gray.shape[0] >= 24
        and gray.shape[1] >= 40
    ):
        gray = _tight_text_region(gray)

    return gray


def _variant_primary(
    base_gray: np.ndarray,
) -> np.ndarray:
    """
    Tier 1 — CLAHE-enhanced grayscale.
    """

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8),
    )

    return clahe.apply(base_gray)


def _variant_fallback(
    enhanced_gray: np.ndarray,
) -> np.ndarray:
    """
    Tier 2 — OTSU binarization.
    """

    _, otsu = cv2.threshold(
        enhanced_gray,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )

    return otsu


def _variant_tertiary(
    enhanced_gray: np.ndarray,
) -> np.ndarray:
    """
    Tier 3 — bilateral filtering + OTSU.

    Last resort for genuinely difficult crops.
    """

    denoised = cv2.bilateralFilter(
        enhanced_gray,
        7,
        45,
        45,
    )

    _, otsu = cv2.threshold(
        denoised,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )

    return otsu


# ============================================================
# MULTI-LINE (TWO-BAND) PLATE SUPPORT
# ============================================================
#
# Indian motorcycle plates stack the state code over the serial
# ("37-N1" / "4635"). The recognizer-only path reads a crop as ONE
# horizontal line, so a two-band crop must be split and each band
# recognized separately.
#
# Memory safety: the splitter is pure OpenCV geometry — grayscale,
# Sobel row-energy, and a projection-profile valley scan over the
# SAME already-allocated crop. No extra model, no torch tensors,
# no CRAFT. Peak added memory is a few grayscale copies of a crop
# that is already capped at _OCR_MAX_DIM.

# A single-line Indian plate is very wide (aspect ~0.18-0.35). A two-line
# motorcycle plate is stacked, so it is noticeably taller — but still wider
# than tall (measured ~0.6-0.9 on real plates). Anything below this ratio
# is never split; anything at or above it is a split candidate.
_MAX_SPLIT_ASPECT = 0.45

# A viable horizontal gap between two text bands.
_MIN_SPLIT_VALLEY_DEPTH = 0.55

# Reject slivers: each band needs enough rows for the recognizer.
_MIN_BAND_ROWS = 12

# Ignore near-empty energy margins (borders / holder shadows).
_MARGIN_FRACTION = 0.10


def _row_energy(gray: np.ndarray) -> np.ndarray:
    """Per-row text energy: horizontal Sobel magnitude, clamped [0, 1].

    d/dy responds to every horizontal character stroke regardless of
    polarity (dark-on-light or light-on-dark), so it survives uneven
    lighting without any binarization threshold.
    """

    sobel_y = cv2.Sobel(
        gray,
        cv2.CV_32F,
        0,
        1,
        ksize=3,
    )

    np.abs(sobel_y, out=sobel_y)

    energy = sobel_y.sum(axis=1)

    max_energy = float(energy.max())

    if max_energy <= 1e-6:
        return energy

    return energy / max_energy


def _split_two_line_bands(
    enhanced_gray: np.ndarray,
) -> list[tuple[int, int]] | None:
    """Find the horizontal gap between the two text bands of a plate crop.

    Returns [(y0, y1), (y0, y1)] row ranges (top band, bottom band), or
    None when the crop does not look like two stacked text lines.
    """

    h, w = enhanced_gray.shape[:2]

    if h < 2 * _MIN_BAND_ROWS + 2 or w < 16:
        return None

    # Skip low-information borders (dark frames, holder edges).
    margin_y = max(1, int(h * _MARGIN_FRACTION))

    energy = _row_energy(enhanced_gray)

    y_lo = margin_y
    y_hi = h - margin_y

    if y_hi - y_lo < 2 * _MIN_BAND_ROWS + 2:
        y_lo = 0
        y_hi = h

    span = energy[y_lo:y_hi]

    if span.size == 0 or float(span.max()) <= 1e-6:
        return None

    # Longest run of low-energy rows = the gap between the bands.
    threshold = _MIN_SPLIT_VALLEY_DEPTH * float(span.max())

    best_start = -1
    best_len = 0
    cur_start = -1
    cur_len = 0

    for i, value in enumerate(span):
        if value <= threshold:
            if cur_start < 0:
                cur_start = i
            cur_len += 1
            if cur_len > best_len:
                best_len = cur_len
                best_start = cur_start
        else:
            cur_start = -1
            cur_len = 0

    # Plausibility: a real two-band plate has a WIDE ink-free gap (the
    # plate's horizontal divider). A shallow/short quiet stretch inside a
    # single text line (quiet rows between glyph strokes) is not a band
    # boundary — splitting there would feed the recognizer glyph halves.
    if best_len < max(6, int((y_hi - y_lo) * 0.06)):
        return None

    gap_top = y_lo + best_start
    gap_bottom = gap_top + best_len

    # Tighten to the ink-free core of the valley.
    while (
        gap_top > y_lo
        and energy[gap_top - 1] <= threshold
    ):
        gap_top -= 1

    while (
        gap_bottom < y_hi
        and energy[gap_bottom] <= threshold
    ):
        gap_bottom += 1

    top_rows = gap_top - y_lo
    bottom_rows = y_hi - gap_bottom

    if top_rows < _MIN_BAND_ROWS or bottom_rows < _MIN_BAND_ROWS:
        return None

    # Both halves must actually contain text energy.
    if (
        float(energy[y_lo:gap_top].max()) < 0.20
        or float(energy[gap_bottom:y_hi].max()) < 0.20
    ):
        return None

    # Split at the valley MIDPOINT: every ink row (including glyph tops
    # that dip into the shallow end of the valley) is assigned to its
    # nearest band. _tight_text_region() then trims each band to its
    # actual text rows/columns.
    mid = (gap_top + gap_bottom) // 2

    return [
        (y_lo, mid),
        (mid, y_hi),
    ]


def _energy_runs(
    energy: np.ndarray,
    threshold: float,
) -> list[tuple[int, int]]:
    """All maximal contiguous above-threshold runs, in order."""

    above = energy > threshold

    runs: list[tuple[int, int]] = []
    start = None

    for i, flag in enumerate(above):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            runs.append((start, i))
            start = None

    if start is not None:
        runs.append((start, len(above)))

    return runs


def _longest_energy_run(
    energy: np.ndarray,
    threshold: float,
    bridge: int = 0,
) -> tuple[int, int] | None:
    """Widest contiguous above-threshold run of rows/columns.

    `bridge` merges runs separated by gaps up to that many cells
    (e.g. the space in "MH 20"), while frame strokes sitting far from
    the text stay isolated and are dropped.
    """

    runs = _energy_runs(energy, threshold)

    if bridge > 0 and runs:
        merged: list[tuple[int, int]] = [runs[0]]

        for s, e in runs[1:]:
            if s - merged[-1][1] <= bridge:
                merged[-1] = (merged[-1][0], e)
            else:
                merged.append((s, e))

        runs = merged

    best = max(
        runs,
        key=lambda r: r[1] - r[0],
    )

    return best


def _max_run_per_column(
    binary_img: np.ndarray,
) -> np.ndarray:
    """Length of the longest vertical True-run for every column."""

    best = np.zeros(
        binary_img.shape[1],
        dtype=np.int32,
    )

    current = np.zeros_like(best)

    for row in binary_img:

        current = np.where(
            row,
            current + 1,
            0,
        )

        np.maximum(
            best,
            current,
            out=best,
        )

    return best


def _zero_frame_lines(
    ink: np.ndarray,
) -> None:
    """Zero out the plate's border-frame lines from the ink image (in place).

    Frame lines are STRUCTURALLY different from text: a thin, essentially
    full-span stroke (the plate rim). Per-column longest vertical ink-run
    flags them (glyph strokes never span the full band height), and only
    columns near the band's left/right edges are eligible — the frame is
    always at the rim, never in the middle.
    """

    h, w = ink.shape[:2]

    ink_max = float(ink.max())

    if ink_max <= 1e-6:
        return

    binary = ink > max(
        12.0,
        ink_max * 0.15,
    )

    edge_zone = max(2, int(w * 0.15))

    # Vertical frame lines -> per-column full-height runs.
    col_runs = _max_run_per_column(binary)

    frame_cols = col_runs > (h * 0.70)

    frame_cols[edge_zone : w - edge_zone] = False

    if frame_cols.any():

        # Text never sits OUTSIDE the border frame, so blank everything
        # from the band edge up to the innermost detected frame line on
        # each side. This also removes high-pass edge artifacts on the
        # dark plate surround (blur padding resurrects ink there).
        left_zone = frame_cols[:edge_zone]

        if left_zone.any():
            ink[:, : int(np.max(np.nonzero(left_zone)[0])) + 1] = 0

        right_zone = frame_cols[w - edge_zone :]

        if right_zone.any():
            ink[
                :,
                w - edge_zone + int(np.min(np.nonzero(right_zone)[0])) :,
            ] = 0

    # Horizontal frame lines -> per-row full-width runs.
    row_runs = _max_run_per_column(binary.T)

    frame_rows = row_runs > (w * 0.70)

    row_zone = max(2, int(h * 0.15))

    frame_rows[row_zone : h - row_zone] = False

    if frame_rows.any():

        top_zone = frame_rows[:row_zone]

        if top_zone.any():
            ink[: int(np.max(np.nonzero(top_zone)[0])) + 1, :] = 0

        bottom_zone = frame_rows[h - row_zone :]

        if bottom_zone.any():
            ink[
                h - row_zone + int(np.min(np.nonzero(bottom_zone)[0])) :,
                :,
            ] = 0


def _tight_text_region(
    band: np.ndarray,
) -> np.ndarray:
    """Trim a band slice to its actual text rows and columns.

    Uses a high-pass residual (band minus large-kernel blur) so smooth
    shading (uneven lighting, reflections) contributes no energy and
    only glyph strokes do. _zero_frame_lines() removes the plate rim's
    full-span strokes first (they would otherwise dominate the energy
    profile and read as phantom I/L/1 characters). Independent 25%
    guards per dimension keep a pathological trim from collapsing the
    band.
    """

    h, w = band.shape[:2]

    if h < 6 or w < 10:
        return band

    # High-pass: glyph strokes survive, smooth gradients/shadows vanish.
    ksize = max(3, (min(h, w) // 6) * 2 + 1)

    background = cv2.GaussianBlur(
        band,
        (ksize, ksize),
        0,
    )

    ink = cv2.absdiff(band, background)

    _zero_frame_lines(ink)

    rows = ink.sum(
        axis=1,
        dtype=np.float64,
    )

    cols = ink.sum(
        axis=0,
        dtype=np.float64,
    )

    max_row = float(rows.max())

    if max_row > 1e-6:
        rows = rows / max_row

    max_col = float(cols.max())

    if max_col > 1e-6:
        cols = cols / max_col

    y0, y1 = 0, h
    x0, x1 = 0, w

    row_run = _longest_energy_run(rows, 0.25)

    # Two stacked text bands (motorcycle plates): the longest
    # above-threshold row run is then ONE band, and trimming to it would
    # discard the other line entirely — the whole-crop canvas would look
    # single-line (low aspect) and the multi-line tier would never fire.
    # When a second substantial run exists, keep the FULL row span so the
    # band splitter sees both lines. Column trimming still applies (the
    # frame-line removal above is layout-independent).
    multi_band = False

    if row_run and (row_run[1] - row_run[0]) >= h * 0.25:
        significant_runs = [
            run
            for run in _energy_runs(rows, 0.25)
            if (run[1] - run[0]) >= _MIN_BAND_ROWS
            and (run[1] - run[0]) >= h * 0.12
        ]

        if len(significant_runs) >= 2:
            multi_band = True

    if multi_band:
        y0, y1 = 0, h
    elif row_run and (row_run[1] - row_run[0]) >= h * 0.25:
        y0, y1 = row_run

    col_run = _longest_energy_run(
        cols,
        0.25,
        bridge=max(2, int(w * 0.08)),
    )

    if (
        col_run
        and (col_run[1] - col_run[0]) >= w * 0.25
    ):
        x0, x1 = col_run

    return band[y0:y1, x0:x1]


# Below this a band read is noise, not evidence; using it would only
# inject junk into the combined candidate.
_MIN_BAND_CONF = 0.10


def _recognize_bands(
    enhanced_gray: np.ndarray,
    plain_gray: np.ndarray | None,
    bands: list[tuple[int, int]],
) -> list[tuple[str, float]]:
    """Recognize each horizontal band separately, then JOIN the reads.

    Band texts are short ("37N1", "4635"), far below the >=6-character
    filter the evidence selector applies — so the combined top-to-bottom
    string is the candidate that matters:

        "37-N1" over "4635"  ->  combined "37N14635"

    Each band is read on the CLAHE enhancement first; the plain
    grayscale is tried ONLY when the CLAHE read came back empty or weak
    (CLAHE amplifies sensor noise on clean/flat plates, which corrupts
    band reads — but on the majority of bands the plain retry would be
    a duplicate pass). Small crops, one recognizer pass per variant
    actually needed. Returns [] when fewer than two bands produced text
    (a split of a true single-line plate).
    """

    band_reads: list[tuple[str, float]] = []

    for y0, y1 in bands:

        best_read: tuple[str, float] | None = None

        for source in (enhanced_gray, plain_gray):

            if source is None:
                continue

            # After a decent CLAHE read, the plain-gray retry is a
            # duplicate pass, not new evidence.
            if (
                source is plain_gray
                and best_read is not None
                and best_read[1] >= 0.35
            ):
                break

            band = source[
                max(0, y0):max(0, y1),
                :,
            ]

            if band.size == 0:
                continue

            # Trim to the actual text rows/columns: removes the plate's
            # border-frame strokes and empty margins that otherwise
            # become phantom characters.
            band = _tight_text_region(band)

            if band.size == 0:
                continue

            # Mild padding keeps ascenders/descenders off the border.
            pad = max(4, band.shape[0] // 5)

            # Padding must blend with the plate background INSIDE the
            # band. A constant frame darker than the local background is
            # read by the recognizer as phantom edge characters, so take
            # the dominant (majority) luminance of the trimmed band:
            # bright for white plates, dark for black ones.
            border_value = int(np.percentile(band, 70))

            band = cv2.copyMakeBorder(
                band,
                pad,
                pad,
                pad,
                pad,
                cv2.BORDER_CONSTANT,
                value=border_value,
            )

            band_candidates = _run_ocr(band)

            if band_candidates:

                candidate = max(
                    band_candidates,
                    key=lambda item: item[1],
                )

                if (
                    best_read is None
                    or candidate[1] > best_read[1]
                ):
                    best_read = candidate

        if (
            best_read is not None
            and best_read[1] >= _MIN_BAND_CONF
        ):
            band_reads.append(best_read)

    if len(band_reads) < 2:
        return []

    combined = "".join(
        text for text, _, _ in band_reads
    )

    if not combined:
        return []

    combined_conf = min(
        conf for _, conf, _ in band_reads
    )

    # Concatenate the per-character probabilities of both bands so the
    # joined candidate keeps full edge evidence (leading char of band 1,
    # trailing char of band 2).
    combined_probs: list[float] | None = []

    for _, _, probs in band_reads:
        if probs is None:
            combined_probs = None
            break

        combined_probs.extend(probs)

    return [(combined, combined_conf, combined_probs)]


# ============================================================
# OCR SCORING
# ============================================================

def _candidate_score(
    text: str,
    ocr_conf: float,
) -> float:
    """Plate-aware OCR score; structure must outweigh confidence alone."""
    text = clean_text(text)
    if not text:
        return 0.0

    format_score = indian_plate_score(text)
    length = len(text)

    if 8 <= length <= 12:
        length_score = 20.0
    elif 6 <= length <= 13:
        length_score = 8.0
    else:
        length_score = 0.0

    return (
        float(ocr_conf) * 35.0
        + format_score * 0.90
        + length_score
        # Real-format validation on top of the legacy heuristic:
        # a strict match (real state code + valid series) is rewarded,
        # near-miss junk is penalized.
        + _strict_format_adjustment(text)
    )


# ============================================================
# OCR
# ============================================================

def _run_ocr(
    image: np.ndarray,
) -> list[tuple[str, float, list[float] | None]]:
    """
    Run exactly one EasyOCR recognizer pass on a plate crop.

    IMPORTANT:
    The reader is built with detector=False (see _ensure_models_loaded).
    YOLO has already isolated the plate, so CRAFT re-detection would only
    duplicate work and cost ~500 MB of RAM — which the free tier does not
    have. recognize() reads the crop as one text line and returns the
    same (text, confidence) evidence the cascade expects.

    Each candidate additionally carries the recognizer's per-character
    CTC probabilities (None when the capture hook is unavailable), used
    by the selectors to distrust weakly-evidenced edge characters.

    Evidence contract (multi-line support):
    recognize() on a multi-band image returns one entry per segment.
    Single-segment results are returned exactly as before. Multi-segment
    results are ALSO joined per pass (top-to-bottom, LTR) as extra
    candidates so two-line plates compete fairly with single-line ones.
    A single-segment pass therefore can never lose its previous winner —
    it can only gain candidates.
    """

    # One-time, lazy: keeps model import out of module import time.
    _install_char_conf_hook()

    try:
        import torch

        with torch.inference_mode():
            results = _ocr_reader.recognize(
                image,
                detail=1,
                paragraph=False,
                # Speed: EasyOCR's default contrast_ths=0.1 reruns the
                # whole recognizer a second time on every low-confidence
                # read (contrast-boosted). Our cascade already retries
                # with genuinely different preprocessing (OTSU, denoise)
                # and a structure-aware selector, so the internal retry
                # is redundant work on the free tier.
                contrast_ths=0.0,
                # "-" is allowed so the separator stamped on many Indian
                # plates ("37-N1") maps to a hyphen glyph instead of
                # being forced onto the nearest alphanumeric (usually a
                # phantom J/I). clean_text() strips it afterwards.
                allowlist=(
                    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                    "0123456789-"
                ),
            )

    except Exception:
        return []

    segments: list[
        tuple[float, float, str, float, list[float] | None]
    ] = []
    # (y_center, x_center, cleaned_text, confidence, char_probs)

    for item in results:

        box, text, conf = item[0], item[1], item[2]

        char_probs: list[float] | None = None

        if len(item) > 3 and isinstance(item[3], (list, tuple)):
            char_probs = [float(p) for p in item[3]]

        cleaned = clean_text(text)

        if not cleaned:
            continue

        # clean_text() strips '-' (and any separator); drop the
        # corresponding probability steps so the list stays aligned
        # with the kept characters.
        if (
            char_probs is not None
            and len(char_probs) != len(cleaned)
        ):
            # Rebuild alignment from the RAW text (before '-' removal):
            # walk raw chars, keep the prob of every char that survives
            # cleaning.
            kept_probs: list[float] = []
            raw_clean = clean_text(text)
            ri = 0

            for ch, p in zip(str(text).upper(), char_probs):
                if ch in ("-", " "):
                    continue

                if ri < len(raw_clean) and ch == raw_clean[ri]:
                    kept_probs.append(p)
                    ri += 1

            char_probs = (
                kept_probs
                if len(kept_probs) == len(cleaned)
                else None
            )

        if box and len(box) > 0:
            ys = [float(p[1]) for p in box]
            xs = [float(p[0]) for p in box]
            y_center = sum(ys) / len(ys)
            x_center = sum(xs) / len(xs)
        else:
            y_center = 0.0
            x_center = 0.0

        segments.append(
            (y_center, x_center, cleaned, float(conf), char_probs)
        )

    if not segments:
        return []

    candidates = [
        (text, conf, probs)
        for _, _, text, conf, probs in segments
    ]

    if len(segments) > 1:
        # Whole-crop reading, top-to-bottom then left-to-right.
        ordered = sorted(
            segments,
            key=lambda s: (
                s[0],
                s[1],
            ),
        )

        joined = "".join(
            text for _, _, text, _, _ in ordered
        )

        joined_conf = min(
            conf for _, _, _, conf, _ in ordered
        )

        joined_probs: list[float] | None = []

        for _, _, _, _, probs in ordered:
            if probs is None:
                joined_probs = None
                break

            joined_probs.extend(probs)

        if joined:
            candidates.append(
                (
                    joined,
                    joined_conf,
                    joined_probs if joined_probs else None,
                )
            )

    return candidates


# ============================================================
# CONSENSUS
# ============================================================

def _sequence_similarity(a: str, b: str) -> float:
    """Simple normalized character-position similarity."""
    a = clean_text(a)
    b = clean_text(b)

    if not a or not b:
        return 0.0

    common = min(len(a), len(b))
    positional = sum(
        1
        for i in range(common)
        if a[i] == b[i]
    )

    length_penalty = abs(len(a) - len(b))
    return max(
        0.0,
        (positional - length_penalty * 0.5)
        / max(len(a), len(b)),
    )


def _log_ocr_candidates(
    plate_index: int,
    tier_name: str,
    candidates: list,
) -> None:
    """Log OCR candidates for debugging without affecting inference."""
    if not candidates:
        logger.info(
            "plate%d %s: no candidates",
            plate_index,
            tier_name,
        )
        return

    formatted = ", ".join(
        f"{item[0]} ({item[1]:.3f})"
        for item in candidates
    )

    logger.info(
        "plate%d %s candidates: %s",
        plate_index,
        tier_name,
        formatted,
    )


def _evidence_select(
    tier_candidates: list[tuple[int, list]],
):
    """Select the most plausible plate using structure, confidence,
    agreement, real-format validation and edge-character evidence."""
    all_candidates: list[tuple[str, float, int, list[float] | None]] = []

    for tier_index, candidates in tier_candidates:
        for item in candidates:
            text = clean_text(item[0])
            if text and len(text) >= 6:
                probs = item[2] if len(item) > 2 else None
                all_candidates.append((text, float(item[1]), tier_index, probs))

    if not all_candidates:
        return None

    groups: dict[str, list[tuple[float, int, list[float] | None]]] = defaultdict(list)
    for text, conf, tier, probs in all_candidates:
        groups[text].append((conf, tier, probs))

    scored = []
    for text, observations in groups.items():
        confs = [c for c, _, _ in observations]
        avg_conf = sum(confs) / len(confs)
        max_conf = max(confs)
        votes = len(observations)
        format_score = indian_plate_score(text)

        primary_conf = max(
            (c for c, tier, _ in observations if tier == 1),
            default=0.0,
        )
        agreement_bonus = min(votes - 1, 2) * 7.0
        prefix_bonus = 25.0 if len(text) >= 2 and text[:2].isalpha() else 0.0
        length_bonus = (
            20.0 if 8 <= len(text) <= 12
            else 8.0 if 6 <= len(text) <= 13
            else 0.0
        )

        # Edge-character evidence: use the per-character probabilities
        # of the best-observed read of this text. Hallucinated leading/
        # trailing characters (frame strokes, crop margins) rest on
        # low-probability CTC steps.
        best_obs = max(observations, key=lambda o: o[0])
        edge = _first_last_conf(text, best_obs[2])

        edge_penalty = 0.0

        if edge is not None and edge[0] < _EDGE_CONF_PENALTY_THRESHOLD:
            edge_penalty = (
                _EDGE_CONF_PENALTY
                * (_EDGE_CONF_PENALTY_THRESHOLD - edge[0])
                / _EDGE_CONF_PENALTY_THRESHOLD
            )

        score = (
            max_conf * 35.0
            + avg_conf * 15.0
            + format_score * 0.90
            + prefix_bonus
            + length_bonus
            + agreement_bonus
            + primary_conf * 8.0
            + _strict_format_adjustment(text)
            - edge_penalty
        )

        scored.append((text, avg_conf, votes, score, format_score))

    scored.sort(key=lambda item: item[3], reverse=True)
    best_text, best_conf, best_votes, best_score, _ = scored[0]

    for text, avg_conf, votes, score, _ in scored[1:]:
        if _sequence_similarity(best_text, text) >= 0.70 and score > best_score + 5.0:
            best_text, best_conf, best_votes, best_score = text, avg_conf, votes, score

    return best_text, best_conf, best_votes


def _finalize_selection(
    tier_evidence: list[tuple[int, list]],
):
    """Production selection pipeline: evidence-gated format repairs join
    as tier 5, then the weighted evidence ranking decides. Shared by the
    API path and the benchmark harness so both measure the same thing.

    Returns (best, repairs).
    """

    repairs = _generate_format_repairs(tier_evidence)

    if repairs:
        tier_evidence.append((5, repairs))

    return _evidence_select(tier_evidence), repairs


def _consensus_select(
    candidates: list,
):
    """Select the best candidate from one OCR pass using plate structure,
    real-format validation and edge-character evidence."""
    if not candidates:
        return None

    groups: dict[str, list[tuple[float, list[float] | None]]] = defaultdict(list)

    for item in candidates:
        text = clean_text(item[0])
        if not text or len(text) < 6:
            continue
        probs = item[2] if len(item) > 2 else None
        groups[text].append((float(item[1]), probs))

    if not groups:
        return None

    scored = []
    for text, observations in groups.items():
        confs = [c for c, _ in observations]
        votes = len(confs)
        avg_conf = sum(confs) / votes
        format_score = indian_plate_score(text)
        consensus_bonus = min(votes - 1, 2) * 7.0

        best_probs = max(
            observations,
            key=lambda o: o[0],
        )[1]

        edge = _first_last_conf(text, best_probs)

        edge_penalty = 0.0

        if edge is not None and edge[0] < _EDGE_CONF_PENALTY_THRESHOLD:
            edge_penalty = (
                _EDGE_CONF_PENALTY
                * (_EDGE_CONF_PENALTY_THRESHOLD - edge[0])
                / _EDGE_CONF_PENALTY_THRESHOLD
            )

        score = (
            avg_conf * 35.0
            + format_score * 0.90
            + (20.0 if 8 <= len(text) <= 12 else 8.0 if 6 <= len(text) <= 13 else 0.0)
            + consensus_bonus
            + _strict_format_adjustment(text)
            - edge_penalty
        )

        scored.append((text, avg_conf, votes, score))

    return max(scored, key=lambda item: item[3])[:3]


# ============================================================
# ADAPTIVE ESCALATION
# ============================================================

def _is_strong_enough(candidate) -> bool:
    """Stop OCR when the candidate is both readable and plate-shaped."""
    if candidate is None:
        return False

    text, conf, votes = candidate
    text = clean_text(text)
    if not text or len(text) < 6:
        return False

    format_score = indian_plate_score(text)

    if conf >= 0.55 and format_score >= 45:
        return True

    if 0.45 <= conf < 0.55 and format_score >= 60 and len(text) <= 13:
        return True

    if votes >= 2 and conf >= 0.40 and format_score >= 60:
        return True

    return False


# ============================================================
# MAIN ENTRY POINT
# ============================================================

def run_detection_on_image(
    image_bytes: bytes,
    conf_threshold: float = 0.25,
) -> dict[str, Any]:
    """
    Run:

        image
          ↓
        YOLO
          ↓
        crop
          ↓
        CLAHE OCR
          ↓
        optional OTSU OCR
          ↓
        optional denoise OCR
          ↓
        scoring
          ↓
        annotation

    API response schema remains unchanged.
    """

    # --------------------------------------------------------
    # Lazy model initialization
    # --------------------------------------------------------
    # Keep one-time model startup out of per-image processing latency.

    _ensure_models_loaded()

    timing = _TimingBucket()

    # --------------------------------------------------------
    # Decode image
    # --------------------------------------------------------

    np_arr = np.frombuffer(
        image_bytes,
        np.uint8,
    )

    image = cv2.imdecode(
        np_arr,
        cv2.IMREAD_COLOR,
    )

    if image is None:
        raise ValueError(
            "Could not decode image. "
            "Is it a valid image file?"
        )

    # Downscale oversized uploads before YOLO (schema/coordinates are
    # reported in the processed image's pixel space, exactly like the
    # video pipeline).
    h, w = image.shape[:2]

    longest = max(h, w)

    if longest > _MAX_INFER_SIDE:
        scale = _MAX_INFER_SIDE / longest

        image = cv2.resize(
            image,
            (
                max(1, int(w * scale)),
                max(1, int(h * scale)),
            ),
            interpolation=cv2.INTER_AREA,
        )

        h, w = image.shape[:2]

    # --------------------------------------------------------
    # YOLO detection
    # --------------------------------------------------------

    import torch

    t0 = time.perf_counter()

    with torch.inference_mode():
        results = _yolo_model.predict(
            source=image,
            imgsz=640,
            conf=conf_threshold,
            verbose=False,
        )

    timing.add(
        "YOLO",
        time.perf_counter() - t0,
    )

    result = results[0]

    detections = []

    if result.boxes is not None:

        for box in result.boxes:

            confidence = float(
                box.conf[0]
            )

            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0].tolist(),
            )

            detections.append(
                (
                    confidence,
                    x1,
                    y1,
                    x2,
                    y2,
                )
            )

    # Ultralytics Results retain the full-resolution frame and overlay
    # tensors; everything needed (conf + xyxy) is already extracted.
    del results
    del result

    # Highest confidence first.
    detections.sort(
        key=lambda d: d[0],
        reverse=True,
    )

    # Bounded OCR effort: at most _MAX_OCR_BOXES boxes ever reach the
    # OCR cascade (real images carry 1-2 plates; the cap only bites on
    # NMS misfires). Lower-confidence boxes beyond the cap are still
    # reported (bbox + YOLO confidence, status OCR_FAILED) so the API
    # response keeps describing every detection.
    ocr_eligible = detections[:_MAX_OCR_BOXES]

    plates = []

    annotated = image.copy()

    ocr_pass_count = 0

    # ========================================================
    # PROCESS EVERY DETECTED PLATE
    # ========================================================

    for idx, (
        yolo_conf,
        x1,
        y1,
        x2,
        y2,
    ) in enumerate(
        detections,
        start=1,
    ):

        # ----------------------------------------------------
        # Clamp bounding box
        # ----------------------------------------------------

        x1c = max(
            0,
            x1,
        )

        y1c = max(
            0,
            y1,
        )

        x2c = min(
            w,
            x2,
        )

        y2c = min(
            h,
            y2,
        )

        crop = image[
            y1c:y2c,
            x1c:x2c,
        ]

        # ----------------------------------------------------
        # Default plate result
        # ----------------------------------------------------

        plate_entry = {
            "plate_id": idx,
            "bbox": [
                x1c,
                y1c,
                x2c,
                y2c,
            ],
            "yolo_confidence": round(
                yolo_conf,
                4,
            ),
            "ocr_text": "",
            "ocr_confidence": 0.0,
            "validation_score": 0,
            "validation": "UNVERIFIED",
            "final_confidence": 0.0,
            "status": "OCR_FAILED",
        }

        # ====================================================
        # OCR
        # ====================================================

        if crop.size > 0 and idx <= len(ocr_eligible):

            # ------------------------------------------------
            # Crop / resize / grayscale
            # ------------------------------------------------

            t0 = time.perf_counter()

            base_gray = _canvas_gray(
                crop
            )

            timing.add(
                "Crop",
                time.perf_counter() - t0,
            )

            # =================================================
            # TIER 1 — CLAHE
            # =================================================

            tier_evidence: list[tuple[int, list[tuple[str, float]]]] = []

            t0 = time.perf_counter()

            enhanced = _variant_primary(
                base_gray
            )

            timing.add(
                "Preprocess",
                time.perf_counter() - t0,
            )

            t0 = time.perf_counter()

            tier1_candidates = _run_ocr(
                enhanced
            )

            ocr_pass_count += 1

            timing.add(
                "OCR",
                time.perf_counter() - t0,
                note=f"plate{idx} tier1(CLAHE)",
            )

            _log_ocr_candidates(
                idx,
                "tier1(CLAHE)",
                tier1_candidates,
            )

            tier_evidence.append(
                (1, tier1_candidates)
            )

            # Use the current tier for the stopping decision.
            t0 = time.perf_counter()

            best = _consensus_select(
                tier1_candidates
            )

            timing.add(
                "Scoring",
                time.perf_counter() - t0,
            )

            # ------------------------------------------------------------
            # Effort policy: the historical batch run recorded every box
            # below 0.45 YOLO confidence as producing junk OCR. Such boxes
            # get one cheap pass; the strong confidence they lack never
            # materializes, so tiers 2-4 are skipped unless tier 1 read
            # literally nothing (then one fallback is spent, bounded).
            # ------------------------------------------------------------
            low_effort = (
                yolo_conf < _MIN_FULL_EFFORT_YOLO_CONF
                and best is not None
            )

            # =================================================
            # TIER 4 — MULTI-LINE SPLIT (two-band plates),
            # runs BEFORE tiers 2/3 by design:
            #
            # A stacked plate read as ONE line produces exactly the
            # production failure being fixed here: the merged read can be
            # long, structured junk (e.g. EKA181897-style strings pass the
            # lenient legacy pattern) that looks "strong" and stops the
            # cascade, while the real content is two short lines the
            # whole-crop pass never resolved. The band split is cheap
            # OpenCV geometry and at most two bounded recognizer passes,
            # so for tall crops it runs first; tiers 2/3 then only run
            # when the band evidence is ALSO weak, so a false split can
            # never remove the old fallback path — it can only add
            # candidates.
            # =================================================

            # Gate on BOTH aspects: the tightened canvas can still
            # under-report height for stacked plates (its row profile is
            # dominated by whichever band has more ink), so the raw YOLO
            # crop's aspect must also be considered.
            raw_crop_aspect = (
                float(y2c - y1c)
                / max(float(x2c - x1c), 1.0)
            )

            crop_aspect = max(
                float(base_gray.shape[0])
                / max(float(base_gray.shape[1]), 1.0),
                raw_crop_aspect,
            )

            # A whole-crop read whose EDGE characters are not backed by
            # the recognizer's own per-character probabilities is weak
            # evidence too (phantom leading/trailing chars).
            whole_crop_weak = (
                best is None
                or not _is_strong_enough(best)
                or (
                    best is not None
                    and not _edge_supported_in(
                        best[0],
                        tier1_candidates,
                    )
                )
            )

            # On a TALL crop a strong-but-not-strict read is NOT trusted:
            # two stacked bands merged into one line can satisfy the
            # lenient legacy pattern (any 2-3 letters + digits), while a
            # genuine read of a stacked plate can never be strict-valid
            # (the motorcycle state code is numeric). This is the gate
            # that keeps merged-garbage reads from stopping the cascade.
            if (
                crop_aspect >= _MAX_SPLIT_ASPECT
                and best is not None
                and not strict_indian_plate(best[0])
            ):
                whole_crop_weak = True

            # NOTE: no low_effort exclusion here. A two-line plate whose
            # whole-crop read is weak (one line only, or merged garbage)
            # must reach the band splitter even at low YOLO confidence —
            # the split is cheap OpenCV geometry and adds at most two
            # bounded recognizer passes. Crops without plausible band
            # geometry return no bands and cost nothing further.
            if (
                crop_aspect >= _MAX_SPLIT_ASPECT
                and whole_crop_weak
            ):

                t0 = time.perf_counter()

                bands = _split_two_line_bands(
                    enhanced
                )

                timing.add(
                    "Preprocess",
                    time.perf_counter() - t0,
                    note=f"plate{idx} multiline-split",
                )

                if bands:

                    t0 = time.perf_counter()

                    band_candidates = _recognize_bands(
                        enhanced,
                        base_gray,
                        bands,
                    )

                    ocr_pass_count += len(bands)

                    timing.add(
                        "OCR",
                        time.perf_counter() - t0,
                        note=(
                            f"plate{idx} tier4(bands="
                            f"{len(bands)})"
                        ),
                    )

                    _log_ocr_candidates(
                        idx,
                        "tier4(multiline)",
                        band_candidates,
                    )

                    if band_candidates:

                        tier_evidence.append(
                            (
                                4,
                                band_candidates,
                            )
                        )

                        t0 = time.perf_counter()

                        best = _evidence_select(
                            tier_evidence
                        )

                        timing.add(
                            "Scoring",
                            time.perf_counter() - t0,
                        )

            # =================================================
            # TIER 2 — OTSU FALLBACK
            # =================================================

            if not low_effort and not _is_strong_enough(best):

                t0 = time.perf_counter()

                fallback_img = _variant_fallback(
                    enhanced
                )

                timing.add(
                    "Preprocess",
                    time.perf_counter() - t0,
                )

                t0 = time.perf_counter()

                tier2_candidates = _run_ocr(
                    fallback_img
                )

                ocr_pass_count += 1

                timing.add(
                    "OCR",
                    time.perf_counter() - t0,
                    note=f"plate{idx} tier2(OTSU)",
                )

                _log_ocr_candidates(
                    idx,
                    "tier2(OTSU)",
                    tier2_candidates,
                )

                tier_evidence.append(
                    (2, tier2_candidates)
                )

                t0 = time.perf_counter()

                best = _evidence_select(
                    tier_evidence
                )

                timing.add(
                    "Scoring",
                    time.perf_counter() - t0,
                )

                # =================================================
                # TIER 3 — DENOISE LAST RESORT
                # =================================================

                # Skip when Tier 2 contributed nothing new: if OTSU
                # produced no candidate that Tier 1 hadn't already
                # produced, a denoised OTSU (nearly the same image)
                # will not either — and Tier 4 handles the layouts
                # where whole-crop reading fundamentally fails.
                tier2_new = any(
                    clean_text(item[0])
                    not in {
                        clean_text(prev[0])
                        for prev in tier1_candidates
                    }
                    for item in tier2_candidates
                )

                # Only run Tier 3 if the combined evidence is still weak.
                if (
                    (best is None or not _is_strong_enough(best))
                    and tier2_new
                ):

                    t0 = time.perf_counter()

                    tertiary_img = _variant_tertiary(
                        enhanced
                    )

                    timing.add(
                        "Preprocess",
                        time.perf_counter() - t0,
                    )

                    t0 = time.perf_counter()

                    tier3_candidates = _run_ocr(
                        tertiary_img
                    )

                    ocr_pass_count += 1

                    timing.add(
                        "OCR",
                        time.perf_counter() - t0,
                        note=f"plate{idx} tier3(denoise)",
                    )

                    _log_ocr_candidates(
                        idx,
                        "tier3(denoise)",
                        tier3_candidates,
                    )

                    tier_evidence.append(
                        (3, tier3_candidates)
                    )

                    t0 = time.perf_counter()

                    best = _evidence_select(
                        tier_evidence
                    )

                    timing.add(
                        "Scoring",
                        time.perf_counter() - t0,
                    )

            # Final evidence-based selection.
            if tier_evidence:
                t0 = time.perf_counter()

                # Narrow, evidence-gated format repairs (O/0, I/1, ...)
                # join as extra candidates before the final ranking.
                evidence_best, repairs = _finalize_selection(
                    tier_evidence
                )

                if repairs:
                    _log_ocr_candidates(
                        idx,
                        "format-repairs",
                        repairs,
                    )

                if evidence_best is not None:
                    best = evidence_best

                timing.add(
                    "Scoring",
                    time.perf_counter() - t0,
                )

            # =================================================
            # FINAL PLATE SCORING
            # =================================================

            if best is not None:

                (
                    best_text,
                    best_ocr_conf,
                    best_votes,
                ) = best

                validation_score = indian_plate_score(
                    best_text
                )

                validation = classify_validation(
                    validation_score
                )

                final_confidence = min(
                    1.0,
                    best_ocr_conf * 0.6
                    + (
                        validation_score
                        / 100
                    ) * 0.4,
                )

                # Per-character evidence of the winning read (for the
                # API/debug consumers; None when unavailable). The win
                # may come from any tier, so scan all collected evidence.
                _win_probs = next(
                    (
                        item[2]
                        for _tier, cands in tier_evidence
                        for item in cands
                        if len(item) > 2
                        and clean_text(item[0]) == best_text
                        and item[2] is not None
                    ),
                    None,
                )
                win_edge = _first_last_conf(
                    best_text,
                    _win_probs,
                )

                plate_entry.update(
                    {
                        "ocr_text": best_text,
                        "ocr_confidence": round(
                            best_ocr_conf,
                            4,
                        ),
                        "validation_score": validation_score,
                        "validation": validation,
                        "final_confidence": round(
                            final_confidence,
                            4,
                        ),
                        "status": status_from_confidence(
                            final_confidence,
                            best_text,
                        ),
                        "strict_format": strict_indian_plate(
                            best_text
                        ),
                        "edge_confidence": (
                            round(win_edge[0], 4)
                            if win_edge is not None
                            else None
                        ),
                    }
                )

                # -------------------------------------------------
                # Debug logging
                # -------------------------------------------------

                if os.environ.get(
                    "LVA_DEBUG_TIMING",
                    "1",
                ) != "0":

                    logger.info(
                        "  plate%d consensus: "
                        "text=%r votes=%d "
                        "avg_conf=%.3f format=%d",
                        idx,
                        best_text,
                        best_votes,
                        best_ocr_conf,
                        validation_score,
                    )

        # ====================================================
        # SAVE PLATE
        # ====================================================

        plates.append(
            plate_entry
        )

        # ====================================================
        # ANNOTATION
        # ====================================================

        if plate_entry["ocr_text"]:

            color = (
                0,
                200,
                100,
            )

        else:

            color = (
                0,
                140,
                255,
            )

        cv2.rectangle(
            annotated,
            (
                x1c,
                y1c,
            ),
            (
                x2c,
                y2c,
            ),
            color,
            3,
        )

        label = (
            f"{plate_entry['ocr_text'] or 'UNKNOWN'}"
            f" | "
            f"{plate_entry['final_confidence'] * 100:.0f}%"
        )

        cv2.putText(
            annotated,
            label,
            (
                x1c,
                max(
                    25,
                    y1c - 10,
                ),
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
            cv2.LINE_AA,
        )

    # ========================================================
    # ENCODE ANNOTATED IMAGE
    # ========================================================

    ok, buf = cv2.imencode(
        ".jpg",
        annotated,
        [cv2.IMWRITE_JPEG_QUALITY, 85],
    )

    annotated_b64 = None

    if ok:

        import base64

        annotated_b64 = base64.b64encode(
            buf.tobytes()
        ).decode(
            "ascii"
        )

    # ========================================================
    # TIMING
    # ========================================================

    elapsed = timing.total()

    timing.report(
        plate_count=len(plates),
        ocr_pass_count=ocr_pass_count,
    )

    # Per-request transient allocations (decoded image, annotated copy,
    # crops, OCR tensors) are garbage by now; hand the heap back so the
    # container's resident set stays close to the model footprint.
    gc.collect()
    _release_memory_to_os()

    # ========================================================
    # API RESPONSE
    # ========================================================

    return {
        "image": {
            "width": w,
            "height": h,
        },
        "plates_detected": len(plates),
        "plates": plates,
        "annotated_image_base64": annotated_b64,
        "processing_time_seconds": round(
            elapsed,
            3,
        ),
    }