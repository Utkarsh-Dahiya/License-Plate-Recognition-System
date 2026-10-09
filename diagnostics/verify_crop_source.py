"""
CROP PROVENANCE CHECK — one-off, read-only diagnostic.

diagnostics/make_gt_crops.py keys plate crops by their numeric prefix
("0001_Cars0.jpg" -> batch_results/0001_plate.jpg), which assumes the
_NNNN_ index used by the original batch run matches the sorted position
of that image in YOLO_dataset/images.  That assumption was questioned
when a transcription didn't match the expected OCR prediction, so this
script verifies it directly: for every sampled crop it template-matches
the crop against its claimed source scene at multiple scales.  A high
score proves the crop really came from that scene.

It reads images and writes nothing.  Usage (from project root):
    venv/Scripts/python.exe diagnostics/verify_crop_source.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app" / "backend"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

SAMPLE = Path(__file__).resolve().parent / "sample.txt"
IMG_DIRS = [ROOT / "YOLO_dataset" / "images"]
EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# matchTemplate score above this on the claimed scene = "same image".
MATCH_OK = 0.6


def scene_index() -> dict[str, Path]:
    """stem -> path for every source scene (first extension wins)."""
    index: dict[str, Path] = {}
    for d in IMG_DIRS:
        if not d.exists():
            continue
        for p in sorted(d.rglob("*")):
            if p.is_file() and p.suffix.lower() in EXTS:
                index.setdefault(p.stem, p)
    return index


def best_score(crop: np.ndarray, scene: np.ndarray) -> float:
    """Best normalized match of crop anywhere in scene, over scales.

    The crop may have been saved before or after the 4x OCR upscale, so
    the crop is shrunk (never enlarged) until it fits the scene; the
    max over all fitting scales is returned.
    """
    cg = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    sg = cv2.cvtColor(scene, cv2.COLOR_BGR2GRAY)
    ch, cw = cg.shape
    sh, sw = sg.shape

    best = -1.0
    # 1.0 = crop is a literal sub-image of the scene; 0.25 undoes the
    # 4x upscale batch_test.py applied before OCR.
    scale = 1.0
    while scale >= 0.15:
        rw, rh = max(1, int(round(cw * scale))), max(1, int(round(ch * scale)))
        if rw <= sw and rh <= sh:
            rc = cv2.resize(cg, (rw, rh), interpolation=cv2.INTER_AREA)
            res = cv2.matchTemplate(sg, rc, cv2.TM_CCOEFF_NORMED)
            best = max(best, float(res.max()))
            if best >= MATCH_OK:
                return best
        scale *= 0.85
    return best


def main() -> int:
    scenes = scene_index()
    if not scenes:
        print("No source scenes found.")
        return 1

    print(f"{len(scenes)} source scenes indexed\n")
    failures = 0

    for line in SAMPLE.read_text(encoding="utf-8").splitlines():
        name = line.strip()
        if not name:
            continue
        stem = name.rsplit(".", 1)[0]
        prefix, scene_stem = stem.split("_", 1)

        crop_path = ROOT / "batch_results" / f"{prefix}_plate.jpg"
        scene_path = scenes.get(scene_stem)

        if not crop_path.exists():
            print(f"{stem}: MISSING crop {crop_path.name}")
            failures += 1
            continue
        if scene_path is None:
            print(f"{stem}: MISSING scene for stem '{scene_stem}'")
            failures += 1
            continue

        crop = cv2.imread(str(crop_path))
        scene = cv2.imread(str(scene_path))
        if crop is None or scene is None:
            print(f"{stem}: UNREADABLE input")
            failures += 1
            continue

        score = best_score(crop, scene)
        verdict = "OK " if score >= MATCH_OK else "MISMATCH"
        if score < MATCH_OK:
            failures += 1
        print(
            f"{verdict} {prefix} crop {crop.shape[1]}x{crop.shape[0]} vs "
            f"{scene_path.name}: score={score:.3f}"
        )

    print(f"\n{'ALL CROPS MATCH THEIR CLAIMED SCENES' if failures == 0 else f'{failures} FAILURES'}")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())