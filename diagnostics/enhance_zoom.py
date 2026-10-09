"""
CONTRAST/SHARPNESS ENHANCEMENT — one-off, read-only diagnostic (writes only to diagnostics/contact_sheets/).

The hardest crops (0034, 0333) are ambiguous even at full contact-sheet
magnification.  This script applies contrast stretching + unsharp masking to
the labelled zoom sheet and writes a fresh *_eN.png file, so a subsequent
image read sees enhanced pixels (and, being a new filename, bypasses the
read cache).

Enhancement cannot invent detail that is not in the source pixels; it only
makes existing contrast easier to see.  Successive runs bump the counter so
each run produces a fresh, uncached image.

Usage (from project root):
    venv/Scripts/python.exe diagnostics/enhance_zoom.py contact_sheets/zoomhard_1.png
    venv/Scripts/python.exe diagnostics/enhance_zoom.py contact_sheets/zoomhard_3.png
"""

from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image, ImageEnhance, ImageFilter

HERE = Path(__file__).resolve().parent
OUT = HERE / "contact_sheets"


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 1
    src = Path(sys.argv[1])
    if not src.is_absolute():
        src = HERE / src
    if not src.exists():
        print(f"MISSING {src}")
        return 1

    out = OUT / f"{src.stem}_e1.png"
    n = 1
    while out.exists():
        n += 1
        out = OUT / f"{src.stem}_e{n}.png"

    im = Image.open(src).convert("RGB")
    im = ImageEnhance.Contrast(im).enhance(1.8)
    im = ImageEnhance.Color(im).enhance(0.6)      # tame the green pad
    im = im.filter(ImageFilter.UnsharpMask(radius=6, percent=180, threshold=2))
    im.save(out)
    print(f"{out.name}: {im.size[0]}x{im.size[1]} from {src.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
