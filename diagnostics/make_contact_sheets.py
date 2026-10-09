"""
CONTACT SHEETS — one-off, read-only diagnostic (writes only to diagnostics/contact_sheets/).

Batch reads of individual crop files arrive out of order, which makes it
ambiguous which transcription belongs to which file.  This script builds
labeled contact sheets: each ground-truth crop is upscaled to a uniform
height with its filename drawn above it, so a single sheet read yields
unambiguous transcriptions for several crops at once.

Options let it zoom into specific crops at higher magnification when the
first pass leaves character-level ambiguity (I vs 1, O vs 0, ...).

Usage (from project root):
    venv/Scripts/python.exe diagnostics/make_contact_sheets.py
    venv/Scripts/python.exe diagnostics/make_contact_sheets.py --prefix zoom \
        --height 360 --per-sheet 2 0001_Cars0.png 0167_Cars293.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
CROPS = HERE / "gt_crops"
OUT = HERE / "contact_sheets"

TARGET_H = 160          # upscaled crop height in px
LABEL_H = 28            # filename bar height
GAP = 12                # spacing between entries
PER_SHEET = 4           # crops per sheet


def font() -> ImageFont.ImageFont:
    for name in ("arialbd.ttf", "arial.ttf", "segoeui.ttf"):
        try:
            return ImageFont.truetype(name, 20)
        except OSError:
            continue
    return ImageFont.load_default()


def main() -> int:
    ap = argparse.ArgumentParser(description="Build labeled contact sheets of gt_crops.")
    ap.add_argument("crops", nargs="*", help="crop filenames (default: all in gt_crops)")
    ap.add_argument("--prefix", default="sheet", help="output filename prefix")
    ap.add_argument("--height", type=int, default=TARGET_H)
    ap.add_argument("--per-sheet", type=int, default=PER_SHEET)
    args = ap.parse_args()

    OUT.mkdir(exist_ok=True)
    if args.crops:
        files = []
        for name in args.crops:
            hit = sorted(CROPS.glob(name.rsplit(".", 1)[0] + ".*"))
            if not hit:
                print(f"MISSING crop {name}")
                return 1
            files.append(hit[0])
    else:
        files = sorted(p for p in CROPS.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"})
    if not files:
        print("No crops found.")
        return 1

    fnt = font()
    entries = []
    for p in files:
        im = Image.open(p).convert("RGB")
        w, h = im.size
        scale = args.height / h
        im = im.resize((max(1, round(w * scale)), args.height), Image.LANCZOS)
        label = p.name
        tw = max(ImageDraw.Draw(Image.new("RGB", (1, 1))).textlength(label, font=fnt), 1)
        bar_w = max(im.width, int(tw) + 16)
        entries.append((label, im, bar_w))

    made = 0
    for si in range(0, len(entries), args.per_sheet):
        chunk = entries[si : si + args.per_sheet]
        sheet_w = max(e[2] for e in chunk)
        sheet_h = sum(LABEL_H + args.height for _ in chunk) + GAP * (len(chunk) + 1)
        sheet = Image.new("RGB", (sheet_w + 2 * GAP, sheet_h), (245, 245, 245))
        d = ImageDraw.Draw(sheet)
        y = GAP
        for label, im, _ in chunk:
            d.rectangle([GAP, y, GAP + sheet_w, y + LABEL_H], fill=(30, 30, 30))
            d.text((GAP + 8, y + 4), label, fill=(255, 255, 0), font=fnt)
            y += LABEL_H
            sheet.paste(im, (GAP, y))
            y += args.height + GAP
        out = OUT / f"{args.prefix}_{si // args.per_sheet + 1}.png"
        sheet.save(out)
        print(f"{out.name}: {sheet.size[0]}x{sheet.size[1]} -> {', '.join(c[0] for c in chunk)}")
        made += 1

    print(f"\n{made} sheets written to {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
