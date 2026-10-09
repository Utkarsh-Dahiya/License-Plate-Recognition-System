"""
AUDIT CONTACT SHEETS — one-off, read-only diagnostic
(writes only to diagnostics/audit/sheets/).

Builds labeled contact sheets of the crops saved by audit_collect.py,
sorted by crop WIDTH (descending) so the readable / medium-quality
plates come first and the low-resolution tail comes last.  Each entry
is stamped with its filename and pixel dimensions so transcriptions
attach unambiguously to a crop.

Usage (from project root):
    venv/Scripts/python.exe diagnostics/audit_sheets.py
    venv/Scripts/python.exe diagnostics/audit_sheets.py --min-w 90 --prefix readable
    venv/Scripts/python.exe diagnostics/audit_sheets.py --max-w 59 --prefix lowres
    venv/Scripts/python.exe diagnostics/audit_sheets.py --offset 30 --count 24
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
AUDIT = HERE / "audit"
CROPS = AUDIT / "crops"
OUT = AUDIT / "sheets"

TARGET_H = 160
LABEL_H = 28
GAP = 12
PER_SHEET = 6


def font() -> ImageFont.ImageFont:
    for name in ("arialbd.ttf", "arial.ttf", "segoeui.ttf"):
        try:
            return ImageFont.truetype(name, 20)
        except OSError:
            continue
    return ImageFont.load_default()


def collect_rows(collection: Path) -> list[dict]:
    """Crop rows joined with their image record (kept for labeling)."""
    data = json.loads(collection.read_text(encoding="utf-8"))
    rows = []
    for im in data["images"]:
        for c in im["crops"]:
            rows.append({
                "file": c["file"],
                "w": c["w"],
                "h": c["h"],
                "aspect": c["aspect"],
                "source": c["source"],
                "gt_index": c["gt_index"],
                "image": im["image"],
                "matched": any(
                    m["matched"] and m["gt_index"] == c["gt_index"]
                    for m in im["gt_match"]
                ),
            })
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description="Labeled contact sheets of audit crops.")
    ap.add_argument("--collection", default=str(AUDIT / "collection.json"))
    ap.add_argument("--prefix", default="sheet")
    ap.add_argument("--height", type=int, default=TARGET_H)
    ap.add_argument("--per-sheet", type=int, default=PER_SHEET)
    ap.add_argument("--min-w", type=int, default=0)
    ap.add_argument("--max-w", type=int, default=10_000)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--count", type=int, default=0, help="0 = all")
    args = ap.parse_args()

    rows = collect_rows(Path(args.collection))
    rows.sort(key=lambda r: (-r["w"], r["file"]))
    rows = [r for r in rows if args.min_w <= r["w"] <= args.max_w]
    rows = rows[args.offset:]
    if args.count:
        rows = rows[: args.count]
    if not rows:
        print("no crops match")
        return 1

    OUT.mkdir(parents=True, exist_ok=True)
    fnt = font()
    entries = []
    for r in rows:
        p = CROPS / r["file"]
        if not p.exists():
            print(f"MISSING {p.name}")
            continue
        im = Image.open(p).convert("RGB")
        w, h = im.size
        scale = args.height / h
        im = im.resize((max(1, round(w * scale)), args.height), Image.LANCZOS)
        label = f"{r['file'][:-4]}  {w}x{h}"
        tw = max(ImageDraw.Draw(Image.new("RGB", (1, 1))).textlength(label, font=fnt), 1)
        entries.append((label, im, max(im.width, int(tw) + 16)))

    made = 0
    for si in range(0, len(entries), args.per_sheet):
        chunk = entries[si: si + args.per_sheet]
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
        made += 1

    print(f"{made} sheets written -> {OUT}  ({len(entries)} crops)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

