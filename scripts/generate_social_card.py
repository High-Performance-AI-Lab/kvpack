#!/usr/bin/env python3
"""Generate the kvpack social card (1200x630 PNG).

Deterministic output: no randomness, no external assets. The committed
assets/kvpack-social-card.png is the render used as the GitHub social
preview. Re-run after changing any text:

    python3 scripts/generate_social_card.py --check   # verify committed PNG
    python3 scripts/generate_social_card.py           # regenerate
"""

import argparse
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

W, H = 1200, 630
BG = (13, 17, 23)          # deep charcoal
FG = (240, 246, 252)       # near-white
ACCENT = (63, 185, 170)    # teal
DIM = (139, 148, 158)      # muted grey

FONT_PATHS = [
    "/System/Library/Fonts/SFNS.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
    "/System/Library/Fonts/SUPPLEMENTARY/Arial Bold.ttf",
]


def font(size: int) -> ImageFont.FreeTypeFont:
    for p in FONT_PATHS:
        if Path(p).exists():
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def render() -> Image.Image:
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)

    # accent rail on the left edge
    d.rectangle([0, 0, 14, H], fill=ACCENT)

    x = 84
    d.text((x, 130), "kvpack", font=font(150), fill=FG)
    d.text((x, 330), "Crash-safe, bitwise-exact KV-cache replay.",
           font=font(52), fill=ACCENT)
    d.text((x, 420), "Save inference state once. Restore it bit-identically —",
           font=font(36), fill=DIM)
    d.text((x, 470), "after a restart, a crash, or on another machine.",
           font=font(36), fill=DIM)
    d.text((x, 555), "github.com/High-Performance-AI-Lab/kvpack",
           font=font(30), fill=DIM)
    return img


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="verify the committed PNG matches a fresh render")
    args = ap.parse_args()

    out = Path(__file__).resolve().parent.parent / "assets" / "kvpack-social-card.png"
    fresh = render()

    if args.check:
        if not out.exists():
            print("FAIL: committed card missing:", out)
            return 1
        committed = Image.open(out).convert("RGB")
        if committed.tobytes() != fresh.tobytes():
            print("FAIL: committed card differs from fresh render")
            return 1
        print("PASS: social card matches render")
        return 0

    out.parent.mkdir(parents=True, exist_ok=True)
    fresh.save(out, format="PNG", optimize=True)
    print("wrote", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
