"""Build gallery/: 520-px thumbnails and four 5x5 contact sheets."""
import re
from pathlib import Path

from PIL import Image, ImageDraw

R = Path(__file__).resolve().parents[1]
F = R / "figures"
G = R / "gallery"
(G / "thumbs").mkdir(parents=True, exist_ok=True)

pat = re.compile(r"^- `(fig(\d+)_[a-z0-9_]+)\.py` \u2014 (.+)$")
entries = []
for line in (F / "README.md").read_text().splitlines():
    m = pat.match(line)
    if m:
        entries.append((int(m.group(2)), m.group(1)))
entries.sort()

for _, stem in entries:
    im = Image.open(F / f"{stem}.png")
    im.thumbnail((520, 520), Image.LANCZOS)
    im.save(G / "thumbs" / f"{stem}.png", optimize=True)

CELL, PAD = 300, 10
for s in range(0, len(entries), 25):
    block = entries[s:s + 25]
    sheet = Image.new("RGB", (5 * CELL + 6 * PAD, 5 * CELL + 6 * PAD),
                      "white")
    d = ImageDraw.Draw(sheet)
    for k, (num, stem) in enumerate(block):
        im = Image.open(G / "thumbs" / f"{stem}.png").convert("RGB")
        im.thumbnail((CELL, CELL - 16), Image.LANCZOS)
        r, c = divmod(k, 5)
        x = PAD + c * (CELL + PAD) + (CELL - im.width) // 2
        y = PAD + r * (CELL + PAD)
        sheet.paste(im, (x, y))
        d.text((PAD + c * (CELL + PAD) + 4, y + CELL - 14),
               f"fig{num:03d}", fill=(90, 90, 90))
    out = G / f"sheet{s // 25 + 1}.png"
    sheet.save(out, optimize=True)
    print(out)
print(f"gallery: {len(entries)} thumbs")
