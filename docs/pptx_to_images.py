"""Convert a .pptx to per-slide JPEGs via the installed MS PowerPoint COM API.

Saves each slide as `slide-N.jpg` in the same directory as the .pptx.
"""
import os
import sys
import glob
import time
import pathlib
import win32com.client

PPTX = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "presentation.pptx").resolve()
OUT_DIR = PPTX.parent
PNG_TMP = OUT_DIR / "_slides_png"

# Clean stale slide-*.jpg
for f in glob.glob(str(OUT_DIR / "slide-*.jpg")):
    os.remove(f)
PNG_TMP.mkdir(exist_ok=True)
for f in glob.glob(str(PNG_TMP / "*")):
    os.remove(f)

print(f"Opening PowerPoint to convert {PPTX} ...")
ppt = win32com.client.Dispatch("PowerPoint.Application")
# PowerPoint requires the app to be at least shown for some COM ops on some
# builds; the WindowState=2 setting minimizes the window.
try:
    ppt.WindowState = 2  # ppWindowMinimized
except Exception:
    pass

pres = ppt.Presentations.Open(str(PPTX), WithWindow=False)
try:
    # Save as PNG sequence into a temp dir
    pres.SaveAs(str(PNG_TMP / "slide"), 18)  # ppSaveAsPNG = 18
finally:
    pres.Close()
    ppt.Quit()
    time.sleep(0.5)

# The above creates either a folder full of PNGs or one PNG per slide directly.
# PowerPoint saves into PNG_TMP/slide/slide1.PNG etc when given a folder name.
# Locate the actual PNGs.
# Use set to dedupe (filesystem is case-insensitive on Windows so .PNG and .png glob hit same files).
candidates = sorted(set(p.resolve() for p in PNG_TMP.glob("**/*.PNG")),
                    key=lambda p: int(''.join(c for c in p.stem if c.isdigit()) or 0))
print(f"Found {len(candidates)} PNG slides at {PNG_TMP}")

if not candidates:
    print("ERROR: PowerPoint did not produce PNGs. Inspect", PNG_TMP)
    sys.exit(2)

# Re-emit as slide-N.jpg next to the pptx (Pillow conversion).
from PIL import Image
for i, src in enumerate(candidates, 1):
    dst = OUT_DIR / f"slide-{i}.jpg"
    img = Image.open(src).convert("RGB")
    # Resize to ~1600 px wide for QA
    target_w = 1600
    ratio = target_w / img.width
    img = img.resize((target_w, int(img.height * ratio)), Image.LANCZOS)
    img.save(dst, "JPEG", quality=85)
    print(f"  wrote {dst.name}")

print("Done.")
