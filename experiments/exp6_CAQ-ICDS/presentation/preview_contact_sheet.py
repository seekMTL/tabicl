"""Assemble rendered PDF slide thumbnails for visual quality checks."""
from pathlib import Path
from PIL import Image, ImageDraw

root = Path(__file__).resolve().parent / "preview"
pages = sorted(root.glob("slide-*.png"))
assert len(pages) == 16
width, height = 590, 352
canvas = Image.new("RGB", (4*width, 4*height), "#dae3e9")
draw = ImageDraw.Draw(canvas)
for i, path in enumerate(pages):
    picture = Image.open(path).convert("RGB")
    picture.thumbnail((width-14, height-27))
    x, y = (i % 4)*width+7, (i//4)*height+22
    canvas.paste(picture, (x,y))
    draw.text((x,y-17), f"Slide {i+1:02d}", fill="#183650")
canvas.save(root / "contact-sheet.png")
print(root / "contact-sheet.png")
