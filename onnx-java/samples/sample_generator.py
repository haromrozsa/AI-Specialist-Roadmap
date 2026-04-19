"""
Generates a sample MNIST-style digit image for the ONNX Java demo.
Produces a 28x28 PNG with a white digit on black background.
"""
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path

OUTPUT = Path(__file__).parent.parent / "" / "digit.png"
OUTPUT.parent.mkdir(parents=True, exist_ok=True)

# Create a 280x280 image (scaled up for anti-aliasing), then shrink to 28x28
size_big = 280
image = Image.new("L", (size_big, size_big), color=0)  # black background
draw = ImageDraw.Draw(image)

digit = "8"  # change this to test other digits: "0", "1", ..., "9"

# Use a large default font; fall back gracefully if specific fonts are unavailable
try:
    font = ImageFont.truetype("arial.ttf", 220)
except IOError:
    font = ImageFont.load_default()

# Center the digit
bbox = draw.textbbox((0, 0), digit, font=font)
w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
x = (size_big - w) // 2 - bbox[0]
y = (size_big - h) // 2 - bbox[1]
draw.text((x, y), digit, fill=255, font=font)  # white digit

# Downscale to 28x28 with smooth resampling
final = image.resize((28, 28), Image.LANCZOS)
final.save(OUTPUT)

print(f"Saved: {OUTPUT}")