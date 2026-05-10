"""
Generate sample MNIST-style digit images for testing.

Creates simple digit images with white digits on black background.
"""

from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
import sys


def generate_digit_image(digit: int, output_path: Path, size: int = 280) -> None:
    """
    Generate a simple digit image for MNIST testing.

    Args:
        digit: Digit to draw (0-9)
        output_path: Path to save the image
        size: Image size (will be downscaled to 28x28 for inference)
    """
    # Create black background
    img = Image.new('L', (size, size), color=0)
    draw = ImageDraw.Draw(img)

    # Try to use a nice font, fallback to default
    try:
        # Try system fonts
        if sys.platform == 'win32':
            font = ImageFont.truetype('arial.ttf', size // 2)
        elif sys.platform == 'darwin':
            font = ImageFont.truetype('/System/Library/Fonts/Helvetica.ttc', size // 2)
        else:
            font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', size // 2)
    except:
        # Fallback to default font
        font = ImageFont.load_default()
        print(f"Warning: Using default font (may look pixelated)")

    # Draw white digit centered
    text = str(digit)

    # Get text size and center it
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]

    x = (size - text_width) // 2
    y = (size - text_height) // 2 - 10  # Slight offset for better centering

    # Draw the digit in white
    draw.text((x, y), text, fill=255, font=font)

    # Save image
    img.save(output_path)
    print(f"Generated: {output_path}")


def main():
    """Generate sample images for all digits 0-9."""
    # Create samples directory
    samples_dir = Path('samples')
    samples_dir.mkdir(exist_ok=True)

    print("Generating MNIST-style sample images...")
    print(f"Output directory: {samples_dir.absolute()}\n")

    # Generate images for digits 0-9
    for digit in range(10):
        output_file = samples_dir / f'digit_{digit}.png'
        generate_digit_image(digit, output_file)

    print(f"\n✓ Successfully generated {10} sample images")
    print(f"\nTo test locally, run:")
    print(f"  python local_test.py")


if __name__ == '__main__':
    main()
