from pathlib import Path
from typing import Sequence

from PIL import Image


Box = Sequence[float]


def crop_with_box(image: Image.Image, box: Box) -> Image.Image:
    """Crop a PIL image with a bounding box [x1, y1, x2, y2]."""
    if box is None or len(box) != 4:
        raise ValueError("Box must be a 4-element sequence: [x1, y1, x2, y2].")

    width, height = image.size
    x1, y1, x2, y2 = [int(round(v)) for v in box]

    x1 = max(0, min(x1, width - 1))
    y1 = max(0, min(y1, height - 1))
    x2 = max(1, min(x2, width))
    y2 = max(1, min(y2, height))

    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"Invalid box after clamping: {[x1, y1, x2, y2]}")

    return image.crop((x1, y1, x2, y2))


def save_crop(image: Image.Image, box: Box, output_path: Path | str) -> Path:
    """Crop the image with the provided box and save it to output_path."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    cropped = crop_with_box(image, box)
    cropped.save(output)
    return output
