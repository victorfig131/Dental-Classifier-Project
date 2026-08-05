from pathlib import Path

import torch
from facenet_pytorch import MTCNN
from PIL import Image, ImageOps


_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_DETECTOR = MTCNN(keep_all=False, device=_DEVICE)


def detect_face(path: str | Path) -> list[float] | None:
    """Detect the primary face and return [x1, y1, x2, y2]."""
    image_path = Path(path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image does not exist: {image_path}")

    image = Image.open(image_path).convert("RGB")
    image = ImageOps.exif_transpose(image)
    boxes, _ = _DETECTOR.detect(image)

    if boxes is None or len(boxes) == 0:
        return None

    x1, y1, x2, y2 = boxes[0].tolist()
    return [float(x1), float(y1), float(x2), float(y2)]


def detect_face_with_prob(path: str | Path) -> tuple[list[float] | None, float | None]:
    """Detect the primary face and return ([x1, y1, x2, y2], prob)."""
    image_path = Path(path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image does not exist: {image_path}")

    image = Image.open(image_path).convert("RGB")
    image = ImageOps.exif_transpose(image)
    boxes, probs = _DETECTOR.detect(image)

    if boxes is None or len(boxes) == 0:
        return None, None

    x1, y1, x2, y2 = boxes[0].tolist()
    prob = float(probs[0]) if probs is not None else None
    return [float(x1), float(y1), float(x2), float(y2)], prob


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run pretrained face detection on a single image.")
    parser.add_argument("image", type=str, help="Path to input image")
    args = parser.parse_args()

    box = detect_face(args.image)
    print(box)
