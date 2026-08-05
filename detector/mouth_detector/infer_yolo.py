from pathlib import Path
import os

from PIL import Image, ImageOps
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[2]
WEIGHTS_PATH = PROJECT_ROOT / "detector" / "mouth_detector" / "weights" / "best.pt"

_MODEL = None


def _load_model() -> YOLO:
    global _MODEL

    if _MODEL is not None:
        return _MODEL

    weights_path = Path(os.getenv("MOUTH_DETECTOR_WEIGHTS_PATH", str(WEIGHTS_PATH)))

    if not weights_path.exists():
        raise FileNotFoundError(
            f"YOLO weights not found: {weights_path}. "
            "Set MOUTH_DETECTOR_WEIGHTS_PATH or include detector/mouth_detector/weights/best.pt in deployment."
        )

    _MODEL = YOLO(str(weights_path))
    return _MODEL


def detect_mouth(path: str | Path) -> list[float] | None:
    """Detect mouth and return [x1, y1, x2, y2] for the highest-confidence box."""
    image_path = Path(path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image does not exist: {image_path}")

    image = Image.open(image_path).convert("RGB")
    image = ImageOps.exif_transpose(image)

    model = _load_model()
    result = model.predict(source=image, verbose=False)[0]

    if result.boxes is None or len(result.boxes) == 0:
        return None

    confidences = result.boxes.conf.cpu().tolist()
    best_idx = max(range(len(confidences)), key=lambda i: confidences[i])
    best_box = result.boxes.xyxy[best_idx].cpu().tolist()
    return [float(v) for v in best_box]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run YOLOv8 mouth detection on a single image.")
    parser.add_argument("image", type=str, help="Path to input image")
    args = parser.parse_args()

    box = detect_mouth(args.image)
    print(box)
