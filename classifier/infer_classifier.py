import sys
import os
from pathlib import Path

import torch
from PIL import Image, ImageOps
from torch import nn
from torchvision import models


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.transforms import get_classifier_eval_transforms

MODEL_PATH = PROJECT_ROOT / "classifier" / "model" / "classifier_3class.pth"

_MODEL = None
_IDX_TO_CLASS = None
_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _build_model() -> nn.Module:
    # Use an uninitialized backbone for inference to avoid any network weight download.
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 3)
    return model


def _load_model() -> tuple[nn.Module, dict[int, str]]:
    global _MODEL, _IDX_TO_CLASS

    if _MODEL is not None and _IDX_TO_CLASS is not None:
        return _MODEL, _IDX_TO_CLASS

    model_path = Path(os.getenv("CLASSIFIER_MODEL_PATH", str(MODEL_PATH)))
    if not model_path.exists():
        raise FileNotFoundError(
            f"Classifier weights not found: {model_path}. "
            "Set CLASSIFIER_MODEL_PATH or include classifier/model/classifier_3class.pth in deployment."
        )

    checkpoint = torch.load(model_path, map_location=_DEVICE)

    model = _build_model().to(_DEVICE)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        class_to_idx = checkpoint.get("class_to_idx", {})
    else:
        # Supports loading a plain state_dict if the user saves one manually.
        model.load_state_dict(checkpoint)
        class_to_idx = {"head_shots": 0, "mouth_shots": 1, "other": 2}

    idx_to_class = {idx: class_name for class_name, idx in class_to_idx.items()}
    model.eval()

    _MODEL = model
    _IDX_TO_CLASS = idx_to_class
    return _MODEL, _IDX_TO_CLASS


def classify_image(path: str | Path) -> str:
    """Classify an image as head_shots, mouth_shots, or other."""
    image_path = Path(path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image does not exist: {image_path}")

    model, idx_to_class = _load_model()
    transform = get_classifier_eval_transforms()

    image = Image.open(image_path).convert("RGB")
    image = ImageOps.exif_transpose(image)
    tensor = transform(image).unsqueeze(0).to(_DEVICE)

    with torch.no_grad():
        logits = model(tensor)
        prediction_idx = int(logits.argmax(dim=1).item())

    if prediction_idx not in idx_to_class:
        raise ValueError(f"Predicted class index {prediction_idx} not found in class mapping.")

    return idx_to_class[prediction_idx]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run inference with the trained 3-class classifier.")
    parser.add_argument("image", type=str, help="Path to input image")
    args = parser.parse_args()

    predicted = classify_image(args.image)
    print(predicted)
