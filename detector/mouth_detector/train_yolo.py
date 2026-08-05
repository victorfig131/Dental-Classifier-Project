import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_MOUTH_DIR = PROJECT_ROOT / "dataset" / "mouth_shots"
DATASET_YAML = DATASET_MOUTH_DIR / "data.yaml"
DATASET_YAML_TXT = DATASET_MOUTH_DIR / "data.yaml.txt"
LOCAL_YAML = PROJECT_ROOT / "detector" / "mouth_detector" / "data.yaml"
WEIGHTS_DIR = PROJECT_ROOT / "detector" / "mouth_detector" / "weights"
BEST_WEIGHT_OUT = WEIGHTS_DIR / "best.pt"


def ensure_dataset_yaml() -> Path:
    """Ensure dataset/mouth_shots/data.yaml exists, copying from data.yaml.txt if needed."""
    if DATASET_YAML.exists():
        return DATASET_YAML

    if DATASET_YAML_TXT.exists():
        shutil.copy2(DATASET_YAML_TXT, DATASET_YAML)
        print(f"Created {DATASET_YAML} from {DATASET_YAML_TXT}")
        return DATASET_YAML

    raise FileNotFoundError("Could not find data.yaml or data.yaml.txt under dataset/mouth_shots")


def sync_detector_yaml(dataset_yaml: Path) -> Path:
    """Write detector/mouth_detector/data.yaml with an absolute dataset path."""
    with dataset_yaml.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    data["path"] = str(DATASET_MOUTH_DIR.resolve())

    LOCAL_YAML.parent.mkdir(parents=True, exist_ok=True)
    with LOCAL_YAML.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)

    return LOCAL_YAML


def train_yolo(epochs: int = 50, imgsz: int = 640, batch_size: int = 16) -> Path:
    """Train YOLOv8 mouth detector and place best.pt under detector/mouth_detector/weights/."""
    dataset_yaml = ensure_dataset_yaml()
    resolved_yaml = sync_detector_yaml(dataset_yaml)

    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

    model = YOLO("yolov8n.pt")
    results = model.train(
        data=str(resolved_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch_size,
        project=str(WEIGHTS_DIR / "runs"),
        name="mouth_yolo",
        exist_ok=True,
    )

    best_src = Path(results.save_dir) / "weights" / "best.pt"
    if not best_src.exists():
        raise FileNotFoundError(f"Training completed but best.pt was not found at: {best_src}")

    shutil.copy2(best_src, BEST_WEIGHT_OUT)
    print(f"Copied trained weights to: {BEST_WEIGHT_OUT}")
    return BEST_WEIGHT_OUT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLOv8 on mouth_shots annotations.")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch-size", type=int, default=16)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_yolo(epochs=args.epochs, imgsz=args.imgsz, batch_size=args.batch_size)
