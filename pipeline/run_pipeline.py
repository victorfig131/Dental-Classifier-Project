import argparse
import shutil
import sys
from pathlib import Path

from PIL import Image, ImageOps


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from classifier.infer_classifier import classify_image
from detector.face_detector.infer_face import detect_face, detect_face_with_prob
from detector.mouth_detector.infer_yolo import detect_mouth
from utils.image_ops import save_crop

DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "pipeline" / "outputs"
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def _copy_original_to_output(image_path: Path, output_dir: str | Path, reason: str) -> Path:
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    out_file = output_root / image_path.name

    # Avoid attempting to copy file onto itself.
    if image_path.resolve() == out_file.resolve():
        print(f"{image_path.name}: {reason} -> already in output ({out_file})")
        return out_file

    shutil.copy2(image_path, out_file)
    print(f"{image_path.name}: {reason} -> original copied to {out_file}")
    return out_file


def process_image(path: str | Path, output_dir: str | Path = DEFAULT_OUTPUT_DIR) -> Path:
    """
    Classify an image and apply class-specific cropping.

    - head_shots: face detector crop
    - mouth_shots: YOLO mouth detector crop
    - other: return original image path unchanged
    """
    image_path = Path(path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image does not exist: {image_path}")

    kind = classify_image(image_path)

    # If the classifier didn't pick head_shots, verify with the face detector.
    # A confident face detection overrides the classifier (handles diverse head shot appearances).
    if kind != "head_shots":
        face_box, face_prob = detect_face_with_prob(image_path)
        if face_box is not None and (face_prob is None or face_prob >= 0.90):
            prob_str = f"{face_prob:.3f}" if face_prob is not None else "N/A"
            print(f"{image_path.name}: classifier said '{kind}' but face detected (prob={prob_str}) -> overriding to 'head_shots'")
            kind = "head_shots"

    if kind == "other":
        return _copy_original_to_output(
            image_path=image_path,
            output_dir=output_dir,
            reason="classified as 'other'",
        )

    if kind == "head_shots":
        box = detect_face(image_path)
    elif kind == "mouth_shots":
        box = detect_mouth(image_path)
    else:
        raise ValueError(f"Unknown classifier output: {kind}")

    if box is None:
        return _copy_original_to_output(
            image_path=image_path,
            output_dir=output_dir,
            reason=f"no detection box for {kind}",
        )

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    out_file = output_root / f"{image_path.stem}_{kind}_crop{image_path.suffix.lower()}"
    image = Image.open(image_path).convert("RGB")
    image = ImageOps.exif_transpose(image)
    saved_path = save_crop(image=image, box=box, output_path=out_file)

    print(f"{image_path.name}: classified as {kind} -> crop saved to {saved_path}")
    return saved_path


def _iter_images(input_path: Path):
    if input_path.is_file():
        yield input_path
        return

    if input_path.is_dir():
        for file in sorted(input_path.rglob("*")):
            if file.is_file() and file.suffix.lower() in VALID_EXTENSIONS:
                yield file
        return

    raise FileNotFoundError(f"Input path does not exist: {input_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Classify and crop dental images end-to-end.")
    parser.add_argument("input", type=str, help="Input image file or directory")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)

    for image_path in _iter_images(input_path):
        process_image(image_path, output_dir=output_dir)
