import io
import os
import sys
import tempfile
from pathlib import Path

from flask import Flask, jsonify, request, send_file, send_from_directory
from PIL import Image, ImageOps

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
ALLOWED_MODES = {"auto", "classify_only", "force_face", "force_mouth", "original"}

PUBLIC_DIR = PROJECT_ROOT / "public"

app = Flask(__name__)


def _mode_from_request() -> str:
    mode = (request.form.get("mode") or request.args.get("mode") or "auto").strip().lower()
    if mode not in ALLOWED_MODES:
        raise ValueError(f"Unsupported mode: {mode}. Allowed modes: {sorted(ALLOWED_MODES)}")
    return mode


def _save_upload(upload, temp_dir: Path) -> Path:
    filename = (upload.filename or "upload.jpg").strip()
    suffix = Path(filename).suffix.lower() or ".jpg"
    if suffix not in ALLOWED_EXTENSIONS:
        raise ValueError(f"Unsupported file extension: {suffix}")

    input_path = temp_dir / f"input{suffix}"
    upload.save(input_path)
    return input_path


def _send_image(image_path: Path, classification: str, mode: str, note: str):
    image = Image.open(image_path).convert("RGB")
    image = ImageOps.exif_transpose(image)

    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=95)
    buffer.seek(0)

    response = send_file(buffer, mimetype="image/jpeg")
    response.headers["X-Classification"] = classification
    response.headers["X-Mode"] = mode
    response.headers["X-Note"] = note
    return response


@app.get("/api/health")
def health_check():
    return jsonify({"status": "ok"})


@app.get("/")
def home():
    index_path = PUBLIC_DIR / "index.html"
    if index_path.exists():
        return send_from_directory(PUBLIC_DIR, "index.html")

    return jsonify({"error": "Frontend not found. Expected public/index.html"}), 404


@app.post("/api/predict")
def predict():
    upload = request.files.get("image")
    if upload is None:
        return jsonify({"error": "Missing file field: image"}), 400

    try:
        mode = _mode_from_request()
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    from classifier.infer_classifier import classify_image
    from detector.face_detector.infer_face import detect_face
    from detector.mouth_detector.infer_yolo import detect_mouth
    from pipeline.run_pipeline import process_image
    from utils.image_ops import crop_with_box

    with tempfile.TemporaryDirectory(prefix="dental_tmp_") as tmp:
        temp_dir = Path(tmp)

        try:
            input_path = _save_upload(upload, temp_dir)
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400

        try:
            classification = classify_image(input_path)

            if mode == "classify_only":
                return jsonify({
                    "classification": classification,
                    "mode": mode,
                    "note": "classification only; image not transformed",
                })

            if mode == "original":
                return _send_image(input_path, classification, mode, "returned original image")

            if mode == "auto":
                output_path = process_image(input_path, output_dir=temp_dir)
                return _send_image(output_path, classification, mode, "pipeline auto mode")

            if mode == "force_face":
                box = detect_face(input_path)
                if box is None:
                    return _send_image(input_path, classification, mode, "no face detected; returned original")

                image = Image.open(input_path).convert("RGB")
                image = ImageOps.exif_transpose(image)
                cropped = crop_with_box(image, box)
                output_path = temp_dir / "force_face.jpg"
                cropped.save(output_path, format="JPEG", quality=95)
                return _send_image(output_path, classification, mode, "forced face crop")

            box = detect_mouth(input_path)
            if box is None:
                return _send_image(input_path, classification, mode, "no mouth detected; returned original")

            image = Image.open(input_path).convert("RGB")
            image = ImageOps.exif_transpose(image)
            cropped = crop_with_box(image, box)
            output_path = temp_dir / "force_mouth.jpg"
            cropped.save(output_path, format="JPEG", quality=95)
            return _send_image(output_path, classification, mode, "forced mouth crop")
        except Exception as exc:  # noqa: BLE001
            return jsonify({"error": str(exc)}), 500


if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    app.run(host="0.0.0.0", port=port, debug=False)
