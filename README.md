# Dental Image Classification + Detection Pipeline

I built this project to classify dental images into three groups and then apply the right follow-up crop for each one:

- `head_shots`
- `mouth_shots`
- `other`

The pipeline works like this:

- `head_shots` -> cropped with a pretrained face detector (`MTCNN`)
- `mouth_shots` -> cropped with a YOLOv8 mouth detector
- `other` -> returned unchanged

## What is included in this repo

The repository contains the code, pipeline, web demo, and the trained model files needed to run inference.

The learned model weights are stored here:

- `classifier/model/classifier_3class.pth`
- `detector/mouth_detector/weights/best.pt`

As long as those files are present, the project will use the trained models already included in the repo.

## Dataset layout used during development

The training code expects this folder structure:

- `dataset/head_shots/`
- `dataset/mouth_shots/`
- `dataset/other/`

The YOLO mouth detector uses:

- `dataset/mouth_shots/images/`
- `dataset/mouth_shots/labels/`
- `dataset/mouth_shots/data.yaml`

Private datasets and test images are intentionally not meant to be committed.

## Setup

Recommended Conda setup:

```bash
conda env create -f environment.yml
conda activate dental311
```

If you prefer `pip`:

```bash
pip install -r requirements.txt
```

## Training

Train the image classifier:

```bash
python classifier/train_classifier.py --epochs 10
```

This writes the classifier weights to `classifier/model/classifier_3class.pth`.

Train the YOLO mouth detector:

```bash
python detector/mouth_detector/train_yolo.py --epochs 50
```

This writes the best detector weights to `detector/mouth_detector/weights/best.pt`.

## Running the pipeline

Run a single image:

```bash
python pipeline/run_pipeline.py path/to/image.jpg
```

Run a whole directory:

```bash
python pipeline/run_pipeline.py path/to/folder --output-dir pipeline/outputs
```

## Frontend demo

This repo also includes a simple browser UI:

- Frontend: `public/index.html`
- Backend: `api/index.py`

Uploads are handled temporarily for processing and are not persisted by the app itself.

Available modes:

- `auto`: full pipeline behavior
- `classify_only`: return only the predicted class
- `force_face`: force the face crop path
- `force_mouth`: force the mouth crop path
- `original`: return the uploaded image unchanged

Run it locally with:

```bash
pip install -r requirements.txt # or use the conda environment
python api/index.py
```

Then open `http://localhost:8000`.

## Notes for anyone using this repository

- The code and deploy-required trained weights are meant to stay in the repo.
- Private training data, personal test images, and generated local outputs are not meant to be committed.
- The repo includes a `.gitignore` for local datasets, test media, detector training artifacts, and pipeline output folders.

## Deployment notes

By default, inference loads these model files:

- `classifier/model/classifier_3class.pth`
- `detector/mouth_detector/weights/best.pt`

If you need different model locations in a hosted environment, set:

- `CLASSIFIER_MODEL_PATH`
- `MOUTH_DETECTOR_WEIGHTS_PATH`

`vercel.json` is included for static frontend plus Python API routing.

One practical caveat: this stack uses Python, Torch, YOLO, and MTCNN, so it can be heavy for serverless limits on platforms like Vercel. It is fine for a lightweight demo, but for more reliable inference workloads I would use a more ML-friendly host such as Render, Railway, Fly.io, Hugging Face Spaces, or a container-based VM service.
