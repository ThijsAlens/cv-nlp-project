# Trash detection with YOLO + TACO

This project builds a complete **object detection** pipeline for litter/trash recognition using the **TACO** dataset and a pretrained **Ultralytics YOLO** model.

The default setup is intentionally biased toward a **working pipeline on modest hardware** rather than squeezing out the last bit of accuracy. The focused default task is bottle-and-cap detection, because TACO already contains separate labels such as:

- `Clear plastic bottle`
- `Other plastic bottle`
- `Glass bottle`
- `Plastic bottle cap`
- `Metal bottle cap`

That makes it possible to train a detector that can later support rule-based instructions such as *remove the cap before recycling the bottle*.

## Project structure

```text
trash_detection_yolo_taco/
├── configs/
│   └── label_maps/
│       └── bottle_focus.json     # Focused class subset
├── data/
│   ├── raw/                      # Downloaded TACO assets
│   └── prepared/                 # YOLO-formatted datasets
├── assets/
│   └── taco_category_summary.json
├── runs/
│   └── train/                    # Ultralytics training outputs
├── scripts/
│   ├── download_assets.py        # Download annotations, images, and warm up weights
│   ├── prepare_taco.py           # Convert COCO TACO to YOLO format
│   ├── train.py                  # Fine-tune YOLO on prepared data
│   ├── predict.py                # Run inference
│   └── full_pipeline.py          # One-command end-to-end pipeline
└── src/trash_detector/
    ├── data/
    ├── inference/
    ├── training/
    └── utils/
```

## Why this setup

A few design choices are deliberate:

1. **Detection, not classification**  
   The TACO task and the bottle-cap use case are spatial: you need object locations, not just one label for the whole image.

2. **YOLOv8n by default**  
   On an RTX 4060 mobile with 8 GB VRAM, `yolov8n.pt` is the safest default for a full working pipeline. Once the pipeline is stable, trying `yolov8s.pt` on a T4 is reasonable.

3. **Focused label subset first**  
   TACO has 60 classes, but many are rare. The default configuration keeps only bottle and cap classes, which is usually a better first milestone than training all 60 classes immediately.

4. **Partial backbone freezing**  
   The default training config freezes the first 10 layers to stabilize early fine-tuning on a relatively small dataset. This is configurable.

5. **Symlinked prepared dataset by default**  
   Preparing the YOLO dataset uses symlinks instead of copying images to avoid doubling disk usage. Use `--copy-images` on Windows if symlinks are inconvenient.

## Recommended environments

### Laptop: RTX 4060 mobile, 8 GB VRAM

Start with:

- model: `yolov8n.pt`
- image size: `640`
- batch size: `16`
- epochs: `60`
- mixed precision: enabled
- workers: `4`

If you hit CUDA out-of-memory, reduce batch size in this order: `16 -> 12 -> 8`.

### School cloud: NVIDIA T4, 16 GB VRAM

Good next step:

- model: `yolov8s.pt`
- image size: `640`
- batch size: `16` or `24`
- epochs: `80`

## Installation

Create a virtual environment and install the package in editable mode.

### Linux / macOS

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```

### Windows PowerShell

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -U pip
pip install -e .
```

## Step 1: download annotations, images, and YOLO weights

This downloads the official TACO annotations JSON, downloads TACO images from the URLs referenced in the annotations, and warms up the YOLO checkpoint cache.

```bash
uv run python scripts/download_assets.py --weights yolov8n.pt
```

For a quick smoke test instead of the full dataset:

```bash
uv run python scripts/download_assets.py --weights yolov8n.pt --max-images 200
```

## Step 2: convert TACO into YOLO format

This converts the COCO-format TACO annotations into YOLO text labels and creates train/val/test splits.

```bash
uv run python scripts/prepare_taco.py
```

The default output dataset will be created at:

```text
data/prepared/taco_bottle_focus/
```

Its YOLO dataset file is:

```text
data/prepared/taco_bottle_focus/dataset.yaml
```

## Step 3: train the detector

```bash
uv run python scripts/train.py `
  --data-yaml data/prepared/taco_bottle_focus/dataset.yaml `
  --model yolov8n.pt `
  --epochs 60 `
  --batch 16 `
  --imgsz 640 `
  --device 0
```

Trained weights will end up under:

```text
runs/train/<run_name>/weights/best.pt
```

## One-command pipeline

If the machine is already set up and you want the end-to-end flow in one command:

```bash
python scripts/full_pipeline.py --model yolov8n.pt --epochs 60 --batch 16 --device 0
```

## Run inference

```bash
python scripts/predict.py runs/train/yolov8n_taco_bottle_focus/weights/best.pt path/to/test_image.jpg --save
```

## Switching to all TACO classes

The project defaults to the focused bottle-cap subset. To train on all classes, modify `scripts/prepare_taco.py` to pass `label_map_path=None`, or create another script that calls `prepare_yolo_dataset(..., label_map_path=None)`.

That is not the recommended first run because many TACO classes are sparse.

## Practical notes

- TACO image hosting depends on external URLs. Some images may fail to download. Failures are logged to `assets/taco_download_failures.json`.
- Very small boxes are dropped by default using `--min-box-pixels 8` because tiny targets tend to create noisy labels for a first-pass detector.
- The dataset split is image-level random with a fixed seed for reproducibility.
- For deployment later, `scripts/train.py --export-onnx` can export the best checkpoint to ONNX.

## Likely next extension

Detection alone will let you identify bottles and caps separately. To produce instructions such as *remove the cap from this bottle first*, the next step is usually a **lightweight relational rule layer** on top of detections, for example:

- detect a `Glass bottle`
- detect a nearby `Plastic bottle cap`
- check whether the cap box overlaps or sits near the bottle neck
- emit a symbolic action: `remove_cap_before_recycling`

That rule layer is intentionally not baked into the training code yet, so the current project remains cleanly scoped to dataset preparation, fine-tuning, and inference.
