# model_training

YOLO-based waste material detection: train a model on the Totaal dataset,
evaluate it, and run inference with automatic bin routing.

## Quick start

```bash
# Install dependencies (requires Python 3.11-3.12 and uv)
uv sync

# Edit the dataset path and model settings
# -> config/train_config.yaml

# Train
uv run python scripts/run_train.py

# Evaluate a checkpoint
# -> edit config/evaluate_config.yaml first
uv run python scripts/run_evaluate.py

# Run inference on one image and save crop visualisation
# -> edit config/inference_config.yaml first
uv run python scripts/run_predict.py
uv run python scripts/run_crop_showcase.py

# (Optional) Download and prepare the TACO dataset
# -> edit the CONFIG block in scripts/run_prepare_taco.py first
uv run python scripts/run_prepare_taco.py
```

## Project layout

```
config/          User-editable YAML configuration files and bin_mapping.json
scripts/         Runner entry points (one per task)
src/
  waste_detector/
    training/    Dataset handling, TrainConfig, YoloTrainer, evaluator
    inference/   GarbageDetector, bin mapper, predictor, crop showcase
    data/        TACO dataset manager and label map utilities
    utils/       Shared JSON/YAML I/O helpers
```

See each subfolder's README.md for details.

## Configuration files

| File | Used by |
|------|---------|
| `config/train_config.yaml` | `run_train.py` |
| `config/evaluate_config.yaml` | `run_evaluate.py` |
| `config/inference_config.yaml` | `run_predict.py`, `run_crop_showcase.py` |
| `config/bin_mapping.json` | inference scripts and `GarbageDetector` |

## Dataset

Place your dataset folder (e.g. `Totaal_dataset`) anywhere and point
`dataset.path` in `config/train_config.yaml` to it. The folder must
contain a `data.yaml` (or `dataset.yaml`) file and `train/`, `valid/`,
and optionally `test/` image subfolders.

## Classes

The default Totaal dataset has 5 material classes that map to Belgian household bins:

| Material | Bin |
|----------|-----|
| Cardboard | Paper |
| Glass | Rest |
| Metal | PMD |
| Paper | Paper |
| Plastic | PMD |

To change the bin routing, edit `config/bin_mapping.json`.
