# How this project works

This document explains the **end-to-end flow** for **material-class detection** on **`data/Totaal_dataset`**: dataset YAML handling, **Ultralytics** fine-tuning, evaluation, inference, and **material-to-bin** routing via **`data/bin_mapping.json`**.

The runnable package is **`trash-detector`** (`pyproject.toml`). Application code lives under **`src/trash_detector/`**; primary runners live under **`scripts/`**.

**TACO-only** download and preparation scripts were moved to **`legacy/cv_training_archive/`** (see the README there). They still import **`trash_detector`** from the repository **`src/`** tree.

---

## What the project does

- **Detection** on waste materials: bounding boxes plus material class names from your dataset YAML.
- **Stable training YAML**: `load_dataset_spec()` writes **`dataset.ultralytics.yaml`** with an absolute `path` (`src/trash_detector/training/yolo_data.py`).
- **Bin routing**: `load_bin_mapping_payload` / `resolve_bin_for_material` in `src/trash_detector/inference/bin_mapping.py` map class strings to bin keys.

---

## Repository layout (conceptual)

| Area | Role |
|------|------|
| `data/Totaal_dataset/` | Images, YOLO-format labels, `data.yaml`, generated `dataset.ultralytics.yaml` |
| `data/bin_mapping.json` | `material_to_bin`, `default_bin`, `bin_descriptions` |
| `src/trash_detector/training/` | `TrainConfig`, `YoloTrainer`, `yolo_data`, `eval_report` |
| `src/trash_detector/inference/` | `TrashPredictor`, `crop_showcase`, `bin_mapping` |
| `src/trash_detector/data/` | `TacoDatasetManager` (used only by legacy TACO scripts) |
| `scripts/` | `run_train.py`, `train.py`, `evaluate.py`, `predict.py`, `inference_crop_showcase.py` |
| `legacy/cv_training_archive/` | TACO scripts, label-map configs, optional artifacts |

---

## Step 1: Dataset on disk

`data/Totaal_dataset/` should follow the layout expected by **`find_dataset_yaml`** / **`load_dataset_spec()`**: `data.yaml` or `dataset.yaml`, `train`/`val` image folders (for example `train/images`, `valid/images`), and label `.txt` files under parallel `labels/` trees.

---

## Step 2: Training and evaluation

**`scripts/run_train.py`** builds **`TrainConfig`**, calls **`YoloTrainer.train()`**, and can run post-training **`evaluate_checkpoint()`** on `val` / `test`.

**`scripts/evaluate.py`** evaluates an existing checkpoint against a dataset YAML.

---

## Step 3: Inference and bin mapping

**`TrashPredictor`** (`predictor.py`) runs **`model.predict()`** and returns dicts with `class_name`, confidence, and boxes.

**`scripts/inference_crop_showcase.py`** combines detections with **`bin_mapping.json`** and optional crop grids under **`runs/inference/`**.

---

## Step 4: Example without a checkpoint

**`main_runner.py`** only loads **`data/bin_mapping.json`** and prints resolved bins (no GPU).

---

## Mental model

```text
data/Totaal_dataset/  ->  dataset.ultralytics.yaml  ->  train / val / test
        |
        v
runs/train/.../best.pt  ->  predict.py / inference_crop_showcase.py
        |
        +->  bin_mapping.json  ->  bin key per material
```

---

## Further reading

- **`README.md`**: commands and install.
- **`legacy/cv_training_archive/README.md`**: TACO-only scripts.
