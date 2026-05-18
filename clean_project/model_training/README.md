# model_training

YOLO-based waste material detection: train a model on the Totaal dataset,
evaluate it, and run inference with automatic bin routing.

## Quick start

```bash
# Install dependencies (requires Python 3.11-3.12 and uv)
uv lock --upgrade
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

# (Optional) Build a class-merged sibling dataset (e.g. Cardboard merged into Paper)
# -> edit the CONFIG block in scripts/run_merge_classes.py first
uv run python scripts/run_merge_classes.py

# (Optional) Fine-tune the trained model on real-world images
# -> edit constants at the top of scripts/run_split_finetune_dataset.py if needed
uv run python scripts/run_split_finetune_dataset.py
uv run python scripts/run_finetune_train.py
uv run python scripts/run_finetune_evaluate.py
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
| `config/finetune_train_config.yaml` | `run_finetune_train.py` |
| `config/finetune_evaluate_config.yaml` | `run_finetune_evaluate.py` |
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

## Class merging

`scripts/run_merge_classes.py` builds a sibling dataset folder (default:
`data/Totaal_dataset_merged`) where some YOLO classes are merged together.
The default merge collapses `Cardboard` into `Paper`, leaving 4 classes:
`Glass`, `Metal`, `Plastic`, `Paper`. Image files are not duplicated: each
image in the new dataset is a hardlink (`os.link`) to the corresponding
source image, and only the YOLO `.txt` label files are rewritten with
remapped class ids. The source and target dataset folders must live on
the same filesystem/drive (hardlinks cannot cross volumes).

After running the merge script, `config/train_config.yaml` and
`config/evaluate_config.yaml` already point at the merged dataset, so the
normal training and evaluation commands pick it up without further edits.

## Fine-tuning on real-world images

`scripts/run_split_finetune_dataset.py` builds a `train/val/test` split
under `data/FINALE_TESTSET_finetune_split/` from the 200 real-world
images in `data/FINALE_TESTSET_merged/`. The split is seeded and
recorded in `split_manifest.json` so the held-out test images can be
audited later.

`scripts/run_finetune_train.py` then fine-tunes the trained checkpoint
on the 100 training images (config in `config/finetune_train_config.yaml`).
The fine-tune config uses heavier backbone freezing, fewer epochs and
gentler augmentation than the from-scratch training config, so the
checkpoint can shift toward real-world performance without overfitting
the tiny set or wiping out what was learned from the Totaal dataset.

`scripts/run_finetune_evaluate.py` evaluates the fine-tuned checkpoint
on the 80 held-out images recorded in `split_manifest.json -> files.test`,
writing the results to `runs/evaluate/finetune_real_results.json`.
