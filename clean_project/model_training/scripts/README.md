# scripts/

Runner entry points. Each script does one task and reads its settings
from a YAML file in `config/`. None of them use command-line arguments;
edit the config YAML instead.

## Scripts

| Script | Config file | What it does |
|--------|-------------|--------------|
| `run_train.py` | `train_config.yaml` | Train a YOLO model from scratch or fine-tune a pretrained one. Optionally evaluates and exports to ONNX afterwards. |
| `run_evaluate.py` | `evaluate_config.yaml` | Evaluate a trained checkpoint on val/test splits. Saves a JSON report with precision, recall, F1, and mAP scores. |
| `run_predict.py` | `inference_config.yaml` | Run detection on a single image. Saves crops and a detections.json to a timestamped output folder. |
| `run_crop_showcase.py` | `inference_config.yaml` | Same as `run_predict.py` but also saves a PNG grid showing each detected crop labelled with its bin. |
| `run_prepare_taco.py` | (inline CONFIG block) | Download and convert the TACO litter dataset to YOLO format. Only needed if using TACO data. |
| `run_merge_classes.py` | (inline CONFIG block) | Build a class-merged sibling copy of an existing YOLO dataset. Images are linked (not copied); only the label `.txt` files and a new `data.yaml` are written. |
| `run_split_finetune_dataset.py` | (inline CONFIG block) | Split `FINALE_TESTSET_merged` (200 real-world images) into a `train/val/test` layout for fine-tuning. Writes `split_manifest.json` recording exactly which images landed in each split so the held-out test set can be audited later. |
| `run_finetune_train.py` | `finetune_train_config.yaml` | Thin wrapper around `run_train.py` that points at the fine-tune YAML. Fine-tunes the trained checkpoint on the split produced by `run_split_finetune_dataset.py`, with hyperparameters tuned for the small (100-image) set. |
| `run_finetune_evaluate.py` | `finetune_evaluate_config.yaml` | Thin wrapper around `run_evaluate.py` that evaluates the fine-tuned checkpoint on the 80 held-out real-world images. |

## How to run

```bash
# From the model_training/ root:
uv run python scripts/run_train.py
uv run python scripts/run_evaluate.py
uv run python scripts/run_predict.py
uv run python scripts/run_crop_showcase.py
uv run python scripts/run_prepare_taco.py
uv run python scripts/run_merge_classes.py
uv run python scripts/run_split_finetune_dataset.py
uv run python scripts/run_finetune_train.py
uv run python scripts/run_finetune_evaluate.py
```

## Fine-tune workflow

To improve the trained model on real-world data while reserving an unseen
test set, run these in order:

```bash
# 1. One-time: split 'data/FINALE_TESTSET_merged/test/' (200 images)
#    into a 100/20/80 train/val/test layout under
#    'data/FINALE_TESTSET_finetune_split/'. Also writes
#    'split_manifest.json' recording which filenames went to each split.
uv run python scripts/run_split_finetune_dataset.py

# 2. Fine-tune the trained checkpoint on the 100 training images.
#    Hyperparameters live in 'config/finetune_train_config.yaml' (heavier
#    backbone freezing, fewer epochs, gentler augmentation; tuned to avoid
#    overfitting and catastrophic forgetting on the tiny set).
uv run python scripts/run_finetune_train.py

# 3. Evaluate on the 80 truly-unseen images (the ones in
#    'split_manifest.json -> files.test').
uv run python scripts/run_finetune_evaluate.py
```
