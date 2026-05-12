# training/

Handles everything needed to go from a dataset folder to a trained YOLO checkpoint.

## Modules

| Module | Purpose |
|--------|---------|
| `config.py` | `TrainConfig` dataclass - holds all training hyperparameters in one place. |
| `dataset.py` | Discovers the source dataset YAML, generates a canonical Ultralytics training YAML with absolute paths, and returns a `YoloDatasetSpec`. |
| `trainer.py` | `YoloTrainer` - loads the YOLO model, builds training kwargs, optionally applies class balancing, and runs `model.train()`. |
| `evaluator.py` | `evaluate_checkpoint()` and `save_evaluation_report()` - run Ultralytics validation and extract metrics (precision, recall, F1, mAP). |

## Data flow

```
train_config.yaml
  -> load_dataset_spec()  (dataset.py)
  -> TrainConfig          (config.py)
  -> YoloTrainer.train()  (trainer.py)
  -> runs/train/<run_name>/weights/best.pt
  -> evaluate_checkpoint() (evaluator.py)
  -> evaluation_metrics.json
```

## Class balancing

Set `balanced_training: true` in `train_config.yaml` to enable inverse-frequency
class weighting via the `cls_pw` parameter. This requires Ultralytics >= 8.4.40.
`balanced_cls_pw` controls the weighting strength (0.25 = mild, 1.0 = full).
