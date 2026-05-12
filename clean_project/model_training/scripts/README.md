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

## How to run

```bash
# From the model_training/ root:
uv run python scripts/run_train.py
uv run python scripts/run_evaluate.py
uv run python scripts/run_predict.py
uv run python scripts/run_crop_showcase.py
uv run python scripts/run_prepare_taco.py
```
