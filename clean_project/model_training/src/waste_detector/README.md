# waste_detector/

Main Python package for waste material detection. Organised into four submodules:

| Submodule | Purpose |
|-----------|---------|
| `training/` | Dataset preparation, TrainConfig dataclass, YoloTrainer, and evaluation metric extraction. |
| `inference/` | GarbageDetector (full pipeline), bin mapper, lightweight predictor, and crop showcase. |
| `data/` | TACO dataset manager and label map utilities. |
| `utils/` | Shared JSON/YAML file I/O helpers used by all other submodules. |

## Typical import patterns

```python
# Full detection pipeline
from waste_detector.inference import GarbageDetector

# Bin lookup only
from waste_detector.inference import load_bin_mapping, resolve_bin

# Training
from waste_detector.training.config import TrainConfig
from waste_detector.training.trainer import YoloTrainer
from waste_detector.training.dataset import load_dataset_spec
```
