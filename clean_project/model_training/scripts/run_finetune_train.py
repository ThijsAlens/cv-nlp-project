"""
Thin wrapper that runs 'run_train.main()' against the fine-tune YAML.

Keeps the canonical 'scripts/run_train.py' + 'config/train_config.yaml'
untouched while still letting the fine-tune workflow live alongside it.

Usage:
  uv run python scripts/run_finetune_train.py
Prerequisite:
  uv run python scripts/run_split_finetune_dataset.py   (one-time, creates
  'data/FINALE_TESTSET_finetune_split/' with the 100/20/80 split.)
"""

import sys
from pathlib import Path

# Add 'src' to the import path so 'waste_detector' can be imported by the
# delegated main(). Also add the 'scripts' folder so 'run_train' resolves.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT / "src"))
sys.path.insert(0, str(_PROJECT_ROOT / "scripts"))

from run_train import main as run_train_main  # noqa: E402

# Path to the fine-tune-specific YAML. Edit that file to change fine-tune
# hyperparameters; this script itself should stay a one-liner.
FINETUNE_CONFIG_PATH = _PROJECT_ROOT / "config" / "finetune_train_config.yaml"


if __name__ == "__main__":
    run_train_main(FINETUNE_CONFIG_PATH)
