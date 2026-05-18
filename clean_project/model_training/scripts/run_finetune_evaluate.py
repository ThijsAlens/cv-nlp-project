"""
Thin wrapper that runs 'run_evaluate.main()' against the fine-tune
evaluation YAML, which targets the 80 held-out real-world images.

Usage:
  uv run python scripts/run_finetune_evaluate.py
Prerequisite:
  uv run python scripts/run_finetune_train.py
"""

import sys
from pathlib import Path

# Add 'src' (for 'waste_detector') and 'scripts' (for 'run_evaluate') to the
# import path so the delegated main() can resolve its dependencies.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT / "src"))
sys.path.insert(0, str(_PROJECT_ROOT / "scripts"))

from run_evaluate import main as run_evaluate_main  # noqa: E402

# Path to the fine-tune evaluation YAML. Edit that file to change which
# checkpoint or split is evaluated.
FINETUNE_EVAL_CONFIG_PATH = _PROJECT_ROOT / "config" / "finetune_evaluate_config.yaml"


if __name__ == "__main__":
    run_evaluate_main(FINETUNE_EVAL_CONFIG_PATH)
