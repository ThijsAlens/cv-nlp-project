# Material waste detection (`Totaal_dataset` + bin mapping)

This project trains and runs **object detection** on **`data/Totaal_dataset`** (Ultralytics-style layout: splits, images, label `.txt` files, and `data.yaml` / generated `dataset.ultralytics.yaml`). Class names are **material** labels (for example Cardboard, Glass, Metal, Paper, Plastic). **`data/bin_mapping.json`** maps each material string to a **bin key** (for example PMD, Paper, Rest).

The installable package is **`trash-detector`** (`pyproject.toml`). Library code is under **`src/trash_detector/`**; runnable scripts are under **`scripts/`**.

For module-level detail and data flow, see **`HOW_IT_WORKS.md`**.

## Project structure

```text
project/
├── data/
│   ├── Totaal_dataset/           # Images, labels, data.yaml, dataset.ultralytics.yaml
│   └── bin_mapping.json          # material_to_bin for demos and apps
├── scripts/
│   ├── run_train.py              # Primary training (edit CONFIG at top)
│   ├── train.py                  # CLI training (defaults to Totaal paths)
│   ├── evaluate.py               # Evaluate a checkpoint
│   ├── predict.py                # Inference JSON (+ optional --save)
│   └── inference_crop_showcase.py  # Crops + material + bin under runs/inference/
├── legacy/cv_training_archive/   # TACO-only scripts, configs, optional artifacts
├── main_runner.py                # Small demo: load bin_mapping.json only
└── src/trash_detector/
```

## Installation

### Linux / macOS

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
pip install -e .
```

### Windows PowerShell

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -U pip
pip install -e .
```

With **`uv`**:

```bash
uv sync
```

## Dataset: `data/Totaal_dataset`

Keep your **Totaal** export under **`data/Totaal_dataset/`** with `data.yaml` (or `dataset.yaml`), image folders per split, and matching `labels/`. The first run of **`load_dataset_spec()`** (via **`scripts/run_train.py`**) writes **`dataset.ultralytics.yaml`** with an absolute `path` for reliable training.

Align **`data/bin_mapping.json`** with the **`names`** list in that YAML so **`material_to_bin`** keys match detector class strings.

## Train

Edit **`CONFIG`** in **`scripts/run_train.py`**, then:

```bash
uv run python scripts/run_train.py
```

Or use the CLI trainer (defaults point at **`data/Totaal_dataset/dataset.ultralytics.yaml`**):

```bash
uv run python scripts/train.py --data-yaml data/Totaal_dataset/dataset.ultralytics.yaml --model yolo11s.pt
```

Weights land under **`runs/train/<run_name>/weights/best.pt`**.

### Imbalanced class counts (optional)

If some materials appear much more often in the training labels than others, the detector can overfit to frequent classes and underperform on rare ones. Ultralytics supports **inverse-frequency weighting** on the **classification** part of the loss via the training argument **`cls_pw`**.

In this project:

- Set **`balanced_training`: `True`** in the **`CONFIG`** dict in **`scripts/run_train.py`** to enable that behavior (it passes **`cls_pw`** to `YOLO.train()`).
- **`balanced_cls_pw`** controls **how strong** the rebalancing is: **`0.0`** means no extra class weighting (Ultralytics default when you do not enable our flag). Values **above `0`** tell Ultralytics to compute per-class weights from label counts (roughly based on inverse frequency, raised to the power **`balanced_cls_pw`**) so rare classes contribute more to the classification loss. **`0.25`** is a common starting point; increase toward **`1.0`** for stronger upweighting of rare classes if metrics on those classes are still poor.

CLI equivalent:

```bash
uv run python scripts/train.py --data-yaml data/Totaal_dataset/dataset.ultralytics.yaml --balanced-training --balanced-cls-pw 0.25
```

Anything you put in **`extra_train_args`** in **`run_train.py`** (including an explicit **`cls_pw`**) is merged **after** these settings and **overrides** them if the same key appears twice. The effective values are also written to **`run_summary.json`** under each run.

**Requirement:** the **`cls_pw`** training argument exists only in **Ultralytics 8.4.40 and newer**. Use **`uv sync`**, **`pip install -U 'ultralytics>=8.4.40'`**, or match **`pyproject.toml`** so balanced training does not raise a configuration error.

## Evaluate

```bash
uv run python scripts/evaluate.py runs/train/<your_run>/weights/best.pt data/Totaal_dataset/dataset.ultralytics.yaml --device 0
```

## Predict

```bash
uv run python scripts/predict.py runs/train/<your_run>/weights/best.pt path/to/image.jpg --save
```

## Material + bin showcase

Edit **`CONFIG`** in **`scripts/inference_crop_showcase.py`**, then:

```bash
uv run python scripts/inference_crop_showcase.py
```

## Bin mapping only

```bash
uv run python main_runner.py
```

## TACO-only tools

TACO download, prepare, and analysis scripts live under **`legacy/cv_training_archive/`** (see **`legacy/cv_training_archive/README.md`**).

## Further reading

- **`HOW_IT_WORKS.md`**: training wrappers, bin resolution, layout table.
- **`legacy/cv_training_archive/README.md`**: TACO entry points only.
