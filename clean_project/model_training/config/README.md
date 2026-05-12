# config/

User-editable configuration files. Edit these before running any script.
No Python code needs to change when you want to adjust settings.

## Files

| File | Purpose |
|------|---------|
| `train_config.yaml` | All training settings: dataset path, model, hyperparameters, evaluation, and export options. |
| `evaluate_config.yaml` | Checkpoint path and evaluation settings for `run_evaluate.py`. |
| `inference_config.yaml` | Checkpoint, target image, and output settings for `run_predict.py` and `run_crop_showcase.py`. |
| `bin_mapping.json` | Maps each detected material class to its disposal bin. Edit to add new materials or change bin assignments. |

## bin_mapping.json schema

```json
{
  "default_bin": "Rest",
  "material_to_bin": {
    "MaterialName": "BinKey"
  }
}
```

- `default_bin`: bin used when a detected material has no entry in `material_to_bin`.
- `material_to_bin`: maps model class names (case-sensitive) to bin keys.
- `bin_descriptions`: optional human-readable descriptions for each bin (not used by code).
