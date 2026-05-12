# models/

Place trained model bundles here. Each bundle is a folder produced by a
training run (from `clean_project/model_training/`).

## Expected structure

```
models/
  <bundle_name>/
    weights/
      best.pt     <- Required: the trained YOLO weights used by the demo
      last.pt     <- Optional: last epoch checkpoint
    args.yaml     <- Optional: training hyperparameters (for reference)
    run_summary.json  <- Optional: training metadata
    ...
```

## How to add a new model

1. Run a training job in `clean_project/model_training/`.
   The output will appear in `model_training/runs/train/<run_name>/`.

2. Copy the entire run folder into this `models/` directory:
   ```
   cp -r model_training/runs/train/yolov8n_garbage_v1  demo/models/
   ```
   Or on Windows:
   ```powershell
   Copy-Item -Recurse model_training\runs\train\yolov8n_garbage_v1 demo\models\
   ```

3. Update `vision.model_bundle` in `config/demo_config.yaml` to the new folder name.

## Current model

Set in `config/demo_config.yaml` under `vision.model_bundle`.
The demo loads `models/<model_bundle>/weights/best.pt` at startup.
