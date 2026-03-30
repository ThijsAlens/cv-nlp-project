uv run python scripts/predict.py `
  .\runs\train\yolov8n_taco_bottle_focus2\weights\best.pt `
  .\data\inference_examples\example1.jpg `
  --save


uv run python scripts/predict.py `
  .\runs\train\yolo11s_taco_all_classes\weights\best.pt `
  .\data\prepared\taco_bottle_focus\images\test `
  --save



uv run python scripts/prepare_taco.py `
  --output-name taco_all_classes `
  --train-ratio 0.8 `
  --val-ratio 0.1 `
  --test-ratio 0.1 `
  --min-box-pixels 4 `
  --copy-images

uv run python scripts/download_assets.py `
  --weights yolo11s.pt

uv run python scripts/train.py `
  --data-yaml .\data\prepared\taco_all_classes\dataset.yaml `
  --model yolo11s.pt `
  --run-name yolo11s_taco_all_classes `
  --epochs 150 `
  --imgsz 832 `
  --batch 8 `
  --device 0 `
  --workers 4 `
  --freeze 0 `
  --patience 50 `
  --cache

