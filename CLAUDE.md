# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Smart Waste Sorter** — a Computer Vision + NLP system that detects waste items from a webcam, maps them to the correct disposal bin, and provides disposal instructions via a RAG-backed chatbot.

## Repository Structure

The repo has two independently runnable components:

- **`/demo`** — The integrated application (vision + chatbot). This is the primary deliverable.
- **`/testing_projects/Yolo-taco_test_v1`** — The model training/evaluation pipeline used to produce the YOLO weights.
- **`/TACO`** — TACO dataset management scripts (reference/download only).

## Commands

### Demo Application

```bash
cd demo

# First-time setup (creates .venv and installs deps)
./setup.sh          # Linux/macOS
# On Windows: python -m venv .venv && .venv\Scripts\activate && pip install -r requirements.txt

# Run the demo (starts webcam + chatbot)
python demo.py      # Press ESC to stop

# Teardown
./remove.sh
```

### Training Pipeline (testing_projects/Yolo-taco_test_v1)

```bash
cd testing_projects/Yolo-taco_test_v1

uv sync                                        # Install dependencies
uv run python scripts/run_train.py             # Train (edit CONFIG in script first)
uv run python scripts/evaluate.py             # Evaluate on test split
uv run python scripts/predict.py <ckpt> <img> # Single-image inference
uv run python scripts/inference_crop_showcase.py  # Visualize detections + bins
```

## Architecture

### Demo Threading Model

`demo.py` launches two threads that communicate via shared temp files in `demo/temp/`:
- **Vision thread** (`vision/vision.py`): Captures webcam frames, runs YOLO, writes `output.png` + `results.json`
- **NLP thread** (`nlp/nlp.py`): Reads `results.json`, queries the RAG system, streams Ollama responses to the terminal

Both threads share the global config in `demo/config.py`.

### Vision Pipeline

```
Webcam → 640×640 crop → YOLO (best.pt) → labels → results.json + output.png
```

YOLO model weights are at `demo/vision/best.pt` (trained by the testing project pipeline).

### NLP / RAG Pipeline

```
results.json → sorting_rules.json (material→bin) → FAISS index query → Ollama LLM → chat response
```

- `nlp/rag_system.py`: Builds/queries a FAISS vector index over Belgian waste law documents using `sentence-transformers`
- `nlp/sorting_rules.json`: Hard-coded material → bin mapping
- `nlp/nlp.py`: Manages Ollama conversation history and integrates RAG context
- LLM default: `qwen2.5:3b` via local Ollama

### Training Architecture

The `trash_detector` package (`testing_projects/Yolo-taco_test_v1/src/trash_detector/`) provides:
- YOLO training wrapper with balanced class weights (`cls_pw`)
- Multi-dataset merging (TACO + custom data → `Totaal_dataset`)
- `bin_mapping.json`: maps detected material classes → disposal bin categories

## Key Configuration

**Demo** (`demo/config.py`):
- `VISION_MODEL_PATH` — path to YOLO weights
- `NLP_MODEL_NAME` — Ollama model name (default: `"qwen2.5:3b"`)
- `TEMP_DIR` — shared temp directory for inter-thread files
- `SORTING_RULES_PATH`, `SYSTEM_PROMPT_PATH`, `START_CONVERSATION_PROMPT_PATH`

**Training** (edit directly in `scripts/run_train.py`):
- Model variant (e.g. `yolo11s`, `yolo8n`)
- Epochs, batch size, `cls_pw` for class balancing

## Dependencies

| Component | Package Manager | Python |
|-----------|----------------|--------|
| demo | pip / uv | ≥3.10 |
| testing_projects | uv | ≥3.11, <3.13 |

Key packages: `ultralytics`, `ollama`, `faiss-cpu`, `sentence-transformers`, `opencv-python`, `torch`/`torchvision`.

On Windows, PyTorch is sourced from the PyTorch index via `uv.sources` in `testing_projects/pyproject.toml` — do not change the source configuration without updating the lock file.

## Runtime Prerequisites

- **Ollama** must be running locally (`ollama serve`) before starting `demo.py`
- The chosen Ollama model must be pulled: `ollama pull qwen2.5:3b`
- A webcam must be available for the vision thread
