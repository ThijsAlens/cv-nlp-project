# config/

User-editable configuration for the demo. No Python code needs to change
when adjusting settings here.

## Files

| File | Purpose |
|------|---------|
| `demo_config.yaml` | All runtime settings: model bundle name, LLM model, RAG parameters, file paths, and debug flag. |

## Changing the vision model

1. Copy your trained model bundle folder into `models/` (see `models/README.md`).
2. Set `vision.model_bundle` in `demo_config.yaml` to the bundle folder name.

## Changing the LLM

1. Pull the new model: `ollama pull <model-name>`.
2. Set `nlp.model` in `demo_config.yaml` to the new model name.
