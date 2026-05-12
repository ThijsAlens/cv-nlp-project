# scripts/

Runner entry points. No command-line arguments; all settings are in `config/nlp_config.yaml`.

## Scripts

| Script | What it does |
|--------|-------------|
| `run_build_index.py` | Builds the FAISS index from `documents/general/` and saves it to `faiss_index/`. Run once before the first chat, and again whenever general documents change. |
| `run_chat.py` | Starts the interactive chatbot loop. Requires Ollama to be running and the FAISS index to be built. |

## Typical order of operations

```bash
# 1. Build the FAISS index (once)
uv run python scripts/run_build_index.py

# 2. Start Ollama (in a separate terminal)
ollama serve
ollama pull granite4.1:3b

# 3. Run the chatbot
uv run python scripts/run_chat.py
```
