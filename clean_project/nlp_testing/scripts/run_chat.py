"""
Chatbot runner for the waste disposal assistant.

Reads 'config/nlp_config.yaml', initialises the RAG system and the LLM
client, then starts an interactive terminal chatbot loop.

Commands inside the chatbot:
  start  - Generate an opening disposal instruction. Detected items are
            read from 'config/test_input.json' (if configured) or prompted.
  quit   - Exit the chatbot.
  (any other text) - Ask a follow-up waste disposal question.

Prerequisites:
  - Ollama must be running: 'ollama serve'
  - The configured model must be pulled: 'ollama pull <model>'
  - The FAISS index must be built: 'uv run python scripts/run_build_index.py'

Usage:
  uv run python scripts/run_chat.py
"""

import sys
from pathlib import Path

# Add 'src' to the import path so 'nlp_assistant' can be found when
# running this script directly without installing the package first.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from nlp_assistant.chatbot import WasteAssistant
from nlp_assistant.rag_system import RAGSystem
from nlp_assistant.utils.io import read_yaml

# ---------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------

CONFIG_PATH = _PROJECT_ROOT / "config" / "nlp_config.yaml"

# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------

def main() -> None:
    # --- Load config ---
    cfg = read_yaml(CONFIG_PATH)

    # Helper to resolve a path from the config (relative paths are
    # resolved relative to the project root).
    def resolve(key: str, section: str = "paths") -> Path:
        raw = cfg[section][key]
        p = Path(raw)
        return p if p.is_absolute() else (_PROJECT_ROOT / p).resolve()

    # --- Resolve all configured paths ---
    general_docs = resolve("general_docs")
    region_docs = resolve("region_docs")
    faiss_index_dir = resolve("faiss_index")
    sorting_rules = resolve("sorting_rules")
    system_prompt = resolve("system_prompt")
    start_prompt = resolve("start_prompt")

    # The test input path is optional; leave as None if not configured.
    raw_test_input = cfg.get("test_input", "")
    test_input_path: Path | None = None
    if raw_test_input:
        candidate = Path(raw_test_input)
        test_input_path = candidate if candidate.is_absolute() else (
            _PROJECT_ROOT / candidate
        ).resolve()

    model_name = cfg["llm"]["model"]
    debug = cfg.get("debug", False)

    # --- Validate that required paths exist ---
    if not general_docs.exists():
        raise FileNotFoundError(
            f"General documents folder not found: {general_docs}\n"
            "Create 'documents/general/' and add at least one .txt file."
        )
    if not sorting_rules.exists():
        raise FileNotFoundError(f"Sorting rules file not found: {sorting_rules}")

    # --- Initialise the RAG system ---
    # This loads the FAISS index from disk (if it exists) and always
    # rebuilds the BM25 index fresh from the region documents folder.
    print(f"Loading RAG system (model: {model_name}) ...")
    rag = RAGSystem(
        general_docs_dir=general_docs,
        region_docs_dir=region_docs,
        faiss_index_dir=faiss_index_dir,
        embedding_model=cfg["rag"]["embedding_model"],
    )

    # Warn if the FAISS index has not been built yet.
    if not rag.has_index():
        print(
            "Warning: FAISS index not found. Run 'uv run python scripts/run_build_index.py' first.\n"
            "The chatbot will still work but cannot retrieve general context."
        )

    # --- Start the chatbot ---
    assistant = WasteAssistant(
        rag=rag,
        model_name=model_name,
        sorting_rules_path=sorting_rules,
        system_prompt_path=system_prompt,
        start_prompt_path=start_prompt,
        debug=debug,
    )

    assistant.run_loop(test_input_path=test_input_path)


if __name__ == "__main__":
    main()
