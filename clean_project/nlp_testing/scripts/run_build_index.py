"""
FAISS index builder for the waste disposal RAG system.

Reads 'config/nlp_config.yaml', loads documents from 'documents/general/',
and saves a new FAISS index to the configured 'faiss_index/' directory.

Run this script:
  - Once before using the chatbot for the first time.
  - Again whenever documents in 'documents/general/' are added or changed.

The BM25 index (for 'documents/regions/') is always rebuilt at chatbot
startup and does NOT need this script.

Usage:
  uv run python scripts/run_build_index.py
"""

import sys
from pathlib import Path

# Add 'src' to the import path so 'nlp_assistant' can be found when
# running this script directly without installing the package first.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

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

    # Resolve paths relative to the project root.
    def resolve(key: str) -> Path:
        raw = cfg["paths"][key]
        p = Path(raw)
        return p if p.is_absolute() else (_PROJECT_ROOT / p).resolve()

    general_docs = resolve("general_docs")
    region_docs = resolve("region_docs")
    faiss_index_dir = resolve("faiss_index")
    embedding_model = cfg["rag"]["embedding_model"]

    # Validate the documents folder exists before trying to build.
    if not general_docs.exists():
        raise FileNotFoundError(
            f"General documents folder not found: {general_docs}\n"
            "Create 'documents/general/' and add at least one .txt file."
        )

    print(f"Building FAISS index from: {general_docs}")
    print(f"Embedding model: {embedding_model}")
    print(f"Index output:    {faiss_index_dir}\n")

    # Initialise the RAG system without loading an existing index
    # (the constructor loads one if it exists, but we will rebuild anyway).
    rag = RAGSystem(
        general_docs_dir=general_docs,
        region_docs_dir=region_docs,
        faiss_index_dir=faiss_index_dir,
        embedding_model=embedding_model,
    )

    # Build and save the FAISS index.
    success = rag.build_index()

    if success:
        print("\nFAISS index built successfully.")
        print("You can now run 'uv run python scripts/run_chat.py'.")
    else:
        print("\nIndex build failed. Check the documents folder and try again.")


if __name__ == "__main__":
    main()
