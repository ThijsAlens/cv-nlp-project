"""
One-time setup script: build the FAISS index for the NLP chatbot.

Reads 'config/demo_config.yaml' and builds a FAISS vector index from the
documents in 'nlp/documents/general/'. Run this once before starting the
demo, and again whenever files in that folder are added or changed.

The BM25 index (for 'nlp/documents/regions/') is always rebuilt automatically
when the demo starts and does NOT require this script.

Usage:
  uv run python build_nlp_index.py
"""

from pathlib import Path

from nlp.rag_system import RAGSystem
from nlp.utils.io import read_yaml

# ---------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parent
CONFIG_PATH = _PROJECT_ROOT / "config" / "demo_config.yaml"

# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------

def main() -> None:
    # --- Load config ---
    cfg = read_yaml(CONFIG_PATH)

    # Resolve all paths relative to the project root.
    def resolve(raw: str) -> Path:
        p = Path(raw)
        return p if p.is_absolute() else (_PROJECT_ROOT / p).resolve()

    general_docs = resolve(cfg["paths"]["general_docs"])
    region_docs  = resolve(cfg["paths"]["region_docs"])
    faiss_dir    = resolve(cfg["paths"]["faiss_index"])
    embed_model  = cfg["nlp"]["embedding_model"]

    if not general_docs.exists():
        raise FileNotFoundError(
            f"General documents folder not found: {general_docs}\n"
            "Add .txt files to 'nlp/documents/general/' before building the index."
        )

    print(f"Building FAISS index from: {general_docs}")
    print(f"Embedding model: {embed_model}")
    print(f"Index output:    {faiss_dir}\n")

    rag = RAGSystem(
        general_docs_dir=general_docs,
        region_docs_dir=region_docs,
        faiss_index_dir=faiss_dir,
        embedding_model=embed_model,
    )

    success = rag.build_index()

    if success:
        print("\nFAISS index built. You can now run 'uv run python demo.py'.")
    else:
        print("\nIndex build failed. Check the documents folder and try again.")


if __name__ == "__main__":
    main()
