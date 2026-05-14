"""
Smart Waste Sorter demo.

Starts two daemon threads concurrently:
  - Vision thread: continuously captures webcam frames, runs YOLO detection,
    and writes results to 'temp/'.
  - NLP thread: runs an interactive chatbot that reads the latest detection
    results when the user types 'start'.

Press ESC to stop both threads and exit.

Usage:
  uv run python demo.py
"""

import os

# Must be set BEFORE any 'import cv2' (directly or transitively via ultralytics).
# Without this, MSMF (the default Windows capture backend) opens USB cameras
# but silently fails to deliver frames, leaving 'output.png' frozen.
os.environ.setdefault("OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS", "0")

import threading
import time
from pathlib import Path

import ollama
from pynput import keyboard
from ultralytics import YOLO

from nlp.chatbot import WasteAssistant
from nlp.rag_system import RAGSystem
from nlp.runner import run_nlp_thread
from nlp.utils.io import read_yaml
from vision.vision import run_vision_thread

# ---------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parent
CONFIG_PATH = _PROJECT_ROOT / "config" / "demo_config.yaml"


# ---------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------

def _resolve(raw: str) -> Path:
    """Resolve a path string relative to the project root if it is not absolute."""
    p = Path(raw)
    return p if p.is_absolute() else (_PROJECT_ROOT / p).resolve()


# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------

def main() -> None:
    # --- Load config YAML ---
    cfg = read_yaml(CONFIG_PATH)

    # --- Locate YOLO model weights from the configured model bundle ---
    model_bundle = cfg["vision"]["model_bundle"]
    weights_path = _PROJECT_ROOT / "models" / model_bundle / "weights" / "best.pt"
    if not weights_path.is_file():
        raise FileNotFoundError(
            f"Model weights not found: {weights_path}\n"
            f"Copy a trained model bundle into 'models/{model_bundle}/' "
            f"(the bundle must contain 'weights/best.pt')."
        )
    model = YOLO(str(weights_path))

    # --- Set up the shared temp directory ---
    temp_dir = _resolve(cfg["paths"]["temp_dir"])
    temp_dir.mkdir(parents=True, exist_ok=True)

    # --- Build the RAG system (loads FAISS index if available, builds BM25 fresh) ---
    print("Loading RAG system...")
    rag = RAGSystem(
        general_docs_dir=_resolve(cfg["paths"]["general_docs"]),
        region_docs_dir=_resolve(cfg["paths"]["region_docs"]),
        faiss_index_dir=_resolve(cfg["paths"]["faiss_index"]),
        embedding_model=cfg["nlp"]["embedding_model"],
    )

    if not rag.has_index():
        print(
            "Warning: FAISS index not found.\n"
            "Run 'uv run python build_nlp_index.py' to build it before starting the demo.\n"
            "The chatbot will still work but cannot retrieve general context."
        )

    # --- Create the NLP assistant ---
    assistant = WasteAssistant(
        rag=rag,
        model_name=cfg["nlp"]["model"],
        sorting_rules_path=_resolve(cfg["paths"]["sorting_rules"]),
        system_prompt_path=_resolve(cfg["paths"]["system_prompt"]),
        start_prompt_path=_resolve(cfg["paths"]["start_prompt"]),
        debug=cfg["nlp"].get("debug", False),
    )

    # --- Warm up the Ollama model so the first user question is fast ---
    # Without this, the first 'start' or chat call would block while Ollama
    # loads the model into memory.
    print(f"Warming up Ollama model '{cfg['nlp']['model']}'...")
    ollama.chat(
        model=cfg["nlp"]["model"],
        messages=[{"role": "user", "content": "ok"}],
    )

    # --- Shared stop event: set when ESC is pressed ---
    stop_event = threading.Event()

    def _on_key_press(key: keyboard.Key) -> bool | None:
        """Return False from the listener callback to stop it."""
        if key == keyboard.Key.esc:
            print("\nESC pressed, stopping...")
            stop_event.set()
            return False
        return None

    # --- Start daemon threads ---
    print("Starting vision and chatbot threads...")
    camera_index = int(cfg["vision"].get("camera_index", 0))
    vision_thread = threading.Thread(
        target=run_vision_thread,
        args=(model, temp_dir, stop_event, camera_index),
        daemon=True,
    )
    nlp_thread = threading.Thread(
        target=run_nlp_thread,
        args=(assistant, temp_dir, stop_event),
        daemon=True,
    )
    vision_thread.start()
    nlp_thread.start()

    # --- Start the keyboard listener for ESC ---
    listener = keyboard.Listener(on_press=_on_key_press)
    listener.start()

    print('All threads started. Press "ESC" to stop.\n')

    # Main thread sleeps while the daemon threads run in the background.
    while not stop_event.is_set():
        time.sleep(1)

    # Give threads a moment to print any final output before exiting.
    time.sleep(2)
    listener.stop()
    print("Exiting.")
    # Force-exit to ensure all daemon threads are killed.
    os._exit(0)


if __name__ == "__main__":
    main()
