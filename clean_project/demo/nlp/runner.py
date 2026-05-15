"""
NLP thread runner for the demo pipeline.

Provides 'run_nlp_thread()', which is the entry point for the NLP daemon thread
started by 'demo.py'. It wraps 'WasteAssistant' with demo-specific behaviour:
  - The 'start' command reads detected items from 'temp/results.json' (written
    by the vision thread) instead of from a test input file.
  - The 'clear' command empties the chat history without needing a detection.
  - Ctrl+W is a global keyboard shortcut that triggers the same behaviour as
    typing 'start'. Useful when both hands are busy holding an item.
  - The loop exits when 'stop_event' is set (ESC key) rather than on a 'quit' command.

This file is intentionally kept separate from 'chatbot.py' so that 'chatbot.py'
can remain identical to the version in 'nlp_testing' for easy copy-paste syncing.
"""

import threading
import time
from pathlib import Path
from typing import List

from pynput import keyboard

from nlp.chatbot import WasteAssistant
from nlp.utils.io import read_json


# ---------------------------------------------------------------
# Demo thread entry point
# ---------------------------------------------------------------

def run_nlp_thread(
    assistant: WasteAssistant,
    temp_dir: Path,
    stop_event: threading.Event,
) -> None:
    """
    Run the NLP chatbot loop as a daemon thread.

    Reads user input from the terminal. The loop runs until 'stop_event'
    is set (triggered by the ESC key in 'demo.py').

    Commands:
      start  - Reads detected items from 'temp/results.json' (written by the
               vision thread) and generates an opening disposal instruction.
      clear  - Wipes the chat history so the next message starts a fresh
               conversation. Does not require an object in view.
      Ctrl+W - Global shortcut that triggers the same behaviour as 'start'.
      (any other text) - Answers a follow-up waste disposal question.
    """
    # Wait for the vision thread to initialise before accepting input.
    time.sleep(2)

    # Chat history is shared between the input loop and the Ctrl+S handler,
    # so all reads and writes go through this lock to keep them consistent.
    chat_history: List[dict] = []
    history_lock = threading.Lock()

    # Local helper so 'start' and Ctrl+S share the exact same logic.
    def _trigger_start() -> None:
        """Read the latest detection, reset history, and run a start turn."""
        # Read the latest vision results from the shared temp file.
        detected_items = _read_detected_items(temp_dir)

        if not detected_items:
            print("No objects detected yet. Point the camera at an object first.\n")
            return

        # Reset history first, then store the start turn so follow-up
        # questions still have the new context.
        bot_response = assistant.start_conversation(detected_items)
        with history_lock:
            chat_history.clear()
            chat_history.append({"role": "user", "content": "start"})
            chat_history.append({"role": "assistant", "content": bot_response})
        print(f"\nBot: {bot_response}\n")

    # Ctrl+W callback. Runs in the keyboard listener thread, so it must be
    # safe to call concurrently with the input loop. The lock inside
    # '_trigger_start' takes care of that for chat_history.
    def _on_shortcut_start() -> None:
        try:
            _trigger_start()
        except Exception as e:
            print(f"Shortcut start failed: {e}")
        # Re-show the input prompt so the user knows they can type again.
        print("You: ", end="", flush=True)

    # Register the global Ctrl+W hotkey. 'GlobalHotKeys' is OS-wide, matching
    # the existing ESC listener in 'demo.py'. It runs in its own thread.
    # Ctrl+W was chosen over Ctrl+S because the latter is widely captured by
    # other apps (browsers, editors) and was being intercepted before reaching here.
    hotkey_listener = keyboard.GlobalHotKeys({"<ctrl>+w": _on_shortcut_start})
    hotkey_listener.start()

    print("Chatbot ready. Type 'start' (or press Ctrl+W) after pointing the camera at an object.")
    print("Type 'clear' to wipe the chat history without needing a detection.\n")

    try:
        while not stop_event.is_set():
            try:
                user_input = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                break

            if not user_input:
                continue

            command = user_input.lower()

            # -------------------------------------------------------
            if command == "start":
                # Same code path as the Ctrl+S shortcut.
                _trigger_start()

            # -------------------------------------------------------
            elif command == "clear":
                # Wipe history without touching the camera or the LLM.
                with history_lock:
                    chat_history.clear()
                print("Chat history cleared. Starting a new conversation.\n")

            # -------------------------------------------------------
            else:
                # Continue the conversation with full RAG context.
                # Snapshot the history under the lock so a concurrent Ctrl+S
                # cannot mutate the list while the LLM call is in flight.
                with history_lock:
                    history_snapshot = list(chat_history)

                bot_response = assistant.chat(user_input, history_snapshot)
                print(f"\nBot: {bot_response}\n")

                # Append the new turn under the lock as well.
                with history_lock:
                    chat_history.append({"role": "user", "content": user_input})
                    chat_history.append({"role": "assistant", "content": bot_response})
    finally:
        # Always release the global hotkey, even if the loop exits via an exception.
        hotkey_listener.stop()


# ---------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------

def _read_detected_items(temp_dir: Path) -> List[str]:
    """
    Read the detected material labels from the vision thread's 'results.json' output.

    Returns an empty list if the file does not exist or cannot be parsed.
    """
    results_file = temp_dir / "results.json"
    if not results_file.exists():
        return []

    try:
        data = read_json(results_file)
        return data.get("labels", [])
    except Exception as e:
        # Log the error but do not crash the NLP thread.
        print(f"Warning: could not read results.json: {e}")
        return []
