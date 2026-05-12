"""
NLP thread runner for the demo pipeline.

Provides 'run_nlp_thread()', which is the entry point for the NLP daemon thread
started by 'demo.py'. It wraps 'WasteAssistant' with demo-specific behaviour:
  - The 'start' command reads detected items from 'temp/results.json' (written
    by the vision thread) instead of from a test input file.
  - The loop exits when 'stop_event' is set (ESC key) rather than on a 'quit' command.

This file is intentionally kept separate from 'chatbot.py' so that 'chatbot.py'
can remain identical to the version in 'nlp_testing' for easy copy-paste syncing.
"""

import threading
import time
from pathlib import Path
from typing import List

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
      (any other text) - Answers a follow-up waste disposal question.
    """
    # Wait for the vision thread to initialise before accepting input.
    time.sleep(2)

    chat_history: List[dict] = []
    print("Chatbot ready. Type 'start' after pointing the camera at an object.\n")

    while not stop_event.is_set():
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not user_input:
            continue

        # -------------------------------------------------------
        if user_input.lower() == "start":
            # Read the latest vision results from the shared temp file.
            detected_items = _read_detected_items(temp_dir)

            if not detected_items:
                print("No objects detected yet. Point the camera at an object first.\n")
                continue

            # Reset history to start a fresh conversation about the new items.
            chat_history = []
            bot_response = assistant.start_conversation(detected_items)
            print(f"\nBot: {bot_response}\n")

            # Store the start turn so the LLM has context for follow-up questions.
            chat_history.append({"role": "user", "content": "start"})
            chat_history.append({"role": "assistant", "content": bot_response})

        # -------------------------------------------------------
        else:
            # Continue the conversation with full RAG context.
            bot_response = assistant.chat(user_input, chat_history)
            print(f"\nBot: {bot_response}\n")

            chat_history.append({"role": "user", "content": user_input})
            chat_history.append({"role": "assistant", "content": bot_response})


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
