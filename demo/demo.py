import os
import threading
import time
from pynput import keyboard

import config
from nlp.nlp import run_nlp
from vision.vision import run_vision

def _on_press(key):
    if key == keyboard.Key.esc:
        print("ESC pressed, stopping...")
        config.IS_RUNNING = False
        return False
    
if __name__ == "__main__":
    print("Starting threads for vision and chatbot...")
    config.IS_RUNNING = True

    vision_thread = threading.Thread(target=run_vision, daemon=True)
    vision_thread.start()

    nlp_thread = threading.Thread(target=run_nlp, daemon=True)
    nlp_thread.start()

    listener = keyboard.Listener(on_press=_on_press)
    listener.start()

    print("All threads started. Press \"ESC\" to stop.")
    
    while config.IS_RUNNING:
        time.sleep(1)  # Main thread sleeps while threads runs in the background

    time.sleep(2)  # Give threads a moment to finish up before exiting
    listener.stop()
    print("\nThreads stopped. Exiting program.")
    os._exit(0)  # Force exit to ensure all threads are killed