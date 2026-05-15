"""
Entry point that wires 'WasteSorterGUI' to the rest of the demo.

Kept as a thin shim so 'demo.py' has a single, stable import line and the
'app' module can be reorganised internally without breaking the entry point.
"""

import threading
from pathlib import Path

from nlp.chatbot import WasteAssistant

from .app import WasteSorterGUI


def run_gui(
    assistant: WasteAssistant,
    temp_dir: Path,
    stop_event: threading.Event,
    refresh_ms: int = 100,
) -> None:
    """
    Build and run the GUI on the calling thread.

    This call blocks until the window is closed (either by the user clicking
    the OS close button, by 'stop_event' being set from elsewhere, or by an
    exception during 'mainloop'). After it returns, the caller is expected to
    tear down the rest of the demo.
    """
    # Construct first, then run. Splitting the two lets future callers attach
    # extra hooks (for example, registering additional shortcuts) between the
    # constructor and the blocking mainloop call.
    gui = WasteSorterGUI(
        assistant=assistant,
        temp_dir=temp_dir,
        stop_event=stop_event,
        refresh_ms=refresh_ms,
    )
    gui.run()
