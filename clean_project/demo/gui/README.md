# gui

Optional Tkinter window for the Smart Waste Sorter demo. When enabled, it
replaces the terminal chat and the separate 'temp/output.png' preview with
a single resizable window that shows the live annotated webcam feed on the
left and the chatbot conversation on the right.

The module is fully optional: deleting this folder and removing the `gui`
section from `config/demo_config.yaml` returns the demo to its original
terminal-only behaviour without further changes.

## Enable

In `config/demo_config.yaml`:

```yaml
gui:
  enabled: true
  refresh_ms: 100   # live feed redraw interval (lower = smoother, more CPU)
```

Then run the demo as usual:

```bash
uv run python demo.py
```

## Controls

| Action       | Effect                                                              |
|--------------|---------------------------------------------------------------------|
| `Start` button | Reads the latest YOLO detections and asks the bot for instructions. |
| `Ctrl+W`     | Same as the `Start` button, works regardless of focus inside the window. |
| `Send` button | Sends the text in the input box as a chat message.                  |
| `Enter` (in the input box) | Same as `Send`.                                       |
| `Clear` button | Wipes the chat transcript and history.                            |
| Close button | Stops the whole demo (sets the same shared stop event as ESC).      |
| `ESC`        | Stops the demo globally, even when the window is not focused.       |

## Architecture

```
demo.py
  ├── vision thread (vision/vision.py)  -> writes temp/output.png + results.json
  └── GUI main loop  (gui/runner.py -> gui/app.py)
        ├── Tk after() tick   -> re-reads output.png + results.json
        └── worker threads    -> WasteAssistant.chat() / start_conversation()
                                 results piped back via queue.Queue
```

Key design points:
- The vision thread is unchanged; it still writes the same files it always did.
- All LLM calls run on daemon worker threads, so the window stays responsive
  while the model is generating its reply.
- All widget access happens on the Tk main thread. Worker threads push their
  results through a `queue.Queue` that the Tk thread drains on every tick.
- The image is re-scaled to the current label size on every tick, so resizing
  the window rescales the feed live.

## Files

- `app.py` - `WasteSorterGUI` class. Builds the window and owns the chat history.
- `runner.py` - `run_gui()` entry point called from `demo.py`.
- `__init__.py` - marks the folder as a package and documents how to remove it.
