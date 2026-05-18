"""
Tkinter window for the Smart Waste Sorter demo.

When 'gui.enabled' is true in 'config/demo_config.yaml', 'demo.py' opens this
window instead of running the chatbot in the terminal. The window shows the
latest annotated webcam frame ('temp/output.png') on the left and the chatbot
conversation on the right.

The vision thread itself is untouched: it still writes 'temp/output.png' and
'temp/results.json' on every iteration. The GUI just polls 'output.png' from
the Tk main loop and rebuilds the image at the current widget size on every
refresh, so resizing the window rescales the feed live.

Layout:
  +-------------------------+----------------------+
  |                         | Chat history (text)  |
  |    Live annotated       |                      |
  |    webcam feed          |                      |
  |                         +----------------------+
  |                         | [ Input entry      ] |
  |                         | [Start] [Send] [Clear]|
  +-------------------------+----------------------+

Keyboard shortcuts:
  Ctrl+W   - Run the 'start' action (matches the terminal mode shortcut).
  Enter    - Send the current input as a chat message.
  ESC      - Stops the whole demo (handled by the pynput listener in 'demo.py').
"""

import queue
import threading
import tkinter as tk
from pathlib import Path
from tkinter import scrolledtext, ttk
from typing import List, Optional

from PIL import Image, ImageTk

from nlp.chatbot import WasteAssistant
from nlp.utils.io import read_json


# ---------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------

def _read_detected_items(temp_dir: Path) -> List[str]:
    """Read the latest detected material labels from 'temp/results.json'."""
    # Mirrors the helper in 'nlp/runner.py' so the GUI does not need to call
    # back into the terminal-mode runner.
    results_file = temp_dir / "results.json"
    if not results_file.exists():
        return []

    try:
        data = read_json(results_file)
        return data.get("labels", [])
    except Exception:
        # The vision thread may be mid-write; the next tick will retry.
        return []


# ---------------------------------------------------------------
# Main window
# ---------------------------------------------------------------

class WasteSorterGUI:
    """
    Tkinter window for the Smart Waste Sorter demo.

    All LLM calls run on a daemon worker thread so the Tk main loop stays
    responsive. Worker results are sent back through a thread-safe queue
    ('_msg_queue') and rendered on the next 'after()' tick.
    """

    def __init__(
        self,
        assistant: WasteAssistant,
        temp_dir: Path,
        stop_event: threading.Event,
        refresh_ms: int = 100,
    ) -> None:
        # Wire up dependencies. Nothing in this constructor calls into Tk yet
        # so the caller can construct the GUI from any thread; only 'run()'
        # must execute on the main thread.
        self._assistant = assistant
        self._temp_dir = temp_dir
        self._stop_event = stop_event
        # Clamp the refresh interval so a very low YAML value cannot peg the CPU.
        self._refresh_ms = max(20, int(refresh_ms))

        # Chat history shared between the Send and Start actions. Touched only
        # from the Tk main thread (worker threads communicate via '_msg_queue').
        self._chat_history: List[dict] = []

        # Queue used by the worker threads to deliver bot replies, error
        # messages and status updates back to the Tk main loop.
        self._msg_queue: "queue.Queue[tuple[str, str]]" = queue.Queue()

        # Cached PhotoImage. Tk only keeps a weak reference to the underlying
        # image data, so we must keep this attribute alive for the duration
        # of the display or the feed flickers to black.
        self._photo: Optional[ImageTk.PhotoImage] = None

        # True while a worker thread is running an LLM call. The 'bind_all'
        # Ctrl+W shortcut bypasses the disabled-button state, so we also
        # check this flag in '_on_start' / '_on_send' to prevent overlap.
        self._worker_busy: bool = False

        # Build the actual widgets last so all attributes used by callbacks exist.
        self._build_window()

    # -----------------------------------------------------------
    # Window construction
    # -----------------------------------------------------------

    def _build_window(self) -> None:
        """Create all Tk widgets and bind the keyboard shortcuts."""
        # Default sizes: the image pane is square (IMAGE_SIDE x IMAGE_SIDE)
        # and the chat pane is 3/4 of that width. Total window width is the
        # sum of both plus a small allowance for pane padding; the height adds
        # a few pixels for the status line below the image and the input row.
        IMAGE_SIDE = 720
        CHAT_WIDTH = int(IMAGE_SIDE * 3 / 4)   # 540 px
        WINDOW_W = IMAGE_SIDE + CHAT_WIDTH + 20
        WINDOW_H = IMAGE_SIDE + 60
        # Remember the initial sash position so '_position_sash' can apply it
        # once the window has gone through its first layout pass.
        self._initial_sash_x = IMAGE_SIDE

        # Top-level window. Default geometry is roomy but the user can resize.
        self._root = tk.Tk()
        self._root.title("Smart Waste Sorter")
        self._root.geometry(f"{WINDOW_W}x{WINDOW_H}")
        # Minimum size keeps both panes usable even when the user drags small.
        self._root.minsize(640, 480)
        # Honour the OS close button by routing it through the stop event.
        self._root.protocol("WM_DELETE_WINDOW", self._on_close)

        # PanedWindow lets the user drag the divider between feed and chat.
        # Using ttk.PanedWindow keeps the look consistent with the buttons.
        self._paned = ttk.PanedWindow(self._root, orient=tk.HORIZONTAL)
        self._paned.pack(fill=tk.BOTH, expand=True)
        # Local alias for the rest of the method.
        paned = self._paned

        # ---- Left pane: live annotated feed ----
        feed_frame = ttk.Frame(paned, padding=4)
        paned.add(feed_frame, weight=3)

        # The image label expands to fill the pane. The 'bg' colour shows up
        # in the unused border around the (square) feed when the pane is wider
        # or taller than the feed itself.
        self._image_label = tk.Label(feed_frame, bg="#202020")
        self._image_label.pack(fill=tk.BOTH, expand=True)

        # Small status line below the image. Reports the latest detection.
        self._status_var = tk.StringVar(value="Waiting for camera...")
        ttk.Label(feed_frame, textvariable=self._status_var).pack(
            anchor=tk.W, pady=(4, 0)
        )

        # ---- Right pane: chat ----
        chat_frame = ttk.Frame(paned, padding=4)
        paned.add(chat_frame, weight=2)

        # Input row is packed first, anchored to the bottom of the chat pane,
        # so the buttons always have reserved space no matter how tall the
        # transcript gets. Packing the transcript first with expand=True can
        # push the input row off-screen on narrow windows.
        input_row = ttk.Frame(chat_frame)
        input_row.pack(side=tk.BOTTOM, fill=tk.X, pady=(4, 0))

        # Read-only chat transcript. Disabled state prevents the user from
        # accidentally editing the bot's replies. Font size 15 is a compromise:
        # 10pt is too small to read from a distance, 20pt cramps the buttons
        # on a narrow pane.
        self._chat_text = scrolledtext.ScrolledText(
            chat_frame,
            wrap=tk.WORD,
            state=tk.DISABLED,
            font=("Segoe UI", 15),
        )
        self._chat_text.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        # Different colours for the two participants make the transcript easier
        # to scan. 'system' is used for lightweight status notes (errors etc).
        self._chat_text.tag_config("user", foreground="#1f6feb")
        self._chat_text.tag_config("bot", foreground="#1a7f37")
        self._chat_text.tag_config(
            "system", foreground="#8a8a8a", font=("Segoe UI", 13, "italic")
        )

        # Action buttons are packed to the right of the input row first so
        # they get their preferred width before the entry. The entry then
        # expands to fill whatever space is left.
        self._clear_btn = ttk.Button(input_row, text="Clear", command=self._on_clear)
        self._clear_btn.pack(side=tk.RIGHT, padx=(4, 0))
        self._send_btn = ttk.Button(input_row, text="Send", command=self._on_send)
        self._send_btn.pack(side=tk.RIGHT, padx=(4, 0))
        self._start_btn = ttk.Button(input_row, text="Start (Ctrl+W)", command=self._on_start)
        self._start_btn.pack(side=tk.RIGHT, padx=(4, 0))

        # Entry widget plus its backing StringVar for easy clearing. Packed
        # last so it only takes the leftover space, leaving room for buttons.
        self._input_var = tk.StringVar()
        self._input_entry = ttk.Entry(
            input_row,
            textvariable=self._input_var,
            font=("Segoe UI", 15),
        )
        self._input_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        # Enter key sends the message, same as in most chat apps.
        self._input_entry.bind("<Return>", lambda _e: self._on_send())
        self._input_entry.focus_set()

        # Bind Ctrl+W at the application level so the shortcut works no matter
        # which widget currently has focus inside the window. Both case forms
        # are bound to cover the case where Caps Lock is on.
        self._root.bind_all("<Control-w>", lambda _e: self._on_start())
        self._root.bind_all("<Control-W>", lambda _e: self._on_start())

        # Initial hint so the chat area is not blank on startup.
        self._append("system", "Point the camera at an object and press Start (Ctrl+W).\n")

    # -----------------------------------------------------------
    # Periodic Tk-thread callbacks
    # -----------------------------------------------------------

    def _tick(self) -> None:
        """Refresh the live feed image and drain the worker-result queue."""
        # If ESC (or another component) flipped the stop event, tear down the
        # window. The Tk main loop will then return and 'demo.py' can exit.
        if self._stop_event.is_set():
            try:
                self._root.destroy()
            except tk.TclError:
                # Window already destroyed; nothing more to do.
                pass
            return

        self._refresh_image()
        self._drain_queue()

        # Re-arm the timer for the next tick. Using 'after' keeps everything
        # on the Tk main thread, avoiding cross-thread widget access.
        self._root.after(self._refresh_ms, self._tick)

    def _refresh_image(self) -> None:
        """Re-read 'output.png' and rescale it to the current label size."""
        output_path = self._temp_dir / "output.png"
        if not output_path.is_file():
            return

        # 'Image.open' is lazy; '.load()' forces the decode and lets us drop
        # the file handle before the vision thread tries to rewrite the file.
        try:
            image = Image.open(output_path)
            image.load()
        except Exception:
            # The vision thread may be mid-write; try again on the next tick.
            return

        # Read the current label size. Returns 1 before the first layout pass,
        # so we guard against that to avoid degenerate resize calls.
        label_w = max(1, self._image_label.winfo_width())
        label_h = max(1, self._image_label.winfo_height())

        # Pick the largest size that fits in the label while preserving aspect.
        scale = min(label_w / image.width, label_h / image.height)
        if scale <= 0:
            return
        new_size = (
            max(1, int(image.width * scale)),
            max(1, int(image.height * scale)),
        )
        # BILINEAR is fast and visually fine for a live feed; LANCZOS would
        # look slightly sharper but doubles the per-frame CPU cost.
        image = image.resize(new_size, Image.BILINEAR)

        # Convert and keep a strong reference so Tk does not drop the image.
        self._photo = ImageTk.PhotoImage(image)
        self._image_label.configure(image=self._photo)

        # Update the small status line with the most recent detection labels.
        labels = _read_detected_items(self._temp_dir)
        if labels:
            self._status_var.set(f"Detected: {', '.join(labels)}")
        else:
            self._status_var.set("No object detected.")

    def _drain_queue(self) -> None:
        """Apply any pending worker-thread results to the chat widget."""
        # Worker threads push tuples of (kind, payload). The Tk loop is the
        # only consumer, so a single drain per tick is enough.
        while True:
            try:
                kind, text = self._msg_queue.get_nowait()
            except queue.Empty:
                return

            # -------------------------------------------------------
            if kind == "bot":
                # Reply to a normal chat message: append to running history.
                self._append("bot", f"Bot: {text}\n\n")
                self._chat_history.append({"role": "assistant", "content": text})
                self._set_buttons_enabled(True)

            # -------------------------------------------------------
            elif kind == "bot-start":
                # 'start' always begins a fresh conversation, so the history
                # is replaced (not appended to). This mirrors 'nlp/runner.py'.
                self._append("bot", f"Bot: {text}\n\n")
                self._chat_history = [
                    {"role": "user", "content": "start"},
                    {"role": "assistant", "content": text},
                ]
                self._set_buttons_enabled(True)

            # -------------------------------------------------------
            elif kind == "error":
                # Surface worker exceptions in the chat as a system note.
                self._append("system", f"[error] {text}\n")
                self._set_buttons_enabled(True)

            # -------------------------------------------------------
            elif kind == "system":
                # Generic status line (currently unused).
                self._append("system", f"{text}\n")

    # -----------------------------------------------------------
    # Button / shortcut handlers (Tk main thread only)
    # -----------------------------------------------------------

    def _on_send(self) -> None:
        """Send the current input as a chat message via the worker thread."""
        # Reject re-entry: a previous LLM call must finish before the next one.
        if self._worker_busy:
            return
        # Ignore empty/whitespace-only input so the user cannot trigger a
        # pointless LLM call by mashing Enter.
        user_input = self._input_var.get().strip()
        if not user_input:
            return
        self._input_var.set("")

        # Render the user message immediately so the UI feels responsive even
        # while the LLM is still generating its reply.
        self._append("user", f"You: {user_input}\n\n")
        self._chat_history.append({"role": "user", "content": user_input})

        # Snapshot the history without the current turn, since 'WasteAssistant.chat'
        # appends the new user message itself.
        history_snapshot = list(self._chat_history[:-1])
        self._set_buttons_enabled(False)

        # The worker thread is daemon so it dies with the process. We do not
        # need to track it - results flow back through the queue.
        threading.Thread(
            target=self._worker_chat,
            args=(user_input, history_snapshot),
            daemon=True,
        ).start()

    def _on_start(self) -> None:
        """Run the 'start' action: re-read detections and ask for instructions."""
        # Reject re-entry: 'bind_all' lets Ctrl+W fire even while the buttons
        # are disabled, so an explicit busy check is needed here too.
        if self._worker_busy:
            return
        # Read the latest detection from the vision thread's JSON output.
        detected = _read_detected_items(self._temp_dir)
        if not detected:
            self._append(
                "system",
                "No objects detected yet. Point the camera at an object first.\n",
            )
            return

        # Echo the implicit 'start' command so the user can see why the bot
        # responded. This matches the terminal flow.
        self._append("user", "You: start\n\n")
        self._set_buttons_enabled(False)

        threading.Thread(
            target=self._worker_start,
            args=(detected,),
            daemon=True,
        ).start()

    def _on_clear(self) -> None:
        """Wipe chat history and clear the transcript widget."""
        self._chat_history = []
        # Temporarily re-enable the widget so we can delete its contents.
        self._chat_text.configure(state=tk.NORMAL)
        self._chat_text.delete("1.0", tk.END)
        self._chat_text.configure(state=tk.DISABLED)
        self._append("system", "Chat history cleared.\n")

    def _on_close(self) -> None:
        """OS close button: ask the rest of the demo to stop, then close."""
        # Setting the flag here tells the next '_tick' to destroy the window.
        # Doing it that way (instead of calling destroy() directly) avoids
        # racing with an in-flight refresh that might still touch the widget.
        self._stop_event.set()

    # -----------------------------------------------------------
    # Worker-thread targets (not in Tk main thread, never touch widgets)
    # -----------------------------------------------------------

    def _worker_chat(self, user_input: str, history_snapshot: list) -> None:
        """Run a 'chat()' call and push the result back through the queue."""
        try:
            reply = self._assistant.chat(user_input, history_snapshot)
            self._msg_queue.put(("bot", reply))
        except Exception as e:
            # Pushing the error through the queue keeps all widget access on
            # the Tk thread, regardless of which thread the failure happened on.
            self._msg_queue.put(("error", f"chat failed: {e}"))

    def _worker_start(self, detected_items: list) -> None:
        """Run a 'start_conversation()' call and push the result through the queue."""
        try:
            reply = self._assistant.start_conversation(detected_items)
            self._msg_queue.put(("bot-start", reply))
        except Exception as e:
            self._msg_queue.put(("error", f"start failed: {e}"))

    # -----------------------------------------------------------
    # Chat widget helpers
    # -----------------------------------------------------------

    def _append(self, tag: str, text: str) -> None:
        """Append 'text' to the chat transcript with the given style tag."""
        # The widget is normally disabled; flip it briefly to insert text.
        self._chat_text.configure(state=tk.NORMAL)
        self._chat_text.insert(tk.END, text, tag)
        self._chat_text.configure(state=tk.DISABLED)
        # Auto-scroll so the newest message is always visible.
        self._chat_text.see(tk.END)

    def _set_buttons_enabled(self, enabled: bool) -> None:
        """Enable or disable input controls while a worker is in flight."""
        # Mirror the visual state in '_worker_busy' so the Ctrl+W shortcut
        # (which bypasses 'bind_all'-disabled-state semantics) can also see it.
        self._worker_busy = not enabled
        # ttk buttons use the state(['!disabled']) / state(['disabled']) API.
        state = ("!disabled",) if enabled else ("disabled",)
        self._send_btn.state(state)
        self._start_btn.state(state)
        # ttk.Entry uses 'configure(state=...)' instead of 'state([...])'.
        self._input_entry.configure(state=tk.NORMAL if enabled else tk.DISABLED)

    # -----------------------------------------------------------
    # Public entry point
    # -----------------------------------------------------------

    def run(self) -> None:
        """Start the Tk main loop. Blocks until the window is destroyed."""
        # Kick off the first refresh tick. Subsequent ticks re-arm themselves.
        self._root.after(self._refresh_ms, self._tick)
        # Position the divider once Tk has finished its first layout pass.
        # 'sashpos' silently no-ops before the panes have a real size, so
        # scheduling via 'after' guarantees it lands at the intended pixel.
        self._root.after(50, self._position_sash)
        self._root.mainloop()

    def _position_sash(self) -> None:
        """Place the pane divider so the image pane starts out square."""
        # Flush any pending layout work so 'winfo_width' returns the real
        # paned-window width instead of the placeholder '1' value it has
        # before the first <Configure> event.
        self._root.update_idletasks()
        paned_w = self._paned.winfo_width()

        # If the paned has not been mapped yet, try again on the next tick.
        # Without this guard, sashpos can be clamped to a tiny width and the
        # chat pane collapses to zero pixels.
        if paned_w <= 1:
            self._root.after(50, self._position_sash)
            return

        # Keep at least 'min_chat' px reserved for the chat pane so the sash
        # cannot be placed past the right edge on small or slow-to-realise
        # windows.
        min_chat = 280
        sash_x = min(self._initial_sash_x, max(0, paned_w - min_chat))
        try:
            self._paned.sashpos(0, sash_x)
        except tk.TclError:
            # Window not fully realised yet; ignore - the user can drag the
            # divider manually, and subsequent resizes work normally.
            pass
