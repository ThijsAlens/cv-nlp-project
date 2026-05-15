# nlp/

NLP module for the demo. Contains the RAG system, chatbot logic, and the
demo-specific thread runner.

## File origins

The core files are kept **identical** to their counterparts in
`clean_project/nlp_testing/src/nlp_assistant/` so changes can be
copy-pasted between the two projects without manual adaptation.

| File | Origin | Notes |
|------|--------|-------|
| `rag_system.py` | `nlp_testing/src/nlp_assistant/rag_system.py` | Identical copy |
| `chatbot.py` | `nlp_testing/src/nlp_assistant/chatbot.py` | Identical copy |
| `utils/io.py` | `nlp_testing/src/nlp_assistant/utils/io.py` | Identical copy |
| `runner.py` | Demo-specific | Wraps `WasteAssistant` for the threading model |
| `documents/` | `nlp_testing/documents/` | Identical copies of all rule documents |
| `sorting_rules.json` | `nlp_testing/config/sorting_rules.json` | Identical copy |
| `system_prompt.txt` | `nlp_testing/config/system_prompt.txt` | Identical copy |
| `start_conversation.txt` | `nlp_testing/config/start_conversation.txt` | Identical copy |

## Syncing changes from nlp_testing

When you update a core NLP file in `nlp_testing`, copy it here with the same relative path:
```
nlp_testing/src/nlp_assistant/rag_system.py  ->  demo/nlp/rag_system.py
nlp_testing/src/nlp_assistant/chatbot.py     ->  demo/nlp/chatbot.py
nlp_testing/src/nlp_assistant/utils/io.py   ->  demo/nlp/utils/io.py
```

Do NOT copy `runner.py` from `nlp_testing` -- that file is demo-specific.

## runner.py

Contains `run_nlp_thread(assistant, temp_dir, stop_event)`, the thread entry
point used by `demo.py`. Key differences from the standalone `WasteAssistant.run_loop()`:
- The 'start' command reads detected items from `temp/results.json` (written by
  the vision thread), not from a test input file or interactive prompt.
- A 'clear' command wipes the chat history without needing a detection. Useful
  for ending the current conversation when no new object is in view.
- A global Ctrl+W keyboard shortcut triggers the same behaviour as typing 'start'.
  Convenient when both hands are holding an item. Ctrl+W was chosen over the
  more obvious Ctrl+S because the latter is widely captured by browsers,
  editors, and other apps before it reaches the demo.
- The loop exits when `stop_event` is set (ESC key), not on a 'quit' command.

### Commands

| Input | What happens |
|-------|-------------|
| `start` | Reads `temp/results.json`, resets history, and runs an opening turn. |
| Ctrl+W | Same as `start`, triggered by the OS-wide hotkey. |
| `clear` | Wipes chat history. No camera read, no LLM call. |
| Any other text | Continues the conversation with full RAG context. |
| ESC (in any window) | Stops the demo (handled by `demo.py`). |

## FAISS index

The `faiss_index/` directory is auto-generated. Build it once before running:
```bash
uv run python build_nlp_index.py
```
Rebuild it after adding files to `documents/general/`.
