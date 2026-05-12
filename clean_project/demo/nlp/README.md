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
point used by `demo.py`. Key difference from the standalone `WasteAssistant.run_loop()`:
- The 'start' command reads detected items from `temp/results.json` (written by
  the vision thread), not from a test input file or interactive prompt.
- The loop exits when `stop_event` is set (ESC key), not on a 'quit' command.

## FAISS index

The `faiss_index/` directory is auto-generated. Build it once before running:
```bash
uv run python build_nlp_index.py
```
Rebuild it after adding files to `documents/general/`.
