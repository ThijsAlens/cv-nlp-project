# nlp_assistant/

Main Python package. Two core modules plus shared utilities.

## Modules

| Module | Purpose |
|--------|---------|
| `rag_system.py` | `RAGSystem` class -- builds/loads the FAISS index, builds the BM25 index, and exposes `retrieve()` (FAISS) and `bm25_retrieve()` (BM25). |
| `chatbot.py` | `WasteAssistant` class -- wraps the RAG system and an Ollama LLM client. Handles `start_conversation()`, `chat()`, and the interactive `run_loop()`. |
| `utils/io.py` | `read_json`, `read_yaml`, `read_text` -- file reading helpers used by both modules. |

## How the two modules connect

```
run_chat.py
  -> RAGSystem(...)           loads FAISS index + builds BM25
  -> WasteAssistant(rag, ...) wraps the RAG system
  -> assistant.run_loop()     interactive terminal loop
        |
        +--> start_conversation(items)
        |       -> _get_rag_context()  FAISS + BM25 retrieval
        |       -> ollama.chat()       LLM generates disposal instruction
        |
        +--> chat(user_input, history)
                -> _extract_location() LLM extracts city from query
                -> _get_rag_context()  FAISS + BM25 retrieval
                -> ollama.chat()       LLM generates answer
```
