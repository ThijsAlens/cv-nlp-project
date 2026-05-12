"""
WasteAssistant: the chatbot that answers waste disposal questions.

Uses two components:
  - RAGSystem: retrieves relevant rule snippets (FAISS + BM25).
  - Ollama: a locally running LLM that generates answers from the retrieved context.

The assistant has two interaction modes:
  - 'start': takes a list of detected material names, maps them to bins,
    retrieves context, and generates an introductory disposal instruction.
  - 'chat': answers a free-form follow-up question with full RAG context.

Typical usage (from a runner script):
  assistant = WasteAssistant(rag, cfg)
  assistant.run_loop()   # blocking interactive loop
"""

from pathlib import Path
from typing import List, Optional, Tuple

import ollama

from .rag_system import RAGSystem
from .utils.io import read_json, read_text


class WasteAssistant:
    """
    Interactive waste disposal chatbot backed by a RAG system and a local LLM.

    All configuration is passed at construction so there is no dependency on
    a global config module.
    """

    def __init__(
        self,
        rag: RAGSystem,
        model_name: str,
        sorting_rules_path: Path,
        system_prompt_path: Path,
        start_prompt_path: Path,
        debug: bool = False,
    ) -> None:
        self._rag = rag
        self._model_name = model_name
        self._sorting_rules_path = sorting_rules_path
        self._system_prompt_path = system_prompt_path
        self._start_prompt_path = start_prompt_path
        self._debug = debug

    # -----------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------

    def _get_bin(self, material_name: str) -> str:
        """
        Look up the disposal bin for a detected material name.

        Reads 'sorting_rules.json' on every call so edits take effect
        without restarting the assistant. Returns 'Rest' if no match found.
        """
        rules = read_json(self._sorting_rules_path)
        # Keys in sorting_rules.json are lowercase; normalise the input.
        return rules.get(material_name.lower(), "Rest")

    def _extract_location(self, query: str) -> Optional[str]:
        """
        Ask the LLM to extract a city or region name from 'query'.

        Returns the location string, or None if no location was mentioned.
        The LLM is instructed to prefer specific cities over country names.
        """
        prompt = (
            "Extract the most specific location name (city, municipality, or intercommunale) from the sentence below.\n"
            "Rules:\n"
            "- Reply with ONLY the name (one or two words).\n"
            "- Always prefer a city or municipality over a country name. "
            "For example, if 'Gent' is mentioned, reply 'Gent', not 'Belgium'.\n"
            "- Only return a country name if NO city or municipality is mentioned at all.\n"
            "- If no location is mentioned whatsoever, reply with exactly: none\n\n"
            f"Sentence: {query}"
        )
        response = ollama.chat(
            model=self._model_name,
            messages=[{"role": "user", "content": prompt}]
        )
        result = response["message"]["content"].strip()
        # The LLM returns the literal string 'none' when no location is found.
        return None if result.lower() == "none" else result

    def _get_rag_context(
        self, query: str
    ) -> Tuple[str, str, List[Tuple[str, str]], Optional[str]]:
        """
        Retrieve relevant context for 'query' using both FAISS and BM25.

        Steps:
          1. FAISS semantic search on general rule documents.
          2. LLM location extraction to find a city/region in the query.
          3. BM25 keyword search on region documents using the extracted location.
          4. Deduplicate: remove any FAISS chunk that also appears in BM25 results
             to avoid repeating the same text in the prompt.

        Returns a tuple of:
          - faiss_context: joined text of FAISS-only chunks
          - bm25_context: joined text of BM25 chunks
          - sources: list of (chunk_text, retrieval_method) for debug output
          - extracted_location: the location string found by the LLM (or None)
        """
        if not self._rag.has_index():
            return "", "", [], None

        # Semantic search for general rules.
        faiss_chunks = self._rag.retrieve(query, top_k=3)

        # Location-based keyword search for region-specific rules.
        location = self._extract_location(query)
        bm25_chunks = self._rag.bm25_retrieve(location, top_k=1) if location else []

        # Remove chunks from FAISS that are already covered by BM25.
        bm25_set = set(bm25_chunks)
        faiss_set = set(faiss_chunks)
        faiss_only = [c for c in faiss_chunks if c not in bm25_set]

        # Build source labels for debug output.
        sources: List[Tuple[str, str]] = []
        for chunk in faiss_only:
            sources.append((chunk, "FAISS"))
        for chunk in bm25_chunks:
            label = "FAISS+BM25" if chunk in faiss_set else f"BM25 (location: {location})"
            sources.append((chunk, label))

        return (
            "\n\n".join(faiss_only),
            "\n\n".join(bm25_chunks),
            sources,
            location,
        )

    def _print_debug(
        self,
        location: Optional[str],
        sources: List[Tuple[str, str]],
    ) -> None:
        """Print retrieved chunks and extracted location when debug mode is on."""
        print("\n" + "=" * 60)
        print(f"[DEBUG] Location agent extracted: {location!r}")
        print(f"[DEBUG] Retrieved {len(sources)} chunk(s):")
        for i, (chunk, source) in enumerate(sources, 1):
            print(f"  [{i}] source={source}")
            # Show only the first 200 characters to keep debug output readable.
            print(f"      {chunk[:200].replace(chr(10), ' ')}...")
        print("=" * 60 + "\n")

    # -----------------------------------------------------------
    # Conversation turns
    # -----------------------------------------------------------

    def start_conversation(self, detected_items: List[str]) -> str:
        """
        Generate an opening disposal instruction for a list of detected materials.

        Maps each material to its bin, retrieves general RAG context, and
        fills the 'start_conversation.txt' prompt template before calling the LLM.
        """
        # Build the disposal instruction lines, e.g. "Metal needs to be disposed in pmd".
        disposal_lines = [
            f"{item} needs to be disposed in {self._get_bin(item)}"
            for item in detected_items
        ]

        # Retrieve general RAG context using the detected item names as the query.
        faiss_ctx, bm25_ctx, _, _ = self._get_rag_context(" ".join(detected_items))
        # Merge FAISS and BM25 context into one block for the start prompt.
        combined_rag = "\n\n".join(filter(None, [faiss_ctx, bm25_ctx]))

        # Fill the prompt template with the disposal lines and context.
        template = read_text(self._start_prompt_path)
        prompt = template.format(
            disposal_items="\n".join(disposal_lines),
            rag_context=(
                f"BELGIAN LAW CONTEXT:\n{combined_rag}\n" if combined_rag else ""
            ),
        )

        response = ollama.chat(
            model=self._model_name,
            messages=[{"role": "user", "content": prompt}]
        )
        return response["message"]["content"].strip()

    def chat(self, user_input: str, chat_history: List[dict]) -> str:
        """
        Generate a reply to 'user_input' using the current chat history and RAG context.

        The system prompt is rebuilt on every call so the retrieved context
        is always specific to the current message.
        """
        faiss_ctx, bm25_ctx, sources, location = self._get_rag_context(user_input)

        if self._debug:
            self._print_debug(location, sources)

        # Build the general RAG context block for the system prompt.
        if not faiss_ctx and not bm25_ctx:
            # No documents loaded - tell the LLM the system is unavailable.
            rag_ctx = "Note: No documents loaded, tell the user that the AI system is broken"
        else:
            rag_ctx = f"GENERAL CONTEXT:\n{faiss_ctx}" if faiss_ctx else ""

        # Build the region-specific context block (shown separately in the prompt).
        bm25_ctx_block = (
            f"Following region-specific information applies to {location} only. "
            f"This information is more important!:\n{bm25_ctx}"
            if bm25_ctx else ""
        )

        # Fill the system prompt template with the retrieved context.
        template = read_text(self._system_prompt_path)
        system_prompt = template.format(
            rag_context=rag_ctx,
            bm25_context=bm25_ctx_block,
        )

        # Assemble the full message list: system prompt + history + current turn.
        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(chat_history)
        messages.append({"role": "user", "content": user_input})

        response = ollama.chat(model=self._model_name, messages=messages)
        return response["message"]["content"].strip()

    # -----------------------------------------------------------
    # Interactive loop
    # -----------------------------------------------------------

    def run_loop(self, test_input_path: Optional[Path] = None) -> None:
        """
        Start an interactive chatbot loop in the terminal.

        Commands:
          start  - Begin a new conversation. If 'test_input_path' is set and the
                   file exists, detected items are read from it; otherwise the user
                   is prompted to enter them manually.
          quit   - Exit the loop.
          (anything else) - Continue the conversation with RAG context.
        """
        chat_history: List[dict] = []
        print("Waste disposal assistant ready.")
        print("Commands: 'start' to begin with detected items, 'quit' to exit.\n")

        while True:
            try:
                user_input = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                # Handle Ctrl+C or piped input ending gracefully.
                break

            if not user_input:
                continue

            if user_input.lower() == "quit":
                break

            # ---------------------------------------------------
            if user_input.lower() == "start":
                # Try to load detected items from the test input file.
                detected_items = self._load_detected_items(test_input_path)

                if not detected_items:
                    print("No items detected. Skipping start.\n")
                    continue

                # Reset history so the 'start' turn begins a fresh conversation.
                chat_history = []
                bot_response = self.start_conversation(detected_items)
                print(f"\nBot: {bot_response}\n")

                # Store the start turn in history for follow-up questions.
                chat_history.append({"role": "user", "content": "start"})
                chat_history.append({"role": "assistant", "content": bot_response})

            # ---------------------------------------------------
            else:
                bot_response = self.chat(user_input, chat_history)
                print(f"\nBot: {bot_response}\n")

                # Append this turn to history so the LLM has full context next turn.
                chat_history.append({"role": "user", "content": user_input})
                chat_history.append({"role": "assistant", "content": bot_response})

    def _load_detected_items(self, test_input_path: Optional[Path]) -> List[str]:
        """
        Return a list of detected material names for the 'start' command.

        If 'test_input_path' points to an existing JSON file with a 'labels' key,
        those labels are returned. Otherwise the user is prompted to enter them.
        """
        # Try reading from the configured test input file.
        if test_input_path and test_input_path.exists():
            data = read_json(test_input_path)
            items = data.get("labels", [])
            if items:
                print(f"(Using test input from '{test_input_path.name}'): {items}")
                return items

        # Fall back to interactive input.
        raw = input("Enter detected items (comma-separated, e.g. 'Metal, Glass'): ").strip()
        if not raw:
            return []
        return [item.strip() for item in raw.split(",") if item.strip()]
