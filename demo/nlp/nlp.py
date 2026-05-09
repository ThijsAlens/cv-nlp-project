"""
This module contains the main logic for the NLP component of the waste sorting assistant. 
It should be used from the "demo.py" script, where the "run_nlp" function is called as a daemon thread.
"""

import json
import time
import ollama

import config

def _get_bin(obj_name: str) -> str:
    """
    Helper function to map an object/material name to the correct bin.
    
    Args:
        obj_name (str): The name of the detected object/material.
        
    Returns:
        str: The name of the bin where this object should be disposed.
    """
    with open(config.SORTING_RULES_PATH, "r") as f:
        sorting_rules = json.load(f)
    return sorting_rules.get(obj_name.lower(), "Rest")

def _extract_location(query: str) -> str | None:
    """Use the LLM to extract a city or country name from the user query.
    Returns the name as a string, or None if no location was found."""
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
        model=config.NLP_MODEL_NAME,
        messages=[{"role": "user", "content": prompt}]
    )
    result = response["message"]["content"].strip()
    return None if result.lower() == "none" else result


def _get_rag_context(query: str) -> tuple[str, str, list[tuple[str, str]], str | None]:
    """Retrieve relevant context: FAISS semantic search + BM25 location search.
    Returns (faiss_context, bm25_context, sources, extracted_location).
    faiss_context: general semantic matches.
    bm25_context: region/city-specific match (empty string if no location found)."""
    if not config.RAG.has_index():
        return "", "", [], None

    faiss_chunks = config.RAG.retrieve(query, top_k=3)

    # Location agent: extract city/country and do a targeted BM25 search
    location = _extract_location(query)
    bm25_chunks = config.RAG.bm25_retrieve(location, top_k=1) if location else []

    bm25_set = set(bm25_chunks)
    faiss_set = set(faiss_chunks)

    # FAISS context: only chunks NOT also returned by BM25 (avoid duplication in prompt)
    faiss_only = [c for c in faiss_chunks if c not in bm25_set]

    # Build source list for debug
    sources: list[tuple[str, str]] = []
    for chunk in faiss_only:
        sources.append((chunk, "FAISS"))
    for chunk in bm25_chunks:
        label = "FAISS+BM25" if chunk in faiss_set else f"BM25 (location: {location})"
        sources.append((chunk, label))

    return "\n\n".join(faiss_only), "\n\n".join(bm25_chunks), sources, location

def _start_conversation() -> str:
    with open(config.TEMP_DIR / "results.json", "r") as f:
        vision_data = json.load(f)
    detected_labels = vision_data.get("labels", [])
    
    disposal_info = [f"{obj} needs to be disposed in {_get_bin(obj)}" for obj in detected_labels]
    faiss_context, bm25_context, _, _ = _get_rag_context(" ".join(detected_labels))
    rag_context = "\n\n".join(filter(None, [faiss_context, bm25_context]))

    with open(config.START_CONVERSATION_PROMPT_PATH, "r") as f:
        start_prompt_template = f.read()
    start_prompt = start_prompt_template.format(
        disposal_items=chr(10).join(disposal_info),
        rag_context=f"BELGIAN LAW CONTEXT:{chr(10)}{rag_context}{chr(10)}" if rag_context else ""
    )

    response = ollama.chat(model=config.NLP_MODEL_NAME, messages=[{"role": "user", "content": start_prompt}])
    return response["message"]["content"].strip()

def _chat(user_input: str, chat_history: list) -> str:
    faiss_context, bm25_context, sources, location = _get_rag_context(user_input)

    with open(config.SYSTEM_PROMPT_PATH, "r") as f:
        system_prompt_template = f.read()

    if not faiss_context and not bm25_context:
        rag_ctx = "Note: No documents loaded, tell the user that the AI system is broken"
    else:
        rag_ctx = f"GENERAL CONTEXT:{chr(10)}{faiss_context}" if faiss_context else ""

    bm25_ctx = (
        f"Following region-specific information applies to {location} only. "
        f"This information is more important!:{chr(10)}{bm25_context}"
        if bm25_context else ""
    )

    system_prompt = system_prompt_template.format(rag_context=rag_ctx, bm25_context=bm25_ctx)

    messages_to_send = [{"role": "system", "content": system_prompt}]
    messages_to_send.extend(chat_history)
    messages_to_send.append({"role": "user", "content": user_input})

    if config.DEBUG:
        print("\n" + "=" * 60)
        print(f"[DEBUG] Location agent extracted: {location!r}")
        print(f"[DEBUG] Retrieved {len(sources)} chunk(s):")
        for i, (chunk, source) in enumerate(sources, 1):
            print(f"  [{i}] source={source}")
            print(f"      {chunk[:200].replace(chr(10), ' ')}...")
        print("=" * 60 + "\n")

    response = ollama.chat(model=config.NLP_MODEL_NAME, messages=messages_to_send)
    return response["message"]["content"].strip()

def run_nlp() -> None:
    """
    Run the NLP part:
        - Start an interactive chatbot loop where users can ask questions about waste disposal.
    This function is meant to be called as a deamon that continuously processes user input.
    """
    time.sleep(2)  # Wait a bit to ensure all threads are up and running

    chat_history = []
    while config.IS_RUNNING:
        user_input = input("You: ").strip()

        if not user_input: continue

        if user_input.lower() == "start":
            # start a "new" conversation in the chatbot, with new context based on the new YOLO input
            chat_history = []  # Clear chat history to start a new conversation
            bot_response = _start_conversation()
            print("\nBot:", bot_response)
            chat_history.append({"role": "user", "content": "start"})
            chat_history.append({"role": "assistant", "content": bot_response})
        else:
            # keep the conversation going with the existing context in chat_history
            bot_response = _chat(user_input, chat_history)
            print("\nBot:", bot_response)
            chat_history.append({"role": "user", "content": user_input})
            chat_history.append({"role": "assistant", "content": bot_response})