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

def _get_rag_context(query: str) -> str:
    """Retrieve relevant Belgian law context for a query."""
    if not config.RAG.has_index():
        return ""
    
    chunks = config.RAG.retrieve(query, top_k=3)
    if not chunks:
        return ""
    
    return "\n\n".join(chunks)

def _start_conversation() -> str:
    with open(config.TEMP_DIR / "results.json", "r") as f:
        vision_data = json.load(f)
    detected_labels = vision_data.get("labels", [])
    
    disposal_info = [f"{obj} needs to be disposed in {_get_bin(obj)}" for obj in detected_labels]
    rag_context = _get_rag_context(" ".join(detected_labels))

    with open(config.START_CONVERSATION_PROMPT_PATH, "r") as f:
        start_prompt_template = f.read()
    start_prompt = start_prompt_template.format(
        disposal_items=chr(10).join(disposal_info),
        rag_context=f"BELGIAN LAW CONTEXT:{chr(10)}{rag_context}{chr(10)}" if rag_context else ""
    )

    response = ollama.chat(model=config.NLP_MODEL_NAME, messages=[{"role": "user", "content": start_prompt}])
    return response["message"]["content"].strip()

def _chat(user_input: str, chat_history: list) -> str:
    rag_context = _get_rag_context(user_input)

    with open(config.SYSTEM_PROMPT_PATH, "r") as f:
        system_prompt_template = f.read()
    system_prompt = system_prompt_template.format(rag_context=f"BELGIAN LAW CONTEXT:{chr(10)}{rag_context}{chr(10)}" if rag_context else "Note: No documents loaded, tell the user that the AI system is broken")

    messages_to_send = [{"role": "system", "content": system_prompt}]
    messages_to_send.extend(chat_history)
    messages_to_send.append({"role": "user", "content": user_input})

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