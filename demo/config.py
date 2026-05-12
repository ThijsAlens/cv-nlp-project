"""
In this file, all global variables for the "demo.py" script are defined.
"""

# ---------------------------------------------------------------------------
# VISION MODEL CONFIGURATION
# ---------------------------------------------------------------------------
VISION_MODEL_PATH = "vision/best.pt"                # Path to the trained YOLO model weights

from ultralytics import YOLO
VISION_MODEL = YOLO(VISION_MODEL_PATH)              # Initialize the YOLO model globally

# ---------------------------------------------------------------------------
# NLP CONFIGURATION
# ---------------------------------------------------------------------------

from nlp.rag_system import RAGSystem
RAG = RAGSystem()                                   # Initialize the RAG system globally

NLP_MODEL_NAME = "granite4.1:3b"                       # Default NLP model to use for the chatbot

SORTING_RULES_PATH = "nlp/sorting_rules.json"       # Path to the JSON file containing sorting rules for different materials

START_CONVERSATION_PROMPT_PATH = "nlp/start_conversation.txt"  # Path to the prompt template for starting a conversation
SYSTEM_PROMPT_PATH = "nlp/system_prompt.txt"       # Path to the system prompt template for the chatbot

# ---------------------------------------------------------------------------
# OTHER CONFIGURATION VARIABLES
# ---------------------------------------------------------------------------
IS_RUNNING = False                                  # Flag to control the main loop and threads
DEBUG = False                                        # Flag to enable debug output (prints retrieved chunks + full prompt)

from pathlib import Path
TEMP_DIR = Path("temp")                             # Temporary directory for intermediate files (e.g., captured webcam frames)

# ---------------------------------------------------------------------------
# Starting operation
# ---------------------------------------------------------------------------
# Ensure the temporary directory exists
import os
os.makedirs(TEMP_DIR, exist_ok=True)