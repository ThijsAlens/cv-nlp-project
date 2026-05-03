"""
Waste Disposal Pipeline - Generates disposal instructions from YOLO detections using Ollama.
Now includes RAG system for Belgian waste disposal laws (Cross-Lingual).
"""
import sys
import json
import ollama
from pathlib import Path

_BASE = Path(__file__).resolve().parent

# trash_detector lives in src/ — add it to the path so it's importable
# without needing to pip-install the package separately.
_SRC = _BASE / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from nlp.rag_system import RAGSystem
from trash_detector.detector.garbage_detector import GarbageDetector
_IMAGE_FOR_PIPELINE = _BASE / "Image_for_pipeline" / "image.jpg"
_INPUT_JSON_PATH    = _BASE / "nlp" / "input.json"

# ---------------------------------------------------------------------------
# Sorting rules: maps model output material names to household bin labels.
# The vision model detects exactly these 5 classes (from data/bin_mapping.json):
# ---------------------------------------------------------------------------
SORTING_RULES = {
    "Cardboard": "Paper",
    "Glass":     "Rest",
    "Metal":     "PMD",
    "Paper":     "Paper",
    "Plastic":   "PMD",
}

# ---------------------------------------------------------------------------
# Lazy-loaded vision detector (model load is expensive, done once on first use)
# ---------------------------------------------------------------------------
_detector: GarbageDetector | None = None


def _get_detector() -> GarbageDetector:
    """Return the shared GarbageDetector instance, loading it on first call."""
    global _detector
    if _detector is None:
        print("Loading vision model...")
        _detector = GarbageDetector()
        print(f"Vision model loaded: {_detector.weights_path.name}")
    return _detector


def _run_vision_detection() -> None:
    """Run GarbageDetector on Image_for_pipeline/image.jpg and write the detected
    material names to NLP-Part_PHI/input.json so run_pipeline() can read them."""
    if not _IMAGE_FOR_PIPELINE.is_file():
        print(f"[detect] Image not found: {_IMAGE_FOR_PIPELINE}")
        return

    detector = _get_detector()
    result = detector.detect(_IMAGE_FOR_PIPELINE, save_crops=False, save_json=False)

    if not result.detections:
        print(f"[detect] No objects detected in '{_IMAGE_FOR_PIPELINE.name}'.")
        objects = []
    else:
        # Deduplicate while preserving order (same material can appear multiple times)
        seen: set[str] = set()
        objects = []
        for det in result.detections:
            if det.material not in seen:
                objects.append(det.material)
                seen.add(det.material)
        print(f"[detect] Detected: {objects}")

    with open(_INPUT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump({"Objects": objects}, f, indent=2)

# Initialize RAG system globally
rag = RAGSystem()


def get_bin(obj: str) -> str:
    """Look up the correct bin for an object."""
    return SORTING_RULES.get(obj, "Unknown")


def get_rag_context(query: str) -> str:
    """Retrieve relevant Belgian law context for a query."""
    if not rag.has_index():
        return ""
    
    chunks = rag.retrieve(query, top_k=3)
    if not chunks:
        return ""
    
    return "\n\n".join(chunks)


def run_pipeline(input_path: str, model: str = "qwen2.5:3b") -> str:
    """Load YOLO output, map to bins, and generate disposal instruction."""
    
    with open(input_path, "r") as f:
        data = json.load(f)
    
    # Build disposal info by looking up each object
    objects = data.get("Objects", [])
    disposal_info = [f"{obj} needs to be disposed in {get_bin(obj)}" for obj in objects]
    
    # Get relevant Belgian law context
    rag_context = get_rag_context(" ".join(objects))

    # Construct prompt with looked-up info and RAG context
    prompt = f"""
You are a helpful waste assistant in Flanders, Belgium. Your goal is to give a short, natural, and human-sounding instruction.

For this you will need the items:
{chr(10).join(disposal_info)}

RULES:
1. Start directly by mentioning the items.
2. Tell the user exactly where they go based on the context.
3. Use a maximum of 2 sentences.
4. Don't change the name of the bins!
5. CRITICAL: Read the Dutch context below, but ALWAYS write your final answer in the language of the user. If language is unclear, for example due to single word questions, default to English.

STRICT RULE: Do NOT add extra advice or warnings. Only talk about the detected items. 
If an item is PMD, call it PMD. Do NOT mention paper unless there is paper.

6. If you have a handy tip that is directly relevant to the items, you can add it as a final third sentence, but only if it is directly relevant to the items. For example, if there is a glass bottle, you can add "Make sure to rinse the glass bottle before recycling!" But if there is only paper, do NOT add a tip about rinsing. Followin context can be used for this, context only used for generating the tips!
{f"BELGIAN LAW CONTEXT:{chr(10)}{rag_context}{chr(10)}" if rag_context else ""}
CURRENT INPUT TO PROCESS:


    """

    response = ollama.chat(model=model, messages=[{"role": "user", "content": prompt}])
    return response["message"]["content"].strip()

def chat(user_message: str, chat_history: list, model: str = "qwen2.5:3b") -> str: # NIEUW: chat_history toegevoegd
    """Answer a user question about waste disposal using RAG context."""
    
    # NIEUW: Slim zoeken. Als de vraag kort is (bijv. "why?"), plakken we het vorige antwoord erbij voor FAISS.
    search_query = user_message
    if len(user_message.split()) < 4 and len(chat_history) > 0:
        search_query = f"{chat_history[-1]['content']} {user_message}"
        
    # Get relevant context from Belgian law documents
    rag_context = get_rag_context(search_query)
    
    # NIEUW: De prompt is nu een 'system' prompt zonder de user_message er hard ingecodeerd.
    system_prompt = f"""
You are an expert waste disposal assistant for Flanders, Belgium. 
Read the context below and answer the user's question based ONLY on this context. Do not invent acronyms or rules.
Answer the questions in the 'you' form, for example "You should put the glass bottle in the glass container".

CRITICAL INSTRUCTIONS BEFORE YOU ANSWER:
1. Keep it to 1-3 sentences and be direct. Do NOT invent information.
2. If you have a handy tip that is directly relevant to the items, you can add it as a final sentence, but only if it is directly relevant to the items. For example, if there is a glass bottle, you can add "Make sure to rinse the glass bottle before recycling!" But if there is only paper, do NOT add a tip about rinsing.

RELEVANT CONTEXT:
{rag_context if rag_context else "Note: No documents loaded, tell the user that the AI system is broken"}
    """
    
    # Hier geschiedenis en laatste vraag meegeven aan ollama.
    messages_to_send = [{"role": "system", "content": system_prompt}]
    messages_to_send.extend(chat_history)
    messages_to_send.append({"role": "user", "content": user_message})
    
    response = ollama.chat(model=model, messages=messages_to_send) # NIEUW: Stuur de hele lijst
    return response["message"]["content"].strip()


def chatbot_loop(model: str = "qwen2.5:3b"):
    """Interactive chatbot loop for asking waste disposal questions."""
    print("\n  Belgian Waste Disposal Chatbot")
    print("=" * 40)
    
    if rag.has_index():
        print(f" Belgian law documents loaded ({len(rag.chunks)} chunks)")
    else:
        print("  No documents indexed yet. Run: python rag_system.py and choose '1' to build index")
    
    print("\nAsk me anything about waste disposal in Belgium!")
    print("Commands: 'quit' to exit, 'detect' to process YOLO input\n")
    
    chat_history = []
    
    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break
        
        if not user_input:
            continue
        
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break
        
        if user_input.lower() == "detect":
            _run_vision_detection()
            bot_response = run_pipeline(str(_INPUT_JSON_PATH), model=model)
            print("\nBot:", bot_response)
            
            # Sla YOLO detectie en bot response op in het geheugen
            chat_history.append({"role": "user", "content": "detect"})
            chat_history.append({"role": "assistant", "content": bot_response})
            
        else:
            # Geef het geheugen mee aan de chat functie
            bot_response = chat(user_input, chat_history, model=model) 
            print("\nBot:", bot_response)
            
            # Sla gewone chat op in het geheugen
            chat_history.append({"role": "user", "content": user_input})
            chat_history.append({"role": "assistant", "content": bot_response})
        print()

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--detect":
        # Original behavior: process input.json
        instruction = run_pipeline(str(_INPUT_JSON_PATH))
        print(f"\n DISPOSAL INSTRUCTION:\n{instruction}")
    else:
        # New behavior: interactive chatbot
        chatbot_loop()
