"""
RAG System for Belgian Waste Disposal Laws
Uses FAISS for vector storage and sentence-transformers for embeddings.
"""
import os
import pickle
import re
from pathlib import Path

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

# Try to import pypdf, fall back gracefully
try:
    from pypdf import PdfReader
    HAS_PYPDF = True
except ImportError:
    HAS_PYPDF = False

# Paths
GENERAL_DOCS_DIR = Path(__file__).parent / "documents" / "general"   # FAISS: general Belgian material rules
REGION_DOCS_DIR  = Path(__file__).parent / "documents" / "regions"   # BM25: region/city/country-specific rules
INDEX_DIR = Path(__file__).parent / "faiss_index"
INDEX_FILE = INDEX_DIR / "index.faiss"
CHUNKS_FILE = INDEX_DIR / "chunks.pkl"

# Embedding model (small, fast, good quality)
EMBEDDING_MODEL = "all-MiniLM-L6-v2" # for multilingual support, consider "paraphrase-multilingual-MiniLM-L12-v2"


def _tokenize(text: str) -> list[str]:
    """Lowercase and extract word tokens, stripping all punctuation."""
    return re.findall(r"[\w]+", text.lower())


class RAGSystem:
    def __init__(self):
        self.model = SentenceTransformer(EMBEDDING_MODEL)
        self.index = None
        self.chunks = []
        self._bm25: BM25Okapi | None = None
        self._bm25_docs: list[str] = []  # one entry per document file, always fresh from disk
        
        # Load existing FAISS index if available
        if INDEX_FILE.exists() and CHUNKS_FILE.exists():
            self.load_index()
        
        # BM25 is always built fresh from disk so it includes any new files
        # even if the FAISS index has not been rebuilt yet
        self._build_bm25_from_disk()
    
    def load_documents(self) -> list[tuple[str, str]]:
        """Load general documents (for FAISS) from documents/general/."""
        if not GENERAL_DOCS_DIR.exists():
            print(f"No general documents folder found at {GENERAL_DOCS_DIR}")
            return []
        
        all_docs = []
        
        for file_path in sorted(GENERAL_DOCS_DIR.iterdir()):
            if file_path.suffix.lower() == ".txt":
                text = file_path.read_text(encoding="utf-8", errors="ignore")
                all_docs.append((file_path.name, text))
                print(f"Loaded: {file_path.name}")
                
            elif file_path.suffix.lower() == ".pdf" and HAS_PYPDF:
                reader = PdfReader(file_path)
                text = "\n".join(page.extract_text() or "" for page in reader.pages)
                all_docs.append((file_path.name, text))
                print(f"Loaded: {file_path.name}")
        
        return all_docs
    
    def build_index(self):
        """Build FAISS index from documents — one chunk per file."""
        documents = self.load_documents()
        if not documents:
            print("No documents found to index.")
            return False
        
        # Each file is one chunk (file-based chunking)
        self.chunks = [text.strip() for _, text in documents if text.strip()]
        
        if not self.chunks:
            print("No text chunks extracted.")
            return False
        
        print(f"Created {len(self.chunks)} chunks")
        
        # Create embeddings
        print("Creating embeddings...")
        embeddings = self.model.encode(self.chunks, show_progress_bar=True)
        embeddings = np.array(embeddings).astype('float32')
        
        # Build FAISS index
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings)
        
        # Save index and chunks
        INDEX_DIR.mkdir(exist_ok=True)
        faiss.write_index(self.index, str(INDEX_FILE))
        with open(CHUNKS_FILE, "wb") as f:
            pickle.dump(self.chunks, f)
        
        print(f"Index saved to {INDEX_DIR}")
        return True
    
    def _build_bm25_from_disk(self):
        """Build BM25 index from documents/regions/ — always fresh from disk."""
        if not REGION_DOCS_DIR.exists():
            return
        docs = []
        for file_path in sorted(REGION_DOCS_DIR.iterdir()):
            if file_path.suffix.lower() == ".txt":
                text = file_path.read_text(encoding="utf-8", errors="ignore").strip()
                if text:
                    docs.append(text)
            elif file_path.suffix.lower() == ".pdf" and HAS_PYPDF:
                reader = PdfReader(file_path)
                text = "\n".join(page.extract_text() or "" for page in reader.pages).strip()
                if text:
                    docs.append(text)
        self._bm25_docs = docs
        tokenized = [_tokenize(doc) for doc in docs]
        self._bm25 = BM25Okapi(tokenized)

    def load_index(self):
        """Load existing FAISS index."""
        self.index = faiss.read_index(str(INDEX_FILE))
        with open(CHUNKS_FILE, "rb") as f:
            self.chunks = pickle.load(f)
        print(f"Loaded index with {len(self.chunks)} chunks")
    
    def retrieve(self, query: str, top_k: int = 3) -> list[str]:
        """Retrieve most relevant chunks for a query."""
        if self.index is None or not self.chunks:
            return []
        
        # Embed query
        query_embedding = self.model.encode([query])
        query_embedding = np.array(query_embedding).astype('float32')
        
        # Search
        distances, indices = self.index.search(query_embedding, min(top_k, len(self.chunks)))
        
        # Return matching chunks
        results = [self.chunks[i] for i in indices[0] if i < len(self.chunks)]
        return results
    
    def bm25_retrieve(self, location: str, top_k: int = 1) -> list[str]:
        """Return the top_k documents that best match the location name via BM25.
        Only returns chunks that actually score > 0 (i.e. contain the keyword)."""
        if self._bm25 is None or not self._bm25_docs:
            return []
        tokens = _tokenize(location)
        scores = self._bm25.get_scores(tokens)
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        return [
            self._bm25_docs[i]
            for i, score in ranked[:top_k]
            if score > 0
        ]

    def has_index(self) -> bool:
        """Check if an index is loaded."""
        return self.index is not None and len(self.chunks) > 0


# Simple usage when run directly
if __name__ == "__main__":
    rag = RAGSystem()
    
    print("\n=== RAG System ===")
    print("1. Build index from documents")
    print("2. Test retrieval")
    
    choice = input("\nChoice: ").strip()
    
    if choice == "1":
        rag.build_index()
    elif choice == "2":
        query = input("Enter query: ")
        results = rag.retrieve(query)
        print(f"\nFound {len(results)} relevant chunks:")
        for i, chunk in enumerate(results, 1):
            print(f"\n--- Chunk {i} ---\n{chunk[:200]}...")
