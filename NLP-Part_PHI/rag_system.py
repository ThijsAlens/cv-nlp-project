"""
RAG System for Belgian Waste Disposal Laws
Uses FAISS for vector storage and sentence-transformers for embeddings.
"""
import os
import pickle
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Try to import pypdf, fall back gracefully
try:
    from pypdf import PdfReader
    HAS_PYPDF = True
except ImportError:
    HAS_PYPDF = False

# Paths
DOCUMENTS_DIR = Path(__file__).parent / "documents"
INDEX_DIR = Path(__file__).parent / "faiss_index"
INDEX_FILE = INDEX_DIR / "index.faiss"
CHUNKS_FILE = INDEX_DIR / "chunks.pkl"

# Embedding model (small, fast, good quality)
EMBEDDING_MODEL = "all-MiniLM-L6-v2" # for multilingual support, consider "paraphrase-multilingual-MiniLM-L12-v2"
CHUNK_SIZE = 500  # characters per chunk
CHUNK_OVERLAP = 50


class RAGSystem:
    def __init__(self):
        self.model = SentenceTransformer(EMBEDDING_MODEL)
        self.index = None
        self.chunks = []
        
        # Load existing index if available
        if INDEX_FILE.exists() and CHUNKS_FILE.exists():
            self.load_index()
    
    def load_documents(self) -> list[str]:
        """Load all documents from the documents folder."""
        if not DOCUMENTS_DIR.exists():
            print(f"No documents folder found at {DOCUMENTS_DIR}")
            return []
        
        all_text = []
        
        for file_path in DOCUMENTS_DIR.iterdir():
            if file_path.suffix.lower() == ".txt":
                text = file_path.read_text(encoding="utf-8", errors="ignore")
                all_text.append(text)
                print(f"Loaded: {file_path.name}")
                
            elif file_path.suffix.lower() == ".pdf" and HAS_PYPDF:
                reader = PdfReader(file_path)
                text = "\n".join(page.extract_text() or "" for page in reader.pages)
                all_text.append(text)
                print(f"Loaded: {file_path.name}")
        
        return all_text
    
    def chunk_text(self, text: str) -> list[str]:
        """Split text into overlapping chunks."""
        chunks = []
        start = 0
        while start < len(text):
            end = start + CHUNK_SIZE
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            start = end - CHUNK_OVERLAP
        return chunks
    
    def build_index(self):
        """Build FAISS index from documents in the documents folder."""
        documents = self.load_documents()
        if not documents:
            print("No documents found to index.")
            return False
        
        # Chunk all documents
        self.chunks = []
        for doc in documents:
            self.chunks.extend(self.chunk_text(doc))
        
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
