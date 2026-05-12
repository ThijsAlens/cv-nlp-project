"""
RAG system for Belgian waste disposal laws.

Uses two complementary retrieval strategies:
  - FAISS (IndexFlatL2): semantic vector search over general rule documents.
    Documents live in 'documents/general/'. The index is built once and
    persisted to disk. Rebuild it when new documents are added.
  - BM25 (BM25Okapi): keyword search over region-specific rule documents.
    Documents live in 'documents/regions/'. Always rebuilt fresh from disk
    at startup so new region files are picked up automatically.

Typical usage:
  rag = RAGSystem(general_docs_dir, region_docs_dir, faiss_index_dir, embedding_model)
  rag.build_index()              # Once (or when documents change)
  chunks = rag.retrieve(query)   # FAISS semantic search
  docs = rag.bm25_retrieve(loc)  # BM25 location search
"""

import os
import pickle
import re
from pathlib import Path
from typing import List, Optional, Tuple

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

# Try to import pypdf for optional PDF support.
try:
    from pypdf import PdfReader
    _HAS_PYPDF = True
except ImportError:
    _HAS_PYPDF = False


# ---------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------

def _tokenize(text: str) -> List[str]:
    """Lowercase the text and extract word tokens, stripping all punctuation."""
    return re.findall(r"[\w]+", text.lower())


def _load_text_files(directory: Path) -> List[Tuple[str, str]]:
    """
    Load all .txt (and optionally .pdf) files from 'directory'.

    Returns a list of (filename, text_content) tuples, sorted by filename
    so the order is deterministic across runs.
    """
    docs = []
    for file_path in sorted(directory.iterdir()):
        if file_path.suffix.lower() == ".txt":
            # Read the full text file.
            text = file_path.read_text(encoding="utf-8", errors="ignore")
            docs.append((file_path.name, text))

        elif file_path.suffix.lower() == ".pdf" and _HAS_PYPDF:
            # Extract text from all PDF pages.
            reader = PdfReader(file_path)
            text = "\n".join(page.extract_text() or "" for page in reader.pages)
            docs.append((file_path.name, text))

    return docs


# ---------------------------------------------------------------
# RAGSystem class
# ---------------------------------------------------------------

class RAGSystem:
    """
    Dual-retrieval RAG system combining FAISS (semantic) and BM25 (keyword) search.

    Pass all path arguments at construction time so the class has no
    implicit dependency on the global config module.
    """

    def __init__(
        self,
        general_docs_dir: Path,
        region_docs_dir: Path,
        faiss_index_dir: Path,
        embedding_model: str = "all-MiniLM-L6-v2",
    ) -> None:
        self._general_dir = general_docs_dir
        self._region_dir = region_docs_dir
        self._index_dir = faiss_index_dir
        self._index_file = faiss_index_dir / "index.faiss"
        self._chunks_file = faiss_index_dir / "chunks.pkl"

        # Load the sentence-transformer model used for FAISS embedding.
        # The HF_TOKEN env var is read if set (needed for gated models).
        self._embed_model = SentenceTransformer(
            embedding_model, token=os.environ.get("HF_TOKEN")
        )

        # FAISS index and the text chunks it maps to.
        self.index: Optional[faiss.IndexFlatL2] = None
        self.chunks: List[str] = []

        # BM25 index and the raw document texts it maps to.
        self._bm25: Optional[BM25Okapi] = None
        self._bm25_docs: List[str] = []

        # Load the persisted FAISS index if it already exists.
        if self._index_file.exists() and self._chunks_file.exists():
            self._load_index()

        # BM25 is always rebuilt fresh from disk so any newly added region
        # files are included without needing to rebuild the FAISS index.
        self._build_bm25_from_disk()

    # -----------------------------------------------------------
    # FAISS index management
    # -----------------------------------------------------------

    def build_index(self) -> bool:
        """
        Build the FAISS index from the general documents directory.

        Each document file becomes one chunk (no sliding-window chunking).
        The index and chunk list are saved to the 'faiss_index/' directory.
        Returns True on success, False if no documents were found.
        """
        if not self._general_dir.exists():
            print(f"General documents folder not found: {self._general_dir}")
            return False

        documents = _load_text_files(self._general_dir)
        if not documents:
            print("No documents found to index.")
            return False

        # Strip whitespace and drop empty files.
        self.chunks = [text.strip() for _, text in documents if text.strip()]
        if not self.chunks:
            print("All documents were empty after stripping whitespace.")
            return False

        print(f"Building FAISS index from {len(self.chunks)} document(s) ...")

        # Encode all chunks into dense vectors.
        embeddings = self._embed_model.encode(self.chunks, show_progress_bar=True)
        embeddings = np.array(embeddings, dtype="float32")

        # Build an exact L2-distance index (no approximation).
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings)

        # Persist the index and chunk list to disk.
        self._index_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(self._index_file))
        with open(self._chunks_file, "wb") as f:
            pickle.dump(self.chunks, f)

        print(f"Index saved to: {self._index_dir} ({len(self.chunks)} chunks)")
        return True

    def _load_index(self) -> None:
        """Load a previously saved FAISS index and its chunk list from disk."""
        self.index = faiss.read_index(str(self._index_file))
        with open(self._chunks_file, "rb") as f:
            self.chunks = pickle.load(f)
        print(f"Loaded FAISS index: {len(self.chunks)} chunk(s)")

    def has_index(self) -> bool:
        """Return True if a non-empty FAISS index is loaded."""
        return self.index is not None and len(self.chunks) > 0

    # -----------------------------------------------------------
    # BM25 index management
    # -----------------------------------------------------------

    def _build_bm25_from_disk(self) -> None:
        """
        Load all region documents and build a fresh BM25 index.

        Called automatically at construction time so new files added to
        'documents/regions/' are always included without manual rebuilding.
        """
        if not self._region_dir.exists():
            return

        documents = _load_text_files(self._region_dir)
        # Keep only non-empty document texts.
        self._bm25_docs = [text.strip() for _, text in documents if text.strip()]

        if not self._bm25_docs:
            return

        # Tokenize each document and build the BM25Okapi index.
        tokenized = [_tokenize(doc) for doc in self._bm25_docs]
        self._bm25 = BM25Okapi(tokenized)

    # -----------------------------------------------------------
    # Retrieval
    # -----------------------------------------------------------

    def retrieve(self, query: str, top_k: int = 3) -> List[str]:
        """
        Retrieve the 'top_k' most semantically relevant general-rule chunks.

        Embeds the query and performs an L2 nearest-neighbour search in the
        FAISS index. Returns an empty list if no index is loaded.
        """
        if not self.has_index():
            return []

        # Embed the query into the same vector space as the chunks.
        query_vec = np.array(self._embed_model.encode([query]), dtype="float32")

        # Search for the nearest neighbours (cap at number of available chunks).
        k = min(top_k, len(self.chunks))
        _, indices = self.index.search(query_vec, k)

        # Map indices back to chunk text, filtering out any out-of-range values.
        return [self.chunks[i] for i in indices[0] if i < len(self.chunks)]

    def bm25_retrieve(self, location: str, top_k: int = 1) -> List[str]:
        """
        Retrieve the 'top_k' region documents that best match 'location'.

        Uses BM25 keyword scoring. Only returns documents with a score > 0,
        meaning at least one token from 'location' was found in the document.
        """
        if self._bm25 is None or not self._bm25_docs:
            return []

        # Tokenize the location query and score all region documents.
        tokens = _tokenize(location)
        scores = self._bm25.get_scores(tokens)

        # Sort by score descending and take the top results with score > 0.
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        return [
            self._bm25_docs[i]
            for i, score in ranked[:top_k]
            if score > 0
        ]
