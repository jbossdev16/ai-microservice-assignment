"""RAG service for document retrieval using embeddings and FAISS."""

from pathlib import Path
from typing import List, Dict
import pickle
import logging

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

from app.config import settings

logger = logging.getLogger(__name__)


class RAGService:
    """Service for Retrieval-Augmented Generation using FAISS and embeddings."""
    
    def __init__(self):
        """Initialize RAG service with embedding model and FAISS index."""
        logger.info(f"Loading embedding model: {settings.embedding_model}")
        self.model = SentenceTransformer(settings.embedding_model)
        
        self.index = None
        self.chunks = []  # List of dicts with {text, product_id, source}
        
        # Try to load existing index, otherwise build new one
        if self._index_exists():
            self._load_index()
        else:
            logger.info("No existing index found, building new index...")
            self.build_index()
    
    def _index_exists(self) -> bool:
        """Check if FAISS index files exist."""
        index_file = settings.faiss_index_dir / "index.faiss"
        chunks_file = settings.faiss_index_dir / "chunks.pkl"
        return index_file.exists() and chunks_file.exists()
    
    def build_index(self):
        """
        Build FAISS index from document files.
        
        Reads all text files from docs directory, chunks them,
        generates embeddings, and creates FAISS index.
        """
        logger.info("Building FAISS index from documents...")
        
        # Read and chunk all documents
        self.chunks = []
        docs_dir = Path(settings.docs_dir)
        
        if not docs_dir.exists():
            logger.error(f"Documents directory not found: {docs_dir}")
            return
        
        for doc_file in docs_dir.glob("*.txt"):
            product_id = doc_file.stem
            logger.debug(f"Processing document: {doc_file.name}")
            
            with open(doc_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Chunk the document
            doc_chunks = self._chunk_document(content, product_id, doc_file.name)
            self.chunks.extend(doc_chunks)
        
        logger.info(f"Created {len(self.chunks)} chunks from {len(list(docs_dir.glob('*.txt')))} documents")
        
        if not self.chunks:
            logger.error("No chunks created, cannot build index")
            return
        
        # Generate embeddings for all chunks
        chunk_texts = [chunk['text'] for chunk in self.chunks]
        logger.info("Generating embeddings...")
        embeddings = self.model.encode(chunk_texts, show_progress_bar=False)
        
        # Create FAISS index
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings.astype('float32'))
        
        logger.info(f"FAISS index created with {self.index.ntotal} vectors")
        
        # Save index and chunks
        self._save_index()
    
    def _chunk_document(self, content: str, product_id: str, source: str) -> List[Dict]:
        """
        Split document into overlapping chunks.
        
        Args:
            content: Full document text
            product_id: Product identifier
            source: Source file name
            
        Returns:
            List of chunk dictionaries
        """
        words = content.split()
        chunks = []
        
        chunk_size = settings.chunk_size
        overlap = settings.chunk_overlap
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk_words = words[i:i + chunk_size]
            if len(chunk_words) < 20:  # Skip very small chunks
                continue
            
            chunk_text = ' '.join(chunk_words)
            chunks.append({
                'text': chunk_text,
                'product_id': product_id,
                'source': source
            })
        
        return chunks
    
    def _save_index(self):
        """Save FAISS index and chunks metadata to disk."""
        try:
            settings.faiss_index_dir.mkdir(parents=True, exist_ok=True)
            
            index_file = settings.faiss_index_dir / "index.faiss"
            chunks_file = settings.faiss_index_dir / "chunks.pkl"
            
            faiss.write_index(self.index, str(index_file))
            
            with open(chunks_file, 'wb') as f:
                pickle.dump(self.chunks, f)
            
            logger.info(f"Index saved to {settings.faiss_index_dir}")
            
        except Exception as e:
            logger.error(f"Failed to save index: {e}")
    
    def _load_index(self):
        """Load FAISS index and chunks metadata from disk."""
        try:
            index_file = settings.faiss_index_dir / "index.faiss"
            chunks_file = settings.faiss_index_dir / "chunks.pkl"
            
            self.index = faiss.read_index(str(index_file))
            
            with open(chunks_file, 'rb') as f:
                self.chunks = pickle.load(f)
            
            logger.info(f"Index loaded: {self.index.ntotal} vectors, {len(self.chunks)} chunks")
            
        except Exception as e:
            logger.error(f"Failed to load index: {e}")
            self.build_index()
    
    def retrieve(self, query: str, product_id: str = None, top_k: int = None) -> List[Dict]:
        """
        Retrieve relevant text chunks for a query.
        
        Args:
            query: Search query
            product_id: Optional product ID to filter results
            top_k: Number of results to return (defaults to settings.top_k_retrieval)
            
        Returns:
            List of relevant chunks with metadata
        """
        if top_k is None:
            top_k = settings.top_k_retrieval
        
        if not self.index or not self.chunks:
            logger.error("Index not initialized")
            return []
        
        try:
            # Generate query embedding
            query_embedding = self.model.encode([query])
            
            # Search in FAISS index
            # Search more than needed if filtering by product_id
            search_k = top_k * 5 if product_id else top_k
            distances, indices = self.index.search(query_embedding.astype('float32'), search_k)
            
            # Retrieve chunks
            results = []
            for idx in indices[0]:
                if idx < len(self.chunks):
                    chunk = self.chunks[idx]
                    
                    # Filter by product_id if specified
                    if product_id and chunk['product_id'] != product_id:
                        continue
                    
                    results.append(chunk)
                    
                    if len(results) >= top_k:
                        break
            
            logger.info(f"Retrieved {len(results)} chunks for query")
            return results
            
        except Exception as e:
            logger.error(f"Retrieval failed: {e}")
            return []


