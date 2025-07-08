import faiss
import numpy as np
import pickle
import os
from sentence_transformers import SentenceTransformer
from typing import List, Tuple
import json
from datetime import datetime

class VectorStore:
    def __init__(self, model_name="all-MiniLM-L6-v2", index_file="vector_index.faiss", metadata_file="metadata.pkl"):
        self.model = SentenceTransformer(model_name)
        self.dimension = 384  # Dimension for all-MiniLM-L6-v2
        self.index_file = index_file
        self.metadata_file = metadata_file
        
        # Initialize or load FAISS index
        if os.path.exists(index_file) and os.path.exists(metadata_file):
            self.index = faiss.read_index(index_file)
            with open(metadata_file, 'rb') as f:
                self.metadata = pickle.load(f)
        else:
            self.index = faiss.IndexFlatIP(self.dimension)  # Inner product for cosine similarity
            self.metadata = []
    
    def add_conversation(self, user_message: str, bot_response: str, session_id: str):
        """Add a conversation pair to the vector store"""
        # Create conversation context
        conversation_text = f"User: {user_message}\nBot: {bot_response}"
        
        # Generate embedding
        embedding = self.model.encode([conversation_text])
        embedding = embedding / np.linalg.norm(embedding, axis=1, keepdims=True)  # Normalize for cosine similarity
        
        # Add to FAISS index
        self.index.add(embedding.astype('float32'))
        
        # Store metadata
        metadata = {
            "user_message": user_message,
            "bot_response": bot_response,
            "session_id": session_id,
            "timestamp": datetime.now().isoformat(),
            "conversation_text": conversation_text
        }
        self.metadata.append(metadata)
        
        # Save to disk
        self._save_index()
    
    def search_similar_conversations(self, query: str, k: int = 5) -> List[Tuple[float, dict]]:
        """Search for similar conversations based on query"""
        if self.index.ntotal == 0:
            return []
        
        # Generate query embedding
        query_embedding = self.model.encode([query])
        query_embedding = query_embedding / np.linalg.norm(query_embedding, axis=1, keepdims=True)
        
        # Search in FAISS index
        scores, indices = self.index.search(query_embedding.astype('float32'), min(k, self.index.ntotal))
        
        # Return results with metadata
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx != -1:  # Valid index
                results.append((float(score), self.metadata[idx]))
        
        return results
    
    def get_conversation_history(self, session_id: str, limit: int = 10) -> List[dict]:
        """Get recent conversation history for a session"""
        session_conversations = [
            meta for meta in self.metadata 
            if meta["session_id"] == session_id
        ]
        return session_conversations[-limit:]
    
    def _save_index(self):
        """Save FAISS index and metadata to disk"""
        faiss.write_index(self.index, self.index_file)
        with open(self.metadata_file, 'wb') as f:
            pickle.dump(self.metadata, f)
    
    def get_stats(self) -> dict:
        """Get statistics about the vector store"""
        return {
            "total_conversations": len(self.metadata),
            "total_vectors": self.index.ntotal,
            "dimension": self.dimension
        }