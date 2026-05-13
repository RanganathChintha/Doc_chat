import chromadb
from pathlib import Path
from typing import List, Dict
import numpy as np
import logging

logger = logging.getLogger(__name__)

class ChromaVectorStore:
    """Manage vector storage with Chroma"""
    
    def __init__(self, db_path: Path, collection_name: str = "documents"):
        self.db_path = db_path
        self.collection_name = collection_name
        
        self.client = chromadb.PersistentClient(path=str(db_path))
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        
        logger.info(f"Initialized Chroma at {db_path}")
    
    def add_chunks(self, embedded_chunks: List[Dict]):
        """Add embedded chunks to vector store"""
        logger.info(f"Adding {len(embedded_chunks)} chunks to vector store")
        if not embedded_chunks:
            raise ValueError("Cannot add chunks to Chroma because no embeddings were created.")
        
        ids = []
        embeddings = []
        documents = []
        metadatas = []
        
        for chunk in embedded_chunks:
            ids.append(chunk["chunk_id"])
            embeddings.append(chunk["embedding"].tolist())
            documents.append(chunk["text"])
            
            # Store images and captions as metadata
            metadatas.append({
                "num_images": len(chunk["images"]),
                "images": str(chunk["images"]),
                "captions": str(chunk["captions"]),
                **chunk["metadata"]
            })
        
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )
        
        logger.info(f"Successfully added {len(ids)} chunks")
    
    def search(self, query_embedding: np.ndarray, n_results: int = 5) -> Dict:
        """Search for relevant chunks"""
        try:
            results = self.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=n_results,
                include=["documents", "metadatas", "embeddings", "distances"]
            )
            return results
        except Exception as e:
            logger.error(f"Search error: {e}")
            return None
    
    def get_stats(self) -> Dict:
        """Get collection statistics"""
        count = self.collection.count()
        return {"total_chunks": count}
