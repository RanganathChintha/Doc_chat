from typing import List, Dict
import numpy as np
import logging

logger = logging.getLogger(__name__)

class MultimodalRAG:
    """Complete RAG system for retrieval and generation"""
    
    def __init__(self, vector_store, embedder):
        self.vector_store = vector_store
        self.embedder = embedder
        logger.info("RAG system initialized")
    
    def retrieve(self, query: str, top_k: int = 5) -> Dict:
        """
        Retrieve relevant chunks for a query
        
        Args:
            query: User query string
            top_k: Number of results to return
            
        Returns:
            Dictionary with retrieved chunks and metadata
        """
        logger.info(f"Retrieving top {top_k} chunks for query: {query}")
        
        try:
            # Embed query
            from utils.chunker import Chunk
            query_chunk = Chunk(
                text=query,
                images=[],
                image_captions=[],
                metadata={"type": "query"},
                chunk_id="query"
            )
            query_embedding = self.embedder.embed_chunk_concatenate(query_chunk)
            
            # Search vector store
            results = self.vector_store.search(query_embedding, n_results=top_k)
            
            return self._format_results(results)
            
        except Exception as e:
            logger.error(f"Retrieval error: {e}")
            return None
    
    def _format_results(self, results: Dict) -> Dict:
        """Format vector store results for display"""
        if not results or not results.get("documents"):
            return {"chunks": []}
        
        formatted_chunks = []
        
        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]
        
        for doc, meta, distance in zip(documents, metadatas, distances):
            formatted_chunks.append({
                "text": doc,
                "images": eval(meta.get("images", "[]")),
                "captions": eval(meta.get("captions", "[]")),
                "metadata": {k: v for k, v in meta.items() 
                           if k not in ["images", "captions"]},
                "similarity_score": 1 - distance  # Convert distance to similarity
            })
        
        return {"chunks": formatted_chunks}
    
    def generate_response(self, query: str, retrieved_chunks: List[Dict]) -> str:
        """Generate response using LLM (integrate your LLM)"""
        context = self._build_context(retrieved_chunks)
        
        # TODO: Call your LLM here
        # response = llm.generate(prompt=f"Query: {query}\n\nContext: {context}")
        
        logger.info("Response generated (placeholder)")
        return context
    
    def _build_context(self, retrieved_chunks: List[Dict]) -> str:
        """Build context from retrieved chunks"""
        context = "Retrieved Information:\n\n"
        
        for i, chunk in enumerate(retrieved_chunks, 1):
            context += f"--- Chunk {i} ---\n"
            context += f"{chunk['text']}\n"
            
            if chunk['images']:
                context += f"Associated images: {', '.join(chunk['images'])}\n"
            
            if chunk['captions']:
                context += f"Image descriptions: {'; '.join(chunk['captions'])}\n"
            
            context += f"Similarity: {chunk['similarity_score']:.3f}\n\n"
        
        return context