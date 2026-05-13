import logging
from pathlib import Path
from config import (
    DOCUMENTS_DIR, IMAGES_DIR, CHROMA_DB_DIR,
    EmbeddingConfig, CaptioningConfig, RAGConfig
)
from utils.document_parser import DocumentParser
from utils.chunker import MultimodalChunker
from utils.captions import ImageCaptioner
from embeddings.embedder import MultimodalEmbedder
from vector_store.chroma_store import ChromaVectorStore
from rag.retrieval import MultimodalRAG

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def setup_pipeline(pdf_path: str):
    """
    Setup and run the complete RAG pipeline
    
    Args:
        pdf_path: Path to input PDF document
    """
    logger.info("=" * 60)
    logger.info("MULTIMODAL RAG PIPELINE - SETUP")
    logger.info("=" * 60)
    
    # Step 1: Parse document
    logger.info("\n[1/6] Parsing document...")
    parser = DocumentParser(IMAGES_DIR)
    parsed_doc = parser.extract_from_pdf(pdf_path)
    
    # Step 2: Chunk document
    logger.info("\n[2/6] Chunking document...")
    chunker = MultimodalChunker()
    chunks = chunker.chunk_document(parsed_doc)
    if not chunks:
        raise ValueError(
            "No chunks were created from the document. The PDF may not contain "
            "extractable text or images; run OCR first or use a text-based PDF."
        )
    
    # Step 3: Generate captions
    logger.info("\n[3/6] Generating image captions...")
    captioner = ImageCaptioner(enable=CaptioningConfig.ENABLE_CAPTIONS)
    chunks = captioner.caption_images(chunks)
    
    # Step 4: Embed chunks
    logger.info("\n[4/6] Creating embeddings...")
    embedder = MultimodalEmbedder(strategy=EmbeddingConfig.STRATEGY)
    embedded_chunks = embedder.embed_chunks(chunks)
    if not embedded_chunks:
        raise ValueError("No embeddings were created; check the embedding errors above.")
    
    # Step 5: Store in vector database
    logger.info("\n[5/6] Storing in vector database...")
    vector_store = ChromaVectorStore(CHROMA_DB_DIR)
    vector_store.add_chunks(embedded_chunks)
    
    # Step 6: Initialize RAG
    logger.info("\n[6/6] Initializing RAG system...")
    rag = MultimodalRAG(vector_store, embedder)
    
    logger.info("\n" + "=" * 60)
    logger.info("PIPELINE SETUP COMPLETE!")
    logger.info("=" * 60)
    
    stats = vector_store.get_stats()
    logger.info(f"Total chunks indexed: {stats['total_chunks']}")
    
    return rag

def run_query(rag, query: str):
    """Run a query through the RAG system"""
    logger.info(f"\nQuery: {query}")
    logger.info("-" * 60)
    
    # Retrieve
    results = rag.retrieve(query, top_k=RAGConfig.TOP_K)
    
    # Display results
    if results and results["chunks"]:
        logger.info(f"Found {len(results['chunks'])} relevant chunks\n")
        
        for i, chunk in enumerate(results["chunks"], 1):
            logger.info(f"Chunk {i} (Similarity: {chunk['similarity_score']:.3f})")
            logger.info(f"Text: {chunk['text'][:200]}...")
            
            if chunk['images']:
                logger.info(f"Images: {chunk['images']}")
            
            logger.info("-" * 40)
    else:
        logger.warning("No relevant chunks found")

if __name__ == "__main__":
    # Example usage
    pdf_file = DOCUMENTS_DIR / "sample.pdf"
    
    if pdf_file.exists():
        # Setup pipeline
        rag = setup_pipeline(str(pdf_file))
        
        # Run sample queries
        queries = [
            "What does the diagram show?",
            "Summarize the key points",
            "What are the main findings?"
        ]
        
        for query in queries:
            run_query(rag, query)
    else:
        logger.error(f"PDF file not found: {pdf_file}")
        logger.info(f"Please place a PDF in {DOCUMENTS_DIR}")
