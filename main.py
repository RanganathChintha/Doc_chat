import logging
from pathlib import Path
from run_pipeline import setup_pipeline, run_query

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Interactive RAG interface"""
    from config import DOCUMENTS_DIR
    
    print("\n" + "="*60)
    print("MULTIMODAL RAG SYSTEM")
    print("="*60)
    
    # Step 1: Select document
    pdf_path = DOCUMENTS_DIR / "sample.pdf"
    
    if not pdf_path.exists():
        print(f"\n[ERROR] Document not found: {pdf_path}")
        print(f"[INFO] Please add a PDF to: {DOCUMENTS_DIR}")
        return
    
    print(f"\n[INFO] Using document: {pdf_path.name}")
    
    # Step 2: Setup pipeline
    print("\n[INFO] Setting up RAG pipeline...")
    rag = setup_pipeline(str(pdf_path))
    
    # Step 3: Interactive queries
    print("\n[OK] RAG system ready!\n")
    print("Enter your queries (type 'quit' to exit):\n")
    
    while True:
        query = input("Query: ").strip()
        
        if query.lower() == "quit":
            print("Goodbye!")
            break
        
        if not query:
            continue
        
        run_query(rag, query)

if __name__ == "__main__":
    main()
