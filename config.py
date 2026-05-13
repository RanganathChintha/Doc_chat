import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
DOCUMENTS_DIR = DATA_DIR / "documents"
IMAGES_DIR = DATA_DIR / "extracted_images"
CHROMA_DB_DIR = DATA_DIR / "chroma_db"

# Create directories if they don't exist
IMAGES_DIR.mkdir(parents=True, exist_ok=True)
CHROMA_DB_DIR.mkdir(parents=True, exist_ok=True)

# Model configurations
class EmbeddingConfig:
    """Embedding model settings"""
    STRATEGY = "concatenate"  # Options: "clip", "hybrid", "concatenate"
    TEXT_MODEL = "all-MiniLM-L6-v2"
    VISION_MODEL = "openai/clip-vit-base-patch32"
    DEVICE = "cuda"  # or "cpu"

class ChunkingConfig:
    """Chunking settings"""
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 100
    SEPARATORS = ["\n\n", "\n", " ", ""]

class VectorStoreConfig:
    """Vector database settings"""
    COLLECTION_NAME = "multimodal_documents"
    SIMILARITY_METRIC = "cosine"
    DB_TYPE = "chroma"  # Options: "chroma", "pinecone", "weaviate"

class CaptioningConfig:
    """Image captioning settings"""
    MODEL = "Salesforce/blip-image-captioning-base"
    ENABLE_CAPTIONS = False

class RAGConfig:
    """RAG retrieval settings"""
    TOP_K = 5
    SIMILARITY_THRESHOLD = 0.5

# Logging
LOGGING_LEVEL = "INFO"
