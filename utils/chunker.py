from dataclasses import dataclass, field
from typing import List, Dict
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import ChunkingConfig
import logging

logger = logging.getLogger(__name__)

@dataclass
class Chunk:
    """Represent a chunk with text and images"""
    text: str
    images: List[str]
    image_captions: List[str] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)
    chunk_id: str = ""

class MultimodalChunker:
    """Intelligently chunk documents preserving image-text relationships"""
    
    def __init__(self, 
                 chunk_size: int = ChunkingConfig.CHUNK_SIZE,
                 overlap: int = ChunkingConfig.CHUNK_OVERLAP):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            separators=ChunkingConfig.SEPARATORS
        )
        self.chunk_counter = 0
    
    def chunk_document(self, parsed_doc: List[Dict]) -> List[Chunk]:
        """
        Chunk document while maintaining image-text proximity
        
        Args:
            parsed_doc: Output from DocumentParser
            
        Returns:
            List of Chunk objects
        """
        logger.info("Starting document chunking")
        chunks = []
        
        for page_data in parsed_doc:
            chunks.extend(
                self._chunk_page(page_data)
            )
        
        logger.info(f"Created {len(chunks)} chunks from document")
        return chunks
    
    def _chunk_page(self, page_data: Dict) -> List[Chunk]:
        """Chunk a single page"""
        text = page_data["text"]
        images = page_data["images"]
        page_num = page_data["page"]
        
        # Split text into chunks. For scanned/image-only pages, keep a page-level
        # placeholder so downstream embedding/vector storage never receives empty input.
        text_chunks = self.text_splitter.split_text(text) if text.strip() else []
        if not text_chunks and images:
            text_chunks = [
                f"Page {page_num + 1} contains {len(images)} extracted image(s), but no extractable text."
            ]
        elif not text_chunks:
            text_chunks = [
                f"Page {page_num + 1} has no extractable text or embedded images. OCR may be required."
            ]
        
        page_chunks = []
        num_chunks = len(text_chunks)
        
        for idx, text_chunk in enumerate(text_chunks):
            # Associate images with chunks
            # Strategy: assign images to chunks they appear near
            chunk_images = self._assign_images_to_chunk(
                idx, num_chunks, images
            )
            
            chunk = Chunk(
                text=text_chunk,
                images=chunk_images,
                image_captions=[],
                metadata={
                    "page": page_num,
                    "chunk_index": idx,
                    "source": page_data.get("source", "unknown")
                },
                chunk_id=f"chunk_{self.chunk_counter}"
            )
            
            page_chunks.append(chunk)
            self.chunk_counter += 1
        
        return page_chunks
    
    def _assign_images_to_chunk(self, 
                                chunk_idx: int, 
                                total_chunks: int, 
                                images: List[Dict]) -> List[str]:
        """
        Assign images to chunks based on spatial proximity
        """
        if not images:
            return []
        
        # Simple strategy: assign all images from page to all chunks
        # Advanced strategy: use bounding boxes to calculate proximity
        return [img["path"] for img in images]
