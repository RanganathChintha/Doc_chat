from typing import List, Dict, Union
import numpy as np
from PIL import Image
import torch
import logging
from config import EmbeddingConfig

logger = logging.getLogger(__name__)

class MultimodalEmbedder:
    """Create multimodal embeddings for text and images"""
    
    def __init__(self, strategy: str = "clip"):
        """
        Args:
            strategy: "clip", "hybrid", or "concatenate"
        """
        self.strategy = strategy
        logger.info(f"Initializing embedder with strategy: {strategy}")
        
        if strategy == "clip":
            self._init_clip()
        else:
            self._init_sentence_transformer()
    
    def _init_clip(self):
        """Initialize CLIP model"""
        try:
            from transformers import CLIPProcessor, CLIPModel
            self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            logger.info("CLIP model loaded")
        except Exception as e:
            logger.error(f"Failed to load CLIP: {e}")
            raise
    
    def _init_sentence_transformer(self):
        """Initialize Sentence Transformer"""
        try:
            from sentence_transformers import SentenceTransformer
            self.text_encoder = SentenceTransformer(
                EmbeddingConfig.TEXT_MODEL,
                local_files_only=True
            )
            logger.info("Sentence Transformer loaded")
        except Exception as e:
            logger.error(f"Failed to load Sentence Transformer: {e}")
            raise
    
    def embed_chunk_clip(self, chunk) -> np.ndarray:
        """Embed using CLIP (text + images in shared space)"""
        text_input = f"{chunk.text}\nImage captions: {' '.join(chunk.image_captions)}"
        images = [Image.open(img_path) for img_path in chunk.images if img_path]
        
        text_inputs = self.processor(
            text=text_input,
            images=images,
            return_tensors="pt",
            padding=True
        )
        
        with torch.no_grad():
            text_features = self.model.get_text_features(**{
                k: v for k, v in text_inputs.items() if k != 'pixel_values'
            })
            
            if images and 'pixel_values' in text_inputs:
                image_features = self.model.get_image_features(
                    pixel_values=text_inputs['pixel_values']
                )
                combined = (text_features + image_features.mean(dim=0, keepdim=True)) / 2
            else:
                combined = text_features
        
        return combined.cpu().numpy().flatten()
    
    def embed_chunk_concatenate(self, chunk) -> np.ndarray:
        """Embed by concatenating text + captions"""
        full_text = f"{chunk.text}\n\nImage descriptions:\n" + \
                   "\n".join(chunk.image_captions)
        return self.text_encoder.encode(full_text)
    
    def embed_chunks(self, chunks: List) -> List[Dict]:
        """Embed all chunks"""
        logger.info(f"Embedding {len(chunks)} chunks")
        embedded_chunks = []
        
        for idx, chunk in enumerate(chunks):
            try:
                if self.strategy == "clip":
                    embedding = self.embed_chunk_clip(chunk)
                else:
                    embedding = self.embed_chunk_concatenate(chunk)
                
                embedded_chunks.append({
                    "chunk_id": chunk.chunk_id,
                    "text": chunk.text,
                    "images": chunk.images,
                    "captions": chunk.image_captions,
                    "embedding": embedding,
                    "metadata": chunk.metadata
                })
                
                if (idx + 1) % 10 == 0:
                    logger.debug(f"Embedded {idx + 1} chunks")
                    
            except Exception as e:
                logger.error(f"Error embedding chunk {chunk.chunk_id}: {e}")
        
        logger.info(f"Successfully embedded {len(embedded_chunks)} chunks")
        return embedded_chunks
