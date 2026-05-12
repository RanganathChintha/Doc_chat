from PIL import Image
from typing import List
from config import CaptioningConfig
import logging

logger = logging.getLogger(__name__)

class ImageCaptioner:
    """Generate descriptions for images"""
    
    def __init__(self, enable: bool = CaptioningConfig.ENABLE_CAPTIONS):
        self.enable = enable
        
        if enable:
            try:
                from transformers import pipeline
                self.captioner = pipeline(
                    "image-to-text",
                    model=CaptioningConfig.MODEL
                )
                logger.info("Image captioner initialized")
            except Exception as e:
                logger.error(f"Failed to initialize captioner: {e}")
                self.enable = False
    
    def caption_images(self, chunks: List) -> List:
        """Add captions to images in chunks"""
        if not self.enable:
            logger.warning("Image captioning disabled")
            return chunks
        
        logger.info("Generating captions for images")
        
        for chunk in chunks:
            captions = []
            
            for img_path in chunk.images:
                try:
                    image = Image.open(img_path)
                    caption = self.captioner(image)[0]["generated_text"]
                    captions.append(caption)
                    logger.debug(f"Captioned {img_path}")
                    
                except Exception as e:
                    logger.error(f"Error captioning {img_path}: {e}")
                    captions.append("Image caption unavailable")
            
            chunk.image_captions = captions
        
        return chunks