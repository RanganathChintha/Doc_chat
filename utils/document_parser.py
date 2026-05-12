import pymupdf
from PIL import Image
from pathlib import Path
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)

class DocumentParser:
    """Extract text and images from documents"""
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def extract_from_pdf(self, pdf_path: str) -> List[Dict]:
        """
        Extract text and images from PDF maintaining spatial relationships
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            List of page data with text and images
        """
        logger.info(f"Parsing PDF: {pdf_path}")
        doc = pymupdf.open(pdf_path)
        chunks = []
        
        for page_num, page in enumerate(doc):
            logger.debug(f"Processing page {page_num + 1}")
            
            try:
                # Extract text
                text = page.get_text()
                
                # Extract images
                images_list = []
                for img_index, img_ref in enumerate(page.get_images(full=True)):
                    xref = img_ref[0]
                    pix = pymupdf.Pixmap(doc, xref)
                    
                    # Save image
                    img_filename = f"page_{page_num}_img_{img_index}.png"
                    img_path = self.output_dir / img_filename
                    pix.save(str(img_path))
                    
                    # Get bounding box
                    img_bbox = page.get_image_bbox(img_ref)
                    
                    images_list.append({
                        "path": str(img_path),
                        "filename": img_filename,
                        "bbox": img_bbox,
                        "page": page_num,
                        "index": img_index
                    })
                
                chunks.append({
                    "page": page_num,
                    "text": text,
                    "images": images_list,
                    "source": pdf_path
                })
                
                logger.debug(f"Page {page_num}: Found {len(images_list)} images")
                
            except Exception as e:
                logger.error(f"Error processing page {page_num}: {e}")
        
        doc.close()
        logger.info(f"Successfully extracted {len(chunks)} pages")
        return chunks
    
    def extract_from_docx(self, docx_path: str) -> List[Dict]:
        """Extract from Word documents (implement similarly)"""
        # Similar structure for DOCX files
        logger.warning("DOCX extraction not implemented yet")
        return []
    
    def extract_from_pptx(self, pptx_path: str) -> List[Dict]:
        """Extract from PowerPoint (implement similarly)"""
        logger.warning("PPTX extraction not implemented yet")
        return []