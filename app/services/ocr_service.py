"""OCR service for text extraction from images."""

import pytesseract
from PIL import Image
import logging
import re

from app.utils.image_processing import load_image_from_bytes, preprocess_for_ocr

logger = logging.getLogger(__name__)


class OCRService:
    """Service for extracting text from images using Tesseract OCR."""
    
    def __init__(self):
        """Initialize OCR service."""
        logger.info("OCR service initialized")
    
    def extract_text(self, image_bytes: bytes) -> str:
        """
        Extract text from image bytes.
        
        Args:
            image_bytes: Raw image bytes
            
        Returns:
            Extracted and cleaned text
        """
        try:
            # Load image from bytes
            image = load_image_from_bytes(image_bytes)
            logger.debug(f"Image loaded: {image.size}, mode: {image.mode}")
            
            # Preprocess image for better OCR results
            processed_image = preprocess_for_ocr(image)
            
            # Extract text using Tesseract
            text = pytesseract.image_to_string(processed_image, config='--psm 6')
            logger.info(f"OCR extracted {len(text)} characters")
            
            # Clean and normalize text
            cleaned_text = self._clean_text(text)
            logger.debug(f"Cleaned text: {cleaned_text[:100]}...")
            
            return cleaned_text
            
        except Exception as e:
            logger.error(f"OCR extraction failed: {e}")
            return ""
    
    def _clean_text(self, text: str) -> str:
        """
        Clean and normalize extracted text.
        
        Args:
            text: Raw extracted text
            
        Returns:
            Cleaned text
        """
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep alphanumeric and basic punctuation
        text = re.sub(r'[^\w\s\-.,]', '', text)
        
        # Strip leading/trailing whitespace
        text = text.strip()
        
        return text


