"""Image preprocessing utilities for OCR."""

from io import BytesIO
from PIL import Image, ImageEnhance, ImageOps
import logging

logger = logging.getLogger(__name__)


def load_image_from_bytes(image_bytes: bytes) -> Image.Image:
    """
    Load an image from bytes.
    
    Args:
        image_bytes: Raw image bytes
        
    Returns:
        PIL Image object
        
    Raises:
        ValueError: If image cannot be loaded
    """
    try:
        image = Image.open(BytesIO(image_bytes))
        return image
    except Exception as e:
        logger.error(f"Failed to load image: {e}")
        raise ValueError(f"Invalid image format: {e}")


def preprocess_for_ocr(image: Image.Image) -> Image.Image:
    """
    Preprocess image for better OCR results.
    
    Steps:
    1. Convert to grayscale
    2. Resize if too large (max 2000px)
    3. Enhance contrast
    4. Auto-adjust levels
    
    Args:
        image: PIL Image object
        
    Returns:
        Preprocessed PIL Image object
    """
    try:
        # Convert to RGB first (handles RGBA, P, etc.)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize if too large to reduce processing time
        max_dimension = 2000
        if max(image.size) > max_dimension:
            ratio = max_dimension / max(image.size)
            new_size = tuple(int(dim * ratio) for dim in image.size)
            image = image.resize(new_size, Image.Resampling.LANCZOS)
            logger.info(f"Resized image to {new_size}")
        
        # Convert to grayscale
        image = ImageOps.grayscale(image)
        
        # Enhance contrast
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(1.5)
        
        # Auto-adjust levels (equalize)
        image = ImageOps.autocontrast(image)
        
        logger.debug("Image preprocessing completed")
        return image
        
    except Exception as e:
        logger.error(f"Image preprocessing failed: {e}")
        # Return original image if preprocessing fails
        return image


