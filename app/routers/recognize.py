"""Recognition endpoint for product identification from images."""

from fastapi import APIRouter, UploadFile, File, HTTPException
from typing import Optional
import logging

from app.models.schemas import RecognitionResponse
from app.services.ocr_service import OCRService
from app.services.matcher_service import MatcherService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="", tags=["recognition"])

# Service instances (will be injected via dependency)
ocr_service: Optional[OCRService] = None
matcher_service: Optional[MatcherService] = None


def set_services(ocr: OCRService, matcher: MatcherService):
    """Set service instances for the router."""
    global ocr_service, matcher_service
    ocr_service = ocr
    matcher_service = matcher


@router.post("/recognize", response_model=RecognitionResponse)
async def recognize_product(image: UploadFile = File(...)):
    """
    Recognize a product from an uploaded image.
    
    Extracts text using OCR and matches against product catalog.
    
    Args:
        image: Uploaded image file (JPG, PNG, etc.)
        
    Returns:
        RecognitionResponse with top candidate products
    """
    try:
        # Validate file type
        if not image.content_type or not image.content_type.startswith('image/'):
            raise HTTPException(
                status_code=400,
                detail="Invalid file type. Please upload an image file."
            )
        
        # Read image bytes
        image_bytes = await image.read()
        logger.info(f"Processing image: {image.filename}, size: {len(image_bytes)} bytes")
        
        # Extract text using OCR
        ocr_text = ocr_service.extract_text(image_bytes)
        
        if not ocr_text:
            logger.warning("No text extracted from image")
            return RecognitionResponse(candidates=[], best_product_id=None)
        
        logger.info(f"OCR text: {ocr_text[:100]}")
        
        # Find matching products
        candidates = matcher_service.find_matches(ocr_text)
        
        # Determine best match
        best_product_id = candidates[0].product_id if candidates else None
        
        return RecognitionResponse(
            candidates=candidates,
            best_product_id=best_product_id
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Recognition failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Recognition failed: {str(e)}"
        )


