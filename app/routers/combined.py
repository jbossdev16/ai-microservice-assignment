"""Combined endpoint for recognition and Q&A in one call."""

from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from typing import Optional
import logging

from app.models.schemas import CombinedResponse, RecognitionResponse, AnswerResponse
from app.services.ocr_service import OCRService
from app.services.matcher_service import MatcherService
from app.services.rag_service import RAGService
from app.services.llm_service import LLMService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="", tags=["combined"])

# Service instances (will be injected via dependency)
ocr_service: Optional[OCRService] = None
matcher_service: Optional[MatcherService] = None
rag_service: Optional[RAGService] = None
llm_service: Optional[LLMService] = None


def set_services(ocr: OCRService, matcher: MatcherService, rag: RAGService, llm: LLMService):
    """Set service instances for the router."""
    global ocr_service, matcher_service, rag_service, llm_service
    ocr_service = ocr
    matcher_service = matcher
    rag_service = rag
    llm_service = llm


@router.post("/recognize-and-answer", response_model=CombinedResponse)
async def recognize_and_answer(
    image: UploadFile = File(...),
    question: Optional[str] = Form(None)
):
    """
    Recognize product from image and optionally answer a question about it.
    
    Args:
        image: Uploaded image file
        question: Optional question about the product
        
    Returns:
        CombinedResponse with recognition results and optional answer
    """
    try:
        # Validate file type
        if not image.content_type or not image.content_type.startswith('image/'):
            raise HTTPException(
                status_code=400,
                detail="Invalid file type. Please upload an image file."
            )
        
        # Step 1: Recognize product
        logger.info(f"Processing combined request with image: {image.filename}")
        
        image_bytes = await image.read()
        ocr_text = ocr_service.extract_text(image_bytes)
        
        if not ocr_text:
            logger.warning("No text extracted from image")
            return CombinedResponse(
                recognition=RecognitionResponse(candidates=[], best_product_id=None),
                answer=None
            )
        
        candidates = matcher_service.find_matches(ocr_text)
        best_product_id = candidates[0].product_id if candidates else None
        
        recognition = RecognitionResponse(
            candidates=candidates,
            best_product_id=best_product_id
        )
        
        # Step 2: Answer question if provided and product was recognized
        answer_response = None
        
        if question and best_product_id:
            logger.info(f"Answering question for recognized product: {best_product_id}")
            
            try:
                # Retrieve context
                context_chunks_data = rag_service.retrieve(
                    query=question,
                    product_id=best_product_id,
                    top_k=3
                )
                
                if context_chunks_data:
                    context_texts = [chunk['text'] for chunk in context_chunks_data]
                    context_sources = list(set([chunk['source'] for chunk in context_chunks_data]))
                    
                    # Generate answer
                    answer = llm_service.generate_answer(question, context_texts)
                    
                    answer_response = AnswerResponse(
                        answer=answer,
                        context_sources=context_sources
                    )
                else:
                    answer_response = AnswerResponse(
                        answer="No relevant information found in the product documentation.",
                        context_sources=[]
                    )
                    
            except Exception as e:
                logger.error(f"Answer generation failed in combined endpoint: {e}")
                answer_response = AnswerResponse(
                    answer=f"Failed to generate answer: {str(e)}",
                    context_sources=[]
                )
        elif question and not best_product_id:
            answer_response = AnswerResponse(
                answer="Cannot answer question: product not recognized from image.",
                context_sources=[]
            )
        
        return CombinedResponse(
            recognition=recognition,
            answer=answer_response
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Combined endpoint failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Processing failed: {str(e)}"
        )


