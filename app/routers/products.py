"""Products endpoint for answering questions about products."""

from fastapi import APIRouter, HTTPException
from typing import Optional
import logging

from app.models.schemas import AnswerRequest, AnswerResponse
from app.services.rag_service import RAGService
from app.services.llm_service import LLMService
from app.services.matcher_service import MatcherService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/products", tags=["products"])

# Service instances (will be injected via dependency)
rag_service: Optional[RAGService] = None
llm_service: Optional[LLMService] = None
matcher_service: Optional[MatcherService] = None


def set_services(rag: RAGService, llm: LLMService, matcher: MatcherService):
    """Set service instances for the router."""
    global rag_service, llm_service, matcher_service
    rag_service = rag
    llm_service = llm
    matcher_service = matcher


@router.post("/{product_id}/answer", response_model=AnswerResponse)
async def answer_question(product_id: str, request: AnswerRequest):
    """
    Answer a question about a specific product using RAG.
    
    Args:
        product_id: Product identifier
        request: Question and LLM usage preference
        
    Returns:
        AnswerResponse with generated answer and sources
    """
    try:
        # Validate product exists
        if not matcher_service.validate_product_id(product_id):
            raise HTTPException(
                status_code=404,
                detail=f"Product not found: {product_id}"
            )
        
        logger.info(f"Answering question for product: {product_id}")
        
        # Retrieve relevant context chunks
        context_chunks_data = rag_service.retrieve(
            query=request.question,
            product_id=product_id,
            top_k=3
        )
        
        if not context_chunks_data:
            return AnswerResponse(
                answer="No relevant information found in the product documentation.",
                context_sources=[]
            )
        
        # Extract text and sources
        context_texts = [chunk['text'] for chunk in context_chunks_data]
        context_sources = list(set([chunk['source'] for chunk in context_chunks_data]))
        
        logger.info(f"Retrieved {len(context_texts)} context chunks from {len(context_sources)} sources")
        
        # Generate answer using LLM
        if request.use_external_llm:
            try:
                answer = llm_service.generate_answer(request.question, context_texts)
            except Exception as e:
                logger.error(f"LLM generation failed: {e}")
                raise HTTPException(
                    status_code=500,
                    detail=str(e)
                )
        else:
            # Fallback: return context without LLM processing
            answer = f"Based on the documentation: {context_texts[0][:300]}..."
        
        return AnswerResponse(
            answer=answer,
            context_sources=context_sources
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Answer generation failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate answer: {str(e)}"
        )


