"""LLM service for generating answers using OpenAI API."""

from openai import OpenAI
from typing import List
import logging

from app.config import settings

logger = logging.getLogger(__name__)


class LLMService:
    """Service for generating answers using OpenAI's LLM."""
    
    def __init__(self, api_key: str = None):
        """
        Initialize LLM service.
        
        Args:
            api_key: OpenAI API key (defaults to settings.openai_api_key)
        """
        self.api_key = api_key or settings.openai_api_key
        
        if not self.api_key:
            logger.warning("No OpenAI API key provided. LLM service will not be functional.")
            self.client = None
        else:
            self.client = OpenAI(api_key=self.api_key)
            logger.info("LLM service initialized with OpenAI")
    
    def generate_answer(self, question: str, context_chunks: List[str]) -> str:
        """
        Generate an answer to a question based on context chunks.
        
        Args:
            question: User's question
            context_chunks: List of relevant text chunks for context
            
        Returns:
            Generated answer
            
        Raises:
            ValueError: If API key is not configured
            Exception: For API errors
        """
        if not self.client:
            raise ValueError(
                "OpenAI API key not configured. Please set OPENAI_API_KEY environment variable."
            )
        
        try:
            # Combine context chunks
            context = "\n\n".join(context_chunks)
            
            # Construct prompt with enhanced instructions for accuracy
            system_prompt = (
                "You are a technical product expert specializing in Apple products. "
                "Answer questions using ONLY the information provided in the context below.\n\n"
                "Critical Rules:\n"
                "1. Quote exact specifications with proper units (mAh, inches, GB, cores, Hz, nits)\n"
                "2. If the context doesn't contain the answer, respond: 'This information is not specified in the documentation'\n"
                "3. Never make assumptions, estimates, or use external knowledge\n"
                "4. For numerical specs, use the exact values from the context\n"
                "5. Keep answers concise but complete - include all relevant details from context\n"
                "6. If multiple variants exist, clarify which one you're describing\n\n"
                "Format your answer clearly and professionally."
            )
            
            user_prompt = f"Context:\n{context}\n\nQuestion: {question}"
            
            logger.debug(f"Generating answer for question: {question[:100]}")
            
            # Call OpenAI API
            response = self.client.chat.completions.create(
                model=settings.openai_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.2,  # Lower temperature for more factual, consistent responses
                max_tokens=400,  # Increased to allow more detailed answers
                top_p=1.0,
                frequency_penalty=0.0,
                presence_penalty=0.0
            )
            
            answer = response.choices[0].message.content.strip()
            logger.info(f"Generated answer: {len(answer)} characters")
            
            return answer
            
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            
            # Provide helpful error messages
            if "invalid_api_key" in str(e).lower():
                raise Exception(
                    "Invalid OpenAI API key. Please check your OPENAI_API_KEY environment variable."
                )
            elif "rate_limit" in str(e).lower():
                raise Exception(
                    "OpenAI API rate limit exceeded. Please try again later."
                )
            else:
                raise Exception(f"Failed to generate answer: {str(e)}")


