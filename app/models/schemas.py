"""Pydantic models for request and response validation."""

from typing import List, Optional
from pydantic import BaseModel, Field, field_validator


class ProductCandidate(BaseModel):
    """A candidate product match with confidence score."""
    
    product_id: str = Field(..., description="Unique product identifier")
    title: str = Field(..., description="Product title")
    score: float = Field(..., ge=0.0, le=1.0, description="Confidence score between 0 and 1")
    evidence: List[str] = Field(default_factory=list, description="List of matching evidence")
    
    class Config:
        json_schema_extra = {
            "example": {
                "product_id": "iphone-15-pro-max",
                "title": "iPhone 15 Pro Max",
                "score": 0.93,
                "evidence": ["OCR: iPhone 15 Pro Max", "Brand: Apple"]
            }
        }


class RecognitionResponse(BaseModel):
    """Response from product recognition endpoint."""
    
    candidates: List[ProductCandidate] = Field(..., description="List of candidate products")
    best_product_id: Optional[str] = Field(None, description="ID of the best matching product")
    
    class Config:
        json_schema_extra = {
            "example": {
                "candidates": [
                    {
                        "product_id": "iphone-15-pro-max",
                        "title": "iPhone 15 Pro Max",
                        "score": 0.93,
                        "evidence": ["OCR: iPhone 15 Pro Max"]
                    }
                ],
                "best_product_id": "iphone-15-pro-max"
            }
        }


class AnswerRequest(BaseModel):
    """Request for answering a question about a product."""
    
    question: str = Field(..., min_length=1, description="Natural language question")
    use_external_llm: bool = Field(True, description="Whether to use external LLM (OpenAI)")
    
    @field_validator('question')
    @classmethod
    def question_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Question cannot be empty")
        return v.strip()
    
    class Config:
        json_schema_extra = {
            "example": {
                "question": "What is the battery capacity of this product?",
                "use_external_llm": True
            }
        }


class AnswerResponse(BaseModel):
    """Response with answer to a product question."""
    
    answer: str = Field(..., description="Generated answer")
    context_sources: List[str] = Field(default_factory=list, description="Sources used for context")
    
    class Config:
        json_schema_extra = {
            "example": {
                "answer": "The iPhone 15 Pro Max has a 4422 mAh battery.",
                "context_sources": ["iphone-15-pro-max.txt"]
            }
        }


class CombinedResponse(BaseModel):
    """Combined response with recognition and answer."""
    
    recognition: RecognitionResponse = Field(..., description="Product recognition results")
    answer: Optional[AnswerResponse] = Field(None, description="Answer to question if provided")
    
    class Config:
        json_schema_extra = {
            "example": {
                "recognition": {
                    "candidates": [
                        {
                            "product_id": "iphone-15-pro-max",
                            "title": "iPhone 15 Pro Max",
                            "score": 0.93,
                            "evidence": ["OCR: iPhone 15 Pro Max"]
                        }
                    ],
                    "best_product_id": "iphone-15-pro-max"
                },
                "answer": {
                    "answer": "The iPhone 15 Pro Max has a 4422 mAh battery.",
                    "context_sources": ["iphone-15-pro-max.txt"]
                }
            }
        }


class HealthResponse(BaseModel):
    """Health check response."""
    
    status: str = Field(..., description="Service status")
    message: str = Field(..., description="Status message")
    
    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "message": "Service is running"
            }
        }


