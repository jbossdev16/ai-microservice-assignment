"""Configuration settings for the AI Product Intelligence microservice."""

import os
from pathlib import Path
from pydantic_settings import BaseSettings
from pydantic import ConfigDict
from typing import Optional


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    model_config = ConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        protected_namespaces=()  # Allow fields with 'model_' prefix
    )
    
    # OpenAI Configuration
    openai_api_key: Optional[str] = None
    openai_model: str = "gpt-4o-mini"  # Upgraded for better accuracy
    
    # Model Configuration
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    
    # Paths
    base_dir: Path = Path(__file__).parent.parent
    data_dir: Path = base_dir / "data"
    docs_dir: Path = data_dir / "docs"
    faiss_index_dir: Path = data_dir / "faiss_index"
    catalog_path: Path = data_dir / "catalog.csv"
    
    # Matching Configuration
    min_confidence: float = 0.6
    top_k_matches: int = 3
    top_k_retrieval: int = 5  # Increased from 3 for better context coverage
    
    # Chunking Configuration
    chunk_size: int = 300  # words - Increased from 200 for more complete context
    chunk_overlap: int = 75  # words - Increased from 50 to preserve context across chunks
    
    # Scoring Weights
    title_weight: float = 0.5
    model_weight: float = 0.3
    brand_weight: float = 0.2


# Global settings instance
settings = Settings()

# Ensure directories exist
settings.data_dir.mkdir(exist_ok=True)
settings.docs_dir.mkdir(exist_ok=True)
settings.faiss_index_dir.mkdir(exist_ok=True)


