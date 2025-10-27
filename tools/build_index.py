"""Standalone script to build FAISS index from product documents."""

import sys
from pathlib import Path

# Add parent directory to path to import app modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.services.rag_service import RAGService
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Build FAISS index from documents."""
    logger.info("Starting FAISS index build process...")
    
    try:
        # Initialize RAG service (will build index if not exists)
        rag_service = RAGService()
        
        # Force rebuild
        logger.info("Building new index...")
        rag_service.build_index()
        
        logger.info("Index build completed successfully!")
        return 0
        
    except Exception as e:
        logger.error(f"Index build failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())


