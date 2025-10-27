"""Product matching service using fuzzy string matching."""

import pandas as pd
from rapidfuzz import fuzz
from typing import List
import logging

from app.config import settings
from app.models.schemas import ProductCandidate

logger = logging.getLogger(__name__)


class MatcherService:
    """Service for matching OCR text to products in catalog."""
    
    def __init__(self):
        """Initialize matcher service by loading catalog."""
        self.catalog = self._load_catalog()
        logger.info(f"Matcher service initialized with {len(self.catalog)} products")
    
    def _load_catalog(self) -> pd.DataFrame:
        """
        Load product catalog from CSV.
        
        Returns:
            DataFrame with product catalog
        """
        try:
            catalog = pd.read_csv(settings.catalog_path)
            logger.info(f"Loaded catalog with {len(catalog)} products")
            return catalog
        except Exception as e:
            logger.error(f"Failed to load catalog: {e}")
            return pd.DataFrame()
    
    def find_matches(self, ocr_text: str, top_k: int = None) -> List[ProductCandidate]:
        """
        Find matching products based on OCR text.
        
        Args:
            ocr_text: Text extracted from image
            top_k: Number of top matches to return (defaults to settings.top_k_matches)
            
        Returns:
            List of ProductCandidate objects sorted by score
        """
        if top_k is None:
            top_k = settings.top_k_matches
        
        if not ocr_text or self.catalog.empty:
            logger.warning("Empty OCR text or catalog")
            return []
        
        # Normalize OCR text
        ocr_text_normalized = ocr_text.lower().strip()
        
        candidates = []
        
        for _, product in self.catalog.iterrows():
            # Calculate scores for different fields
            title_score = fuzz.token_set_ratio(ocr_text_normalized, str(product['title']).lower()) / 100.0
            model_score = fuzz.token_set_ratio(ocr_text_normalized, str(product['model']).lower()) / 100.0
            brand_score = fuzz.token_set_ratio(ocr_text_normalized, str(product['brand']).lower()) / 100.0
            
            # Calculate weighted combined score
            combined_score = (
                title_score * settings.title_weight +
                model_score * settings.model_weight +
                brand_score * settings.brand_weight
            )
            
            # Generate evidence list
            evidence = []
            if title_score > 0.6:
                evidence.append(f"Title match: {product['title']} ({title_score:.2f})")
            if model_score > 0.6:
                evidence.append(f"Model match: {product['model']} ({model_score:.2f})")
            if brand_score > 0.6:
                evidence.append(f"Brand match: {product['brand']} ({brand_score:.2f})")
            
            # Only include if above minimum confidence
            if combined_score >= settings.min_confidence:
                candidate = ProductCandidate(
                    product_id=product['product_id'],
                    title=product['title'],
                    score=round(combined_score, 3),
                    evidence=evidence if evidence else [f"OCR: {ocr_text[:50]}"]
                )
                candidates.append(candidate)
        
        # Sort by score descending and return top-k
        candidates.sort(key=lambda x: x.score, reverse=True)
        top_candidates = candidates[:top_k]
        
        logger.info(f"Found {len(top_candidates)} matching products (min score: {settings.min_confidence})")
        
        return top_candidates
    
    def validate_product_id(self, product_id: str) -> bool:
        """
        Check if product ID exists in catalog.
        
        Args:
            product_id: Product identifier
            
        Returns:
            True if product exists, False otherwise
        """
        return product_id in self.catalog['product_id'].values
    
    def get_product_info(self, product_id: str) -> dict:
        """
        Get product information by ID.
        
        Args:
            product_id: Product identifier
            
        Returns:
            Dictionary with product information or None
        """
        product = self.catalog[self.catalog['product_id'] == product_id]
        if not product.empty:
            return product.iloc[0].to_dict()
        return None


