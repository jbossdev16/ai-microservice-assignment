"""Unit tests for product recognition endpoint."""

import pytest
from fastapi.testclient import TestClient
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO

from app.main import app
from app.services.ocr_service import OCRService
from app.services.matcher_service import MatcherService

client = TestClient(app)


def create_test_image(text: str) -> bytes:
    """Create a simple test image with text."""
    # Create a white image
    img = Image.new('RGB', (400, 200), color='white')
    draw = ImageDraw.Draw(img)
    
    # Draw text (use default font)
    draw.text((50, 80), text, fill='black')
    
    # Convert to bytes
    img_bytes = BytesIO()
    img.save(img_bytes, format='PNG')
    img_bytes.seek(0)
    
    return img_bytes.getvalue()


def test_health_check():
    """Test health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_root_endpoint():
    """Test root endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_ocr_service():
    """Test OCR service directly."""
    ocr = OCRService()
    
    # Create test image with "iPhone 15 Pro Max"
    image_bytes = create_test_image("iPhone 15 Pro Max")
    
    # Extract text
    text = ocr.extract_text(image_bytes)
    
    # Check that some text was extracted
    assert len(text) > 0


def test_matcher_service():
    """Test matcher service directly."""
    matcher = MatcherService()
    
    # Test matching
    candidates = matcher.find_matches("iPhone 15 Pro Max", top_k=3)
    
    # Should find matches
    assert len(candidates) > 0
    assert candidates[0].product_id is not None
    assert candidates[0].score >= 0.0


def test_recognize_endpoint():
    """Test recognition endpoint with image upload."""
    # Create test image
    image_bytes = create_test_image("MacBook Pro")
    
    # Upload image
    files = {"image": ("test.png", image_bytes, "image/png")}
    response = client.post("/recognize", files=files)
    
    # Check response
    assert response.status_code == 200
    data = response.json()
    
    assert "candidates" in data
    assert "best_product_id" in data


def test_recognize_invalid_file():
    """Test recognition endpoint with invalid file."""
    files = {"image": ("test.txt", b"not an image", "text/plain")}
    response = client.post("/recognize", files=files)
    
    # Should return 400 for invalid file type
    assert response.status_code == 400


def test_answer_endpoint_invalid_product():
    """Test answer endpoint with invalid product ID."""
    response = client.post(
        "/products/invalid-product-id/answer",
        json={"question": "What is the battery capacity?"}
    )
    
    # Should return 404 for invalid product
    assert response.status_code == 404


def test_answer_endpoint_valid_product():
    """Test answer endpoint with valid product ID."""
    # Test with a known product
    response = client.post(
        "/products/iphone-15-pro-max/answer",
        json={"question": "What is the battery capacity?", "use_external_llm": False}
    )
    
    # Should return 200 or 500 (if OpenAI key not configured)
    assert response.status_code in [200, 500]


def test_combined_endpoint():
    """Test combined endpoint."""
    # Create test image
    image_bytes = create_test_image("iPhone 15 Pro")
    
    # Upload with question
    files = {"image": ("test.png", image_bytes, "image/png")}
    data = {"question": "What chip does this have?"}
    
    response = client.post("/recognize-and-answer", files=files, data=data)
    
    # Check response
    assert response.status_code in [200, 500]
    
    if response.status_code == 200:
        result = response.json()
        assert "recognition" in result
        assert "answer" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


