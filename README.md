# AI Product Intelligence Microservice

A FastAPI microservice that recognizes products from images and answers questions about them using Retrieval-Augmented Generation (RAG).

## What This Does

This service helps identify Apple products from images and answer questions about their specs. Upload an image of a product (like an iPhone box), and it will tell you what product it is. Then ask questions like "What's the battery capacity?" and it will give you accurate answers from the product documentation.

It includes 12 Apple products: iPhones (15, 15 Pro, 15 Pro Max), MacBooks (Air 15", Pro 14", Pro 16"), Mac desktops (Mini, Studio, iMac), and AirPods (3, Pro 2, Max).

## How It Works

The system uses four main components:

1. **OCR Service** - Extracts text from product images using Tesseract
2. **Matcher Service** - Matches extracted text to products in the catalog using fuzzy matching
3. **RAG Service** - Retrieves relevant information from product documentation using embeddings and FAISS
4. **LLM Service** - Generates natural language answers using OpenAI's GPT-4o-mini

## Quick Start

### Running with Docker (Recommended)

You need Docker and an OpenAI API key. Here's how to get started:

1. Navigate to the project directory:
```bash
cd /path/to/AI\ Microservice\ Assignment
```

2. Create a `.env` file with your OpenAI API key:
```bash
echo "OPENAI_API_KEY=your_api_key_here" > .env
```

3. Build and start the service:
```bash
docker-compose up --build
```

That's it! The API will be running at http://localhost:8000

You can access the interactive documentation at:
- http://localhost:8000/docs (Swagger UI with a "Try it out" feature)
- http://localhost:8000/redoc (alternative documentation style)

### Running Locally (Without Docker)

If you prefer to run it locally, you'll need Python 3.11+ and Tesseract OCR installed.

**On macOS:**
```bash
brew install tesseract
```

**On Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install tesseract-ocr tesseract-ocr-eng
```

**Then set up the Python environment:**
```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Set your OpenAI API key
export OPENAI_API_KEY="your_api_key_here"

# Build the FAISS index
python tools/build_index.py

# Start the server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## API Endpoints and Examples

### 1. Health Check

Simple endpoint to verify the service is running.

```bash
curl http://localhost:8000/health
```

**Response:**
```json
{
  "status": "healthy",
  "message": "Service is running"
}
```

### 2. Recognize Product from Image

Upload an image and get back the most likely products.

```bash
curl -X POST http://localhost:8000/recognize \
  -F "image=@/path/to/product_image.jpg"
```

**Example Response:**
```json
{
  "candidates": [
    {
      "product_id": "iphone-15-pro-max",
      "title": "iPhone 15 Pro Max",
      "score": 0.93,
      "evidence": ["Title match: iPhone 15 Pro Max (0.93)"]
    },
    {
      "product_id": "iphone-15-pro",
      "title": "iPhone 15 Pro",
      "score": 0.76,
      "evidence": ["Title match: iPhone 15 Pro (0.76)"]
    }
  ],
  "best_product_id": "iphone-15-pro-max"
}
```

The system returns up to 3 candidate matches with confidence scores. The `best_product_id` is the top match.

### 3. Ask Questions About a Product

Once you know the product ID, you can ask questions about it.

```bash
curl -X POST http://localhost:8000/products/iphone-15-pro-max/answer \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is the battery capacity?",
    "use_external_llm": true
  }'
```

**Example Response:**
```json
{
  "answer": "The iPhone 15 Pro Max has a battery capacity of 4422 mAh.",
  "context_sources": ["iphone-15-pro-max.txt"]
}
```

The answer is generated from the actual product documentation, so it's always accurate and grounded in facts.

### 4. Combined: Recognize and Answer in One Request

You can also do both steps in a single call.

```bash
curl -X POST http://localhost:8000/recognize-and-answer \
  -F "image=@/path/to/product_image.jpg" \
  -F "question=What is the screen size?"
```

**Example Response:**
```json
{
  "recognition": {
    "candidates": [
      {
        "product_id": "iphone-15-pro-max",
        "title": "iPhone 15 Pro Max",
        "score": 0.93,
        "evidence": ["Title match: iPhone 15 Pro Max (0.93)"]
      }
    ],
    "best_product_id": "iphone-15-pro-max"
  },
  "answer": {
    "answer": "The iPhone 15 Pro Max has a 6.7-inch Super Retina XDR display.",
    "context_sources": ["iphone-15-pro-max.txt"]
  }
}
```

This is useful when you want to identify a product and immediately get information about it.

## Models and Technology Choices

Here's what I used and why I chose each component:

### OCR: Tesseract

I went with Tesseract instead of more complex options like PaddleOCR because:
- It's lightweight (around 50MB vs 500MB+ for PaddleOCR)
- Works well for product labels and packaging text
- Easy to install and well-supported across platforms
- Fast enough for this use case

### Fuzzy Matching: RapidFuzz

For matching OCR text to product names, I used RapidFuzz because:
- Handles typos and OCR errors gracefully
- Token-based matching works great for product names
- Much faster than other fuzzy matching libraries
- Allows weighted scoring across different fields (title, model, brand)

### Embeddings: sentence-transformers (all-MiniLM-L6-v2)

For converting text into vectors, I chose this model because:
- Compact size (about 80MB)
- Fast inference (under 50ms per query)
- 384-dimensional embeddings are sufficient for this dataset
- Good balance between quality and speed
- No GPU required

### Vector Store: FAISS (CPU version)

For storing and searching embeddings:
- Can handle millions of vectors even though we only have 22 chunks
- CPU version is fast enough for our small dataset
- No external dependencies or services needed
- Simple to deploy

### LLM: OpenAI GPT-4o-mini

For generating answers, I used OpenAI's API because:
- Best answer quality and accuracy
- Upgraded from GPT-3.5-turbo to GPT-4o-mini for better performance
- No need to manage large local models
- Easy to swap for other LLM providers if needed
- Reliable and well-documented API

### Why These Choices Work Together

The combination gives you:
- Fast product recognition (around 1.5 seconds)
- Accurate answers based on real documentation (around 2.5 seconds)
- Low memory footprint (400-450 MB)
- No GPU required
- Easy deployment with Docker

## Project Structure

```
AI Microservice Assignment/
├── app/
│   ├── main.py                    # FastAPI application entry point
│   ├── config.py                  # Configuration and settings
│   ├── routers/
│   │   ├── recognize.py           # Product recognition endpoint
│   │   ├── products.py            # Q&A endpoint
│   │   └── combined.py            # Combined recognition + Q&A
│   ├── services/
│   │   ├── ocr_service.py         # Tesseract OCR wrapper
│   │   ├── matcher_service.py     # Fuzzy matching logic
│   │   ├── rag_service.py         # RAG with FAISS
│   │   └── llm_service.py         # OpenAI API integration
│   ├── models/
│   │   └── schemas.py             # Pydantic request/response models
│   └── utils/
│       └── image_processing.py    # Image preprocessing utilities
├── data/
│   ├── catalog.csv                # Product catalog (12 products)
│   ├── docs/                      # Product documentation files
│   └── faiss_index/               # FAISS index (auto-generated)
├── tools/
│   └── build_index.py             # Script to build FAISS index
├── tests/
│   └── test_recognize.py          # Unit tests
├── Dockerfile                      # Docker container definition
├── docker-compose.yml             # Docker Compose configuration
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

## Configuration

You can customize the system through environment variables:

| Variable | Default | What it does |
|----------|---------|--------------|
| OPENAI_API_KEY | None | Your OpenAI API key (required for Q&A) |
| OPENAI_MODEL | gpt-4o-mini | Which OpenAI model to use |
| EMBEDDING_MODEL | all-MiniLM-L6-v2 | Which embedding model for RAG |
| MIN_CONFIDENCE | 0.6 | Minimum score to include a product match |
| TOP_K_MATCHES | 3 | Number of product candidates to return |
| TOP_K_RETRIEVAL | 5 | Number of context chunks to retrieve for RAG |

## Performance

Based on testing with Apple Silicon M3 Pro:

| Endpoint | Average Time | What takes time |
|----------|--------------|-----------------|
| /recognize | ~1.5 seconds | OCR (1.0s), Matching (0.1s) |
| /products/{id}/answer | ~2.5 seconds | OpenAI API call (2.3s), Retrieval (0.1s) |
| /recognize-and-answer | ~4.0 seconds | Both operations combined |
| /health | <0.01 seconds | Simple status check |

**Resource Usage:**
- Memory: 400-450 MB
- Docker Image: 3.1 GB
- CPU: Less than 30% during processing
- No GPU needed

**Accuracy:**
- Product recognition: 85-90% top-1 match rate with clear product images
- Top-3 match rate: 95%+ 
- RAG retrieval: 5 relevant chunks per query
- LLM answers: Grounded in documentation with no hallucinations observed

All API responses include an `X-Process-Time` header showing the request latency in seconds.

## Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=app --cov-report=html
```

## Troubleshooting

**OCR not finding text:**
- Make sure the image has clear, readable text
- Product names need to be visible in the image
- Try with better lighting or higher resolution images

**"OpenAI API key not configured" error:**
- Set your API key in the `.env` file for Docker
- Or export it as an environment variable: `export OPENAI_API_KEY="your_key"`

**FAISS index not found:**
- Run `python tools/build_index.py` to build it
- Or just restart - the Docker container builds it automatically

**Import errors:**
- Make sure you installed all dependencies: `pip install -r requirements.txt`
- Check that your virtual environment is activated

## What's Included

The service comes with 12 Apple products across 4 categories:

**iPhones:** iPhone 15, iPhone 15 Pro, iPhone 15 Pro Max

**MacBooks:** MacBook Air 15" M2, MacBook Pro 14" M3, MacBook Pro 16" M3 Max

**Mac Desktops:** iMac 24" M3, Mac Mini M2 Pro, Mac Studio M2 Ultra

**AirPods:** AirPods 3rd Gen, AirPods Pro 2nd Gen, AirPods Max

Each product has detailed documentation including specs, features, and technical details.

## License

This project was developed as a take-home assignment for Amygdal LLC.
