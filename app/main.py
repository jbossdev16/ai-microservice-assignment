"""Main FastAPI application for AI Product Intelligence microservice."""

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import logging
import time

from app.config import settings
from app.services.ocr_service import OCRService
from app.services.matcher_service import MatcherService
from app.services.rag_service import RAGService
from app.services.llm_service import LLMService
from app.routers import recognize, products, combined
from app.models.schemas import HealthResponse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="AI Product Intelligence API",
    description="Product recognition and Q&A using OCR and RAG",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Add latency tracking middleware
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """Track request latency and add to response headers."""
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    
    # Add header for clients
    response.headers["X-Process-Time"] = f"{process_time:.3f}"
    
    # Log for monitoring
    logger.info(f"{request.method} {request.url.path} - {process_time:.3f}s")
    
    return response


# Global service instances
ocr_service = None
matcher_service = None
rag_service = None
llm_service = None


@app.on_event("startup")
async def startup_event():
    """Initialize services on application startup."""
    global ocr_service, matcher_service, rag_service, llm_service
    
    logger.info("Initializing AI Product Intelligence API...")
    
    try:
        # Initialize services
        logger.info("Loading OCR service...")
        ocr_service = OCRService()
        
        logger.info("Loading matcher service...")
        matcher_service = MatcherService()
        
        logger.info("Loading RAG service...")
        rag_service = RAGService()
        
        logger.info("Loading LLM service...")
        llm_service = LLMService()
        
        # Inject services into routers
        recognize.set_services(ocr_service, matcher_service)
        products.set_services(rag_service, llm_service, matcher_service)
        combined.set_services(ocr_service, matcher_service, rag_service, llm_service)
        
        logger.info("All services initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize services: {e}")
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on application shutdown."""
    logger.info("Shutting down AI Product Intelligence API...")


@app.get("/", response_model=HealthResponse)
async def root():
    """Root endpoint."""
    return HealthResponse(
        status="healthy",
        message="AI Product Intelligence API is running"
    )


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        message="Service is running"
    )


# Include routers
app.include_router(recognize.router)
app.include_router(products.router)
app.include_router(combined.router)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


