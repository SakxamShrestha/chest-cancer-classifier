import os
import sys
from pathlib import Path
import logging
import datetime
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import torch

# Setup logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Get absolute paths
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent.parent.parent.parent
MODEL_PATH = PROJECT_ROOT / "src" / "models" / "best_model.pth"

# Add project root to Python path
sys.path.append(str(PROJECT_ROOT))

logger.debug(f"Current directory: {CURRENT_DIR}")
logger.debug(f"Project root: {PROJECT_ROOT}")
logger.debug(f"Model path: {MODEL_PATH}")
logger.debug(f"Python path: {sys.path}")

try:
    from app.routers.models.model import PredictionPipeline
    logger.debug("Successfully imported PredictionPipeline")
except ImportError as e:
    logger.error(f"Failed to import PredictionPipeline: {e}")
    raise

# Global pipeline variable
pipeline = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline
    try:
        logger.info(f"Current working directory: {os.getcwd()}")
        logger.info(f"Looking for model at: {MODEL_PATH}")
        logger.info(f"Model file exists: {MODEL_PATH.exists()}")
        
        if not MODEL_PATH.exists():
            logger.error(f"Model file not found at: {MODEL_PATH}")
            pipeline = None
            yield
            return

        logger.info("Loading model...")
        pipeline = PredictionPipeline(str(MODEL_PATH))
        logger.info("Model loaded successfully")
        
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        pipeline = None
    
    yield
    
    if pipeline is not None:
        del pipeline
        logger.info("Model cleaned up")

app = FastAPI(lifespan=lifespan)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_status": "loaded" if pipeline is not None else "not loaded",
        "model_path": str(MODEL_PATH),
        "model_exists": MODEL_PATH.exists(),
        "timestamp": datetime.datetime.now().isoformat()
    }

# Include routers
from app.routers.predict import router as predict_router
app.include_router(predict_router, prefix="/api")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)