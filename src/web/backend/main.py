# src/web/backend/main.py
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import torch
import sys
import os
from pathlib import Path
from PIL import Image
import io
import logging
import datetime
from contextlib import asynccontextmanager
from model import PredictionPipeline

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))

# Global pipeline variable
pipeline = None

# Lifespan context manager
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load model on startup
    global pipeline
    try:
        model_path = os.path.join(project_root, "src", "models", "best_model.pth")
        
        # Check if model file exists
        if not os.path.exists(model_path):
            logger.error(f"Model file not found at: {model_path}")
            logger.info("Please make sure you have trained and saved the model first.")
            pipeline = None
            yield
            return

        # Load model
        logger.info(f"Loading model from: {model_path}")
        pipeline = PredictionPipeline(model_path)
        logger.info("Model loaded successfully")
        
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        pipeline = None
    
    yield
    
    # Cleanup on shutdown
    if pipeline is not None:
        del pipeline
        logger.info("Model cleaned up")

app = FastAPI(
    title="Chest Cancer Classifier API",
    description="API for classifying chest CT scan images",
    version="1.0.0",
    lifespan=lifespan
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    """Root endpoint to check API status."""
    return {
        "status": "online",
        "model_loaded": pipeline is not None,
        "message": "Welcome to the Chest Cancer Classifier API"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_status": "loaded" if pipeline is not None else "not loaded",
        "timestamp": datetime.datetime.now().isoformat()
    }

@app.get("/model-info")
async def model_info():
    """Get detailed information about the model status."""
    model_path = os.path.join(project_root, "models", "best_model.pth")
    return {
        "status": "online",
        "model_loaded": pipeline is not None,
        "model_path": str(model_path),
        "model_exists": os.path.exists(model_path),
        "project_root": str(project_root),
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "message": "Model is ready for predictions" if pipeline is not None else "Model not loaded - please train model first"
    }

@app.get("/classes")
async def get_classes():
    """Get the list of available classes for prediction."""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "status": "success",
        "classes": pipeline.class_names
    }

@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    """
    Predict the class of a chest CT scan image.
    
    Parameters:
    - file: UploadFile - The image file to classify
    
    Returns:
    - JSON object containing prediction results
    """
    # Check if model is loaded
    if pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please ensure the model is properly trained and saved."
        )
    
    # Validate file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="Invalid file type. Please upload an image file."
        )
    
    try:
        # Read image file
        contents = await file.read()
        
        # Make prediction
        result = pipeline.predict(contents)
        
        # Return results
        return {
            "status": "success",
            "filename": file.filename,
            "prediction": result["predicted_class"],
            "confidence": result["confidence"],
            "class_probabilities": result["class_probabilities"],
            "timestamp": datetime.datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing image: {str(e)}"
        )

@app.get("/system-info")
async def system_info():
    """Get system information including GPU availability."""
    gpu_info = {
        "available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None
    }
    
    return {
        "status": "success",
        "gpu_info": gpu_info,
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "timestamp": datetime.datetime.now().isoformat()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.web.backend.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        reload_dirs=[str(project_root)]
    )