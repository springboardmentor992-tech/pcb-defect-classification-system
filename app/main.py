from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
import uvicorn
import base64
import cv2
import numpy as np

from app.schemas.defect import DefectResult, InferenceResponse, ErrorResponse
from app.services.inference import PCBInferenceEngine
from app.services.utils import read_image_file, encode_image_to_base64
from app.core.config import MODEL_PATH, CLASS_NAMES
from app.core.logging import logger

app = FastAPI(
    title="PCB Defect Detection API",
    description="RESTful API for PCB defect detection and classification",
    version="2.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Set this to specific domains in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global inference engine
inference_engine: Optional[PCBInferenceEngine] = None

@app.on_event("startup")
async def startup_event():
    """Initialize inference engine on startup"""
    global inference_engine
    
    try:
        inference_engine = PCBInferenceEngine(
            model_path=MODEL_PATH,
            class_names=CLASS_NAMES,
            device=None  # Auto-detect
        )
        logger.info("PCB Defect Detection API ready!")
    except Exception as e:
        logger.error(f"Failed to initialize inference engine: {e}")
        # In a real app we might want to shut down, but here we just log
        pass

@app.get("/")
async def root():
    """Root endpoint - API info"""
    return {
        "service": "PCB Defect Detection API",
        "version": "2.0.0",
        "status": "running"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    if inference_engine is None:
        raise HTTPException(status_code=503, detail="Inference engine not initialized")
    
    return {
        "status": "healthy",
        "model_loaded": True,
        "device": str(inference_engine.device),
        "classes": inference_engine.class_names
    }

@app.post("/predict", response_model=InferenceResponse)
async def predict(
    defect_image: UploadFile = File(..., description="PCB image with defects"),
    template_image: UploadFile = File(..., description="Defect-free template PCB image"),
    return_images: bool = Form(True, description="Include base64 encoded images in response")
):
    """
    Main prediction endpoint - accepts two image files
    """
    if inference_engine is None:
        raise HTTPException(status_code=503, detail="Inference engine not initialized")
    
    try:
        # Read images
        logger.info(f"Received files: {defect_image.filename}, {template_image.filename}")
        defect_img = read_image_file(defect_image)
        template_img = read_image_file(template_image)
        
        # Process
        result = inference_engine.process_images(
            defect_img, 
            template_img,
            defect_image.filename,
            template_image.filename
        )
        
        # Prepare response
        defects = [
            DefectResult(
                id=d['id'],
                class_name=d['class'],
                confidence=round(d['confidence'], 4),
                bbox=list(d['bbox'])
            )
            for d in result['defects']
        ]
        
        response = InferenceResponse(
            success=True,
            image_name=result['image_name'],
            template_name=result['template_name'],
            num_defects=result['num_defects'],
            defects=defects,
            processing_time=round(result['processing_time'], 3)
        )
        
        # Add images if requested
        if return_images:
            response.annotated_image_base64 = encode_image_to_base64(result['annotated_image'])
            response.threshold_image_base64 = encode_image_to_base64(result['threshold_image'])
        
        logger.info(f"Successfully processed {defect_image.filename}: {result['num_defects']} defects found")
        
        return response
        
    except ValueError as e:
        logger.warning(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Prediction failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

# This endpoint is kept for compatibility if needed, but keeping it simple for now as it wasn't primary
@app.post("/predict_base64")
async def predict_base64(
    defect_image_b64: str = Form(..., description="Base64 encoded defect image"),
    template_image_b64: str = Form(..., description="Base64 encoded template image"),
    defect_filename: str = Form("defect.jpg"),
    template_filename: str = Form("template.jpg")
):
    """
    Alternative prediction endpoint - accepts base64 encoded images
    """
    if inference_engine is None:
        raise HTTPException(status_code=503, detail="Inference engine not initialized")
    
    try:
        # Decode base64 images
        defect_data = base64.b64decode(defect_image_b64)
        template_data = base64.b64decode(template_image_b64)
        
        defect_img = cv2.imdecode(np.frombuffer(defect_data, np.uint8), cv2.IMREAD_COLOR)
        template_img = cv2.imdecode(np.frombuffer(template_data, np.uint8), cv2.IMREAD_COLOR)
        
        if defect_img is None or template_img is None:
            raise ValueError("Could not decode base64 images")
        
        # Process
        result = inference_engine.process_images(
            defect_img, 
            template_img,
            defect_filename,
            template_filename
        )
        
        # Prepare response (dict for flexibility here, or could use Pydantic)
        defects = [
            {
                'id': d['id'],
                'class': d['class'],
                'confidence': round(d['confidence'], 4),
                'bbox': list(d['bbox'])
            }
            for d in result['defects']
        ]
        
        return {
            'success': True,
            'image_name': result['image_name'],
            'template_name': result['template_name'],
            'num_defects': result['num_defects'],
            'defects': defects,
            'processing_time': round(result['processing_time'], 3),
            'annotated_image_base64': encode_image_to_base64(result['annotated_image']),
            'threshold_image_base64': encode_image_to_base64(result['threshold_image'])
        }
        
    except Exception as e:
        logger.error(f"Prediction failed (Base64): {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
