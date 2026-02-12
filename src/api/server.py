"""
PCB Defect Detection - FastAPI Server
======================================

Production-ready REST API server for PCB defect detection.

Features:
---------
- RESTful API endpoints for defect detection
- Image upload support (multipart/form-data)
- Async processing with background tasks
- CORS support for frontend integration
- Health checks and model status endpoints
- Request validation with Pydantic
- Comprehensive error handling
- Swagger/OpenAPI documentation

Endpoints:
----------
- GET  /                    : API info
- GET  /health              : Health check
- GET  /model/info          : Model information
- POST /detect              : Detect defects in image pair
- POST /detect/single       : Analyze single image
- GET  /classes             : List defect classes

Usage:
------
    # Start server
    uvicorn src.api.server:app --host 0.0.0.0 --port 8000 --reload
    
    # Or run directly
    python -m src.api.server

Author: PCB Defect Detection Team
Version: 1.0.0
"""

import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any
import io
import base64
import time
import asyncio
from contextlib import asynccontextmanager

import numpy as np
import cv2
from PIL import Image

from fastapi import FastAPI, File, UploadFile, HTTPException, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

# Import pipeline
from pipeline.inference_pipeline import (
    PCBDefectPipeline,
    PipelineConfig,
    create_pipeline
)


# ============================================================
# PYDANTIC MODELS (Request/Response Schemas)
# ============================================================

class HealthResponse(BaseModel):
    """Health check response."""
    status: str = Field(..., description="Server status")
    timestamp: str = Field(..., description="Current timestamp")
    model_loaded: bool = Field(..., description="Whether the model is loaded")
    version: str = Field(default="1.0.0", description="API version")


class ModelInfoResponse(BaseModel):
    """Model information response."""
    model_path: str
    device: str
    num_classes: int
    class_names: List[str]
    input_size: List[int]
    confidence_threshold: float


class DefectBbox(BaseModel):
    """Bounding box for a detected defect."""
    x: int
    y: int
    width: int
    height: int


class DefectCenter(BaseModel):
    """Center point of a defect."""
    x: int
    y: int


class DetectedDefectResponse(BaseModel):
    """Single detected defect."""
    index: int
    class_name: str = Field(..., alias="class")
    confidence: float
    bbox: DefectBbox
    center: DefectCenter
    area: int
    
    class Config:
        populate_by_name = True


class DetectionSummary(BaseModel):
    """Summary of detection results."""
    total_defects: int
    defect_types: List[str]
    avg_confidence: float


class ImageSize(BaseModel):
    """Image dimensions."""
    width: int
    height: int


class DetectionResponse(BaseModel):
    """Complete detection response."""
    success: bool
    num_defects: int
    defects: List[Dict[str, Any]]
    processing_time: float
    image_size: ImageSize
    timestamp: str
    error_message: Optional[str] = None
    summary: DetectionSummary
    annotated_image_base64: Optional[str] = None
    difference_map_base64: Optional[str] = None
    mask_base64: Optional[str] = None


class DefectClassInfo(BaseModel):
    """Information about a defect class."""
    name: str
    display_name: str
    color: str
    description: str


class DefectClassesResponse(BaseModel):
    """List of available defect classes."""
    classes: List[DefectClassInfo]
    total: int


class APIInfoResponse(BaseModel):
    """API information."""
    name: str
    version: str
    description: str
    documentation_url: str
    endpoints: Dict[str, str]


# ============================================================
# GLOBAL STATE
# ============================================================

class AppState:
    """Application state container."""
    pipeline: Optional[PCBDefectPipeline] = None
    initialized: bool = False
    initialization_error: Optional[str] = None
    request_count: int = 0
    last_request_time: Optional[str] = None


app_state = AppState()


# ============================================================
# LIFECYCLE EVENTS
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.
    
    Handles startup and shutdown events.
    """
    # Startup
    print("\n" + "="*60)
    print("  PCB DEFECT DETECTION API SERVER")
    print("="*60)
    print(f"\nStartup Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        print("\n🔄 Loading model...")
        model_path = str(PROJECT_ROOT / 'models' / 'best_model_weights.pth')
        
        if not Path(model_path).exists():
            app_state.initialization_error = f"Model not found: {model_path}"
            print(f"⚠️  Model not found at: {model_path}")
            print("    Running in demo mode (detection will fail)")
        else:
            config = PipelineConfig(model_path=model_path)
            app_state.pipeline = PCBDefectPipeline(config)
            
            if app_state.pipeline.initialize():
                app_state.initialized = True
                print(f"✓ Model loaded successfully")
                print(f"  Device: {app_state.pipeline.device}")
            else:
                app_state.initialization_error = "Failed to initialize model"
                print("❌ Failed to initialize model")
        
        print("\n" + "-"*60)
        print("  Server is ready to accept requests")
        print("-"*60 + "\n")
        
    except Exception as e:
        app_state.initialization_error = str(e)
        print(f"❌ Startup error: {e}")
    
    yield  # Server is running
    
    # Shutdown
    print("\n🛑 Shutting down server...")
    app_state.pipeline = None
    print("✓ Cleanup complete")


# ============================================================
# FASTAPI APP
# ============================================================

app = FastAPI(
    title="PCB Defect Detection API",
    description="""
    REST API for detecting and classifying defects in Printed Circuit Boards.
    
    ## Features
    - Upload template and test images for defect detection
    - Returns bounding boxes, class labels, and confidence scores
    - Supports multiple image formats (JPEG, PNG)
    - Returns annotated images with visualized defects
    
    ## Defect Classes
    - **Missing Hole**: Holes that should be present but are missing
    - **Mouse Bite**: Irregular edge breaks
    - **Open Circuit**: Breaks in copper traces
    - **Short**: Unwanted copper connections
    - **Spur**: Protruding copper extensions
    - **Spurious Copper**: Unwanted copper deposits
    """,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (customize for production)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def image_to_base64(image: np.ndarray) -> str:
    """Convert numpy image to base64 string."""
    # Ensure BGR for encoding
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    
    # Encode to JPEG
    success, buffer = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not success:
        raise ValueError("Failed to encode image")
    
    return base64.b64encode(buffer).decode('utf-8')


async def read_image_file(file: UploadFile) -> np.ndarray:
    """Read uploaded file as numpy array."""
    contents = await file.read()
    
    # Convert to numpy array
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if image is None:
        raise ValueError(f"Failed to decode image: {file.filename}")
    
    return image


def get_defect_descriptions() -> Dict[str, str]:
    """Get descriptions for each defect class."""
    return {
        'Missing_hole': 'Holes that should be present but are missing',
        'Mouse_bite': 'Irregular edge breaks resembling mouse bites',
        'Open_circuit': 'Breaks in copper traces causing open circuits',
        'Short': 'Unwanted copper connections causing short circuits',
        'Spur': 'Protruding copper extensions from traces',
        'Spurious_copper': 'Unwanted copper deposits on the PCB'
    }


# ============================================================
# API ENDPOINTS
# ============================================================

@app.get("/", response_model=APIInfoResponse, tags=["Info"])
async def root():
    """Get API information and available endpoints."""
    return APIInfoResponse(
        name="PCB Defect Detection API",
        version="1.0.0",
        description="REST API for PCB defect detection using EfficientNet-B3",
        documentation_url="/docs",
        endpoints={
            "GET /": "API information",
            "GET /health": "Health check",
            "GET /model/info": "Model information",
            "GET /classes": "List defect classes",
            "POST /detect": "Detect defects (template + test images)",
        }
    )


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Health check endpoint.
    
    Returns the current status of the server and model.
    """
    return HealthResponse(
        status="healthy" if app_state.initialized else "degraded",
        timestamp=datetime.now().isoformat(),
        model_loaded=app_state.initialized,
        version="1.0.0"
    )


@app.get("/model/info", response_model=ModelInfoResponse, tags=["Model"])
async def get_model_info():
    """
    Get information about the loaded model.
    
    Returns model architecture, device, and configuration.
    """
    if not app_state.initialized or app_state.pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. " + (app_state.initialization_error or "")
        )
    
    info = app_state.pipeline.get_model_info()
    
    return ModelInfoResponse(
        model_path=info['model_path'],
        device=info['device'],
        num_classes=info['num_classes'],
        class_names=info['class_names'],
        input_size=list(info['input_size']),
        confidence_threshold=info['confidence_threshold']
    )


@app.get("/classes", response_model=DefectClassesResponse, tags=["Model"])
async def get_defect_classes():
    """
    Get list of available defect classes.
    
    Returns names, colors, and descriptions for each class.
    """
    descriptions = get_defect_descriptions()
    
    colors = {
        'Missing_hole': '#EF4444',
        'Mouse_bite': '#22C55E',
        'Open_circuit': '#3B82F6',
        'Short': '#EAB308',
        'Spur': '#A855F7',
        'Spurious_copper': '#06B6D4'
    }
    
    classes = []
    for name in descriptions.keys():
        classes.append(DefectClassInfo(
            name=name,
            display_name=name.replace('_', ' '),
            color=colors.get(name, '#888888'),
            description=descriptions[name]
        ))
    
    return DefectClassesResponse(classes=classes, total=len(classes))


@app.post("/detect", response_model=DetectionResponse, tags=["Detection"])
async def detect_defects(
    template: UploadFile = File(..., description="Template (defect-free) image"),
    test: UploadFile = File(..., description="Test image to inspect"),
    confidence_threshold: float = Query(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Minimum confidence threshold for detections"
    ),
    include_images: bool = Query(
        default=True,
        description="Include base64-encoded result images"
    )
):
    """
    Detect defects in a PCB image.
    
    This endpoint accepts a template (defect-free) image and a test image,
    performs defect detection, and returns the results with optional
    annotated images.
    
    ## Request
    - **template**: Defect-free reference image (JPEG/PNG)
    - **test**: Image to inspect for defects (JPEG/PNG)
    - **confidence_threshold**: Minimum confidence (0.0-1.0, default: 0.5)
    - **include_images**: Whether to include base64 images in response
    
    ## Response
    Returns detected defects with bounding boxes, class labels,
    confidence scores, and optionally annotated images.
    """
    # Check model is loaded
    if not app_state.initialized or app_state.pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please wait for initialization."
        )
    
    # Update request stats
    app_state.request_count += 1
    app_state.last_request_time = datetime.now().isoformat()
    
    try:
        # Read images
        template_image = await read_image_file(template)
        test_image = await read_image_file(test)
        
        # Run detection
        result = app_state.pipeline.process(
            template_image,
            test_image,
            confidence_threshold=confidence_threshold
        )
        
        # Build response
        response_data = {
            'success': result.success,
            'num_defects': result.num_defects,
            'defects': [d.to_dict() for d in result.defects],
            'processing_time': result.processing_time,
            'image_size': ImageSize(
                width=result.image_size[0],
                height=result.image_size[1]
            ),
            'timestamp': result.timestamp,
            'error_message': result.error_message,
            'summary': DetectionSummary(
                total_defects=result.num_defects,
                defect_types=list(set(d.class_name for d in result.defects)),
                avg_confidence=float(np.mean([d.confidence for d in result.defects]))
                    if result.defects else 0.0
            ),
            'annotated_image_base64': None,
            'difference_map_base64': None,
            'mask_base64': None
        }
        
        # Add images if requested
        if include_images and result.annotated_image is not None:
            response_data['annotated_image_base64'] = image_to_base64(result.annotated_image)
            
            if result.difference_map is not None:
                response_data['difference_map_base64'] = image_to_base64(result.difference_map)
            
            if result.mask is not None:
                response_data['mask_base64'] = image_to_base64(result.mask)
        
        return DetectionResponse(**response_data)
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Detection failed: {str(e)}")


@app.get("/stats", tags=["Info"])
async def get_stats():
    """Get server statistics."""
    return {
        "total_requests": app_state.request_count,
        "last_request_time": app_state.last_request_time,
        "model_initialized": app_state.initialized,
        "uptime_check": datetime.now().isoformat()
    }


# ============================================================
# ERROR HANDLERS
# ============================================================

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle uncaught exceptions."""
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": "Internal server error",
            "detail": str(exc),
            "timestamp": datetime.now().isoformat()
        }
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    import uvicorn
    
    print("\n" + "="*60)
    print("  Starting PCB Defect Detection API Server")
    print("="*60 + "\n")
    
    uvicorn.run(
        "src.api.server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
