"""
PCB Defect Detection & Classification - FastAPI Backend
RESTful API for single image inference with file uploads
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms, models
from pathlib import Path
import json
import io
import base64
from datetime import datetime
import tempfile
import tempfile
import shutil
import os
from dotenv import load_dotenv

load_dotenv()


# ============================================================================
# PYDANTIC MODELS FOR API RESPONSES
# ============================================================================

class DefectResult(BaseModel):
    id: int
    class_name: str
    confidence: float
    bbox: List[int]


class InferenceResponse(BaseModel):
    success: bool
    image_name: str
    template_name: Optional[str]
    num_defects: int
    defects: List[DefectResult]
    processing_time: float
    annotated_image_base64: Optional[str] = None
    threshold_image_base64: Optional[str] = None


class ErrorResponse(BaseModel):
    success: bool
    error: str


# ============================================================================
# PCB INFERENCE ENGINE (Core Logic - Unchanged)
# ============================================================================

class PCBInferenceEngine:
    """
    Core inference engine - maintains all original logic
    Modified to work with in-memory images instead of file paths
    """
    
    BLUR_KERNELS = {
        'Open_circuit': (7, 7),
        'Short': (7, 7),
        'Mouse_bite': (7, 7),
        'Spur': (9, 9),
        'Spurious_copper': (9, 9),
        'Missing_hole': (9, 9),
        'default': (9, 9)
    }
    
    def __init__(self, model_path: str, class_names: List[str], device: str = None):
        self.class_names = class_names
        self.num_classes = len(class_names)
        
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.model = self._load_model(model_path)
        
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        print(f"[INFO] PCB Inference Engine initialized on {self.device}")
        print(f"[INFO] Classes: {', '.join(class_names)}")
    
    def _load_model(self, model_path: str) -> nn.Module:
        """Load trained ResNet18 model"""
        model = models.resnet18(pretrained=False)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, self.num_classes)
        
        checkpoint = torch.load(model_path, map_location=self.device)
        
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model.to(self.device)
        model.eval()
        
        return model
    
    def align_images(self, defect_img: np.ndarray, 
                    template_img: np.ndarray) -> np.ndarray:
        """Align defect image with template using ORB"""
        try:
            defect_gray = cv2.cvtColor(defect_img, cv2.COLOR_BGR2GRAY)
            template_gray = cv2.cvtColor(template_img, cv2.COLOR_BGR2GRAY)
            
            orb = cv2.ORB_create()
            kp1, des1 = orb.detectAndCompute(defect_gray, None)
            kp2, des2 = orb.detectAndCompute(template_gray, None)
            
            if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
                return defect_img.copy()
            
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = bf.match(des1, des2)
            matches = sorted(matches, key=lambda x: x.distance)
            
            good_matches = matches[:min(50, len(matches))]
            
            if len(good_matches) > 4:
                points1 = np.float32([kp1[m.queryIdx].pt for m in good_matches])
                points2 = np.float32([kp2[m.trainIdx].pt for m in good_matches])
                
                h, _ = cv2.findHomography(points1, points2, cv2.RANSAC, 5.0)
                
                if h is not None:
                    height, width = template_img.shape[:2]
                    return cv2.warpPerspective(defect_img, h, (width, height))
            
            return defect_img.copy()
        except:
            return defect_img.copy()
    
    def detect_defects(self, aligned_img: np.ndarray, 
                      template_img: np.ndarray) -> tuple:
        """Detect defect regions using image subtraction"""
        defect_gray = cv2.cvtColor(aligned_img, cv2.COLOR_BGR2GRAY)
        template_gray = cv2.cvtColor(template_img, cv2.COLOR_BGR2GRAY)
        
        defect_blur = cv2.GaussianBlur(defect_gray, (5, 5), 0)
        template_blur = cv2.GaussianBlur(template_gray, (5, 5), 0)
        
        diff_map = cv2.absdiff(defect_blur, template_blur)
        
        kernel = self.BLUR_KERNELS['default']
        
        # Double blur-normalize sequence
        diff_map = cv2.GaussianBlur(diff_map, kernel, 0)
        diff_map = cv2.normalize(diff_map, None, 0, 255, cv2.NORM_MINMAX)
        diff_map = cv2.GaussianBlur(diff_map, kernel, 0)
        diff_map = cv2.normalize(diff_map, None, 0, 255, cv2.NORM_MINMAX)
        
        _, thresh = cv2.threshold(diff_map, 0, 255, 
                                 cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        kernel_morph = np.ones((3, 3), np.uint8)
        thresh = cv2.dilate(thresh, kernel_morph, iterations=1)
        thresh = cv2.erode(thresh, kernel_morph, iterations=1)
        
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, 
                                       cv2.CHAIN_APPROX_NONE)
        
        bboxes = []
        height, width = thresh.shape
        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            
            if (5 <= w <= 1000 and 5 <= h <= 1000 and
                0 <= x < width and 0 <= y < height and
                x + w <= width and y + h <= height):
                bboxes.append((x, y, x + w, y + h))
        
        return thresh, bboxes
    
    def extract_roi(self, image: np.ndarray, bbox: tuple, 
                   padding: int = 10) -> np.ndarray:
        """Extract and resize ROI to 128x128"""
        x1, y1, x2, y2 = bbox
        height, width = image.shape[:2]
        
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(width, x2 + padding)
        y2 = min(height, y2 + padding)
        
        roi = image[y1:y2, x1:x2]
        
        if roi.size > 0:
            return cv2.resize(roi, (128, 128), interpolation=cv2.INTER_AREA)
        
        return None
    
    def classify_roi(self, roi_img: np.ndarray) -> Dict:
        """Classify a single ROI image"""
        roi_rgb = cv2.cvtColor(roi_img, cv2.COLOR_BGR2RGB)
        input_tensor = self.transform(roi_rgb).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output = self.model(input_tensor)
            probs = torch.softmax(output, dim=1)
            confidence, pred_idx = torch.max(probs, 1)
        
        predicted_class = self.class_names[pred_idx.item()]
        confidence_score = confidence.item()
        
        return {
            'class': predicted_class,
            'confidence': confidence_score
        }
    
    def annotate_image(self, image: np.ndarray, 
                      detections: List[Dict]) -> np.ndarray:
        """Create annotated image with bounding boxes and labels"""
        annotated = image.copy()
        
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            class_name = det['class']
            confidence = det['confidence']
            defect_id = det['id']
            
            color = (0, 0, 255)  # Red
            
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)
            
            label_class = f"#{defect_id} {class_name}"
            label_conf = f"{confidence*100:.1f}%"
            
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.7
            thickness = 2
            
            (w_class, h_class), _ = cv2.getTextSize(label_class, font, 
                                                     font_scale, thickness)
            (w_conf, h_conf), _ = cv2.getTextSize(label_conf, font, 
                                                   font_scale - 0.1, thickness - 1)
            
            label_height = h_class + h_conf + 15
            cv2.rectangle(annotated, (x1, y1 - label_height - 5), 
                         (x1 + max(w_class, w_conf) + 10, y1), color, -1)
            
            cv2.putText(annotated, label_class, (x1 + 5, y1 - h_conf - 10), 
                       font, font_scale, (255, 255, 255), thickness)
            cv2.putText(annotated, label_conf, (x1 + 5, y1 - 5), 
                       font, font_scale - 0.1, (255, 255, 255), thickness - 1)
        
        return annotated
    
    def process_images(self, defect_img: np.ndarray, 
                      template_img: np.ndarray,
                      defect_name: str = "defect.jpg",
                      template_name: str = "template.jpg") -> Dict:
        """
        Main processing method - works with numpy arrays
        """
        import time
        start_time = time.time()
        
        print(f"[INFO] Processing: {defect_name}")
        
        # Step 1: Align
        aligned = self.align_images(defect_img, template_img)
        
        # Step 2: Detect
        thresh, bboxes = self.detect_defects(aligned, template_img)
        print(f"[INFO] Found {len(bboxes)} defect region(s)")
        
        if len(bboxes) == 0:
            elapsed = time.time() - start_time
            return {
                'image_name': defect_name,
                'template_name': template_name,
                'num_defects': 0,
                'defects': [],
                'processing_time': elapsed,
                'annotated_image': aligned,
                'threshold_image': thresh
            }
        
        # Step 3: Extract ROIs
        rois = []
        for bbox in bboxes:
            roi = self.extract_roi(aligned, bbox)
            if roi is not None:
                rois.append(roi)
        
        # Step 4: Classify
        detections = []
        for i, (bbox, roi) in enumerate(zip(bboxes, rois)):
            prediction = self.classify_roi(roi)
            
            detection = {
                'id': i + 1,
                'bbox': bbox,
                'class': prediction['class'],
                'confidence': prediction['confidence']
            }
            detections.append(detection)
            print(f"[INFO] #{i+1}: {prediction['class']} ({prediction['confidence']*100:.1f}%)")
        
        # Create annotated image
        annotated = self.annotate_image(aligned, detections)
        
        elapsed = time.time() - start_time
        
        return {
            'image_name': defect_name,
            'template_name': template_name,
            'num_defects': len(detections),
            'defects': detections,
            'processing_time': elapsed,
            'annotated_image': annotated,
            'threshold_image': thresh
        }


# ============================================================================
# FASTAPI APPLICATION
# ============================================================================

app = FastAPI(
    title="PCB Defect Detection API",
    description="RESTful API for PCB defect detection and classification",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global inference engine (initialized on startup)
inference_engine: Optional[PCBInferenceEngine] = None


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def read_image_file(file: UploadFile) -> np.ndarray:
    """Read uploaded file as OpenCV image"""
    contents = file.file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Could not decode image: {file.filename}")
    return img


def encode_image_to_base64(image: np.ndarray) -> str:
    """Encode OpenCV image to base64 string"""
    _, buffer = cv2.imencode('.png', image)
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    return img_base64


# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize inference engine on startup"""
    global inference_engine
    
    # Configuration - update these paths
    MODEL_PATH = os.getenv("MODEL_PATH")
    if not MODEL_PATH:
         # Fallback or error if not set, though for now we assume .env is standard
         print("[WARNING] MODEL_PATH not set in .env, checking default...")
         MODEL_PATH = "models/pcb_defect_resnet18.pth"
    
    # Ensure absolute path for safety if needed, or rely on relative
    if not os.path.isabs(MODEL_PATH):
        MODEL_PATH = os.path.join(os.getcwd(), MODEL_PATH)

    print(f"[INFO] Loading model from: {MODEL_PATH}")
    CLASS_NAMES = [
        'Missing_hole',
        'Mouse_bite',
        'Open_circuit',
        'Short',
        'Spur',
        'Spurious_copper'
    ]
    
    try:
        inference_engine = PCBInferenceEngine(
            model_path=MODEL_PATH,
            class_names=CLASS_NAMES,
            device=None  # Auto-detect
        )
        print("[SUCCESS] PCB Defect Detection API ready!")
    except Exception as e:
        print(f"[ERROR] Failed to initialize inference engine: {e}")
        raise


@app.get("/")
async def root():
    """Root endpoint - API info"""
    return {
        "service": "PCB Defect Detection API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "health": "/health",
            "predict": "/predict (POST)",
            "predict_with_base64": "/predict_base64 (POST)"
        }
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
    
    Args:
        defect_image: PCB image file with defects (jpg, png)
        template_image: Template PCB image file (jpg, png)
        return_images: Whether to return base64 encoded annotated images
        
    Returns:
        JSON response with detection results and optionally images
    """
    if inference_engine is None:
        raise HTTPException(status_code=503, detail="Inference engine not initialized")
    
    try:
        # Read images
        print(f"[INFO] Received files: {defect_image.filename}, {template_image.filename}")
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
        
        print(f"[SUCCESS] Processed {defect_image.filename}: {result['num_defects']} defects")
        
        return response
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        print(f"[ERROR] Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict_base64")
async def predict_base64(
    defect_image_b64: str = Form(..., description="Base64 encoded defect image"),
    template_image_b64: str = Form(..., description="Base64 encoded template image"),
    defect_filename: str = Form("defect.jpg"),
    template_filename: str = Form("template.jpg")
):
    """
    Alternative prediction endpoint - accepts base64 encoded images
    Useful for integration with web frontends
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
        
        # Prepare response
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
        print(f"[ERROR] Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.get("/download_annotated/{request_id}")
async def download_annotated(request_id: str):
    """
    Download annotated image as PNG file
    (Could be extended to cache results by request_id)
    """
    # Implementation would require result caching
    raise HTTPException(status_code=501, detail="Not implemented")


# ============================================================================
# RUN SERVER
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    print("="*70)
    print("PCB DEFECT DETECTION API")
    print("="*70)
    print("Starting server...")
    print("API Documentation: http://localhost:8000/docs")
    print("="*70)
    
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000,
        log_level="info"
    )