"""
PCB Defect Detection - Inference Pipeline
==========================================

End-to-end pipeline for PCB defect detection using the trained
EfficientNet-B3 classifier.

This module handles:
- Image preprocessing and alignment
- Difference map computation
- Defect region detection
- Classification with confidence scores
- Result annotation

Author: PCB Defect Detection Team
Version: 1.0.0
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime
import time
import logging
import sys

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================
# DATA CLASSES
# ============================================================

@dataclass
class DetectedDefect:
    """Represents a single detected defect."""
    index: int
    class_name: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # (x, y, width, height)
    center: Tuple[int, int]
    area: int
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'index': self.index,
            'class': self.class_name,
            'confidence': float(self.confidence),
            'bbox': {
                'x': int(self.bbox[0]),
                'y': int(self.bbox[1]),
                'width': int(self.bbox[2]),
                'height': int(self.bbox[3])
            },
            'center': {'x': int(self.center[0]), 'y': int(self.center[1])},
            'area': int(self.area)
        }


@dataclass
class DetectionResult:
    """Complete detection result for an image pair."""
    success: bool
    num_defects: int
    defects: List[DetectedDefect]
    processing_time: float
    image_size: Tuple[int, int]
    timestamp: str
    error_message: Optional[str] = None
    
    # Optional image data (as base64 or numpy arrays)
    annotated_image: Optional[np.ndarray] = None
    difference_map: Optional[np.ndarray] = None
    mask: Optional[np.ndarray] = None
    aligned_image: Optional[np.ndarray] = None
    
    def to_dict(self, include_images: bool = False) -> Dict:
        """Convert to dictionary."""
        result = {
            'success': self.success,
            'num_defects': self.num_defects,
            'defects': [d.to_dict() for d in self.defects],
            'processing_time': float(self.processing_time),
            'image_size': {'width': self.image_size[0], 'height': self.image_size[1]},
            'timestamp': self.timestamp,
            'error_message': self.error_message,
            'summary': {
                'total_defects': self.num_defects,
                'defect_types': list(set(d.class_name for d in self.defects)),
                'avg_confidence': float(np.mean([d.confidence for d in self.defects])) 
                    if self.defects else 0.0
            }
        }
        return result


@dataclass
class PipelineConfig:
    """Configuration for the inference pipeline."""
    model_path: str
    device: str = 'auto'
    confidence_threshold: float = 0.5
    min_defect_area: int = 12  # Calibrated: 12px (was 5, too noisy)
    max_defect_area: int = 50000
    expand_bbox_pixels: int = 15
    input_size: Tuple[int, int] = (128, 128)
    
    # Preprocessing settings
    blur_kernel_size: int = 5
    morph_kernel_size: int = 5  # Increased to 5 to clean up noise
    threshold_value: int = 25  # Increased to 25 (was 15, too sensitive)
    
    # Class names
    class_names: Tuple[str, ...] = (
        'Missing_hole',
        'Mouse_bite',
        'Open_circuit',
        'Short',
        'Spur',
        'Spurious_copper'
    )
    
    # Colors for visualization (RGB)
    class_colors: Dict[str, Tuple[int, int, int]] = field(default_factory=lambda: {
        'Missing_hole': (239, 68, 68),      # Red
        'Mouse_bite': (34, 197, 94),        # Green
        'Open_circuit': (59, 130, 246),     # Blue
        'Short': (234, 179, 8),             # Yellow
        'Spur': (168, 85, 247),             # Purple
        'Spurious_copper': (6, 182, 212)    # Cyan
    })


# ============================================================
# INFERENCE PIPELINE
# ============================================================

class PCBDefectPipeline:
    """
    End-to-end pipeline for PCB defect detection.
    
    This pipeline:
    1. Aligns test image to template
    2. Computes difference map
    3. Generates binary mask
    4. Detects contours and extracts ROIs
    5. Classifies each ROI using the trained model
    6. Returns annotated results
    
    Example:
        >>> pipeline = PCBDefectPipeline('models/best_model_weights.pth')
        >>> result = pipeline.process(template_image, test_image)
        >>> print(f"Found {result.num_defects} defects")
    """
    
    def __init__(self, config: Optional[PipelineConfig] = None):
        """
        Initialize the pipeline.
        
        Args:
            config: Pipeline configuration. If None, uses defaults.
        """
        if config is None:
            # Use default config with model path
            model_path = str(PROJECT_ROOT / 'models' / 'best_model_weights.pth')
            config = PipelineConfig(model_path=model_path)
        
        self.config = config
        self.device = self._get_device()
        self.model = None
        self.transform = None
        self._initialized = False
        
        logger.info(f"PCBDefectPipeline initialized with device: {self.device}")
    
    def _get_device(self) -> torch.device:
        """Determine the best available device."""
        if self.config.device == 'auto':
            if torch.cuda.is_available():
                return torch.device('cuda')
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return torch.device('mps')
            else:
                return torch.device('cpu')
        return torch.device(self.config.device)
    
    def initialize(self) -> bool:
        """
        Initialize the model and transforms.
        
        Returns:
            True if initialization successful, False otherwise.
        """
        try:
            logger.info("Initializing pipeline...")
            
            # Load model
            self._load_model()
            
            # Setup transforms
            self._setup_transforms()
            
            self._initialized = True
            logger.info("Pipeline initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize pipeline: {e}")
            return False
    
    def _load_model(self):
        """Load the trained classification model."""
        from models.cnn_model import EfficientNetClassifier, ModelConfig
        
        model_path = Path(self.config.model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        # Create model
        model_config = ModelConfig(
            model_name='efficientnet_b3',
            num_classes=len(self.config.class_names)
        )
        self.model = EfficientNetClassifier(model_config)
        
        # Load weights
        state_dict = torch.load(str(model_path), map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()
        
        logger.info(f"Model loaded from: {model_path}")
    
    def _setup_transforms(self):
        """Setup image transforms for model input."""
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(self.config.input_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def process(
        self,
        template_image: np.ndarray,
        test_image: np.ndarray,
        confidence_threshold: Optional[float] = None
    ) -> DetectionResult:
        """
        Process a template-test image pair to detect defects.
        
        Args:
            template_image: Defect-free reference image (BGR or RGB)
            test_image: Image to inspect (BGR or RGB)
            confidence_threshold: Override default confidence threshold
            
        Returns:
            DetectionResult containing all detection information
        """
        start_time = time.time()
        timestamp = datetime.now().isoformat()
        
        # Initialize if not already done
        if not self._initialized:
            if not self.initialize():
                return DetectionResult(
                    success=False,
                    num_defects=0,
                    defects=[],
                    processing_time=time.time() - start_time,
                    image_size=(0, 0),
                    timestamp=timestamp,
                    error_message="Failed to initialize pipeline"
                )
        
        threshold = confidence_threshold or self.config.confidence_threshold
        
        try:
            # Step 1: Preprocess images
            template_proc, test_proc = self._preprocess_images(template_image, test_image)
            
            # Step 2: Align images
            aligned = self._align_images(template_proc, test_proc, test_image)
            
            # Step 3: Compute difference map
            diff_map = self._compute_difference(template_proc, aligned)
            
            # Step 4: Generate binary mask
            mask = self._generate_mask(diff_map)
            
            # Step 5: Detect contours and extract ROIs
            contours = self._detect_contours(mask)
            
            if len(contours) == 0:
                # No defects found
                return DetectionResult(
                    success=True,
                    num_defects=0,
                    defects=[],
                    processing_time=time.time() - start_time,
                    image_size=(test_image.shape[1], test_image.shape[0]),
                    timestamp=timestamp,
                    annotated_image=test_image.copy(),
                    difference_map=diff_map,
                    mask=mask,
                    aligned_image=aligned
                )
            
            # Step 6: Extract and classify each defect
            defects = []
            annotated = test_image.copy()
            
            for idx, contour in enumerate(contours):
                # Get bounding box
                x, y, w, h = cv2.boundingRect(contour)
                
                # Expand bounding box
                x, y, w, h = self._expand_bbox(
                    x, y, w, h,
                    test_image.shape[:2],
                    self.config.expand_bbox_pixels
                )
                
                # Extract ROI
                roi = test_image[y:y+h, x:x+w]
                
                if roi.size == 0:
                    continue
                
                # Classify ROI
                class_name, confidence = self._classify_roi(roi)
                
                # Filter by confidence
                if confidence < threshold:
                    continue
                
                # Calculate center and area
                center = (x + w // 2, y + h // 2)
                area = cv2.contourArea(contour)
                
                # Create defect object
                defect = DetectedDefect(
                    index=len(defects) + 1,
                    class_name=class_name,
                    confidence=confidence,
                    bbox=(x, y, w, h),
                    center=center,
                    area=area
                )
                defects.append(defect)
                
                # Draw on annotated image
                self._draw_detection(annotated, defect)
            
            processing_time = time.time() - start_time
            
            return DetectionResult(
                success=True,
                num_defects=len(defects),
                defects=defects,
                processing_time=processing_time,
                image_size=(test_image.shape[1], test_image.shape[0]),
                timestamp=timestamp,
                annotated_image=annotated,
                difference_map=diff_map,
                mask=mask,
                aligned_image=aligned
            )
            
        except Exception as e:
            logger.error(f"Error during processing: {e}")
            return DetectionResult(
                success=False,
                num_defects=0,
                defects=[],
                processing_time=time.time() - start_time,
                image_size=(test_image.shape[1], test_image.shape[0]),
                timestamp=timestamp,
                error_message=str(e)
            )
    
    def _preprocess_images(
        self,
        template: np.ndarray,
        test: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Preprocess images for alignment."""
        # Convert to grayscale
        if len(template.shape) == 3:
            template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        else:
            template_gray = template.copy()
            
        if len(test.shape) == 3:
            test_gray = cv2.cvtColor(test, cv2.COLOR_BGR2GRAY)
        else:
            test_gray = test.copy()
        
        # Apply Gaussian blur
        ksize = self.config.blur_kernel_size
        template_blur = cv2.GaussianBlur(template_gray, (ksize, ksize), 0)
        test_blur = cv2.GaussianBlur(test_gray, (ksize, ksize), 0)
        
        return template_blur, test_blur
    
    def _align_images(
        self,
        template: np.ndarray,
        test: np.ndarray,
        test_color: np.ndarray
    ) -> np.ndarray:
        """Align test image to template using ORB features."""
        try:
            # Create ORB detector
            orb = cv2.ORB_create(nfeatures=5000)
            
            # Detect keypoints and descriptors
            kp1, des1 = orb.detectAndCompute(template, None)
            kp2, des2 = orb.detectAndCompute(test, None)
            
            if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
                logger.warning("Not enough features for alignment, using original image")
                return test_color.copy()
            
            # Match features
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = bf.match(des1, des2)
            
            if len(matches) < 4:
                logger.warning("Not enough matches for alignment")
                return test_color.copy()
            
            # Sort matches by distance
            matches = sorted(matches, key=lambda x: x.distance)
            
            # Use top matches
            good_matches = matches[:min(100, len(matches))]
            
            # Extract matching points
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            
            # Find homography
            H, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
            
            if H is None:
                logger.warning("Failed to compute homography")
                return test_color.copy()
            
            # Warp test image
            h, w = template.shape[:2]
            aligned = cv2.warpPerspective(test_color, H, (w, h))
            
            return aligned
            
        except Exception as e:
            logger.warning(f"Alignment failed: {e}, using original image")
            return test_color.copy()
    
    def _compute_difference(
        self,
        template: np.ndarray,
        aligned: np.ndarray
    ) -> np.ndarray:
        """Compute absolute difference between template and aligned image."""
        # Ensure grayscale
        if len(aligned.shape) == 3:
            aligned_gray = cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY)
        else:
            aligned_gray = aligned
        
        # Resize if necessary
        if template.shape != aligned_gray.shape:
            aligned_gray = cv2.resize(aligned_gray, (template.shape[1], template.shape[0]))
        
        # Compute difference
        diff = cv2.absdiff(template, aligned_gray)
        
        return diff
    
    def _generate_mask(self, diff_map: np.ndarray) -> np.ndarray:
        """Generate binary mask from difference map."""
        # Apply threshold
        _, mask = cv2.threshold(
            diff_map,
            self.config.threshold_value,
            255,
            cv2.THRESH_BINARY
        )
        
        # Morphological operations
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (self.config.morph_kernel_size, self.config.morph_kernel_size)
        )
        
        # Close small gaps
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # Remove noise
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        return mask
    
    def _detect_contours(self, mask: np.ndarray) -> List[np.ndarray]:
        """Detect and filter contours from mask."""
        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        # Filter by area
        filtered = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if self.config.min_defect_area <= area <= self.config.max_defect_area:
                filtered.append(contour)
        
        return filtered
    
    def _expand_bbox(
        self,
        x: int,
        y: int,
        w: int,
        h: int,
        image_shape: Tuple[int, int],
        expand_pixels: int
    ) -> Tuple[int, int, int, int]:
        """Expand bounding box while keeping within image bounds."""
        x = max(0, x - expand_pixels)
        y = max(0, y - expand_pixels)
        w = min(image_shape[1] - x, w + 2 * expand_pixels)
        h = min(image_shape[0] - y, h + 2 * expand_pixels)
        return x, y, w, h
    
    def _classify_roi(self, roi: np.ndarray) -> Tuple[str, float]:
        """Classify a single ROI using the model."""
        # Ensure RGB
        if len(roi.shape) == 2:
            roi = cv2.cvtColor(roi, cv2.COLOR_GRAY2RGB)
        elif roi.shape[2] == 4:
            roi = cv2.cvtColor(roi, cv2.COLOR_BGRA2RGB)
        elif roi.shape[2] == 3:
            roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        
        # Apply transforms
        tensor = self.transform(roi).unsqueeze(0).to(self.device)
        
        # Run inference
        with torch.no_grad():
            output = self.model(tensor)
            probabilities = F.softmax(output, dim=1)
            confidence, predicted = probabilities.max(1)
        
        class_name = self.config.class_names[predicted.item()]
        confidence_value = confidence.item()
        
        return class_name, confidence_value
    
    def _draw_detection(self, image: np.ndarray, defect: DetectedDefect):
        """Draw detection visualization on image."""
        x, y, w, h = defect.bbox
        color = self.config.class_colors.get(defect.class_name, (128, 128, 128))
        
        # Draw bounding box
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
        
        # Draw label background
        label = f"{defect.class_name}: {defect.confidence:.0%}"
        (text_w, text_h), baseline = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            1
        )
        
        # Label position (above bbox if possible)
        label_y = y - 5 if y > text_h + 10 else y + h + text_h + 5
        
        # Draw label background
        cv2.rectangle(
            image,
            (x, label_y - text_h - 5),
            (x + text_w + 10, label_y + 5),
            color,
            -1
        )
        
        # Draw label text
        cv2.putText(
            image,
            label,
            (x + 5, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )
        
        # Draw defect index
        cv2.circle(image, defect.center, 3, color, -1)
    
    def get_model_info(self) -> Dict:
        """Get information about the loaded model."""
        if not self._initialized:
            return {'error': 'Pipeline not initialized'}
        
        return {
            'model_path': self.config.model_path,
            'device': str(self.device),
            'num_classes': len(self.config.class_names),
            'class_names': list(self.config.class_names),
            'input_size': self.config.input_size,
            'confidence_threshold': self.config.confidence_threshold
        }


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def create_pipeline(model_path: Optional[str] = None, **kwargs) -> PCBDefectPipeline:
    """
    Factory function to create a pipeline instance.
    
    Args:
        model_path: Path to the trained model
        **kwargs: Additional configuration options
        
    Returns:
        Initialized PCBDefectPipeline instance
    """
    if model_path is None:
        model_path = str(PROJECT_ROOT / 'models' / 'best_model_weights.pth')
    
    config = PipelineConfig(model_path=model_path, **kwargs)
    pipeline = PCBDefectPipeline(config)
    pipeline.initialize()
    
    return pipeline


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":
    # Quick test
    print("Testing PCBDefectPipeline...")
    
    pipeline = create_pipeline()
    
    print("\nModel Info:")
    info = pipeline.get_model_info()
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    print("\n✓ Pipeline test complete!")
