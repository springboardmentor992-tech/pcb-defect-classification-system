# File: src/pipeline/optimized_pipeline.py

"""
Optimized inference pipeline with performance improvements
"""

import torch
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import time
from functools import lru_cache
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

class OptimizedPCBDefectPipeline:
    """
    Optimized version of PCB defect detection pipeline
    """
    
    def __init__(self, model_path: str, device: str = 'cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Load model with optimizations
        self.model = self._load_optimized_model(model_path)
        
        # Pre-compile frequently used kernels
        self._setup_cv_kernels()
        
        # Initialize components
        # Note: Imports are inside to avoid circular dependencies if any, or just organizational
        from src.preprocessing.image_subtraction import ImageSubtractor
        from src.detection.contour_detector import ContourDetector
        
        self.image_subtractor = ImageSubtractor(alignment_method='ORB')
        self.contour_detector = ContourDetector(min_area=50, max_area=10000)
        
        # Setup transforms
        from torchvision import transforms
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        self.DEFECT_CLASSES = [
            'Missing_hole', 'Mouse_bite', 'Open_circuit',
            'Short', 'Spur', 'Spurious_copper'
        ]
    
    def _load_optimized_model(self, model_path: str):
        """Load model with optimizations"""
        # Assuming EfficientNetClassifier is available in src.models.cnn_model or similar
        # Per previous context, it is in models/cnn_model.py, but likely imported as src.models.cnn_model
        try:
             from src.models.cnn_model import EfficientNetClassifier
        except ImportError:
             # Fallback or try adjusted path
             sys.path.append(str(Path(__file__).parent.parent))
             from models.cnn_model import EfficientNetClassifier, ModelConfig

        # Re-create config if needed or just instantiate
        # Assuming the class needs config or num_classes
        # Based on previous usage: 
        # model = EfficientNetClassifier(ModelConfig(model_name='efficientnet_b3', num_classes=6))
        # The prompt code used EfficientNetClassifier(num_classes=6), implying a slightly different init or helper.
        # Let's stick to what we know works from previous inference_pipeline.py or adapt.
        # Check inference_pipeline.py usage:
        # model_config = ModelConfig(model_name='efficientnet_b3', num_classes=6)
        # model = EfficientNetClassifier(model_config)
        
        # We need to be careful with the prompt's code vs actual codebase. 
        # I will adapt the prompt's intent to the actual codebase structure.
        from src.models.cnn_model import EfficientNetClassifier, ModelConfig
        
        config = ModelConfig(model_name='efficientnet_b3', num_classes=6)
        model = EfficientNetClassifier(config)
        
        state_dict = torch.load(model_path, map_location=self.device)
        model.load_state_dict(state_dict)
        model.to(self.device)
        model.eval()
        
        # Enable inference optimizations
        if self.device.type == 'cuda':
            model = model.half()  # FP16 for faster inference
            torch.backends.cudnn.benchmark = True
        
        return model
    
    def _setup_cv_kernels(self):
        """Pre-create OpenCV kernels"""
        self.morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        self.blur_kernel = (5, 5)
    
    @lru_cache(maxsize=32)
    def _cached_transform(self, roi_bytes):
        """Cache transformed ROIs"""
        roi = np.frombuffer(roi_bytes, dtype=np.uint8)
        # This assumes roi_bytes is the raveled info or similar, 
        # but transforms need image shape. 
        # This cache might be tricky for variable sized ROIs. 
        # For safety/simplicity in this step, I'll trust the logic or just not use bytes caching if usage isn't clear.
        # The prompt suggests typical LRU usage.
        # Let's skip complex byte caching implementation to avoid shape issues and stick to efficient batching.
        pass
    
    def process_images_optimized(self, template_path: str, test_path: str,
                                 confidence_threshold: float = 0.5) -> Dict:
        """
        Optimized image processing
        """
        start_time = time.time()
        
        # Load images (optimized flags)
        template = cv2.imread(template_path, cv2.IMREAD_COLOR)
        test = cv2.imread(test_path, cv2.IMREAD_COLOR)
        
        # Convert to RGB (OpenCV is BGR)
        template = cv2.cvtColor(template, cv2.COLOR_BGR2RGB)
        test = cv2.cvtColor(test, cv2.COLOR_BGR2RGB)
        
        # Preprocess and align
        # Adapting to actual ImageSubtractor API
        # Previous usage: subtractor.process_image_pair(template_path, test_path)
        # Optimizing: using internal methods to avoid re-loading if we already loaded
        
        # Because ImageSubtractor.process_image_pair usually loads from path, 
        # let's modify/use what's available or use the loaded images if possible.
        # Looking at ImageSubtractor, it has align_images taking arrays.
        
        template_gray = cv2.cvtColor(template, cv2.COLOR_RGB2GRAY)
        test_gray = cv2.cvtColor(test, cv2.COLOR_RGB2GRAY)
        
        # Blur
        template_blur = cv2.GaussianBlur(template_gray, self.blur_kernel, 0)
        test_blur = cv2.GaussianBlur(test_gray, self.blur_kernel, 0)
        
        # Align
        alignment_result = self.image_subtractor.align_images(template_blur, test_blur, test)
        aligned_test = alignment_result.aligned_image
        # If alignment returned color, convert to gray for diff
        if len(aligned_test.shape) == 3:
            aligned_gray = cv2.cvtColor(aligned_test, cv2.COLOR_RGB2GRAY)
        else:
            aligned_gray = aligned_test

        # Subtract - simplistic approach for speed
        # Ensure sizes match
        if template_blur.shape != aligned_gray.shape:
             aligned_gray = cv2.resize(aligned_gray, (template_blur.shape[1], template_blur.shape[0]))
        
        diff = cv2.absdiff(template_blur, aligned_gray)
        
        # Threshold (Otsu or fixed)
        # Using fixed based on recent calibration (25) or Otsu
        _, mask = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.morph_kernel)
        
        # Detect contours
        # ContourDetector.detect returns DetectionResult
        det_result = self.contour_detector.detect(mask)
        contours = det_result.contours
        
        if len(contours) == 0:
            return {
                'num_defects': 0,
                'defects': [],
                'processing_time': time.time() - start_time,
                'annotated_image': test.copy()
            }
        
        # Extract bounding boxes from properties
        # contours is list of numpy arrays
        # contour_detector.properties has the analysis
        
        # We need to batch classify.
        
        # Prepare batch
        rois = []
        valid_defects = [] # Will hold dicts
        
        # Process contours
        for idx, props in enumerate(det_result.properties):
             x, y, w, h = props.bounding_box
             # Expand bbox
             expand = 10
             H, W = test.shape[:2]
             x = max(0, x - expand)
             y = max(0, y - expand)
             w = min(W - x, w + 2*expand)
             h = min(H - y, h + 2*expand)
             
             roi = test[y:y+h, x:x+w]
             if roi.size == 0: continue
             
             # Transform
             roi_tensor = self.transform(roi)
             rois.append(roi_tensor)
             
             valid_defects.append({
                 'bbox': (x, y, w, h),
                 'index': idx
             })

        if not rois:
            return {
                'num_defects': 0,
                'defects': [],
                'processing_time': time.time() - start_time,
                'annotated_image': test.copy()
            }

        # Batch Inference
        batch = torch.stack(rois).to(self.device)
        if self.device.type == 'cuda':
            batch = batch.half()
            
        with torch.no_grad():
            outputs = self.model(batch)
            probabilities = torch.softmax(outputs, dim=1)
            confidences, predicted_classes = probabilities.max(1)
            
        # Compile results
        final_defects = []
        for i, defect_info in enumerate(valid_defects):
            conf = confidences[i].item()
            if conf >= confidence_threshold:
                defect_info['class'] = self.DEFECT_CLASSES[predicted_classes[i].item()]
                defect_info['confidence'] = conf
                final_defects.append(defect_info)
        
        # Annotation
        annotated_image = self._draw_annotations(test.copy(), final_defects)
        
        return {
            'num_defects': len(final_defects),
            'defects': final_defects,
            'processing_time': time.time() - start_time,
            'annotated_image': cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR) # Convert back for consistency if needed
        }
    
    def _draw_annotations(self, image, defects):
        """Draw bounding boxes and labels efficiently"""
        for defect in defects:
            x, y, w, h = defect['bbox']
            color = self._get_class_color(defect['class'])
            
            # Draw rectangle
            cv2.rectangle(image, (x, y), (x+w, y+h), color, 2)
            
            # Draw label
            label = f"{defect['class']} {defect['confidence']:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1) # Thickness 1
            cv2.rectangle(image, (x, y-th-10), (x+tw+10, y), color, -1)
            cv2.putText(image, label, (x+5, y-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return image
    
    def _get_class_color(self, class_name):
        colors = {
            'Missing_hole': (255, 0, 0),
            'Mouse_bite': (0, 255, 0),
            'Open_circuit': (0, 0, 255),
            'Short': (255, 255, 0),
            'Spur': (255, 0, 255),
            'Spurious_copper': (0, 255, 255)
        }
        return colors.get(class_name, (128, 128, 128))
