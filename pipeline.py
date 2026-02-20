"""
PCB Defect Detection & Classification - Single Image Inference
Streamlined pipeline for processing individual PCB images
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms, models
from pathlib import Path
import json
from typing import Dict, List, Tuple, Optional
import time


# ============================================================================
# MODEL ARCHITECTURE
# ============================================================================

# Note: Model architecture is defined directly in _load_model using torchvision.models


# ============================================================================
# SINGLE IMAGE INFERENCE PIPELINE
# ============================================================================

class PCBInference:
    """
    Complete pipeline for single PCB image inference.
    Handles detection, classification, and annotation in one streamlined class.
    """
    
    # Class-specific blur kernels
    BLUR_KERNELS = {
        'Open_circuit': (7, 7),
        'Short': (7, 7),
        'Mouse_bite': (7, 7),
        'Spur': (9, 9),
        'Spurious_copper': (9, 9),
        'Missing_hole': (9, 9),
        'default': (9, 9)
    }
    
    def __init__(self, model_path: str, class_names: List[str], 
                 template_dir: str, device: str = None):
        """
        Initialize inference pipeline.
        
        Args:
            model_path: Path to trained model checkpoint (.pth file)
            class_names: List of defect class names (must match training order)
            template_dir: Directory containing template PCB images
            device: 'cuda', 'cpu', or None (auto-detect)
        """
        self.class_names = class_names
        self.num_classes = len(class_names)
        self.template_dir = Path(template_dir)
        
        # Setup device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Load model
        self.model = self._load_model(model_path)
        
        # Image transforms (must match training)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        print("="*70)
        print("PCB DEFECT INFERENCE PIPELINE")
        print("="*70)
        print(f"Device: {self.device}")
        print(f"Classes: {', '.join(class_names)}")
        print(f"Model loaded: {Path(model_path).name}")
        print("="*70 + "\n")
    
    def _load_model(self, model_path: str) -> nn.Module:
        """Load trained ResNet18 model from checkpoint"""
        # Initialize ResNet18 (must match training architecture)
        model = models.resnet18(pretrained=False)
        
        # Replace the final fully connected layer
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
    
    # ========================================================================
    # IMAGE PROCESSING METHODS
    # ========================================================================
    
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
                      template_img: np.ndarray) -> Tuple[np.ndarray, List[Tuple]]:
        """
        Detect defect regions using image subtraction.
        Returns threshold image and bounding boxes.
        """
        # Convert to grayscale and blur
        defect_gray = cv2.cvtColor(aligned_img, cv2.COLOR_BGR2GRAY)
        template_gray = cv2.cvtColor(template_img, cv2.COLOR_BGR2GRAY)
        
        defect_blur = cv2.GaussianBlur(defect_gray, (5, 5), 0)
        template_blur = cv2.GaussianBlur(template_gray, (5, 5), 0)
        
        # Compute difference
        diff_map = cv2.absdiff(defect_blur, template_blur)
        
        # Apply class-specific blur (using default kernel)
        kernel = self.BLUR_KERNELS['default']
        
        # Double blur-normalize sequence (critical!)
        diff_map = cv2.GaussianBlur(diff_map, kernel, 0)
        diff_map = cv2.normalize(diff_map, None, 0, 255, cv2.NORM_MINMAX)
        diff_map = cv2.GaussianBlur(diff_map, kernel, 0)
        diff_map = cv2.normalize(diff_map, None, 0, 255, cv2.NORM_MINMAX)
        
        # Threshold
        _, thresh = cv2.threshold(diff_map, 0, 255, 
                                 cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Morphological operations
        kernel_morph = np.ones((3, 3), np.uint8)
        thresh = cv2.dilate(thresh, kernel_morph, iterations=1)
        thresh = cv2.erode(thresh, kernel_morph, iterations=1)
        
        # Extract contours and bounding boxes
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, 
                                       cv2.CHAIN_APPROX_NONE)
        
        bboxes = []
        height, width = thresh.shape
        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            
            # Filter by size
            if (5 <= w <= 1000 and 5 <= h <= 1000 and
                0 <= x < width and 0 <= y < height and
                x + w <= width and y + h <= height):
                bboxes.append((x, y, x + w, y + h))
        
        return thresh, bboxes
    
    def extract_roi(self, image: np.ndarray, bbox: Tuple, 
                   padding: int = 10) -> np.ndarray:
        """Extract and resize ROI to 128x128"""
        x1, y1, x2, y2 = bbox
        height, width = image.shape[:2]
        
        # Add padding
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(width, x2 + padding)
        y2 = min(height, y2 + padding)
        
        roi = image[y1:y2, x1:x2]
        
        if roi.size > 0:
            return cv2.resize(roi, (128, 128), interpolation=cv2.INTER_AREA)
        
        return None
    
    # ========================================================================
    # CLASSIFICATION METHODS
    # ========================================================================
    
    def classify_roi(self, roi_img: np.ndarray) -> Dict:
        """Classify a single ROI image"""
        # Convert BGR to RGB
        roi_rgb = cv2.cvtColor(roi_img, cv2.COLOR_BGR2RGB)
        
        # Transform and add batch dimension
        input_tensor = self.transform(roi_rgb).unsqueeze(0).to(self.device)
        
        # Inference
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
    
    # ========================================================================
    # ANNOTATION METHODS
    # ========================================================================
    
    def annotate_image(self, image: np.ndarray, 
                      detections: List[Dict]) -> np.ndarray:
        """Create annotated image with bounding boxes and labels"""
        annotated = image.copy()
        
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            class_name = det['class']
            confidence = det['confidence']
            defect_id = det['id']
            '''
             # Color based on confidence
            if confidence >= 0.9:
                color = (0, 255, 0)  # Green - high confidence
            elif confidence >= 0.7:
                color = (0, 165, 255)  # Orange - medium
            else:
                color = (0, 0, 255)  # Red - low confidence
                '''
            # Color based on confidence (Overridden to always be Red as requested)
            color = (0, 0, 255)  # Red
            
            # Draw bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)
            
            # Prepare labels
            label_class = f"#{defect_id} {class_name}"
            label_conf = f"{confidence*100:.1f}%"
            
            # Calculate label box size
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.7
            thickness = 2
            
            (w_class, h_class), _ = cv2.getTextSize(label_class, font, 
                                                     font_scale, thickness)
            (w_conf, h_conf), _ = cv2.getTextSize(label_conf, font, 
                                                   font_scale - 0.1, thickness - 1)
            
            # Draw label background
            label_height = h_class + h_conf + 15
            cv2.rectangle(annotated, (x1, y1 - label_height - 5), 
                         (x1 + max(w_class, w_conf) + 10, y1), color, -1)
            
            # Draw text
            cv2.putText(annotated, label_class, (x1 + 5, y1 - h_conf - 10), 
                       font, font_scale, (255, 255, 255), thickness)
            cv2.putText(annotated, label_conf, (x1 + 5, y1 - 5), 
                       font, font_scale - 0.1, (255, 255, 255), thickness - 1)
        
        return annotated
    
    # ========================================================================
    # MAIN INFERENCE METHOD
    # ========================================================================
    
    def predict(self, defect_image_path: str, 
               template_image_path: Optional[str] = None,
               save_output: bool = False,
               output_dir: str = "inference_output") -> Dict:
        """
        Complete inference pipeline for a single PCB image.
        
        Args:
            defect_image_path: Path to the PCB image with defects
            template_image_path: Path to template (auto-detected if None)
            save_output: Whether to save annotated image and JSON
            output_dir: Directory to save outputs
            
        Returns:
            Dictionary with complete inference results
        """
        start_time = time.time()
        
        defect_path = Path(defect_image_path)
        
        print(f"\n{'='*70}")
        print(f"PROCESSING: {defect_path.name}")
        print(f"{'='*70}")
        
        # Load defect image
        defect_img = cv2.imread(str(defect_path))
        if defect_img is None:
            return {'error': f'Could not read image: {defect_path}'}
        
        # Get template image
        if template_image_path is None:
            pcb_num = defect_path.name.split('_')[0]
            template_path = None
            
            for ext in ['.JPG', '.jpg', '.png', '.PNG']:
                temp_path = self.template_dir / f"{pcb_num}{ext}"
                if temp_path.exists():
                    template_path = temp_path
                    break
            
            if template_path is None:
                return {'error': f'Template not found for PCB #{pcb_num}'}
        else:
            template_path = Path(template_image_path)
        
        template_img = cv2.imread(str(template_path))
        if template_img is None:
            return {'error': f'Could not read template: {template_path}'}
        
        print(f"Template: {template_path.name}")
        
        # Step 1: Align images
        print("\n[1/4] Aligning images...")
        aligned = self.align_images(defect_img, template_img)
        
        # Step 2: Detect defects
        print("[2/4] Detecting defects...")
        thresh, bboxes = self.detect_defects(aligned, template_img)
        print(f"  → Found {len(bboxes)} defect region(s)")
        
        if len(bboxes) == 0:
            elapsed = time.time() - start_time
            result = {
                'image_name': defect_path.name,
                'template_name': template_path.name,
                'num_defects': 0,
                'defects': [],
                'processing_time': elapsed,
                'annotated_image': aligned
            }
            print(f"\n✓ No defects detected ({elapsed:.2f}s)")
            return result
        
        # Step 3: Extract ROIs
        print("[3/4] Extracting ROIs...")
        rois = []
        for bbox in bboxes:
            roi = self.extract_roi(aligned, bbox)
            if roi is not None:
                rois.append(roi)
        
        # Step 4: Classify defects
        print("[4/4] Classifying defects...")
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
            
            print(f"  #{i+1}: {prediction['class']} "
                  f"({prediction['confidence']*100:.1f}%)")
        
        # Create annotated image
        annotated = self.annotate_image(aligned, detections)
        
        elapsed = time.time() - start_time
        
        result = {
            'image_name': defect_path.name,
            'template_name': template_path.name,
            'num_defects': len(detections),
            'defects': detections,
            'processing_time': elapsed,
            'annotated_image': annotated,
            'threshold_image': thresh
        }
        
        print(f"\n✓ Processing complete in {elapsed:.2f}s")
        print(f"{'='*70}\n")
        
        # Save outputs if requested
        if save_output:
            self.save_results(result, output_dir)
        
        return result
    
    def save_results(self, result: Dict, output_dir: str):
        """Save inference results to disk"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)
        
        image_name = Path(result['image_name']).stem
        
        # Save annotated image
        img_path = output_path / f"{image_name}_annotated.png"
        cv2.imwrite(str(img_path), result['annotated_image'])
        
        # Save threshold visualization (for debugging)
        thresh_path = output_path / f"{image_name}_threshold.png"
        cv2.imwrite(str(thresh_path), result['threshold_image'])
        
        # Save JSON results
        json_data = {
            'image_name': result['image_name'],
            'template_name': result['template_name'],
            'num_defects': result['num_defects'],
            'processing_time': round(result['processing_time'], 3),
            'defects': [
                {
                    'id': d['id'],
                    'class': d['class'],
                    'confidence': round(d['confidence'], 4),
                    'bbox': d['bbox']
                }
                for d in result['defects']
            ]
        }
        
        json_path = output_path / f"{image_name}_results.json"
        with open(json_path, 'w') as f:
            json.dump(json_data, f, indent=2)
        
        print(f"Results saved to: {output_path}/")
        print(f"  • {img_path.name}")
        print(f"  • {thresh_path.name}")
        print(f"  • {json_path.name}")


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    
    # Configuration
    MODEL_PATH = "/Users/arya/Documents/Infosys/pcbS/models/pcb_defect_resnet18.pth"
    TEMPLATE_DIR = "/Users/arya/Documents/Infosys/pcbS/PCB_DATASET/PCB_USED"
    
    # Define class names (MUST match training order!)
    CLASS_NAMES = [
        'Missing_hole',
        'Mouse_bite',
        'Open_circuit',
        'Short',
        'Spur',
        'Spurious_copper'
    ]
    
    # Initialize inference pipeline
    inference = PCBInference(
        model_path=MODEL_PATH,
        class_names=CLASS_NAMES,
        template_dir=TEMPLATE_DIR,
        device='cpu'  # Use 'cuda' for GPU
    )
    
    # ========================================================================
    # SINGLE IMAGE INFERENCE
    # ========================================================================
    
    # Option 1: Auto-detect template
    result = inference.predict(
        defect_image_path="/Users/arya/Documents/Infosys/pcbS/PCB_DATASET/images/Mouse_bite/06_mouse_bite_09.jpg",
        save_output=True,
        output_dir="inference_output"
    )
    
    # Option 2: Specify template manually
    """
    result = inference.predict(
        defect_image_path="/path/to/defect.jpg",
        template_image_path="/path/to/template.jpg",
        save_output=True
    )
    """
    
    # Access results
   
    print(f"\nResults Summary:")
    print(f"  Defects detected: {result['num_defects']}")
    print(f"  Processing time: {result['processing_time']:.2f}s")
    
    for defect in result['defects']:
        print(f"\n  Defect #{defect['id']}:")
        print(f"    Class: {defect['class']}")
        print(f"    Confidence: {defect['confidence']:.2%}")
        print(f"    Location: {defect['bbox']}")
    
    # Display annotated image (if using GUI/Jupyter)
    # cv2.imshow('Result', result['annotated_image'])
    # cv2.waitKey(0)
 