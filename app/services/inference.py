import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms, models
from typing import List, Dict, Tuple
import time
from app.core.logging import logger

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
        logger.info(f"PCB Inference Engine initialized on {self.device}")
        logger.info(f"Classes: {', '.join(class_names)}")
    
    def _load_model(self, model_path: str) -> nn.Module:
        """Load trained ResNet18 model"""
        model = models.resnet18(pretrained=False)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, self.num_classes)
        
        # Determine strict loading based on whether it is a full checkpoint or state dict
        # Handling logic from original file
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
        except Exception as e:
            logger.error(f"Error loading model from {model_path}: {e}")
            raise
        
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
                      template_img: np.ndarray) -> Tuple[np.ndarray, List[Tuple[int, int, int, int]]]:
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
                   padding: int = 10) -> Optional[np.ndarray]:
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
            
            # Increase box size for annotations
            padding = 50
            h_img, w_img = annotated.shape[:2]
            
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(w_img, x2 + padding)
            y2 = min(h_img, y2 + padding)
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
        start_time = time.time()
        
        print(f"[INFO] Processing: {defect_name}")
        logger.info(f"Processing image pair: {defect_name} & {template_name}")
        
        # Step 1: Align
        aligned = self.align_images(defect_img, template_img)
        
        # Step 2: Detect
        thresh, bboxes = self.detect_defects(aligned, template_img)
        logger.info(f"Found {len(bboxes)} defect region(s)")
        
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
            logger.info(f"Detected #{i+1}: {prediction['class']} ({prediction['confidence']*100:.1f}%)")
        
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
