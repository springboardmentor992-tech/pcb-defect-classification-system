"""
PCB Defect Detection & ROI Extraction Pipeline
Strict adherence to logic from 'pcb_defect_classification_pipeline.py'
"""

import cv2
import numpy as np
from pathlib import Path
import json
from tqdm import tqdm
import xml.etree.ElementTree as ET
from collections import defaultdict
import os

class PCBDefectPipeline:
    """
    Pipeline that strictly implements the logic provided in the user's prototype file.
    Phase 1: Alignment (ORB + Homography)
    Phase 2: Subtraction -> Blur Diff -> Normalize -> Otsu -> Dilate -> Erode
    Phase 3: Contour Extraction (RETR_TREE) -> ROI Slicing
    """
    
    # Logic from your file comments:
    # "For open circuit, use (7,7); For mouse bite, spur, spurious copper, use (9,9); For short, use (7,7)"
    BLUR_KERNELS = {
        'Open_circuit': (7, 7),
        'Short': (7, 7),
        'Mouse_bite': (7, 7),
        'Spur': (9, 9),
        'Spurious_copper': (9, 9),
        'Missing_hole': (9, 9),
        'default': (9, 9)
    }
    
    def __init__(self, dataset_root=None, output_dir="output", template_dir=None, annotation_dir=None, 
                 roi_size=(128, 128)):
        self.dataset_root = Path(dataset_root) if dataset_root else None
        self.output_dir = Path(output_dir)
        self.template_dir = Path(template_dir) if template_dir else None
        self.annotation_dir = Path(annotation_dir) if annotation_dir else None
        self.roi_size = roi_size
        
        # Create output directories
        self.roi_dir = self.output_dir / 'rois'
        self.viz_dir = self.output_dir / 'visualizations'
        
        self.roi_dir.mkdir(parents=True, exist_ok=True)
        self.viz_dir.mkdir(parents=True, exist_ok=True)
        
        self.metadata = []
        self.stats = defaultdict(int)
    
    def align_images(self, defect_img, template_img):
        """
        Phase 1: Alignment using ORB as defined in uploaded file.
        """
        try:
            # Convert to grayscale for feature detection
            defect_gray = cv2.cvtColor(defect_img, cv2.COLOR_BGR2GRAY)
            template_gray = cv2.cvtColor(template_img, cv2.COLOR_BGR2GRAY)
            
            # Initialize ORB detector
            orb = cv2.ORB_create()
            
            # Find keypoints and descriptors
            kp1, des1 = orb.detectAndCompute(defect_gray, None)
            kp2, des2 = orb.detectAndCompute(template_gray, None)
            
            # Safety check from your script
            if des1 is None or des2 is None or len(kp1) < 2 or len(kp2) < 2:
                # print("Could not find enough keypoints for alignment.")
                return defect_img.copy()
            
            # Match descriptors
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = bf.match(des1, des2)
            matches = sorted(matches, key=lambda x: x.distance)
            
            # "Only take a subset of best matches if there are too many"
            num_good_matches = min(50, len(matches))
            good_matches = matches[:num_good_matches]
            
            # "Need at least 4 points to find homography"
            if len(good_matches) > 4:
                points1 = np.float32([kp1[m.queryIdx].pt for m in good_matches])
                points2 = np.float32([kp2[m.trainIdx].pt for m in good_matches])
                
                # Find homography
                h, mask = cv2.findHomography(points1, points2, cv2.RANSAC, 5.0)
                
                if h is not None:
                    height, width = template_img.shape[:2]
                    aligned_defect = cv2.warpPerspective(defect_img, h, (width, height))
                    return aligned_defect
                else:
                    # print("Could not find homography. Alignment failed.")
                    return defect_img.copy()
            else:
                # print("Not enough good matches. Alignment failed.")
                return defect_img.copy()
                
        except Exception as e:
            print(f"Alignment Exception: {e}")
            return defect_img.copy()
    
    def compute_difference_map(self, aligned_defect, template_img, class_label):
        """
        Phase 2: Subtraction, Blur, Normalize, Otsu, Morphology.
        Strictly follows logic from 'processing' section of uploaded file.
        """
        # Convert to grayscale
        defect_gray = cv2.cvtColor(aligned_defect, cv2.COLOR_BGR2GRAY)
        template_gray = cv2.cvtColor(template_img, cv2.COLOR_BGR2GRAY)
        
        # 1. "defect_blur = cv2.GaussianBlur(defect_gray, (5, 5), 0)"
        defect_blur = cv2.GaussianBlur(defect_gray, (5, 5), 0)
        template_blur = cv2.GaussianBlur(template_gray, (5, 5), 0)
        
        # 2. "diff_map = cv2.absdiff(defect_blur, template_blur)"
        diff_map = cv2.absdiff(defect_blur, template_blur)
        
        # 3. Class-specific Blur on Diff Map
        # "For open circuit use (7,7)... For mouse bite use (9,9)"
        kernel_size = self.BLUR_KERNELS.get(class_label, self.BLUR_KERNELS['default'])
        print(f"kernel_size:{kernel_size}")
        diff_map = cv2.GaussianBlur(diff_map, kernel_size, 0)
        
        # 4. "diff_map = cv2.normalize(diff_map, None, 0, 255, cv2.NORM_MINMAX)"
        diff_map = cv2.normalize(diff_map, None, 0, 255, cv2.NORM_MINMAX)
        diff_map = cv2.GaussianBlur(diff_map, kernel_size, 0)
        
        # 4. "diff_map = cv2.normalize(diff_map, None, 0, 255, cv2.NORM_MINMAX)"
        diff_map = cv2.normalize(diff_map, None, 0, 255, cv2.NORM_MINMAX)
        # 5. Otsu Thresholding
        # "ret_val, thresh = cv2.threshold(diff_map, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)"
        ret_val, thresh = cv2.threshold(diff_map, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        print(f"ret_val:{ret_val}")
        # 6. Morphological Operations (Dilation then Erosion)
        kernel = np.ones((3, 3), np.uint8)
        
        # "dilated_thresh = cv2.dilate(thresh, kernel, iterations = 1)"
        dilated_thresh = cv2.dilate(thresh, kernel, iterations=1)
        
        # "eroded_dilated_thresh = cv2.erode(dilated_thresh, kernel, iterations = 1)"
        eroded_dilated_thresh = cv2.erode(dilated_thresh, kernel, iterations=1)
        
        # "Update thresh to the morphologically processed image"
        final_thresh = eroded_dilated_thresh
        
        return final_thresh
    
    def extract_contours_and_filter(self, thresh_img, aligned_defect, 
                                    min_width=5, max_width=1000, 
                                    min_height=5, max_height=1000):
        """
        Extracts contours and filters by dimension.
        Strictly follows: cv2.findContours(..., cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        """
        # "contours, hierarchy = cv2.findContours(image=thresh, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)"
        contours, hierarchy = cv2.findContours(image=thresh_img, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
        
        height, width = aligned_defect.shape[:2]
        pcb_roi_bbox = (0, 0, width, height)
        x_pcb_min, y_pcb_min, x_pcb_max, y_pcb_max = pcb_roi_bbox
        
        filtered_bounding_boxes = []
        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            
            x_contour_min, y_contour_min = x, y
            x_contour_max, y_contour_max = x + w, y + h
            
            # "Check if the contour's bounding box is entirely within the PCB ROI"
            is_within_pcb_roi = (
                x_contour_min >= x_pcb_min and
                y_contour_min >= y_pcb_min and
                x_contour_max <= x_pcb_max and
                y_contour_max <= y_pcb_max
            )
            
            # "Check if the bounding box dimensions meet your criteria"
            if is_within_pcb_roi and min_width <= w <= max_width and min_height <= h <= max_height:
                filtered_bounding_boxes.append((x, y, x + w, y + h))
                
        return filtered_bounding_boxes
    
    def extract_rois(self, image, bboxes, padding=10):
        rois = []
        height, width = image.shape[:2]
        
        for idx, bbox in enumerate(bboxes):
            x1, y1, x2, y2 = bbox
            x1_pad, y1_pad = max(0, x1 - padding), max(0, y1 - padding)
            x2_pad, y2_pad = min(width, x2 + padding), min(height, y2 + padding)
            
            roi = image[y1_pad:y2_pad, x1_pad:x2_pad]
            
            if roi.size > 0:
                roi_resized = cv2.resize(roi, self.roi_size, interpolation=cv2.INTER_AREA)
                rois.append({
                    'image': roi_resized,
                    'bbox': (x1_pad, y1_pad, x2_pad, y2_pad),
                    'defect_id': idx
                })
        return rois

    def visualize_detections(self, image, detected_bboxes, ground_truth_bboxes=None):
        viz_img = image.copy()
        # "color = (0, 0, 255) # Yellow color (Note: Code says BGR Red, comments say Yellow)"
        # We will use Red (0,0,255) as per the code values in your file
        for idx, bbox in enumerate(detected_bboxes):
            x1, y1, x2, y2 = bbox
            cv2.rectangle(viz_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(viz_img, f"D{idx+1}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
        if ground_truth_bboxes:
            for idx, bbox in enumerate(ground_truth_bboxes):
                x_min, y_min, x_max, y_max = bbox
                cv2.rectangle(viz_img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                
        return viz_img
    
    def load_ground_truth_annotations(self, defect_image_path):
        if not self.annotation_dir: return []
        
        # Logic to find XML file
        image_name = defect_image_path.stem
        # Check standard path structure
        possible_paths = [
            self.annotation_dir / defect_image_path.parent.name / f"{image_name}.xml",
            self.annotation_dir / f"{image_name}.xml"
        ]
        
        xml_path = None
        for p in possible_paths:
            if p.exists():
                xml_path = p
                break
        
        true_bboxes = []
        if xml_path:
            try:
                tree = ET.parse(xml_path)
                root = tree.getroot()
                for obj in root.findall('object'):
                    bndbox = obj.find('bndbox')
                    xmin = int(bndbox.find('xmin').text)
                    ymin = int(bndbox.find('ymin').text)
                    xmax = int(bndbox.find('xmax').text)
                    ymax = int(bndbox.find('ymax').text)
                    true_bboxes.append([xmin, ymin, xmax, ymax])
            except Exception:
                pass
        return true_bboxes

    # =========================================================================
    # SINGLE IMAGE MODE
    # =========================================================================
    def process_single_file_mode(self, defect_path, template_path, class_label="Unknown"):
        print(f"\n--- PROCESSING SINGLE IMAGE: {Path(defect_path).name} ---")
        
        d_path = Path(defect_path)
        t_path = Path(template_path)
        
        if not d_path.exists() or not t_path.exists():
            print("Error: Image files not found.")
            return

        defect_img = cv2.imread(str(d_path))
        template_img = cv2.imread(str(t_path))
        
        # Load GT
        gt = self.load_ground_truth_annotations(d_path)
        if gt: print(f"Loaded {len(gt)} Ground Truth annotations.")

        # Phase 1: Alignment
        aligned = self.align_images(defect_img, template_img)
        
        # Phase 2: Processing (Strict adherence)
        thresh = self.compute_difference_map(aligned, template_img, class_label)
        
        # Phase 3: Extraction
        bboxes = self.extract_contours_and_filter(thresh, aligned)
        print(f"Found {len(bboxes)} candidate defects.")
        
        # Extract ROIs
        rois = self.extract_rois(aligned, bboxes)
        
        # Save Outputs
        single_out = self.roi_dir / "single_test"
        single_out.mkdir(exist_ok=True, parents=True)
        
        for i, r in enumerate(rois):
            out_name = single_out / f"{d_path.stem}_roi_{i+1}.png"
            cv2.imwrite(str(out_name), r['image'])
        
        viz_img = self.visualize_detections(aligned, bboxes, gt)
        viz_out = self.viz_dir / "single_test" / f"{d_path.stem}_viz.png"
        (self.viz_dir / "single_test").mkdir(exist_ok=True, parents=True)
        cv2.imwrite(str(viz_out), viz_img)
        
        print(f"Saved {len(rois)} ROIs to {single_out}")
        print(f"Saved visualization to {viz_out}")

    # =========================================================================
    # BATCH MODE
    # =========================================================================
    def process_dataset(self):
        print("=" * 60)
        print("PCB DEFECT PIPELINE - BATCH MODE")
        print("=" * 60)
        
        class_dirs = [d for d in self.dataset_root.iterdir() if d.is_dir()]
        
        for class_dir in class_dirs:
            class_label = class_dir.name
            images = list(class_dir.glob('*.[jpJP][pnPN]*'))
            
            if not images: continue
            
            print(f"\nProcessing Class: {class_label} ({len(images)} images)")
            
            for img_path in tqdm(images):
                pcb_id = img_path.name.split('_')[0]
                # Try common template extensions
                t_path = None
                for ext in ['.JPG', '.jpg', '.png']:
                    if (self.template_dir / f"{pcb_id}{ext}").exists():
                        t_path = self.template_dir / f"{pcb_id}{ext}"
                        break
                
                if not t_path: continue
                    
                d_img = cv2.imread(str(img_path))
                t_img = cv2.imread(str(t_path))
                
                aligned = self.align_images(d_img, t_img)
                thresh = self.compute_difference_map(aligned, t_img, class_label)
                bboxes = self.extract_contours_and_filter(thresh, aligned)
                rois = self.extract_rois(aligned, bboxes)
                
                # Save
                save_dir = self.roi_dir / class_label
                save_dir.mkdir(exist_ok=True)
                for r in rois:
                    fname = save_dir / f"{img_path.stem}_roi_{r['defect_id']+1}.png"
                    cv2.imwrite(str(fname), r['image'])
                    
                # Viz
                viz = self.visualize_detections(aligned, bboxes)
                viz_save = self.viz_dir / class_label
                viz_save.mkdir(exist_ok=True)
                cv2.imwrite(str(viz_save / f"{img_path.stem}_viz.png"), viz)

# ============================================================================
# MAIN EXECUTION
# ============================================================================
if __name__ == "__main__":
    
    # 1. SETUP PATHS
    DATASET_ROOT = "/Users/arya/Documents/Infosys/pcbS/PCB_DATASET/images"
    TEMPLATE_DIR = "/Users/arya/Documents/Infosys/pcbS/PCB_DATASET/PCB_USED"
    OUTPUT_DIR   = "/Users/arya/Documents/Infosys/pcbS/processed_rois"
    ANNOTATION_DIR = "/Users/arya/Documents/Infosys/pcbS/PCB_DATASET/Annotations"

    # Initialize Pipeline
    pipeline = PCBDefectPipeline(
        dataset_root=DATASET_ROOT,
        output_dir=OUTPUT_DIR,
        template_dir=TEMPLATE_DIR,
        annotation_dir=ANNOTATION_DIR
    )

    # ---------------------------------------------------------
    # OPTION 1: SINGLE IMAGE MODE (Uncomment to test one file)
    # ---------------------------------------------------------
    '''
    pipeline.process_single_file_mode(
        defect_path="/Users/arya/Documents/Infosys/pcbS/PCB_DATASET/images/Mouse_bite/01_mouse_bite_01.jpg",
        template_path="/Users/arya/Documents/Infosys/pcbS/PCB_DATASET/PCB_USED/01.JPG",
        class_label="Mouse_bite"
    )
    '''
    # ---------------------------------------------------------
    # OPTION 2: BATCH MODE (Uncomment to run everything)
    # ---------------------------------------------------------
pipeline.process_dataset()