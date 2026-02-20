"""
PCB Defect Inference Script
Loads a saved PyTorch model and classifies a single input image.
"""

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
from pathlib import Path

# ============================================================================
# CONFIGURATION
# ============================================================================
MODEL_PATH = "/Users/arya/Documents/Infosys/pcbS/models/pcb_defect_resnet18.pth"
TEST_IMAGE_PATH = "/Users/arya/Documents/Infosys/pcbS/processed_rois/rois/Open_circuit/04_open_circuit_13_roi_1.png"
# IMPORTANT: These must match the folder names in your training dataset exactly, in alphabetical order.
# If you aren't sure, check your 'rois' folder.
CLASS_NAMES = [
    'Missing_hole', 
    'Mouse_bite', 
    'Open_circuit', 
    'Short', 
    'Spur', 
    'Spurious_copper'
]

# Set device (Auto-detects M1 Mac, CUDA, or CPU)
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

def load_trained_model(model_path, num_classes):
    """
    Reconstructs the ResNet18 architecture and loads saved weights.
    """
    print(f"Loading model from {model_path}...")
    
    # 1. Initialize the same architecture used in training
    model = models.resnet18(weights=None) # No need for ImageNet weights now, we are loading our own
    
    # 2. Re-configure the final layer to match the number of classes
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    
    # 3. Load the state dictionary (the weights)
    # map_location ensures weights load correctly even if trained on GPU and running on CPU
    model.load_state_dict(torch.load(model_path, map_location=device))
    
    # 4. Set to Evaluation Mode (Crucial: turns off Dropout and Batch Norm updates)
    model = model.to(device)
    model.eval()
    
    return model

def process_image(image_path):
    """
    Applies the same transformations used during validation/testing.
    """
    transform = transforms.Compose([
        transforms.Resize((128, 128)), # Must match training size
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Open image (PIL is standard for PyTorch transforms)
    image = Image.open(image_path).convert('RGB')
    
    # Apply transforms and add Batch Dimension (C, H, W) -> (1, C, H, W)
    image_tensor = transform(image).unsqueeze(0)
    
    return image_tensor.to(device)

def predict_defect(model, image_tensor, class_names):
    """
    Runs inference and returns prediction with confidence.
    """
    with torch.no_grad(): # Disable gradient calculation for speed
        outputs = model(image_tensor)
        
        # Apply Softmax to get probabilities (0-100%)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        
        # Get the highest probability class
        confidence, preds = torch.max(probs, 1)
        
        predicted_idx = preds.item()
        predicted_prob = confidence.item()
        predicted_class = class_names[predicted_idx]
        
        return predicted_class, predicted_prob

# ============================================================================
# MAIN EXECUTION
# ============================================================================
if __name__ == "__main__":
    # Check paths
    if not Path(MODEL_PATH).exists():
        print(f"Error: Model not found at {MODEL_PATH}")
        exit()
    if not Path(TEST_IMAGE_PATH).exists():
        print(f"Error: Image not found at {TEST_IMAGE_PATH}")
        exit()

    # 1. Load Model
    model = load_trained_model(MODEL_PATH, len(CLASS_NAMES))
    
    # 2. Process Image
    img_tensor = process_image(TEST_IMAGE_PATH)
    
    # 3. Predict
    print(f"\nTesting Image: {Path(TEST_IMAGE_PATH).name}")
    label, confidence = predict_defect(model, img_tensor, CLASS_NAMES)
    
    # 4. Output Results
    print("-" * 30)
    print(f"Prediction: {label}")
    print(f"Confidence: {confidence * 100:.2f}%")
    print("-" * 30)
    
    if confidence < 0.60:
        print("⚠️ Warning: Low confidence. Model is unsure.")