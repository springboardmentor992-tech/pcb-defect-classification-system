import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
import numpy as np
from skimage.metrics import structural_similarity as ssim
import os

# ============================================================
# CONFIG
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "/Users/cherukurajesh/Desktop/pcb_defect_system/backend/pcb_resnet_model.pth ")

CLASS_NAMES = [
    "Missing_hole",
    "Mouse_bite",
    "Open_circuit",
    "Short",
    "Spur",
    "Spurious_copper"
]

CONF_THRESHOLD = 0.6
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# ============================================================
# LOAD MODEL (LOAD ONLY ONCE)
# ============================================================

model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 6)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

print("✅ Model loaded successfully")

# ============================================================
# TRANSFORM
# ============================================================

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

# ============================================================
# MAIN FUNCTION
# ============================================================

def run_inspection(template_path, test_path):

    template = cv2.imread(template_path)
    test = cv2.imread(test_path)

    if template is None or test is None:
        return []

    test = cv2.resize(test, (template.shape[1], template.shape[0]))

    gray_template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    gray_test = cv2.cvtColor(test, cv2.COLOR_BGR2GRAY)

    gray_template = cv2.GaussianBlur(gray_template, (5,5), 0)
    gray_test = cv2.GaussianBlur(gray_test, (5,5), 0)

    _, diff = ssim(gray_template, gray_test, full=True)

    diff = (1 - diff) * 255
    diff = diff.astype("uint8")

    _, thresh = cv2.threshold(
        diff, 0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    contours, _ = cv2.findContours(
        thresh,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    image_area = template.shape[0] * template.shape[1]
    detections = []

    for contour in contours:
        area = cv2.contourArea(contour)

        if area < 150 or area > image_area * 0.15:
            continue

        x, y, w, h = cv2.boundingRect(contour)

        pad = 12
        x = max(0, x - pad)
        y = max(0, y - pad)
        w = min(test.shape[1] - x, w + 2*pad)
        h = min(test.shape[0] - y, h + 2*pad)

        roi = test[y:y+h, x:x+w]
        if roi.size == 0:
            continue

        roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        tensor = transform(roi_rgb).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            output_pred = model(tensor)
            probs = F.softmax(output_pred, dim=1)
            conf, pred = torch.max(probs, 1)

        confidence = conf.item()

        if confidence < CONF_THRESHOLD:
            continue

        class_name = CLASS_NAMES[pred.item()]

        detections.append({
            "label": class_name,
            "confidence": round(confidence, 3),
            "box": [int(x), int(y), int(w), int(h)]
        })

    return detections