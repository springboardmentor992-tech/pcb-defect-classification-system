import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
import numpy as np
from skimage.metrics import structural_similarity as ssim
import os

# ============================================================
# CONFIGURATION
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

CONF_THRESHOLD = 0.4   # Lowered for better recall
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# ============================================================
# LOAD MODEL
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
# ROTATION FUNCTION
# ============================================================

def rotate_image(image, angle):
    if angle == 0:
        return image
    elif angle == 90:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    elif angle == 180:
        return cv2.rotate(image, cv2.ROTATE_180)
    elif angle == 270:
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)

# ============================================================
# MAIN FUNCTION
# ============================================================

def run_inspection(template_path, test_path):

    template = cv2.imread(template_path)
    test_original = cv2.imread(test_path)

    if template is None or test_original is None:
        return [], None

    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    template_gray = cv2.GaussianBlur(template_gray, (5,5), 0)

    # ============================================================
    # STEP 1: AUTOMATIC ROTATION ALIGNMENT
    # ============================================================

    best_score = -1
    best_test = None
    best_angle = 0

    for angle in [0, 90, 180, 270]:

        rotated = rotate_image(test_original, angle)
        rotated = cv2.resize(rotated, (template.shape[1], template.shape[0]))

        gray_rotated = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)
        gray_rotated = cv2.GaussianBlur(gray_rotated, (5,5), 0)

        score, _ = ssim(template_gray, gray_rotated, full=True)

        if score > best_score:
            best_score = score
            best_test = rotated
            best_angle = angle

    print("Selected Rotation:", best_angle)

    test = best_test.copy()

    # ============================================================
    # STEP 2: SSIM DIFFERENCE
    # ============================================================

    gray_test = cv2.cvtColor(test, cv2.COLOR_BGR2GRAY)
    gray_test = cv2.GaussianBlur(gray_test, (5,5), 0)

    _, diff = ssim(template_gray, gray_test, full=True)

    diff = (1 - diff) * 255
    diff = diff.astype("uint8")

    # Threshold
    _, thresh = cv2.threshold(
        diff, 0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # Dilation to enhance small defects
    kernel = np.ones((3,3), np.uint8)
    thresh = cv2.dilate(thresh, kernel, iterations=1)

    # ============================================================
    # STEP 3: FIND CONTOURS
    # ============================================================

    contours, _ = cv2.findContours(
        thresh,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    image_area = template.shape[0] * template.shape[1]
    detections = []

    # ============================================================
    # STEP 4: CLASSIFICATION
    # ============================================================

    for contour in contours:

        area = cv2.contourArea(contour)

        # Improved filtering
        if area < 50 or area > image_area * 0.20:
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

    return detections, test
