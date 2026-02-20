import cv2
import numpy as np
import os
import sys

# =========================================================
# 🔹 BASE DATASET PATH
# =========================================================
BASE_PATH = "/Users/cherukurajesh/Desktop/pcb_defect_system/PCB_DATASET"

images_path = os.path.join(BASE_PATH, "images")
template_path = os.path.join(BASE_PATH, "PCB_USED")

output_base = "/Users/cherukurajesh/Desktop/pcb_defect_system/PCB_DATASET_PREPROCESSED"
output_images_path = os.path.join(output_base, "images")
output_template_path = os.path.join(output_base, "PCB_USED")

# =========================================================
# 🔹 CHECK PATHS
# =========================================================
if not os.path.exists(images_path):
    print("❌ Images folder not found:", images_path)
    sys.exit()

if not os.path.exists(template_path):
    print("❌ PCB_USED folder not found:", template_path)
    sys.exit()

os.makedirs(output_images_path, exist_ok=True)
os.makedirs(output_template_path, exist_ok=True)

print("✅ Dataset Found Successfully!")

# =========================================================
# 🔹 ROTATION (Stable Method)
# Only force portrait orientation
# =========================================================
def correct_rotation(image):
    h, w = image.shape[:2]

    # If image is landscape, rotate to portrait
    if w > h:
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)

    return image

# =========================================================
# 🔹 PREPROCESS FUNCTION
# =========================================================
def preprocess(img):

    # 1️⃣ Ensure consistent orientation
    aligned = correct_rotation(img)

    # 2️⃣ Resize to fixed size
    resized = cv2.resize(aligned, (800, 800))

    # 3️⃣ Convert to grayscale
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

    # 4️⃣ Noise removal
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # 5️⃣ Contrast enhancement (important for PCB traces)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(blur)

    # 6️⃣ Normalize intensity
    normalized = cv2.normalize(enhanced, None, 0, 255, cv2.NORM_MINMAX)

    return normalized

# =========================================================
# 🔹 PROCESS TEMPLATE IMAGES
# =========================================================
print("\n🔹 Processing Template Images...")

for filename in os.listdir(template_path):

    if filename.lower().endswith(('.jpg', '.png', '.jpeg')):

        img_path = os.path.join(template_path, filename)
        img = cv2.imread(img_path)

        if img is None:
            continue

        processed = preprocess(img)

        save_path = os.path.join(output_template_path, filename)
        cv2.imwrite(save_path, processed)

        print(f"   ✅ Template: {filename}")

# =========================================================
# 🔹 PROCESS ALL DEFECT CLASSES
# =========================================================
print("\n🔹 Processing All PCB Types...")

for class_name in os.listdir(images_path):

    class_input_path = os.path.join(images_path, class_name)

    if not os.path.isdir(class_input_path):
        continue

    class_output_path = os.path.join(output_images_path, class_name)
    os.makedirs(class_output_path, exist_ok=True)

    print(f"\n📂 Processing Class: {class_name}")

    for filename in os.listdir(class_input_path):

        if filename.lower().endswith(('.jpg', '.png', '.jpeg')):

            img_path = os.path.join(class_input_path, filename)
            img = cv2.imread(img_path)

            if img is None:
                continue

            processed = preprocess(img)

            save_path = os.path.join(class_output_path, filename)
            cv2.imwrite(save_path, processed)

            print(f"   ✅ {filename}")

print("\n🎉 All PCB types preprocessed successfully and consistently!")
