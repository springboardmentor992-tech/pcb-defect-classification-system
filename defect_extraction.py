import cv2
import numpy as np
from pathlib import Path
from skimage.metrics import structural_similarity as ssim

# ============================================================
# 🔹 PATH CONFIGURATION
# ============================================================

BASE_PATH = Path("/Users/cherukurajesh/Desktop/pcb_defect_system/PCB_DATASET")

DEFECT_PARENT = BASE_PATH / "images"
TEMPLATE_FOLDER = BASE_PATH / "PCB_USED"

OUTPUT_PARENT = Path("/Users/cherukurajesh/Desktop/pcb_defect_system/PCB_DATASET_Extracted")
OUTPUT_PARENT.mkdir(parents=True, exist_ok=True)

PATCH_SIZE = (128, 128)
MIN_DEFECT_AREA = 300
MAX_DEFECT_RATIO = 0.10   # prevents full PCB detection

# ============================================================
# 🔹 LOAD ALL TEMPLATE IMAGES
# ============================================================

templates = []

for file in TEMPLATE_FOLDER.iterdir():
    if file.suffix.lower() in [".jpg", ".png", ".jpeg"]:
        img = cv2.imread(str(file))
        if img is not None:
            templates.append((file.name, img))

print(f"Loaded {len(templates)} templates\n")

# ============================================================
# 🔹 ALIGN FUNCTION
# ============================================================

def align_images(template, image):

    gray_t = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    gray_i = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create(4000)
    kp1, des1 = orb.detectAndCompute(gray_t, None)
    kp2, des2 = orb.detectAndCompute(gray_i, None)

    if des1 is None or des2 is None:
        return image

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(des1, des2)

    if len(matches) < 20:
        return image

    matches = sorted(matches, key=lambda x: x.distance)[:50]

    src_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1,1,2)
    dst_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1,1,2)

    H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    if H is None:
        return image

    aligned = cv2.warpPerspective(
        image,
        H,
        (template.shape[1], template.shape[0])
    )

    return aligned

# ============================================================
# 🔹 FIND BEST TEMPLATE USING SSIM
# ============================================================

def find_best_template(test_img):

    best_template = None
    best_score = -1

    for _, template in templates:

        resized_test = cv2.resize(
            test_img,
            (template.shape[1], template.shape[0])
        )

        template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        test_gray = cv2.cvtColor(resized_test, cv2.COLOR_BGR2GRAY)

        score, _ = ssim(template_gray, test_gray, full=True)

        if score > best_score:
            best_score = score
            best_template = template

    return best_template

# ============================================================
# 🔹 MAIN PROCESS
# ============================================================

print("Starting Accurate Defect Extraction...\n")

total_defects = 0

for class_folder in DEFECT_PARENT.iterdir():

    if not class_folder.is_dir():
        continue

    defect_class = class_folder.name
    print(f"Processing Class: {defect_class}")

    output_class_dir = OUTPUT_PARENT / defect_class
    output_class_dir.mkdir(parents=True, exist_ok=True)

    for image_path in class_folder.iterdir():

        if image_path.suffix.lower() not in [".jpg", ".png", ".jpeg"]:
            continue

        test_img = cv2.imread(str(image_path))
        if test_img is None:
            continue

        template_img = find_best_template(test_img)
        if template_img is None:
            continue

        # Resize
        test_img = cv2.resize(
            test_img,
            (template_img.shape[1], template_img.shape[0])
        )

        # Align
        aligned = align_images(template_img, test_img)

        # Grayscale + blur
        template_gray = cv2.cvtColor(template_img, cv2.COLOR_BGR2GRAY)
        test_gray = cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY)

        template_gray = cv2.GaussianBlur(template_gray, (5,5), 0)
        test_gray = cv2.GaussianBlur(test_gray, (5,5), 0)

        # SSIM difference
        score, diff = ssim(template_gray, test_gray, full=True)

        diff = (1 - diff)
        diff = (diff * 255).astype("uint8")

        # Threshold
        _, thresh = cv2.threshold(
            diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        kernel = np.ones((5,5), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        image_area = template_gray.shape[0] * template_gray.shape[1]
        defect_count = 0

        for cnt in contours:

            area = cv2.contourArea(cnt)

            if area < MIN_DEFECT_AREA:
                continue

            if area > MAX_DEFECT_RATIO * image_area:
                continue

            x, y, w, h = cv2.boundingRect(cnt)

            pad = 6
            x = max(0, x - pad)
            y = max(0, y - pad)
            w = min(aligned.shape[1] - x, w + 2*pad)
            h = min(aligned.shape[0] - y, h + 2*pad)

            crop = aligned[y:y+h, x:x+w]
            crop = cv2.resize(crop, PATCH_SIZE)

            save_name = f"{image_path.stem}_defect_{defect_count}.png"
            cv2.imwrite(str(output_class_dir / save_name), crop)

            defect_count += 1
            total_defects += 1

        print(f"   {image_path.name} → {defect_count} defects")

print("\nExtraction Completed!")
print(f"Total defects extracted: {total_defects}")
