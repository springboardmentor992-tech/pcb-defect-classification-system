import os
import random
import shutil
from pathlib import Path

# ============================================================
# 🔹 PATHS
# ============================================================

SOURCE_DIR = Path("/Users/cherukurajesh/Desktop/pcb_defect_system/PCB_DATASET_Extracted")
OUTPUT_DIR = Path("/Users/cherukurajesh/Desktop/pcb_defect_system/PCB_DATASET_SPLIT")

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

random.seed(42)

# ============================================================
# 🔹 CREATE OUTPUT STRUCTURE
# ============================================================

for split in ["train", "val", "test"]:
    for class_folder in SOURCE_DIR.iterdir():
        if class_folder.is_dir():
            (OUTPUT_DIR / split / class_folder.name).mkdir(parents=True, exist_ok=True)

print("Created split folder structure.\n")

# ============================================================
# 🔹 SPLIT PER CLASS
# ============================================================

total_counts = {"train": 0, "val": 0, "test": 0}

for class_folder in SOURCE_DIR.iterdir():

    if not class_folder.is_dir():
        continue

    class_name = class_folder.name
    images = list(class_folder.glob("*.*"))

    random.shuffle(images)

    total_images = len(images)
    train_end = int(total_images * TRAIN_RATIO)
    val_end = train_end + int(total_images * VAL_RATIO)

    train_images = images[:train_end]
    val_images = images[train_end:val_end]
    test_images = images[val_end:]

    print(f"{class_name}: {total_images} images")
    print(f"   Train: {len(train_images)}")
    print(f"   Val:   {len(val_images)}")
    print(f"   Test:  {len(test_images)}\n")

    for img in train_images:
        shutil.copy(img, OUTPUT_DIR / "train" / class_name / img.name)
        total_counts["train"] += 1

    for img in val_images:
        shutil.copy(img, OUTPUT_DIR / "val" / class_name / img.name)
        total_counts["val"] += 1

    for img in test_images:
        shutil.copy(img, OUTPUT_DIR / "test" / class_name / img.name)
        total_counts["test"] += 1

print("Split Completed!\n")
print("Total Images:")
print(total_counts)
