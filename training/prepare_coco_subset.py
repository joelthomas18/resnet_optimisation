import os
import json
import cv2
import numpy as np
from collections import defaultdict

# ==============================
# CONFIG
# ==============================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ANNOTATIONS_DIR = os.path.join(BASE_DIR, "data", "coco", "annotations")
IMAGES_DIR = os.path.join(BASE_DIR, "data", "coco", "images")
OUTPUT_DIR = os.path.join(BASE_DIR, "data", "coco", "classification")

TARGET_CLASSES = ["person", "car", "bus", "truck", "bicycle", "motorcycle"]
MAX_IMAGES_PER_CLASS = 1200 # Slightly increased for better generalization
MIN_BBOX_SIZE = 50 # Ignore very small objects that are blurry

os.makedirs(OUTPUT_DIR, exist_ok=True)
for cls in TARGET_CLASSES:
    os.makedirs(os.path.join(OUTPUT_DIR, cls), exist_ok=True)

# ==============================
# PROCESS DATASET
# ==============================
annotation_files = [
    ("instances_train2017.json", "train2017"),
    ("instances_val2017.json", "val2017")
]

class_counts = defaultdict(int)

for ann_file, image_folder in annotation_files:
    print(f"Processing {ann_file}...")
    ann_path = os.path.join(ANNOTATIONS_DIR, ann_file)
    img_root = os.path.join(IMAGES_DIR, image_folder)

    with open(ann_path, "r") as f:
        coco_data = json.load(f)

    category_map = {cat["id"]: cat["name"] for cat in coco_data["categories"] if cat["name"] in TARGET_CLASSES}
    image_id_to_filename = {img["id"]: img["file_name"] for img in coco_data["images"]}

    for ann in coco_data["annotations"]:
        category_id = ann["category_id"]
        if category_id not in category_map:
            continue

        class_name = category_map[category_id]
        if class_counts[class_name] >= MAX_IMAGES_PER_CLASS:
            continue

        # Check bbox size to ensure quality
        bbox = ann["bbox"] # [x, y, width, height]
        if bbox[2] < MIN_BBOX_SIZE or bbox[3] < MIN_BBOX_SIZE:
            continue

        image_id = ann["image_id"]
        file_name = image_id_to_filename[image_id]
        src_path = os.path.join(img_root, file_name)

        if os.path.exists(src_path):
            img = cv2.imread(src_path)
            if img is None: continue
            
            # Crop logic
            x, y, w, h = map(int, bbox)
            crop = img[max(0, y):y+h, max(0, x):x+w]
            
            if crop.size == 0: continue

            dst_name = f"{image_id}_{ann['id']}.jpg"
            dst_path = os.path.join(OUTPUT_DIR, class_name, dst_name)
            cv2.imwrite(dst_path, crop)
            
            class_counts[class_name] += 1

print("\nObject Crop Complete!")
for cls, count in class_counts.items():
    print(f"{cls}: {count}")