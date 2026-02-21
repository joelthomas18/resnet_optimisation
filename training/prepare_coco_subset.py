import json
import os
import shutil

ANNOTATION_FILE = "data/coco/annotations/instances_val2017.json"
IMAGE_DIR = "data/coco/images"
OUTPUT_DIR = "data/coco/subset"

TARGET_CLASSES = {
    "person", "car", "bus", "truck", "bicycle", "motorcycle"
}

os.makedirs(OUTPUT_DIR, exist_ok=True)

with open(ANNOTATION_FILE, "r") as f:
    coco = json.load(f)

# Safety check
assert "categories" in coco, "ERROR: Not an instance annotation file!"

category_id_to_name = {
    cat["id"]: cat["name"] for cat in coco["categories"]
}

target_category_ids = {
    cid for cid, name in category_id_to_name.items()
    if name in TARGET_CLASSES
}

selected_image_ids = set()
for ann in coco["annotations"]:
    if ann["category_id"] in target_category_ids:
        selected_image_ids.add(ann["image_id"])

print("Images with surveillance objects:", len(selected_image_ids))

copied = 0
for img in coco["images"]:
    if img["id"] in selected_image_ids:
        src = os.path.join(IMAGE_DIR, img["file_name"])
        dst = os.path.join(OUTPUT_DIR, img["file_name"])
        if os.path.exists(src):
            shutil.copy(src, dst)
            copied += 1

print("Images copied:", copied)