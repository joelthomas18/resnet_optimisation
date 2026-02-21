import json
import os
import shutil

ANNOTATION_FILE = "data/coco/annotations/instances_val2017.json"
IMAGE_DIR = "data/coco/images"
OUTPUT_BASE = "data/coco/classification"

TARGET_CLASSES = [
    "person",
    "car",
    "bus",
    "truck",
    "bicycle",
    "motorcycle"
]

# Priority order
CLASS_PRIORITY = [
    "person",
    "car",
    "bus",
    "truck",
    "bicycle",
    "motorcycle"
]

with open(ANNOTATION_FILE, "r") as f:
    coco = json.load(f)

category_id_to_name = {
    cat["id"]: cat["name"] for cat in coco["categories"]
}

# Map image_id -> list of classes in that image
image_classes = {}

for ann in coco["annotations"]:
    class_name = category_id_to_name.get(ann["category_id"])
    if class_name in TARGET_CLASSES:
        image_id = ann["image_id"]
        image_classes.setdefault(image_id, set()).add(class_name)

copied = 0

for img in coco["images"]:
    img_id = img["id"]
    if img_id in image_classes:

        # choose highest priority class
        classes_in_image = image_classes[img_id]
        for cls in CLASS_PRIORITY:
            if cls in classes_in_image:
                chosen_class = cls
                break

        src = os.path.join(IMAGE_DIR, img["file_name"])
        dst = os.path.join(OUTPUT_BASE, chosen_class, img["file_name"])

        if os.path.exists(src):
            shutil.copy(src, dst)
            copied += 1

print("Total classified images:", copied)