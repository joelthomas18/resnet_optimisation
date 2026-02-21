import os
import cv2

SUBSET_DIR = "data/coco/subset"
images = os.listdir(SUBSET_DIR)

for img_name in images[:10]:
    img = cv2.imread(os.path.join(SUBSET_DIR, img_name))
    if img is not None:
        cv2.imshow("COCO Surveillance Subset", img)
        cv2.waitKey(600)

cv2.destroyAllWindows()
