import cv2
import os

img_dir = "data/train/images"
mask_dir = "data/train/masks"

files = os.listdir(img_dir)

for i in range(10):  # check 10 samples
    img_name = files[i]

    img_path = os.path.join(img_dir, img_name)
    mask_name = os.path.splitext(img_name)[0] + ".png"
    mask_path = os.path.join(mask_dir, mask_name)

    img = cv2.imread(img_path)
    mask = cv2.imread(mask_path, 0)

    if img is None or mask is None:
        print("Error loading:", img_name)
        continue

    cv2.imshow("IMAGE", img)
    cv2.imshow("MASK", mask)

    print("Showing:", img_name)
    cv2.waitKey(0)

cv2.destroyAllWindows()