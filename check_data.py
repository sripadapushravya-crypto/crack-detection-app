import os

img_dir = "data/train/images"
mask_dir = "data/train/masks"

img_files = os.listdir(img_dir)
mask_files = os.listdir(mask_dir)

print("Total images:", len(img_files))
print("Total masks:", len(mask_files))

# remove extensions
img_names = set([os.path.splitext(f)[0] for f in img_files])
mask_names = set([os.path.splitext(f)[0] for f in mask_files])

missing_masks = img_names - mask_names
extra_masks = mask_names - img_names

print("\nMissing masks:", len(missing_masks))
for name in list(missing_masks)[:5]:
    print("  ", name)

print("\nExtra masks:", len(extra_masks))
for name in list(extra_masks)[:5]:
    print("  ", name)