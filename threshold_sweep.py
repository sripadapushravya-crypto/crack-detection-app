import os
import cv2
import torch
import numpy as np
import segmentation_models_pytorch as smp

# ======================
# CONFIG
# ======================
MODEL_PATH = "best_model.pth"
IMG_DIR = "data/val/images"
MASK_DIR = "data/val/masks"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

THRESHOLDS = np.arange(0.1, 1.0, 0.1)
IMG_SIZE = 320


# ======================
# LOAD MODEL (MATCH TRAINING EXACTLY)
# ======================
model = smp.Unet(
    encoder_name="efficientnet-b3",
    encoder_weights=None,
    in_channels=3,
    classes=1
)

checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)

if "model" in checkpoint:
    state_dict = checkpoint["model"]
else:
    state_dict = checkpoint

model.load_state_dict(state_dict)
model.to(DEVICE)
model.eval()

print("✔ Model loaded")


# ======================
# METRICS (GLOBAL, NOT PER-BATCH)
# ======================
def compute_metrics(pred_bin, mask_bin):
    tp = np.logical_and(pred_bin == 1, mask_bin == 1).sum()
    fp = np.logical_and(pred_bin == 1, mask_bin == 0).sum()
    fn = np.logical_and(pred_bin == 0, mask_bin == 1).sum()

    iou = tp / (tp + fp + fn + 1e-6)
    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)

    return tp, fp, fn, iou, precision, recall, f1


# ======================
# PREPROCESS
# ======================
def preprocess(img_path):
    img = cv2.imread(img_path)
    if img is None:
        return None

    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))

    return torch.tensor(img).unsqueeze(0).to(DEVICE)


def load_mask(mask_path):
    mask = cv2.imread(mask_path, 0)
    if mask is None:
        return None

    mask = cv2.resize(mask, (IMG_SIZE, IMG_SIZE))
    mask = (mask > 0).astype(np.uint8)

    return mask


# ======================
# MAIN SWEEP
# ======================
results = {}

image_list = os.listdir(IMG_DIR)

valid_images = 0

# initialize accumulators
for t in THRESHOLDS:
    results[t] = {"tp": 0, "fp": 0, "fn": 0}

for img_name in image_list:

    img_path = os.path.join(IMG_DIR, img_name)

    base = os.path.splitext(img_name)[0]
    mask_path = os.path.join(MASK_DIR, base + ".png")

    if not os.path.exists(mask_path):
        print(f"Skipping missing mask: {img_name}")
        continue

    x = preprocess(img_path)
    mask = load_mask(mask_path)

    if x is None or mask is None:
        print(f"Skipping corrupted: {img_name}")
        continue

    with torch.no_grad():
        pred = torch.sigmoid(model(x)).cpu().numpy()[0][0]

    valid_images += 1

    for t in THRESHOLDS:
        pred_bin = (pred > t).astype(np.uint8)

        tp = np.logical_and(pred_bin == 1, mask == 1).sum()
        fp = np.logical_and(pred_bin == 1, mask == 0).sum()
        fn = np.logical_and(pred_bin == 0, mask == 1).sum()

        results[t]["tp"] += tp
        results[t]["fp"] += fp
        results[t]["fn"] += fn


# ======================
# FINAL REPORT
# ======================
print("\n===== THRESHOLD SWEEP =====")

best_f1 = 0
best_t = 0

for t in THRESHOLDS:
    tp = results[t]["tp"]
    fp = results[t]["fp"]
    fn = results[t]["fn"]

    iou = tp / (tp + fp + fn + 1e-6)
    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)

    print(f"T={t:.2f} | IoU={iou:.4f} | P={precision:.4f} | R={recall:.4f} | F1={f1:.4f}")

    if f1 > best_f1:
        best_f1 = f1
        best_t = t
        best_metrics = (iou, precision, recall, f1)


# ======================
# BEST RESULT
# ======================
print("\n🔥 BEST THRESHOLD (by F1)")
print(f"Threshold: {best_t:.2f}")
print(f"IoU:       {best_metrics[0]:.4f}")
print(f"Precision: {best_metrics[1]:.4f}")
print(f"Recall:    {best_metrics[2]:.4f}")
print(f"F1 Score:  {best_metrics[3]:.4f}")

print(f"\nImages evaluated: {valid_images}")