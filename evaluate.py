import torch
import cv2
import numpy as np
import os
import segmentation_models_pytorch as smp

# ======================
# CONFIG (MATCH TRAINING)
# ======================
MODEL_PATH = "best_model.pth"
ENCODER = "efficientnet-b3"
IMG_SIZE = 320
THRESHOLD = 0.35
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

IMG_DIR = "data/val/images"
MASK_DIR = "data/val/masks"

# ======================
# LOAD MODEL
# ======================
def load_model():
    model = smp.Unet(
        encoder_name=ENCODER,
        encoder_weights=None,
        in_channels=3,
        classes=1
    ).to(DEVICE)

    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
    state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint

    model.load_state_dict(state_dict)
    model.eval()

    print("✔ Model loaded")
    return model

# ======================
# PREPROCESS
# ======================
def preprocess(img):
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    img = np.transpose(img, (2, 0, 1)).astype(np.float32)
    return torch.tensor(img).unsqueeze(0).to(DEVICE)

# ======================
# LOAD MASK
# ======================
def load_mask(base_name):
    m1 = os.path.join(MASK_DIR, base_name + ".png")
    m2 = os.path.join(MASK_DIR, base_name + "_mask.png")

    path = m1 if os.path.exists(m1) else m2

    if not os.path.exists(path):
        return None

    mask = cv2.imread(path, 0)
    mask = cv2.resize(mask, (IMG_SIZE, IMG_SIZE))
    mask = (mask > 0).astype(np.uint8)

    return mask

# ======================
# METRICS (PER IMAGE)
# ======================
def compute_metrics(pred, gt):
    pred = pred.astype(np.uint8)
    gt = gt.astype(np.uint8)

    TP = np.sum((pred == 1) & (gt == 1))
    FP = np.sum((pred == 1) & (gt == 0))
    FN = np.sum((pred == 0) & (gt == 1))

    # IoU
    iou = TP / (TP + FP + FN + 1e-6)

    # Precision
    precision = TP / (TP + FP + 1e-6)

    # Recall
    recall = TP / (TP + FN + 1e-6)

    return iou, precision, recall

# ======================
# EVALUATION LOOP
# ======================
def evaluate():
    model = load_model()

    img_list = [f for f in os.listdir(IMG_DIR) if f.endswith(".jpg")]

    if len(img_list) == 0:
        raise ValueError("No validation images found")

    total_iou = 0
    total_precision = 0
    total_recall = 0
    count = 0

    for img_name in img_list:
        img_path = os.path.join(IMG_DIR, img_name)
        base = os.path.splitext(img_name)[0]

        img = cv2.imread(img_path)
        if img is None:
            print(f"Skipping corrupted: {img_name}")
            continue

        gt = load_mask(base)
        if gt is None:
            print(f"Skipping missing mask: {img_name}")
            continue

        # ===== Predict =====
        x = preprocess(img)

        with torch.no_grad():
            pred = model(x)
            pred = torch.sigmoid(pred).cpu().numpy()[0][0]

        pred_bin = (pred > THRESHOLD).astype(np.uint8)

        # ===== Metrics =====
        iou, precision, recall = compute_metrics(pred_bin, gt)

        total_iou += iou
        total_precision += precision
        total_recall += recall
        count += 1

    # ======================
    # FINAL RESULTS
    # ======================
    avg_iou = total_iou / count
    avg_precision = total_precision / count
    avg_recall = total_recall / count

    print("\n===== VALIDATION RESULTS =====")
    print(f"Images evaluated: {count}")
    print(f"IoU:       {avg_iou:.4f}")
    print(f"Precision: {avg_precision:.4f}")
    print(f"Recall:    {avg_recall:.4f}")

# ======================
# ENTRY
# ======================
if __name__ == "__main__":
    evaluate()