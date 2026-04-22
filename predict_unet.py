import cv2
import torch
import numpy as np
import os
import random
import segmentation_models_pytorch as smp

device = "cuda" if torch.cuda.is_available() else "cpu"

# ===== CONFIG =====
MODEL_PATH = "best_model.pth"
ENCODER = "efficientnet-b3"
IMG_SIZE = 320
NUM_SAMPLES = 5

# ===== LOAD MODEL =====
def load_model():
    model = smp.Unet(
        encoder_name=ENCODER,
        encoder_weights=None,
        in_channels=3,
        classes=1
    ).to(device)

    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=True)
    state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint

    model.load_state_dict(state_dict)
    model.eval()

    print("✔ Model loaded")
    return model

# ===== PREPROCESS =====
def preprocess(img):
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    img = np.transpose(img, (2, 0, 1)).astype(np.float32)
    return torch.tensor(img).unsqueeze(0).to(device)

# ===== MAIN =====
def main():
    model = load_model()

    img_dir = "data/val/images"
    mask_dir = "data/val/masks"

    if not os.path.exists(img_dir):
        raise FileNotFoundError(f"Image directory not found: {img_dir}")

    img_list = [f for f in os.listdir(img_dir) if f.endswith(".jpg")]

    if len(img_list) == 0:
        raise ValueError("No images found in directory")

    sample_images = random.sample(img_list, min(NUM_SAMPLES, len(img_list)))

    os.makedirs("outputs", exist_ok=True)

    for i, img_name in enumerate(sample_images):

        img_path = os.path.join(img_dir, img_name)
        base = os.path.splitext(img_name)[0]

        mask_path1 = os.path.join(mask_dir, base + ".png")
        mask_path2 = os.path.join(mask_dir, base + "_mask.png")
        mask_path = mask_path1 if os.path.exists(mask_path1) else mask_path2

        # ===== Load image =====
        img = cv2.imread(img_path)
        if img is None:
            print(f"Skipping corrupted image: {img_name}")
            continue

        img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

        # ===== Predict =====
        x = preprocess(img)

        with torch.no_grad():
            pred = model(x)
            pred = torch.sigmoid(pred).cpu().numpy()[0][0]

        # ===== Threshold =====
        pred_bin = (pred > 0.35).astype(np.uint8) * 255

        # ===== Post-processing =====
        kernel = np.ones((3, 3), np.uint8)
        pred_bin = cv2.morphologyEx(pred_bin, cv2.MORPH_OPEN, kernel)

        # ===== Load GT (optional) =====
        if os.path.exists(mask_path):
            mask = cv2.imread(mask_path, 0)
            mask = cv2.resize(mask, (IMG_SIZE, IMG_SIZE))
        else:
            mask = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)

        # ===== Save =====
        cv2.imwrite(f"outputs/{i}_input.jpg", img_resized)
        cv2.imwrite(f"outputs/{i}_gt.png", mask)
        cv2.imwrite(f"outputs/{i}_pred.png", pred_bin)

        combined = np.hstack([
            img_resized,
            cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR),
            cv2.cvtColor(pred_bin, cv2.COLOR_GRAY2BGR)
        ])

        cv2.imwrite(f"outputs/{i}_combined.jpg", combined)

        print(f"✔ Saved sample {i}: {img_name}")

    print("✔ Batch inference complete")

# ===== ENTRY =====
if __name__ == "__main__":
    main()