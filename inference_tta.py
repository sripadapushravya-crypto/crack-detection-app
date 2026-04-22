import os
import cv2
import torch
import numpy as np
import segmentation_models_pytorch as smp

# ======================
# CONFIG
# ======================
MODEL_PATH = "best_model.pth"
INPUT_PATH = "data/val/images"
OUTPUT_DIR = "outputs"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

IMG_SIZE = 320
STRIDE = 160
THRESHOLD = 0.70

USE_TTA = True
USE_SLIDING_WINDOW = True

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ======================
# LOAD MODEL
# ======================
def load_model():
    model = smp.Unet(
        encoder_name="efficientnet-b3",
        encoder_weights=None,
        in_channels=3,
        classes=1
    )

    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
    state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint

    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()

    print("✔ Model loaded")
    return model


# ======================
# PREPROCESS
# ======================
def preprocess(img):
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    img = np.transpose(img, (2, 0, 1))
    return torch.tensor(img, dtype=torch.float32).unsqueeze(0).to(DEVICE)


# ======================
# SINGLE PRED
# ======================
def predict_single(model, img):
    x = preprocess(img)
    with torch.no_grad():
        y = model(x)
        y = torch.sigmoid(y).cpu().numpy()[0, 0]
    return y


# ======================
# TTA
# ======================
def predict_tta(model, img):

    transforms = [
        lambda x: x,
        lambda x: cv2.flip(x, 1),
        lambda x: cv2.flip(x, 0),
        lambda x: cv2.flip(x, -1),
    ]

    preds = []

    for t in transforms:
        aug = t(img)
        pred = predict_single(model, aug)
        pred = t(pred)  # inverse = same flip
        preds.append(pred)

    return np.mean(preds, axis=0)


# ======================
# SLIDING WINDOW
# ======================
def predict_sliding_window(model, img):

    h, w = img.shape[:2]

    prob_map = np.zeros((h, w), dtype=np.float32)
    count_map = np.zeros((h, w), dtype=np.float32)

    for y in range(0, h, STRIDE):
        for x in range(0, w, STRIDE):

            patch = img[y:y+IMG_SIZE, x:x+IMG_SIZE]
            ph, pw = patch.shape[:2]

            if ph == 0 or pw == 0:
                continue

            if ph < IMG_SIZE or pw < IMG_SIZE:
                padded = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
                padded[:ph, :pw] = patch
                patch = padded

            if USE_TTA:
                pred = predict_tta(model, patch)
            else:
                pred = predict_single(model, patch)

            pred = pred[:ph, :pw]

            prob_map[y:y+ph, x:x+pw] += pred
            count_map[y:y+ph, x:x+pw] += 1

    prob_map /= (count_map + 1e-6)

    return prob_map


# ======================
# POST-PROCESSING (CRITICAL)
# ======================
def post_process(mask):

    kernel = np.ones((3, 3), np.uint8)

    # remove noise
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # connect cracks
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # remove small components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)

    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] < 50:
            mask[labels == i] = 0

    return mask


# ======================
# PROCESS IMAGE
# ======================
def process_image(model, img_path):

    img = cv2.imread(img_path)
    if img is None:
        print(f"❌ Skipping: {img_path}")
        return

    if USE_SLIDING_WINDOW:
        prob = predict_sliding_window(model, img)
    else:
        prob = predict_tta(model, img) if USE_TTA else predict_single(model, img)

    mask = (prob > THRESHOLD).astype(np.uint8) * 255

    # 🔥 APPLY POST PROCESSING
    mask = post_process(mask)

    name = os.path.basename(img_path)
    cv2.imwrite(os.path.join(OUTPUT_DIR, name), mask)


# ======================
# MAIN
# ======================
def main():

    model = load_model()

    if os.path.isfile(INPUT_PATH):
        process_image(model, INPUT_PATH)

    elif os.path.isdir(INPUT_PATH):
        files = os.listdir(INPUT_PATH)
        print(f"Processing {len(files)} images...")

        for f in files:
            process_image(model, os.path.join(INPUT_PATH, f))

    else:
        raise ValueError("Invalid input path")

    print("✔ Done")


if __name__ == "__main__":
    main()