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
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ======================
# LOAD MODEL
# ======================
def load_model():
    model = smp.Unet(
        encoder_name=ENCODER,
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
def preprocess(image_path):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"File not found: {image_path}")

    img = cv2.imread(image_path)

    if img is None:
        raise ValueError(f"Failed to read image: {image_path}")

    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0

    img = np.transpose(img, (2, 0, 1)).astype(np.float32)
    img = np.expand_dims(img, axis=0)

    return torch.tensor(img).to(DEVICE), img

# ======================
# PREDICT
# ======================
def predict(model, image_path, threshold=0.35):
    x, original = preprocess(image_path)

    with torch.no_grad():
        y = model(x)
        y = torch.sigmoid(y)
        y = y.squeeze().cpu().numpy()

    binary = (y > threshold).astype(np.uint8) * 255

    return original, y, binary

# ======================
# SAVE OUTPUT
# ======================
def save_results(original, prob, mask):
    os.makedirs("outputs", exist_ok=True)

    orig_img = (np.transpose(original[0], (1, 2, 0)) * 255).astype(np.uint8)

    cv2.imwrite("outputs/input.jpg", orig_img)
    cv2.imwrite("outputs/probability.png", (prob * 255).astype(np.uint8))
    cv2.imwrite("outputs/mask.png", mask)

    combined = np.hstack([
        orig_img,
        cv2.cvtColor((prob * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR),
        cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    ])

    cv2.imwrite("outputs/combined.jpg", combined)

    print("✔ Results saved in /outputs")

# ======================
# MAIN
# ======================
if __name__ == "__main__":
    IMAGE_PATH = "data/val/images/20160222_164936_641_361.jpg"  # ← CHANGE THIS

    model = load_model()
    original, prob, mask = predict(model, IMAGE_PATH)

    save_results(original, prob, mask)