import streamlit as st
import torch
import cv2
import numpy as np
import segmentation_models_pytorch as smp

# ======================
# CONFIG
# ======================
MODEL_PATH = "best_model.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 320

# ======================
# LOAD MODEL
# ======================
@st.cache_resource
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

    return model

model = load_model()

# ======================
# PREPROCESS
# ======================
def preprocess(img):
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    img = np.transpose(img, (2, 0, 1))
    return torch.tensor(img, dtype=torch.float32).unsqueeze(0).to(DEVICE)

# ======================
# PREDICT
# ======================
def predict(img):
    h, w = img.shape[:2]

    x = preprocess(img)

    with torch.no_grad():
        pred = model(x)
        pred = torch.sigmoid(pred).cpu().numpy()[0, 0]

    # resize back to original
    pred = cv2.resize(pred, (w, h))

    return pred

# ======================
# POSTPROCESS
# ======================
def create_mask(prob, threshold):
    return (prob > threshold).astype(np.uint8) * 255

def overlay_mask(image, mask):
    if image.shape[:2] != mask.shape:
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]))

    overlay = image.copy()
    overlay[mask == 255] = [0, 0, 255]

    return cv2.addWeighted(image, 0.7, overlay, 0.3, 0)

def crack_percentage(mask):
    return (np.sum(mask > 0) / mask.size) * 100

def crack_length(mask):
    kernel = np.ones((3,3), np.uint8)
    skeleton = cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, kernel)
    return np.sum(skeleton > 0)

def create_heatmap(prob):
    heatmap = (prob * 255).astype(np.uint8)
    return cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

# ======================
# UI
# ======================
st.set_page_config(page_title="Crack Detection", layout="wide")

st.title("Crack Detection App")

st.sidebar.header("Settings")
threshold = st.sidebar.slider("Threshold", 0.1, 0.9, 0.7, 0.05)

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:

    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if img is None:
        st.error("Failed to read image")

    else:
        prob = predict(img)
        mask = create_mask(prob, threshold)
        overlay = overlay_mask(img, mask)

        area_pct = crack_percentage(mask)
        length = crack_length(mask)
        heatmap = create_heatmap(prob)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Input Image")
            st.image(img, channels="BGR")

        with col2:
            st.subheader("Overlay (Detected Cracks)")
            st.image(overlay, channels="BGR")

        st.subheader("Crack Metrics")
        st.write(f"Cracked Area: {area_pct:.2f}%")
        st.write(f"Crack Length (relative): {length}")

        st.subheader("Confidence Heatmap")
        st.image(heatmap, channels="BGR")

        # download
        _, buffer = cv2.imencode(".png", overlay)
        st.download_button(
            label="Download Result",
            data=buffer.tobytes(),
            file_name="crack_result.png",
            mime="image/png"
        )