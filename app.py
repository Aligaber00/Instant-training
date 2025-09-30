import streamlit as st
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model

# --------------------------
# Load Models
# --------------------------
@st.cache_resource
def load_models():
    detection_model = load_model('brain_tumor_inception.h5', compile=False)
    segmentation_model = load_model("mobileunet_segmentation.h5", compile=False)
    return detection_model, segmentation_model

detection_model, segmentation_model = load_models()

# --------------------------
# Preprocessing Functions
# --------------------------
def preprocess_for_detection(img: Image.Image):
    img = img.resize((224, 224))  # ✅ FIXED for InceptionV3
    img_array = np.array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

def preprocess_for_segmentation(img: Image.Image):
    img = img.resize((128, 128))  # segmentation model input
    img_array = np.array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

# --------------------------
# Prediction Functions
# --------------------------
def predict_detection(img: Image.Image):
    processed = preprocess_for_detection(img)
    prob = detection_model.predict(processed)[0][0]  # sigmoid output
    label = "Tumor Detected" if prob >= 0.5 else "No Tumor"
    return label, prob

def predict_segmentation(img: Image.Image):
    processed = preprocess_for_segmentation(img)
    pred_mask = segmentation_model.predict(processed)[0]
    pred_mask = (pred_mask > 0.5).astype(np.uint8)

    # Resize mask back to original image size
    mask_resized = cv2.resize(pred_mask, (img.width, img.height))
    return mask_resized

# --------------------------
# Streamlit UI
# --------------------------
st.set_page_config(page_title="Brain Tumor Detection & Segmentation", layout="centered")

st.title("🧠 Brain Tumor Detection & Segmentation")
st.markdown("Upload an MRI scan and choose whether to run **Detection** or **Segmentation**.")

uploaded_file = st.file_uploader("Upload MRI Image", type=["jpg", "jpeg", "png"])

task = st.radio("Select Task", ["Detection", "Segmentation"], horizontal=True)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    # Display smaller image preview
    st.image(image, caption="Uploaded Image", width=300)

    if task == "Detection":
        label, prob = predict_detection(image)

        st.subheader("🔍 Detection Result")
        st.markdown(
            f"""
            **Prediction:** {label}  
            **Confidence:** {prob:.2f}
            """
        )

        if label == "Tumor Detected":
            st.error("⚠️ Tumor Detected")
        else:
            st.success("✅ No Tumor Detected")

    elif task == "Segmentation":
       mask = predict_segmentation(image)

       st.subheader("🩻 Segmentation Result")

      # Create a black & white mask image
       mask_img = (mask * 255).astype(np.uint8)  # white tumor on black background
       mask_pil = Image.fromarray(mask_img)

      # Display original and mask side by side
       col1, col2 = st.columns(2)

       with col1:
          st.image(image, caption="Uploaded Image", use_container_width=True)
       with col2:
          st.image(mask_pil, caption="Tumor Mask", use_container_width=True)

