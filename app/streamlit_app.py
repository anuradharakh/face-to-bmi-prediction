import streamlit as st

st.set_page_config(page_title="BMI Prediction", layout="centered")

st.title("BMI Prediction")

st.markdown("### Select model")

model = st.radio(
    "Model",
    [
        "M1: VGG16 ImageNet",
        "M2: ResNet50 Multi-Task",
        "M3: EfficientNet Fine-Tuned",
    ],
    horizontal=True,
)

st.markdown("### Input image")
uploaded = st.file_uploader("Drop a face/headshot image here", type=["jpg", "jpeg", "png", "bmp"])

if uploaded:
    st.image(uploaded, caption="Uploaded image", use_container_width=True)
    st.info("Prediction will be connected after model training is complete.")
else:
    st.caption("Upload an image to preview the frontend flow.")
