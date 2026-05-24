import time
import streamlit as st
from PIL import Image


MODEL_REGISTRY = {
    "vgg16_baseline": {"label": "VGG16", "subtitle": "Baseline", "tag": "M1", "best_r": "0.4227"},
    "resnet50_multitask": {
        "label": "ResNet50",
        "subtitle": "Multi-Task",
        "tag": "M2",
        "best_r": "0.6811",
    },
   "efficientnet_finetune": {
        "label": "EfficientNet",
        "subtitle": "Fine-Tune",
        "tag": "M3",
        "best_r": "0.5532",
    },
    "resnet50_bmi_only": {"label": "ResNet50", "subtitle": "BMI-Only", "tag": "M2B", "best_r": "0.5771"},
}

st.set_page_config(page_title="BMI Prediction", page_icon="⚖️", layout="wide")

st.markdown(
    """
<style>
.stApp {
    background: radial-gradient(circle at top, #171a20 0%, #08090b 55%, #030303 100%);
    color: #f5f5f5;
}

.block-container {
    max-width: 1100px;
    padding-top: 2rem;
}

.main-title {
    font-size: 4.6rem;
    font-weight: 900;
    text-align: center;
    margin-bottom: 0rem;
}

.subtitle {
    color: #b8beca;
    text-align: center;
    margin-bottom: 2rem;
}

.section-box {
    border: 1px solid #2c313a;
    border-radius: 12px;
    background: rgba(18, 21, 27, 0.92);
    padding: 1.4rem;
    margin-bottom: 1.5rem;
}

.model-card, .result-card {
    border: 1px solid #303642;
    border-radius: 10px;
    background: linear-gradient(145deg, #171b22, #101319);
    padding: 1rem;
    min-height: 145px;
}

.active-card {
    border: 1px solid #a7f35b;
}

.model-tag {
    display: inline-block;
    border: 1px solid #7ad66b;
    color: #9cff6d;
    border-radius: 5px;
    font-size: 0.75rem;
    padding: 0.1rem 0.4rem;
    margin-bottom: 0.5rem;
    font-weight: 800;
}

.prediction-value {
    font-size: 2.1rem;
    font-weight: 900;
    color: #8cff6d;
}

.muted {
    color: #aeb4bf;
    font-size: 0.9rem;
}

.green { color: #8cff6d; font-weight: 800; }
.blue { color: #86aaff; font-weight: 800; }
.yellow { color: #ffd84d; font-weight: 800; }

.stButton > button {
    border-radius: 8px;
    border: 1px solid #83e35e;
    background: rgba(80, 160, 70, 0.12);
    color: #b6ff81;
    font-weight: 700;
}
</style>
""",
    unsafe_allow_html=True,
)


def init_state():
    for key, value in {
        "results": {},
        "input_image": None,
        "uploader_key": 0,
        "camera_key": 0,
    }.items():
        if key not in st.session_state:
            st.session_state[key] = value


def reset_all():
    st.session_state["results"] = {}
    st.session_state["input_image"] = None
    st.session_state["uploader_key"] += 1
    st.session_state["camera_key"] += 1


def fake_predict(model_key: str) -> float:
    return {
        "vgg16_baseline": 24.31,
        "resnet50_multitask": 23.85,
        "efficientnet_finetune": 24.02,
        "resnet50_bmi_only": 23.70,
    }[model_key]


init_state()

st.markdown("<h1 class='main-title'>BMI Prediction</h1>", unsafe_allow_html=True)
st.markdown(
    "<p class='subtitle'>Predict BMI from a face image using deep learning models</p>",
    unsafe_allow_html=True,
)

with st.container(border=True):
    st.subheader("1. Select Models")
    st.caption("Toggle models on/off to include them in prediction.")

    selected_models = []
    cols = st.columns(4)

    for i, (model_key, meta) in enumerate(MODEL_REGISTRY.items()):
        with cols[i]:
            enabled = st.toggle(
                f"{meta['tag']} enabled",
                value=model_key == "resnet50_multitask",
                key=f"toggle_{model_key}",
            )

            if enabled:
                selected_models.append(model_key)

            card_class = "model-card active-card" if enabled else "model-card"
            color = "green"
            if model_key == "vgg16_baseline":
                color = "blue"
            elif model_key == "efficientnet_finetune":
                color = "yellow"

            st.markdown(
                f"""
                <div class="{card_class}">
                    <span class="model-tag">{meta['tag']}</span>
                    <h3>{meta['label']}</h3>
                    <p class="muted">{meta['subtitle']}</p>
                    <p class="muted">Best r</p>
                    <p class="{color}">{meta['best_r']}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

with st.container(border=True):
    st.subheader("2. Gender")
    st.caption("Used by the ResNet50 Multi-Task model only.")

    gender = st.radio(
        "Gender",
        ["Male", "Female"],
        horizontal=True,
        label_visibility="collapsed",
    )

with st.container(border=True):
    st.subheader("3. Input Image")

    input_mode = st.radio(
        "Input Mode",
        ["Upload Image", "Webcam"],
        horizontal=True,
        label_visibility="collapsed",
    )

    left, right = st.columns(2)

    with left:
        if input_mode == "Upload Image":
            uploaded = st.file_uploader(
                "Upload image",
                type=["jpg", "jpeg", "png", "bmp"],
                key=f"uploader_{st.session_state['uploader_key']}",
            )

            if uploaded is not None:
                st.session_state["input_image"] = Image.open(uploaded).convert("RGB")
        else:
            captured = st.camera_input(
                "Capture image",
                key=f"camera_{st.session_state['camera_key']}",
            )

            if captured is not None:
                st.session_state["input_image"] = Image.open(captured).convert("RGB")

    with right:
        if st.session_state["input_image"] is not None:
            st.image(
                st.session_state["input_image"],
                caption="Selected image",
                use_container_width=True,
            )
        else:
            st.info("Upload or capture an image to preview it here.")

with st.container(border=True):
    st.subheader("4. Prediction Results")

    run_col, reset_col, clear_col = st.columns(3)

    with run_col:
        run_prediction = st.button("Run Prediction", use_container_width=True)

    with reset_col:
        if st.button("Reset / Clear Results", use_container_width=True):
            reset_all()
            st.rerun()

    with clear_col:
        if st.button("Clear Image", use_container_width=True):
            reset_all()
            st.rerun()

    if run_prediction:
        if st.session_state["input_image"] is None:
            st.warning("Please upload or capture an image first.")
        elif not selected_models:
            st.warning("Please select at least one model.")
        else:
            with st.spinner("Processing... Please wait."):
                time.sleep(1)
                st.session_state["results"] = {
                    model_key: fake_predict(model_key)
                    for model_key in selected_models
                }

    result_cols = st.columns(4)

    for i, (model_key, meta) in enumerate(MODEL_REGISTRY.items()):
        with result_cols[i]:
            prediction = st.session_state["results"].get(model_key)

            pred_text = "—" if prediction is None else f"{prediction:.2f}"
            status = "Not Run" if prediction is None else "Completed"

            color = "green"
            if model_key == "vgg16_baseline":
                color = "blue"
            elif model_key == "efficientnet_finetune":
                color = "yellow"

            st.markdown(
                f"""
                <div class="result-card">
                    <span class="model-tag">{meta['tag']}</span>
                    <h4>{meta['label']}</h4>
                    <p class="muted">{meta['subtitle']}</p>
                    <p class="muted">Predicted BMI</p>
                    <p class="prediction-value">{pred_text}</p>
                    <p class="muted">Status: {status}</p>
                    <p class="muted">Best r: <span class="{color}">{meta['best_r']}</span></p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.caption(
        "BMI is predicted in kg/m². Current UI uses placeholder predictions; real model inference will be connected next."
    )