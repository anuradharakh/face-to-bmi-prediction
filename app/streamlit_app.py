import time
import streamlit as st
from PIL import Image


MODEL_REGISTRY = {
    "vgg16_baseline": {
        "label": "VGG16",
        "subtitle": "Baseline",
        "tag": "M1",
        "best_r": "0.4227",
    },
    "resnet50_multitask": {
        "label": "ResNet50",
        "subtitle": "Multi-Task",
        "tag": "M2",
        "best_r": "0.5798",
    },
    "efficientnet_finetune": {
        "label": "EfficientNet",
        "subtitle": "Fine-Tune",
        "tag": "M3",
        "best_r": "0.4942",
    },
    "resnet50_bmi_only": {
        "label": "ResNet50",
        "subtitle": "BMI-Only",
        "tag": "M2B",
        "best_r": "0.5771",
    },
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
    font-size: 5rem;
    font-weight: 900;
    letter-spacing: -3px;
    text-align: center;
    margin-bottom: 0;
}
.subtitle {
    color: #b8beca;
    text-align: center;
    margin-bottom: 2rem;
}
.panel {
    border: 1px solid #2c313a;
    border-radius: 10px;
    background: rgba(18, 21, 27, 0.92);
    padding: 1.5rem;
    margin-bottom: 1.4rem;
}
.section-title {
    font-size: 1.25rem;
    font-weight: 800;
    margin-bottom: 0.6rem;
}
.model-card, .result-card {
    border: 1px solid #303642;
    border-radius: 8px;
    background: linear-gradient(145deg, #171b22, #101319);
    padding: 1rem;
    min-height: 150px;
}
.model-card-active {
    border: 1px solid #a7f35b;
}
.model-tag {
    display: inline-block;
    border: 1px solid #7ad66b;
    color: #9cff6d;
    border-radius: 5px;
    font-size: 0.75rem;
    padding: 0.1rem 0.4rem;
    margin-bottom: 0.6rem;
    font-weight: 800;
}
.prediction-value {
    font-size: 2.2rem;
    font-weight: 900;
    color: #8cff6d;
}
.small-muted {
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
    defaults = {
        "results": {},
        "input_image": None,
        "uploader_key": 0,
        "camera_key": 0,
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def reset_all():
    st.session_state["results"] = {}
    st.session_state["input_image"] = None
    st.session_state["uploader_key"] += 1
    st.session_state["camera_key"] += 1


def fake_predict(model_key: str) -> float:
    predictions = {
        "vgg16_baseline": 24.31,
        "resnet50_multitask": 23.85,
        "efficientnet_finetune": 24.02,
        "resnet50_bmi_only": 23.70,
    }
    return predictions[model_key]


init_state()


st.markdown("<h1 class='main-title'>BMI Prediction</h1>", unsafe_allow_html=True)
st.markdown(
    "<p class='subtitle'>Predict BMI from a face image using deep learning models</p>",
    unsafe_allow_html=True,
)


st.markdown("<div class='panel'>", unsafe_allow_html=True)
st.markdown("<div class='section-title'>1. Select Models</div>", unsafe_allow_html=True)

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

        card_class = "model-card model-card-active" if enabled else "model-card"

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
                <p class="small-muted">{meta['subtitle']}</p>
                <p class="small-muted">Best r</p>
                <p class="{color}">{meta['best_r']}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.markdown("</div>", unsafe_allow_html=True)


st.markdown("<div class='panel'>", unsafe_allow_html=True)
st.markdown(
    "<div class='section-title'>2. Gender <span class='small-muted'>(used by M2 only)</span></div>",
    unsafe_allow_html=True,
)

gender = st.radio(
    "Gender",
    ["Male", "Female"],
    horizontal=True,
    label_visibility="collapsed",
)

st.caption("Gender is only used by the ResNet50 multi-task model.")
st.markdown("</div>", unsafe_allow_html=True)


st.markdown("<div class='panel'>", unsafe_allow_html=True)
st.markdown("<div class='section-title'>3. Input Image</div>", unsafe_allow_html=True)

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

st.markdown("</div>", unsafe_allow_html=True)


st.markdown("<div class='panel'>", unsafe_allow_html=True)
st.markdown("<div class='section-title'>4. Prediction Results</div>", unsafe_allow_html=True)

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

        if prediction is None:
            pred_text = "—"
            status = "Not Run"
        else:
            pred_text = f"{prediction:.2f}"
            status = "Completed"

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
                <p class="small-muted">{meta['subtitle']}</p>
                <p class="small-muted">Predicted BMI</p>
                <p class="prediction-value">{pred_text}</p>
                <p class="small-muted">Status: {status}</p>
                <p class="small-muted">Best r: <span class="{color}">{meta['best_r']}</span></p>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.caption("BMI is predicted in kg/m². Current UI uses placeholder predictions; model inference will be connected next.")
st.markdown("</div>", unsafe_allow_html=True)