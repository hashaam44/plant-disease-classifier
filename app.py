import streamlit as st
import tensorflow as tf
import numpy as np
import json
from PIL import Image


st.set_page_config(
    page_title="PlantVision AI",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)


st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background:
        radial-gradient(circle at 10% 10%, rgba(34,197,94,0.08), transparent 30%),
        radial-gradient(circle at 90% 20%, rgba(59,130,246,0.08), transparent 30%),
        #07111a;
    color: #f8fafc;
}

.block-container {
    max-width: 1250px;
    padding-top: 2rem;
    padding-bottom: 4rem;
}

[data-testid="stSidebar"] {
    background: #0a1722;
    border-right: 1px solid rgba(255,255,255,0.08);
}

[data-testid="stSidebar"] * {
    color: #dbeafe;
}

.hero {
    padding: 45px 45px 40px 45px;
    border-radius: 28px;
    background:
        linear-gradient(
            135deg,
            rgba(16,185,129,0.16),
            rgba(15,23,42,0.85)
        );
    border: 1px solid rgba(52,211,153,0.20);
    margin-bottom: 30px;
}

.badge {
    display: inline-block;
    padding: 7px 14px;
    border-radius: 999px;
    background: rgba(16,185,129,0.12);
    border: 1px solid rgba(52,211,153,0.25);
    color: #6ee7b7;
    font-size: 13px;
    font-weight: 700;
    letter-spacing: 0.5px;
}

.hero h1 {
    font-size: 48px;
    font-weight: 800;
    margin: 15px 0 10px 0;
    letter-spacing: -1.5px;
}

.hero p {
    color: #94a3b8;
    font-size: 17px;
    max-width: 720px;
    line-height: 1.7;
}

.metric-card {
    padding: 24px;
    border-radius: 20px;
    background: rgba(15,23,42,0.75);
    border: 1px solid rgba(255,255,255,0.07);
    min-height: 125px;
}

.metric-label {
    color: #94a3b8;
    font-size: 13px;
    font-weight: 600;
}

.metric-value {
    font-size: 30px;
    font-weight: 800;
    margin-top: 8px;
}

.section-title {
    font-size: 24px;
    font-weight: 800;
    margin-top: 35px;
    margin-bottom: 15px;
}

.upload-box {
    padding: 25px;
    border-radius: 22px;
    background: rgba(15,23,42,0.65);
    border: 1px solid rgba(255,255,255,0.08);
}

.prediction-card {
    padding: 28px;
    border-radius: 22px;
    background:
        linear-gradient(
            145deg,
            rgba(16,185,129,0.12),
            rgba(15,23,42,0.9)
        );
    border: 1px solid rgba(52,211,153,0.18);
}

.prediction-name {
    font-size: 27px;
    font-weight: 800;
    color: #6ee7b7;
}

.confidence {
    font-size: 42px;
    font-weight: 800;
    margin-top: 8px;
}

.info-card {
    padding: 20px;
    border-radius: 18px;
    background: rgba(15,23,42,0.65);
    border: 1px solid rgba(255,255,255,0.06);
    margin-bottom: 12px;
}

.footer {
    margin-top: 50px;
    padding-top: 25px;
    border-top: 1px solid rgba(255,255,255,0.08);
    text-align: center;
    color: #64748b;
    font-size: 13px;
}

div.stButton > button {
    width: 100%;
    border-radius: 12px;
    height: 48px;
    font-weight: 700;
    border: 1px solid rgba(52,211,153,0.3);
    background: rgba(16,185,129,0.12);
}

div.stButton > button:hover {
    border-color: #34d399;
    background: rgba(16,185,129,0.22);
}

</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():

    return tf.keras.models.load_model(
        "model/plant_disease_model.keras"
    )


@st.cache_data
def load_classes():

    with open("model/classes.json", "r") as file:
        return json.load(file)


model = load_model()
classes = load_classes()


def format_class_name(name):

    return name.replace("_", " ")


def predict_image(image):

    image = image.convert("RGB")

    image = image.resize((224, 224))

    image_array = np.array(image)

    image_array = tf.keras.applications.mobilenet_v2.preprocess_input(
        image_array
    )

    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    predictions = model.predict(
        image_array,
        verbose=0
    )[0]

    top_indices = np.argsort(
        predictions
    )[::-1][:3]

    results = []

    for index in top_indices:

        results.append({
            "class": classes[index],
            "confidence": float(predictions[index] * 100)
        })

    return results


with st.sidebar:

    st.markdown("## 🌿 PlantVision AI")

    st.caption("CNN-powered plant disease analysis")

    st.markdown("---")

    st.markdown("### Model")

    st.write("Architecture")
    st.caption("MobileNetV2 + Transfer Learning")

    st.write("Classes")
    st.caption("15 plant disease categories")

    st.write("Input")
    st.caption("224 × 224 RGB image")

    st.markdown("---")

    st.markdown("### Model Performance")

    st.metric(
        "Test Accuracy",
        "89.05%"
    )

    st.metric(
        "Macro F1",
        "0.87"
    )

    st.markdown("---")

    st.caption(
        "For educational and research purposes. "
        "Predictions should be verified by an expert."
    )


st.markdown("""
<div class="hero">

<span class="badge">AI-POWERED PLANT ANALYSIS</span>

<h1>PlantVision AI 🌿</h1>

<p>
Upload a plant leaf image and let a deep learning model
analyze it across 15 plant disease categories.
The system uses MobileNetV2 transfer learning trained
on the PlantVillage dataset.
</p>

</div>
""", unsafe_allow_html=True)


col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-label">MODEL ACCURACY</div>
        <div class="metric-value">89.05%</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-label">DISEASE CLASSES</div>
        <div class="metric-value">15</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-label">MODEL TYPE</div>
        <div class="metric-value">CNN</div>
    </div>
    """, unsafe_allow_html=True)


st.markdown(
    '<div class="section-title">🔬 Analyze Plant</div>',
    unsafe_allow_html=True
)


left, right = st.columns([1, 1], gap="large")


with left:

    st.markdown(
        '<div class="upload-box">',
        unsafe_allow_html=True
    )

    uploaded_file = st.file_uploader(
        "Upload a plant leaf image",
        type=["jpg", "jpeg", "png"],
        help="Upload a clear image of a plant leaf."
    )

    st.markdown(
        "</div>",
        unsafe_allow_html=True
    )


with right:

    if uploaded_file:

        image = Image.open(uploaded_file)

        st.image(
            image,
            caption="Uploaded Leaf",
            use_container_width=True
        )

    else:

        st.info(
            "Upload an image to start the AI analysis."
        )


if uploaded_file:

    st.markdown(
        '<div class="section-title">🧠 AI Analysis</div>',
        unsafe_allow_html=True
    )

    if st.button("🚀 Analyze Image"):

        with st.spinner("Running neural network inference..."):

            results = predict_image(image)

        top_result = results[0]

        st.markdown(
            '<div class="prediction-card">',
            unsafe_allow_html=True
        )

        st.markdown(
            "### Primary Prediction"
        )

        st.markdown(
            f'<div class="prediction-name">'
            f'{format_class_name(top_result["class"])}'
            f'</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            f'<div class="confidence">'
            f'{top_result["confidence"]:.2f}%'
            f'</div>',
            unsafe_allow_html=True
        )

        st.caption("Model confidence")

        st.progress(
            min(top_result["confidence"] / 100, 1.0)
        )

        st.markdown(
            '</div>',
            unsafe_allow_html=True
        )


        st.markdown(
            '<div class="section-title">📊 Top 3 Predictions</div>',
            unsafe_allow_html=True
        )

        for rank, result in enumerate(results, start=1):

            col_a, col_b = st.columns([3, 1])

            with col_a:

                st.markdown(
                    f"**#{rank} "
                    f"{format_class_name(result['class'])}**"
                )

            with col_b:

                st.write(
                    f"{result['confidence']:.2f}%"
                )

            st.progress(
                min(result["confidence"] / 100, 1.0)
            )


        st.markdown(
            '<div class="section-title">💡 Interpretation</div>',
            unsafe_allow_html=True
        )

        if top_result["confidence"] >= 90:

            st.success(
                "The model has high confidence in this prediction. "
                "However, confidence is not a guarantee of correctness."
            )

        elif top_result["confidence"] >= 70:

            st.warning(
                "The model has moderate confidence. "
                "Consider comparing the top predictions."
            )

        else:

            st.error(
                "The model has low confidence. "
                "Try a clearer leaf image or verify the result manually."
            )


st.markdown(
    '<div class="footer">'
    'PlantVision AI • CNN + MobileNetV2 Transfer Learning • '
    'Educational AI Project'
    '</div>',
    unsafe_allow_html=True
)