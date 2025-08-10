import streamlit as st
from PIL import Image
from utils import predict

# Page
st.set_page_config(
    page_title="AI Face Age & Gender Detector",
    page_icon="🧠",
    layout="centered"
)

# css
st.markdown("""
    <style>
        /* Background and font */
        body {
            font-family: 'Segoe UI', sans-serif;
            background-color: #0f1117;
        }

        /* Hide Streamlit default footer */
        footer {visibility: hidden;}

        /* Title styling */
        .main > div:first-child {
            padding-top: 2rem;
        }

        .title {
            text-align: center;
            color: #FFFFFF;
            font-size: 2.5rem;
            font-weight: 700;
            margin-bottom: 1.5rem;
        }

        /* File uploader styling */
        .stFileUploader label {
            color: #c7c7c7;
            font-weight: 600;
        }

        /* Result box styling */
        .result-box {
            background-color: #1f2937;
            border-radius: 12px;
            padding: 20px;
            margin-top: 20px;
            color: #ffffff;
            font-size: 1.2rem;
            text-align: center;
        }

        /* Spinner and success message */
        .stSpinner, .stSuccess {
            text-align: center;
        }
    </style>
""", unsafe_allow_html=True)

#Title
st.markdown('<div class="title">👤 AI-Powered Age & Gender Prediction</div>', unsafe_allow_html=True)

# File uploader
uploaded_file = st.file_uploader("📤 Upload a clear photo (jpg/png)", type=["jpg", "jpeg", "png"])

# Prediction logic
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='🖼️ Uploaded Image', use_container_width=True)

    with st.spinner("🧠 Analyzing face features..."):
        age, gender = predict(image)

    # Result display
    st.markdown(f"""
        <div class="result-box">
            <p>🎂 <strong>Estimated Age:</strong> <code>{age:.1f}</code> years</p>
            <p>👥 <strong>Predicted Gender:</strong> <code>{gender}</code></p>
        </div>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
        <div class="result-box">
            <p>🔍 Please upload an image to get predictions.</p>
        </div>
    """, unsafe_allow_html=True)