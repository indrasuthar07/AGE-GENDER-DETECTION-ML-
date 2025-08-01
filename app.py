import streamlit as st
from PIL import Image
from utils import predict

st.set_page_config(page_title="Age & Gender Prediction", layout='centered')
st.title("👤 Age and Gender Prediction from Face Data")

uploaded_file = st.file_uploader("Upload a File", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image',use_container_width=True)
    with st.spinner("Processing..."):
        age,gender = predict(image)
        st.success(f"Predicted Age: {age:.2f} years")
        st.success(f"Predicted Gender: {gender:s}")
        st.markdown(f"### 👩‍🦰 Gender: `{gender}`")
        st.markdown(f"### 🎂 Estimated Age: `{age}` years")

#footer
st.markdown("""
    <style>
        footer {
            visibility: hidden;
        }
    </style>
""", unsafe_allow_html=True)
