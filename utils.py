import numpy as np
from keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import InputLayer



#load model
try:
    model = load_model("model.keras", compile=False)
except Exception as e:
    import streamlit as st
    st.error("Error loading model: {}".format(str(e)))
    raise

def preprocess_image(image):
    image = image.resize((128,128)).convert('L')
    image = img_to_array(image)
    image = image.reshape((1,128,128,1))/255.0
    return image

def predict(image):
    proccesed = preprocess_image(image)
    pred_gender, pred_age = model.predict(proccesed)
    age = pred_age[0][0]
    gender = 'Male' if pred_gender[0][0] < 0.5 else 'Female'
    return age,gender
   