import numpy as np
from keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array

#load model
model = load_model('age_gender_model.h5') 
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
   