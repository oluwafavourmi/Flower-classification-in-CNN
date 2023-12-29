import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
import tensorflow as tf
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.imagenet_utils import decode_predictions
from keras.models import load_model
from keras.preprocessing import image
import time
import matplotlib.pyplot as plt
import random
import os
from keras.applications import inception_v3

final_model = load_model('C:\\Users\\TeeFaith\\Desktop\\ML PROJECTS\\FLOWERS CLASSIFICATION\\Flower-classification-in-CNN\\Flower Model.h5')

@st.cache(allow_output_mutation=True)
def predict_function(input_image, final_model):
    #solution two
    size = (224, 224)
    image = ImageOps.fit(input_image, size, Image.ANTIALIAS)
    image_array = np.asarray(image)
    image_reshape = image_array[np.newaxis,...]
    prediction = final_model.predict(image_reshape)

    return prediction


st.title('Flower Classification App')
file_image = st.file_uploader('Upload your Image', type=['jpeg', 'jpg', 'png', 'gif'])
if file_image is None:
    st.write('No file is uploaded here')
else:
    input_image = Image.open(file_image)
    st.image(input_image, caption='uploaded image', use_column_width=True)
    predictions = predict_function(input_image, final_model)
    string = ''
    if [np.argmax(predictions)] == [0]:
        string = 'This is the picture of a ROSE'
    elif [np.argmax(predictions)] == [1]:
        string = 'This is the picture of a SUNFLOWER'
    elif [np.argmax(predictions)] == [2]:
        string = 'This is the picture of a LILY' 
    else:
        string = 'This is not any of the three'

    st.success(string)
