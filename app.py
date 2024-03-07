

import streamlit as st
from PIL import Image
from pickle import load
from numpy import argmax
from keras.preprocessing.sequence import pad_sequences
from keras.applications.resnet50 import ResNet50
from keras.models import Model
from keras.models import load_model
from os import listdir
from tensorflow.keras.applications.resnet50 import ResNet50
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import Model

def laod_image(image_file):
	img = Image.open(image_file)
	return img

st.title("Image Captioning")

image_file = st.file_uploader("Upload Your File Here")


def generate_desc(model, tokenizer, photo, max_length):
    in_text = 'startseq'
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        yhat = model.predict([photo,sequence], verbose=0)
        yhat = argmax(yhat)
        word = word_for_id(yhat, tokenizer)
        if word is None:
            break
        in_text += ' ' + word
        if word == 'endseq':
            break
    return in_text

def word_for_id(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

def extract_features(directory):
    model = ResNet50()
    model = Model(inputs=model.inputs, outputs=model.layers[-2].output)
    print(model.summary())
    features = dict()
    for name in listdir(directory):
        filename = directory + '/' + name
        image = load_img(filename, target_size=(224,224))
        image = img_to_array(image)
        image = image.reshape(1,image.shape[0], image.shape[1], image.shape[2])
        image = preprocess_input(image)
        feature = model.predict(image, verbose=0)
        image_id = name.split('.')[0]
        features[image_id]= feature
    return features

tokenizer = load(open(r"C:\Users\kevin\PycharmProjects\tokenizer.pkl", 'rb'))
max_length = 34
model = load_model(r"C:\Users\kevin\OneDrive\Desktop\models_resnet_50\model_9.h5")

# print(model50.summary())

if image_file is not None:
    image = Image.open(image_file)
    st.image(image)
    photo = extract_features(r"C:\Users\kevin\OneDrive\Desktop\example")
    description = generate_desc(model, tokenizer, photo, max_length)
    st.header(description[8:-6])
