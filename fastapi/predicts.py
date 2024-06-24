import os
from keras.models import load_model  # type: ignore
from keras.preprocessing import image # type: ignore
import numpy as np
import pickle

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model_path = os.path.join(os.path.dirname(__file__), '../model/model.h5')
model = load_model(model_path, compile=False)

# Load the labels
label_path = os.path.join(os.path.dirname(__file__), '../Face Images/ResultMap.pkl') #"Face Images/ResultMap.pkl"

with open(label_path, 'rb') as file:
    label_map  = pickle.load(file)

def prepare_image(img_path, target_size=(224, 224)):
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def model_prediction(input_data, model, label_map, from_image=True):
    img_array = prepare_image(input_data)
   
    prediction = model.predict(img_array)
    predicted_class_index = np.argmax(prediction, axis=1)[0]
    predicted_class_label = label_map[predicted_class_index]
    
    return predicted_class_label

# Function to predict from an image file
def predict_from_image(image_path):
    predicted_class_label = model_prediction(image_path, model, label_map, from_image=True)
    print(f"Predicted class from image: {predicted_class_label}")


