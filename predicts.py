from keras.models import load_model 
from keras.preprocessing import image
import cv2 
import numpy as np
import pickle
import streamlit as st

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = load_model("model/model.h5", compile=False)

# Load the labels
label_path = "Face Images/ResultMap.pkl"

with open(label_path, 'rb') as file:
    label_map  = pickle.load(file)

# CAMERA can be 0 or 1 based on default camera of your computer
camera = cv2.VideoCapture(0)

def prepare_image(img_path, target_size=(224, 224)):
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# def prepare_image_upload(image_data,target_size=(224, 224)):
#     # Resize or preprocess the image as required by your model
#     img = image.load_img(image_data, target_size=target_size)
#     img_array = image.img_to_array(img)
#     img_array = np.expand_dims(img_array, axis=0) 
#     return img_array

def prepare_frame(frame, target_size=(224, 224)):
    frame = cv2.resize(frame, target_size)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_array = image.img_to_array(frame)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def model_prediction(input_data, model, label_map, from_image=True):
    if from_image:
        img_array = prepare_image(input_data)
    else:
        img_array = prepare_frame(input_data)
    
    prediction = model.predict(img_array)
    predicted_class_index = np.argmax(prediction, axis=1)[0]
    predicted_class_label = label_map[predicted_class_index]
    
    return predicted_class_label

# Function to predict from an image file
def predict_from_image(image_path):
    predicted_class_label = model_prediction(image_path, model, label_map, from_image=True)
    st.success(f"Predicted class from image: {predicted_class_label}")

# Function to predict from the webcam
def predict_from_camera_loop():
    # Initialize the webcam
    camera = cv2.VideoCapture(0)
    
    while True:
        # Capture frame-by-frame
        ret, frame = camera.read()
        if not ret:
            print("Failed to grab frame")
            break
        
        # Predict
        predicted_class_label = model_prediction(frame, model, label_map, from_image=False)
        
        # Display the prediction on the frame
        cv2.putText(frame, f"Predicted: {predicted_class_label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        
        # Display the resulting frame
        cv2.imshow('Webcam', frame)
        
        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # When everything is done, release the capture
    camera.release()
    cv2.destroyAllWindows()


def predict():
    mode = input("Enter 'image' to predict from an image file or 'camera' to predict from the webcam: ").strip().lower()

    if mode == 'image':
        image_path = input("Enter the path to the image file: ").strip()
        predict_from_image(image_path)
    elif mode == 'camera':
        predict_from_camera_loop()
    else:
        print("Invalid mode. Please enter 'image' or 'camera'.")

