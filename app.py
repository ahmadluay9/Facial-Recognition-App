import os
import streamlit as st
from streamlit_option_menu import option_menu
import cv2
from io import BytesIO
from datetime import datetime
import sys
import requests
from predicts import *

# Set the default encoding to UTF-8
sys.stdout.reconfigure(encoding='utf-8')

# Define the directory where images will be saved
SAVE_DIR = "Face Images\captured_images"
os.makedirs(SAVE_DIR, exist_ok=True)

PREDICTION_ENDPOINT = 'http://localhost:8000/predict/image' 

with st.sidebar:
    selected = option_menu("Main Menu", ["Login", 'Register','Database'], 
       default_index=1)
    selected

if selected =='Login':
    # Reading Camera Image
    cam_input= st.camera_input("Take a picture")

    if cam_input is not None:
        img_bytes = cam_input.getvalue()
        img_np = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)

        # Generate a timestamp for the filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        img_filename = f"captured_image_{timestamp}.jpg"
        img_path = os.path.join(SAVE_DIR, img_filename)

        # Save the image
        cv2.imwrite(img_path, img)

        st.success(f"Image saved at {img_path}") 
        # predict_from_image(img_path)
                # Send the image to the prediction endpoint
        # with open(img_path, 'rb') as img_file:
        response = requests.post(PREDICTION_ENDPOINT, params={'image_path': img_path})
        st.write(response)

        if response.status_code == 200:
            prediction = response.json().get('predicted_class', 'Unknown')
            st.write(f"Predicted class: {prediction}")
        else:
            st.write(f"Prediction failed with status code {response.status_code}: {response.text}")


