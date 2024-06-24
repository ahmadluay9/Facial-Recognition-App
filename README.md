# Facial Recognition Application

This application helps company track when employees arrive and leave work. This application Utilizes facial recognition technology for employee identification through a trained deep learning model built using TensorFlow. 

# File Explanation
This repository consists of several files :
```
    ┌── Face Images/
            ├── Attendance Tracker/
                ├── Attend/
                ├── Leave/
            ├── Test/
            ├── Train/
            └── ResultMap.pkl
    ├── fastapi/
            ├── attendance.db
            ├── database.ipynb
            ├── database.py
            ├── main.py
            └── predict.py
    ├── model/
            └── model.h5
    ├── streamlit/
            ├── app.py
            └── utils.py
    ├── .gitignore
    ├── README.md
    ├── model_training.ipynb
    └── requirements.txt

```
## Folders

- `Face Images/`: This folder stores images of faces, used for training a facial recognition model or for attendance purposes.
  
    - `Attendance Tracker/`: This subfolder hold data related to attendance tracking, containing:
      
          - Attend/: This subfolder contain images captured during attendance registration or clock-in.
      
          - Leave/: This subfolder contain images captured during attendance deregistration or clock-out.
      
    - `Test/`: This folder hold images used for testing the facial recognition model's performance.
 
    - `Train/`: This most likely stores images used to train the facial recognition model.

    - `ResultMap.pkl`: This file ontain mappings of employee face IDs, stored using Python's pickle module.
      
- `fastapi/`: This folder contains codes for FastAPI web framework.
  
    - `attendance.db`: This file is the database used by the application, used for storing attendance data.
      
    - `database.ipynb`: This is a Jupyter notebook for interacting with the attendance database.
      
    - `database.py`: This file contains code for interacting with sqlite database.
      
    - `main.py:` This is the main Python script that runs the FastAPI application.
 
    - `predict.py:` This file contain functions for making predictions using the trained model.
     
- `model/`: This folder holds the trained facial recognition model
    
    - `model.h5`: This file is the trained facial recognition model in HDF5 format.
      
- `streamlit/`: This folder contains codes for Streamlit application. Streamlit is a framework for building data apps. 
    
    - `app.py`: This is the main Python script for the Streamlit application.

    - `utils.py`: This file contain helper functions used by the Streamlit application.

## File

- `.gitignore`: This file specifies patterns to be ignored by the Git version control system.

- `README.md`: This file likely contains project documentation or instructions.

- `model_training.ipynb`: This might be a Jupyter notebook for training the facial recognition model.

- `requirements.txt`: A file listing the project's dependencies. It is used by pip to install all the necessary packages specified in this file.

# Libraries

Libraries used for this project:

- **FastAPI**: A high-performance web framework for building APIs in Python.

- **Streamlit**: A framework for building data apps in Python.

- **sqlite3**: The standard Python library for accessing SQLite databases.

- **TensorFlow**: This project uses TensorFlow for tasks related to the facial recognition model, such as model definition, training, and making predictions.

- **opencv-python**: This project uses OpenCV for pre-processing facial images (e.g., resizing, normalization) and for other computer vision tasks needed for the model or application.

- **matplotlib**: This project use matplotlib to visualize attendance data and model performance metrics.

- **pandas**: This project use pandas for data manipulation, cleaning, and analysis related to attendance tracking.

- **numpy**: This project use numpy for or numerical computations related to the facial recognition model or data analysis tasks.


