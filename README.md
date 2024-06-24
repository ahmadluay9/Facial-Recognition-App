# Facial Recognition Application

This application helps company track when employees arrive and leave work. This application Utilizes facial recognition technology for employee identification through a trained deep learning model built using TensorFlow. 

## Setup
1. Clone this repository
```
git clone https://github.com/ahmadluay9/Facial-Recognition-App.git
```

2. Change the directory to `Facial-Recognition-App`
```
cd Facial-Recognition-App
```

3. Create virtual environment
```
python -m venv my_env
```

4. Activating the virtual environment

**For Windows**
 ```bash
 my_env\Scripts\activate.bat
 ```
**For macOS and Linux**
 ```bash
 source my_env/bin/activate
 ```

5. Install package from requirements.txt
**For Windows**
```bash
pip install -r requirements.txt
```

6. Change the directory to `fastapi`
```
cd fastapi
```

7. Run `fastapi` framework
```
uvicorn main:app --reload
```

8. Change the directory to `streamlit`
```
cd streamlit
```

9. Run `streamlit`
```
streamlit run app.py
```

---

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

- `README.md`: This file contains project documentation or instructions.

- `model_training.ipynb`: This is a Jupyter notebook for training the facial recognition model.

- `requirements.txt`: A file listing the project's dependencies. It is used by pip to install all the necessary packages specified in this file.

--- 

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

---

# Dataset

The [dataset](https://vis-www.cs.umass.edu/lfw/) used consists of images of the faces of company employees. There are 15 employees in the dataset, each with multiple images.

## Data Visualization
- Below are some of the images used for model training.
![image](https://github.com/ahmadluay9/Facial-Recognition-App/assets/123846438/b667c4ac-9b4a-46af-bb72-baa6af42f5d2)

- Below are the images used to test the model's performance.
![image](https://github.com/ahmadluay9/Facial-Recognition-App/assets/123846438/1864cbf2-643e-49a5-a7b8-85bec83a2ae8)

---

# Model Training
Here are the parameters used to train the model.
![image](https://github.com/ahmadluay9/Facial-Recognition-App/assets/123846438/a8be9642-14fd-4038-8d33-e70f3ea0e0d7)

1. **conv2d (Conv2D)**, Output Shape: (None, 220, 220, 16), Param #: 1,216
     
    - This is a 2D convolutional layer with 16 filters, each of size 3x3 (or another small filter size). The input shape is  224x224 image with the specified stride and padding.
      
2. **max_pooling2d (MaxPooling2D)**, Output Shape: (None, 110, 110, 16), Param #: 0
      
    - This is a max-pooling layer with a 2x2 pool size, which halves the spatial dimensions of the input (from 220x220 to 110x110) while keeping the number of channels (16) the same. There are no parameters to learn in a pooling layer.
      
3. **conv2d_1 (Conv2D)**, Output Shape: (None, 108, 108, 32), Param #: 4,640
      
    - Another 2D convolutional layer, this time with 32 filters. The output shape a slight reduction in the spatial dimensions, due to no padding being applied (valid padding).
  
4. **max_pooling2d_1 (MaxPooling2D)**, Output Shape: (None, 54, 54, 32), Param #: 0

    - Another max-pooling layer that reduces the spatial dimensions by half again, from 108x108 to 54x54.
      
5. **flatten (Flatten)**, Output Shape: (None, 93312), Param #: 0

    - This layer flattens the 3D tensor output from the previous layer into a 1D tensor. The number of elements in the flattened layer is 54 * 54 * 32 = 93312.
      
6. **dense (Dense)**, Output Shape: (None, 64), Param #: 5,972,032

    - This is a fully connected (dense) layer with 64 units. The number of parameters is calculated as (93312 + 1) * 64 = 5,972,032, where the additional 1 accounts for the bias term.
      
7. **dense_1 (Dense)**, Output Shape: (None, 16), Param #: 1,040

    - Another fully connected layer with 16 units. The number of parameters is (64 + 1) * 16 = 1,040.
      
---
# Model Evaluation

![image](https://github.com/ahmadluay9/Facial-Recognition-App/assets/123846438/2453bc45-5aa7-4205-9d39-894062a5749f)

The image shows two plots that evaluate the performance of a Sequential model on training and validation data. The left plot displays accuracy over epochs, and the right plot displays loss over epochs.

**Left Plot: Training and Validation Accuracy**

- X-axis: Epochs (training iterations)

- Y-axis: Accuracy

- Blue Line: Training accuracy
  
- Orange Line: Validation accuracy
  
**interpretation**:

- Initially, both training and validation accuracy are low, indicating that the model is not performing well.

- As the number of epochs increases, both training and validation accuracy improve significantly, showing that the model is learning and generalizing well to the validation data.

- After about 10 epochs, the accuracy for both training and validation stabilizes and reaches a high level (close to 1.0), indicating good model performance with minimal overfitting, as both training and validation accuracy are similar.

**Right Plot: Training and Validation Loss**

- X-axis: Epochs (training iterations)

- Y-axis: Loss

- Blue Line: Training loss
- 
- Orange Line: Validation loss
  
**interpretation**:

- Initially, the training loss is very high, indicating that the model's predictions are far from the actual values.

- The loss drops rapidly within the first few epochs, showing that the model quickly learns the underlying patterns in the data.

- After about 10 epochs, the training and validation losses converge and remain low, indicating that the model has effectively minimized the error in both training and validation sets.

---

# Database

This project uses SQLite database. There are two tables in this database: the attendance and employee tables.

**Table: Employees**

This table stores information about employees.

- ID: An integer that uniquely identifies each employee.
- Name: A text field that stores the name of the employee.
- Position: A text field that stores the job position of the employee.
- Department: A text field that stores the department in which the employee works.
- ImageID: A text field that stores an identifier for the employee's image.

**Table: Attendance**

This table tracks the attendance records of employees.

- EmployeeID: An integer that uniquely identifies each employee.
- EmployeeName: A text field that stores the name of the employee.
- Date: A text field that stores the date of the attendance record. 
- CheckInTime:  A text field that stores the check-in time of the employee.
- CheckOutTime: A text field that stores the check-out time of the employee.

---
# Application
In this section, a demo of this application will be shown. As explained earlier, this application uses the FastAPI framework and Streamlit.

- **FastAPI**
  
Below is an image of the FastAPI Swagger UI.
![image](https://github.com/ahmadluay9/Facial-Recognition-App/assets/123846438/8a61f6d9-3a37-4c9d-8257-40a0f1ad2dd9)

- **Streamlit**
  
Below is an image of the Streamlit UI.
![image](https://github.com/ahmadluay9/Facial-Recognition-App/assets/123846438/d08d0fd9-cef7-4dbb-9e18-2d46ac018182)
  
## Application Demo

### 1. Record Attendance
   
![Attend-ezgif com-video-to-gif-converter (1)](https://github.com/ahmadluay9/Facial-Recognition-App/assets/123846438/4d9edbe6-0546-4127-b170-4a11143f7cb2)

To record employee attendance:

1. Employees can select either 'attend' for attendance check-in or 'leave' for attendance check-out.

2. Next, employees should click the 'take photo' button.

3. When the 'take a photo' button is clicked, Streamlit will send the data to the API for the model to perform face recognition (`POST /predict/image`) and record the attendance in the Attendance table in the database (`POST /attendance`).
   
### 2. Register new employee

![Register-ezgif com-video-to-gif-converter](https://github.com/ahmadluay9/Facial-Recognition-App/assets/123846438/a8071931-1e4d-49ad-8355-c2f8835e98be)

To register new employees:

1. Employees can click "Register" and then enter their name, position, and department.

2. Next, employees should take two photos. The first photo will be saved to the folder `Face Images/Train/{ID_Employee_Name}.jpg`, and the second photo will be saved to the folder `Face Images/Test/{ID_Employee_Name}.jpg`. These two photos will be used to retrain the existing model.

3. Finally, employees should click the "Register" button. When the "Register" button is clicked, Streamlit will send the data to the API to add the data to the Employee table in the database (`/POST /register`) and also record the attendance in the Attendance table in the database (`/POST /attendance`).









   
 
