import os
import streamlit as st
import cv2
from io import BytesIO
from datetime import datetime
import numpy as np
import pandas as pd
import requests

def attend(PREDICTION_ENDPOINT, EMPLOYEE_DETAILS_ENDPOINT, current_time):
    cam_input= st.camera_input("Take a picture")

    if cam_input is not None:
        img_bytes = cam_input.getvalue()
        img_np = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
        img_resized = cv2.resize(img, (304, 304))

        # Generate a timestamp for the filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        img_filename = f"captured_image_{timestamp}.jpg"
        save_dir = 'Face Images\Attendance Tracker\Attend'
        os.makedirs(save_dir, exist_ok=True)
        img_path = os.path.join(save_dir, img_filename)

        # Save the image
        cv2.imwrite(img_path, img_resized)

        response = requests.post(PREDICTION_ENDPOINT, params={'image_path': img_path})

        if response.status_code == 200:
            prediction = response.json().get('predicted_class', 'Unknown')

            # Fetch employee details from the FastAPI endpoint
            response = requests.post(EMPLOYEE_DETAILS_ENDPOINT, json={"image_id": prediction})
            if response.status_code == 200:
                employee_details = response.json()
                st.success(f"""Welcome {employee_details['name']}!,
                        Employee ID: {employee_details['id']},  
                        Position: {employee_details['position']},  
                        Department: {employee_details['department']},
                        Check-in Time: {current_time}""")
                
                id = employee_details['id']
                name = employee_details['name']

                # Prepare the data to be sent
                attendance_data  = {
                    "employee_id": id,
                    "employee_name": name,
                    "check_out_time": False
                }

                 # URL and headers
                url = 'http://localhost:8000/attendance'

                headers = {
                    'accept': 'application/json',
                    'Content-Type': 'application/json',
                }

                # Send the POST request
                response = requests.post(url, headers=headers, json=attendance_data)

                # Print the response from the server
                if response.status_code == 200:
                    st.success("Attendance recorded successfully!")
                else:
                    st.error(f"Failed to record attendance. Status code: {response.status_code} {response.status_code}")
            
            else:
                st.write(f"Employee not found. Status code: {response.status_code}")

        else:
            st.write(f"Prediction failed with status code {response.status_code}: {response.text}")

def leave(PREDICTION_ENDPOINT, EMPLOYEE_DETAILS_ENDPOINT, current_time):
        cam_input= st.camera_input("Take a picture")

        if cam_input is not None:
            img_bytes = cam_input.getvalue()
            img_np = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
            img_resized = cv2.resize(img, (304, 304))

            # Generate a timestamp for the filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            img_filename = f"captured_image_{timestamp}.jpg"
            save_dir = 'Face Images\Attendance Tracker\Leave'
            os.makedirs(save_dir, exist_ok=True)            
            img_path = os.path.join(save_dir, img_filename)

            # Save the image
            cv2.imwrite(img_path, img_resized)

            response = requests.post(PREDICTION_ENDPOINT, params={'image_path': img_path})

            if response.status_code == 200:
                prediction = response.json().get('predicted_class', 'Unknown')

                # Fetch employee details from the FastAPI endpoint
                response = requests.post(EMPLOYEE_DETAILS_ENDPOINT, json={"image_id": prediction})
                if response.status_code == 200:
                    employee_details = response.json()
                    st.success(f"""Good Bye {employee_details['name']}!,
                            Employee ID: {employee_details['id']}, 
                            Position: {employee_details['position']},  
                            Department: {employee_details['department']},
                            Check-out Time: {current_time}""")
                    
                    id = employee_details['id']
                    name = employee_details['name']

                    # Prepare the data to be sent
                    checkout_data = {
                        "employee_id": id,
                        "employee_name": name
                    }

                    # URL and headers
                    url = 'http://localhost:8000/attendance/checkout'
                    headers = {
                        'accept': 'application/json',
                        'Content-Type': 'application/json',
                    }

                    # Send the POST request
                    response = requests.post(url, headers=headers, json=checkout_data)

                    # Print the response from the server
                    if response.status_code == 200:
                        st.success("Check-out time recorded successfully!")
                    else:
                        st.error(f"Failed to record check-out time. Status code: {response.status_code}")
                    
                else:
                    st.write(f"Employee not found. Status code: {response.status_code}")

            else:
                st.write(f"Prediction failed with status code {response.status_code}: {response.text}")

# Function to get database length
def get_database_length_st():
    url = 'http://localhost:8000/employees_count'  
    response = requests.get(url)
    if response.status_code == 200:
        return response.json().get('count', 0)
    else:
        st.error("Failed to retrieve database length")
        return 0
    
# Register Employees
def register_st():
    name =  st.text_input('Name: ')
    department = st. selectbox('Department: ',['Accounting','Human Resources','I.T.','Logistics','Marketing','Production','Sales','Works'])
    position =  st.text_input('Position: ')

    # Determine ID
    database_length = get_database_length_st()
    new_id = database_length + 1

    # Create the image_path
    imageid = f"{new_id}_{name.replace(' ', '')}"

    # Prepare the data to be sent
    data = {
        "id": new_id,
        "name": name,
        "position": position,
        "department": department,  
        "imageid": imageid   
    }

    attendance_data  = {
        "employee_id": new_id,
        "employee_name": name,
        "check_out_time": False
    }


    # For Training
    cam_input_train= st.camera_input("Take a picture (**For Training Data**)")

    if cam_input_train is not None:
        img_bytes = cam_input_train.getvalue()
        img_np = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
        img_resized = cv2.resize(img, (304, 304))

        # Generate a timestamp for the filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        img_filename = f"captured_image_{timestamp}.jpg"
        
        # Save directories
        save_dir_train = f'Face Images\Train\{imageid}'
        os.makedirs(save_dir_train, exist_ok=True)

        # Save to Train folder        
        img_path_train = os.path.join(save_dir_train, img_filename)
        cv2.imwrite(img_path_train, img_resized)

    # For Test
    cam_input_test= st.camera_input("Take a picture (**For Test Data**)")

    if cam_input_test is not None:
        img_bytes = cam_input_test.getvalue()
        img_np = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
        img_resized = cv2.resize(img, (304, 304))

        # Generate a timestamp for the filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        img_filename = f"captured_image_{timestamp}.jpg"
        
        # Save directories
        save_dir_test = f'Face Images\Test\{imageid}'
        os.makedirs(save_dir_test, exist_ok=True)

        # Save to Test folder
        img_path_test = os.path.join(save_dir_test, img_filename)
        cv2.imwrite(img_path_test, img_resized)

    # URL and headers
    url_register = 'http://localhost:8000/register'
    url_checkin = 'http://localhost:8000/attendance'
    headers = {
        'accept': 'application/json',
        'Content-Type': 'application/json',
    }

    # Add a submit button
    submitted = st.button("Register Employee")

    # Send the POST request
    if submitted:
        response = requests.post(url_register, headers=headers, json=data)
        response_checkin = requests.post(url_checkin, headers=headers, json=attendance_data)

        # Print the response from the server
        if response.status_code == 200 and response_checkin.status_code == 200:
            st.success(f"Registration successful! and Welcome {name}")
        else:
            st.error(f"Failed to register. Status code: {response.status_code}")

# Get all attendance
def get_all_attendance():
    url = 'http://localhost:8000/all_attendance' 
    response = requests.get(url)
    if response.status_code == 200:
            attendance_data = response.json()
            
            df = pd.DataFrame(attendance_data)
           
            new_column_names = {
            0: "EmployeeID",
            1: "EmployeeName",
            2: "Date",
            3: "CheckInTime",
            4: "CheckOutTime"
            }
            
            df = df.rename(columns=new_column_names)
            
            st.success("Successfully retrieved attendance data.")
            st.dataframe(df)
    else:
            st.error(f"Failed to retrieve data. Status code: {response.status_code}")

# Get all attendance
def get_all_employee():
    url = 'http://localhost:8000/all_employee' 
    response = requests.get(url)
    if response.status_code == 200:
            employee_data  = response.json()
            
            df = pd.DataFrame(employee_data )
            
            new_column_names = {
            0: "ID",
            1: "Name",
            2: "Position",
            3: "Department",
            4: "ImageID"
            }
            
            df = df.rename(columns=new_column_names)
            
            st.success("Successfully retrieved employee data.")
            st.dataframe(df)
    else:
            st.error(f"Failed to retrieve data. Status code: {response.status_code}")