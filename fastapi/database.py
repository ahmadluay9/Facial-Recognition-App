import sqlite3
import pandas as pd
import requests
from datetime import datetime

# function to initialize db
def init_db():
    conn = sqlite3.connect('attendance.db')
    cursor = conn.cursor()
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS Employees (
        ID INTEGER PRIMARY KEY,
        Name TEXT NOT NULL,
        Position TEXT NOT NULL,
        Department TEXT NOT NULL,
        ImageID TEXT NOT NULL          
    )
    ''')

    cursor.execute('''
    CREATE TABLE IF NOT EXISTS Attendance (
        EmployeeID INTEGER NOT NULL,
        EmployeeName TEXT NOT NULL,
        Date TEXT NOT NULL,
        CheckInTime TEXT,
        CheckOutTime TEXT,
        FOREIGN KEY (EmployeeID) REFERENCES Employees(ID)
    )
    ''')
    
    conn.commit()
    conn.close()

# Function to fetch and print table contents
def fetch_table_contents(table_name):
    conn = sqlite3.connect('attendance.db')
    cursor = conn.cursor()
    cursor.execute(f'SELECT * FROM {table_name}')
    rows = cursor.fetchall()
    conn.close()

    # Convert rows to DataFrame
    df = pd.DataFrame(rows, columns=[col[0] for col in cursor.description])
    return df

# Example usage:
def print_table_contents(table_name):
    df = fetch_table_contents(table_name)
    print(f"Contents of table '{table_name}':")
    print(df)

# Function to get database length
def get_database_length():
    url = 'http://localhost:8000/employees_count'  # Example endpoint
    response = requests.get(url)
    if response.status_code == 200:
        return response.json().get('count', 0)
    else:
        print("Failed to retrieve database length")
        return 0

# Register Employees
def register():
    name =  input('Employee Name: ')
    position =  input('Employee Position: ')
    department = input ('Employee Department: ')

    # Determine ID
    database_length = get_database_length()
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

    # URL and headers
    url = 'http://localhost:8000/register'
    headers = {
        'accept': 'application/json',
        'Content-Type': 'application/json',
    }

    # Send the POST request
    response = requests.post(url, headers=headers, json=data)

    # Print the response from the server
    if response.status_code == 200:
        print("Registration successful!")
    else:
        print(f"Failed to register. Status code: {response.status_code}")
    
    print(response.json())

# Record Attendance
def record_attendance():
    id =  int(input('Employee id: '))
    name = str(input('Employee name: '))

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
        print("Attendance recorded successfully!")
    else:
        print(f"Failed to record attendance. Status code: {response.status_code} {response.status_code}")
    
    print(response.json())

def record_check_out():
    id = int(input('Employee ID: '))  # Input for employee ID
    name = str(input('Employee name: '))

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
        print("Check-out time recorded successfully!")
    else:
        print(f"Failed to record check-out time. Status code: {response.status_code}")
    
    print(response.json())

# Drop Tables
def drop_table(table_name):
    conn = sqlite3.connect('attendance.db')
    cursor = conn.cursor()
    cursor.execute(f'DROP TABLE IF EXISTS {table_name}')
    conn.commit()
    conn.close()
    print(f"Table '{table_name}' has been dropped successfully.")
