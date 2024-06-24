from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import sqlite3
from datetime import time

from predicts import *
from database import *

app = FastAPI()

# Define Pydantic models for data validation
class Employee(BaseModel):
    id: int
    name: str
    position: str
    department: str
    imageid : str

class ImageIDRequest(BaseModel):
    image_id: str

class Attendance(BaseModel):
    employee_id: int
    employee_name : str
    check_out_time: bool = False

class CheckOut(BaseModel):
    employee_id: int
    employee_name : str

init_db()

# Function to establish a connection to the database
def get_db_connection():
    conn = sqlite3.connect('attendance.db')
    return conn

@app.get('/')
def index():
    return {'message': 'Face Recognition Application'}

# Endpoint to register a new employee
@app.post("/register")
def register_employee(employee: Employee):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('INSERT INTO Employees (ID, Name, Position, Department, ImageID) VALUES (?, ?, ?, ?,?)',
                   (employee.id, employee.name, employee.position, employee.department, employee.imageid))
    conn.commit()
    conn.close()
    return {"status": "success", "message": "Employee registered"}

# Endpoint for employee details
@app.post("/employee_details/")
def fetch_employee_details(imageidrequest: ImageIDRequest):
  conn = get_db_connection()
  cursor = conn.cursor()
  image_id = imageidrequest.image_id
  try:
    cursor.execute("SELECT ID, Name, Position, Department FROM Employees WHERE ImageID=?", (image_id,))
    result = cursor.fetchone()

    if result:
      return {
        "status": "success",
        "id": result[0],
        "name": result[1],
        "position": result[2],
        "department": result[3]
      }
    else:
      raise HTTPException(status_code=404, detail="Employee not found")
  except (sqlite3.Error, Exception) as e:  # Catch both sqlite and generic exceptions
    raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")
  finally:
    conn.close()

# Endpoint to get the count of employees
@app.get("/employees_count")
def get_employee_count():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT COUNT(*) FROM Employees')
    count = cursor.fetchone()[0]
    conn.close()
    return {"count": count}

# Endpoint to get all employee
@app.get("/all_employee")
def get_all_employee():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM Employees')
    select = cursor.fetchall()
    conn.close()
    return select

# Endpoint to get all attendance
@app.get("/all_attendance")
def get_all_attendance():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM Attendance')
    select = cursor.fetchall()
    conn.close()
    return select

# Endpoint to record attendance
@app.post("/attendance")
def record_attendance(attendance: Attendance):
    conn = get_db_connection()
    cursor = conn.cursor()

    # Validate if the employee_id exists in Employees table
    cursor.execute('SELECT Name FROM Employees WHERE ID = ?', (attendance.employee_id,))
    employee = cursor.fetchone()
    if not employee:
        conn.close()
        raise HTTPException(status_code=404, detail="Employee not found")

    today_date = datetime.today().date().isoformat()
    current_time = datetime.now().strftime("%H:%M:%S")
    check_in_time_str = current_time  # Use current time for check-in
    check_out_time_str = current_time if attendance.check_out_time else None  # Use current time for check-out if provided
    
    cursor.execute(
        'INSERT INTO Attendance (EmployeeID, EmployeeName, Date, CheckInTime, CheckOutTime) VALUES (?, ?, ?, ?, ?)',
        (attendance.employee_id, attendance.employee_name, today_date, check_in_time_str, check_out_time_str)
    )
    conn.commit()
    conn.close()
    
    return {"status": "success", "message": "Attendance recorded"}

# Endpoint to record check-out time
@app.post("/attendance/checkout")
def record_check_out(checkout: CheckOut):
    conn = get_db_connection()
    cursor = conn.cursor()

   # Validate if the employee_id exists in Attendance table for today
    today_date = datetime.today().date().isoformat()
    cursor.execute('SELECT * FROM Attendance WHERE EmployeeID = ? AND Date = ?', (checkout.employee_id, today_date))
    attendance_record = cursor.fetchone()
    if not attendance_record:
        conn.close()
        raise HTTPException(status_code=404, detail="Attendance record not found for today")
    
    current_time = datetime.now().strftime("%H:%M:%S")

    cursor.execute(
        'UPDATE Attendance SET CheckOutTime = ? WHERE EmployeeID = ? AND Date = ?',
        (current_time, checkout.employee_id, today_date)
    )
    conn.commit()
    conn.close()


# Endpoint to recognize a face and record attendance
@app.post("/recognize")
def recognize_face(attendance: Attendance):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM Employees WHERE ID = ?', (attendance.employee_id,))
    employee = cursor.fetchone()
    if not employee:
        conn.close()
        raise HTTPException(status_code=404, detail="Employee not found")

    # Simulate face matching and record attendance
    date = datetime.date.today().isoformat()
    check_in_time = datetime.datetime.now().time().isoformat()
    cursor.execute('INSERT INTO Attendance (EmployeeID, Date, CheckInTime) VALUES (?, ?, ?)',
                   (attendance.employee_id, date, check_in_time))
    conn.commit()
    conn.close()
    return {"status": "success", "message": "Attendance recorded"}

# Endpoint to get attendance records for an employee
@app.get("/attendance/{employee_id}")
def get_attendance(employee_id: int):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM Attendance WHERE EmployeeID = ?', (employee_id,))
    attendances = cursor.fetchall()
    conn.close()
    return {"status": "success", "data": attendances}

# Endpoint for prediction from an image file
@app.post('/predict/image')
def predict_from_image(image_path: str):
    try:
        predicted_class_label = model_prediction(image_path, model, label_map, from_image=True)
        return {'predicted_class': predicted_class_label}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



