# Import Library
import streamlit as st
from streamlit_option_menu import option_menu
from io import BytesIO
from datetime import datetime
import sys

from predicts import *
from utils import *

# Set the default encoding to UTF-8
sys.stdout.reconfigure(encoding='utf-8')

# Set the page config
st.set_page_config(
    page_title="Employee Attendace App",
    page_icon="💬 ",
    layout="centered",
    initial_sidebar_state="auto",
    menu_items={
        'About': "This application helps company track when employees arrive and leave work. "
    }
)

st.sidebar.header("💬 Employee Attendance App")

PREDICTION_ENDPOINT = 'http://localhost:8000/predict/image' 
EMPLOYEE_DETAILS_ENDPOINT = "http://localhost:8000/employee_details/"

current_time = datetime.now().strftime("%H:%M:%S")

with st.sidebar:
    selected = option_menu("Main Menu", ["Record Attendance", 'Register','Database'], 
       default_index=1)

st.sidebar.markdown('''
        ## About
        This application helps company track when employees arrive and leave work. This application Utilizes facial recognition technology for employee identification through a trained deep learning model built using TensorFlow. 
                 
        ''')
st.sidebar.write("---")

if selected =='Record Attendance':
    st.header("Attendance Tracker")
    st.write("---")
    attend_or_leave  = st.radio('Select: ',["Attend","Leave"])
    if attend_or_leave  == "Attend":
        attend( PREDICTION_ENDPOINT, EMPLOYEE_DETAILS_ENDPOINT, current_time)

    else:
        leave( PREDICTION_ENDPOINT, EMPLOYEE_DETAILS_ENDPOINT, current_time)

elif selected =='Register':
    st.header("Register New Employee")
    st.write("---")
    register_st()

else:
    st.header("Database")
    st.write("---")

    database = st.selectbox('Select Database: ', ['Attendance', 'Employee'])
    username = st.text_input("Username: ",type='password')
    password = st.text_input("Password: ",type='password')

    if database == 'Attendance':
        if st.button('Login'):
            if username == 'admin' and password == 'admin':
                get_all_attendance()

        else:
            st.warning("Please Login")

    else:
        if st.button('Login'):
            if username == 'admin' and password == 'admin':
                get_all_employee()
        else:
            st.warning("Please Login")

st.sidebar.markdown(''' 
## Created by: 
Ahmad Luay Adnani - [GitHub](https://github.com/ahmadluay9) 
''')