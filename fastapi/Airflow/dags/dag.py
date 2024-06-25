# import libraries 
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime, timedelta
import os
import PIL
import scipy
import tensorflow as tf
import numpy as np
import pickle
import time

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.backend import clear_session
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense
from tensorflow.keras.callbacks import EarlyStopping

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'face_recognition_pipeline',
    default_args=default_args,
    description='A simple face recognition pipeline',
    schedule_interval=timedelta(days=1),
    start_date=datetime(2023, 1, 1),
    catchup=False,
)

def model_training():
    train_images = r"data/Face Images/Train"
    test_images = r"data/Face Images/Test"

    train_gen = ImageDataGenerator(
        shear_range=0.1, 
        zoom_range=0.1,
        horizontal_flip=True
    )

    test_gen = ImageDataGenerator()

    training_data = train_gen.flow_from_directory(
        train_images, 
        target_size=(224,224),
        batch_size=32,
        class_mode='categorical'
    )

    testing_data = test_gen.flow_from_directory(
        test_images, 
        target_size=(224,224),
        batch_size=32,
        class_mode='categorical'
    )

    Train_class = training_data.class_indices

    Result_class = {}
    for value_tag, face_tag in zip(Train_class.values(), Train_class.keys()):
        Result_class[value_tag] = face_tag

    with open('data/Face Images/ResultMap.pkl', 'wb') as Final_mapping:
        pickle.dump(Result_class, Final_mapping)

    Output_Neurons=len(Result_class)

    seed = 50
    clear_session()
    np.random.seed(seed)
    tf.random.set_seed(seed)

    Model = Sequential()

    Model.add(Conv2D(16, kernel_size=(5,5), strides=(1,1), input_shape=(224,224,3), activation='relu'))
    Model.add(MaxPool2D(pool_size=(2,2)))
    Model.add(Conv2D(32, kernel_size=(3,3), strides=(1,1), activation='relu'))
    Model.add(MaxPool2D(pool_size=(2,2)))
    Model.add(Flatten())
    Model.add(Dense(64, activation='relu'))
    Model.add(Dense(Output_Neurons, activation='softmax'))

    Model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['Accuracy'])

    call = EarlyStopping(
        min_delta=0.005,
        patience=5,
        verbose=1
    )
    StartTime = time.time()

    Model.fit(training_data,
                        epochs=50,
                        validation_data=testing_data,
                        callbacks=call)

    EndTime = time.time()
    print('Total Training Time taken: ', round((EndTime - StartTime) / 60), 'Minutes')

    model_directory = 'models'

    if not os.path.exists(model_directory):
        os.makedirs(model_directory)

    model_path = os.path.join(model_directory, 'model.h5')

    Model.save(model_path)

# Define the task
model_training_task = PythonOperator(
    task_id='load_data',
    python_callable=model_training,
    provide_context=True,
    dag=dag,
)

model_training
