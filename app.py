import numpy as np
import pickle
import streamlit as st
from pathlib import Path


MODEL_PATH = Path(__file__).with_name("trained_model.sav")


@st.cache_resource
def load_model():
    with MODEL_PATH.open("rb") as model_file:
        return pickle.load(model_file)


# creating a function for Prediction

def diabetes_prediction(model, input_data):

    # changing the input_data to numpy array
    input_data_as_numpy_array = np.asarray(input_data, dtype=float)

    # reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

    prediction = model.predict(input_data_reshaped)

    if prediction[0] == 0:
        return 'The person is not diabetic'

    return 'The person is diabetic'


def main():

    # giving a title
    st.title('Diabetes Prediction Web App')

    try:
        loaded_model = load_model()
    except FileNotFoundError:
        st.error(f"Model file not found: {MODEL_PATH}")
        st.stop()
    except Exception as error:
        st.error(f"Could not load the model: {error}")
        st.stop()

    # getting the input data from the user

    Pregnancies = st.number_input('Number of Pregnancies', min_value=0, step=1)
    Glucose = st.number_input('Glucose Level', min_value=0.0)
    BloodPressure = st.number_input('Blood Pressure value', min_value=0.0)
    SkinThickness = st.number_input('Skin Thickness value', min_value=0.0)
    Insulin = st.number_input('Insulin Level', min_value=0.0)
    BMI = st.number_input('BMI value', min_value=0.0)
    DiabetesPedigreeFunction = st.number_input(
        'Diabetes Pedigree Function value', min_value=0.0)
    Age = st.number_input('Age of the Person', min_value=0, step=1)

    # code for Prediction
    diagnosis = ''

    # creating a button for Prediction

    if st.button('Diabetes Test Result'):
        diagnosis = diabetes_prediction(
            loaded_model,
            [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age])

    if diagnosis:
        st.success(diagnosis)


if __name__ == '__main__':
    main()
