import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="Diabetes Prediction",
    page_icon="medical_symbol",
    layout="wide",
)

MODEL_PATH = Path(__file__).with_name("trained_model.sav")


@st.cache_resource
def load_model():
    with MODEL_PATH.open("rb") as model_file:
        return pickle.load(model_file)


def diabetes_prediction(model, input_data):
    input_array = np.asarray(input_data, dtype=float).reshape(1, -1)
    prediction = model.predict(input_array)

    probability = None
    if hasattr(model, "predict_proba"):
        probability = float(model.predict_proba(input_array)[0][1])

    if prediction[0] == 0:
        return "The person is not diabetic", probability

    return "The person is diabetic", probability


def main():
    st.sidebar.title("About")
    st.sidebar.info(
        """
        Diabetes Prediction App

        Built using:
        - Streamlit
        - Scikit-Learn
        - NumPy
        """
    )

    st.title("Diabetes Prediction Web App")

    try:
        loaded_model = load_model()
    except FileNotFoundError:
        st.error(f"Model file not found: {MODEL_PATH}")
        st.stop()
    except Exception as error:
        st.error(f"Could not load the model: {error}")
        st.stop()

    st.subheader("Patient Details")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        pregnancies = st.number_input(
            "Number of Pregnancies", min_value=0, step=1)
        skin_thickness = st.number_input("Skin Thickness value", min_value=0.0)
    with col2:
        glucose = st.number_input("Glucose Level", min_value=0.0)
        insulin = st.number_input("Insulin Level", min_value=0.0)
    with col3:
        blood_pressure = st.number_input("Blood Pressure value", min_value=0.0)
        bmi = st.number_input("BMI value", min_value=0.0)
    with col4:
        diabetes_pedigree_function = st.number_input(
            "Diabetes Pedigree Function value",
            min_value=0.0,
        )
        age = st.number_input("Age of the Person", min_value=0, step=1)

    metric_col1, metric_col2, metric_col3 = st.columns(3)
    metric_col1.metric("Glucose", glucose)
    metric_col2.metric("BMI", bmi)
    metric_col3.metric("Age", age)

    features = {
        "Pregnancies": pregnancies,
        "Glucose": glucose,
        "BloodPressure": blood_pressure,
        "SkinThickness": skin_thickness,
        "Insulin": insulin,
        "BMI": bmi,
        "DPF": diabetes_pedigree_function,
        "Age": age,
    }

    st.subheader("Patient Health Parameters")
    chart_data = pd.DataFrame(
        {"Feature": features.keys(), "Value": features.values()}
    ).set_index("Feature")
    st.bar_chart(chart_data)

    if st.button("Diabetes Test Result", type="primary"):
        diagnosis, probability = diabetes_prediction(
            loaded_model,
            [
                pregnancies,
                glucose,
                blood_pressure,
                skin_thickness,
                insulin,
                bmi,
                diabetes_pedigree_function,
                age,
            ],
        )

        st.success(diagnosis)

        if probability is not None:
            risk = probability * 100
            st.subheader("Prediction Confidence")
            st.progress(probability)
            st.write(f"Diabetes Risk Probability: {risk:.2f}%")

            if risk < 30:
                st.success(f"Low Risk: {risk:.2f}%")
            elif risk < 70:
                st.warning(f"Moderate Risk: {risk:.2f}%")
            else:
                st.error(f"High Risk: {risk:.2f}%")
        else:
            st.info("This model does not provide probability scores.")


if __name__ == "__main__":
    main()
