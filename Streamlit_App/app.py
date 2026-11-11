import pickle
from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Titanic Survival Predictor", page_icon="Titanic", layout="centered")

MODEL_PATH = Path(__file__).resolve().parent.parent / "Models" / "titanic_model.pkl"


@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


try:
    model = load_model()
except FileNotFoundError:
    st.error(f"Model not found at {MODEL_PATH}. Run the notebook cell that saves the .pkl first.")
    st.stop()

st.title("Titanic Survival Predictor")
st.write("Enter passenger details to predict whether they survived.")

col1, col2 = st.columns(2)

with col1:
    pclass = st.selectbox("Ticket class", options=[1, 2, 3], format_func=lambda v: {1: "1st", 2: "2nd", 3: "3rd"}[v])
    sex = st.radio("Sex", options=["male", "female"])
    age = st.number_input("Age", min_value=1, max_value=80, value=30, step=1)
    sibsp = st.number_input("Siblings/Spouses aboard", min_value=0, max_value=8, value=0, step=1)

with col2:
    parch = st.number_input("Parents/Children aboard", min_value=0, max_value=6, value=0, step=1)
    fare = st.number_input("Fare", min_value=0.0, max_value=520.0, value=32.0, step=1.0)
    embarked = st.selectbox("Port of embarkation", options=["S", "C", "Q"])
    alone = st.checkbox("Travelling alone")

if st.button("Predict"):
    passenger = pd.DataFrame(
        [
            {
                "pclass": pclass,
                "sex": 1 if sex == "male" else 0,
                "age": age,
                "sibsp": sibsp,
                "parch": parch,
                "fare": fare,
                "embarked": {"C": 0, "Q": 1, "S": 2}[embarked],
                "alone": int(alone),
            }
        ]
    )

    prediction = int(model.predict(passenger)[0])

    st.divider()
    if prediction == 1:
        st.success("Prediction: Survived")
    else:
        st.error("Prediction: Did not survive")
