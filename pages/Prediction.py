import numpy as np
import pandas as pd
import pickle
import streamlit as st

# 1. Use raw string (r"...") or forward slashes to prevent escape character errors in Windows paths
model = pickle.load(open("iris_model.pkl", "rb"))

st.title("Iris Flower Prediction App")

sepal_length = st.slider("Sepal Length(cm)", 4.0, 8.0, 5.0)
sepal_width = st.slider("Sepal Width(cm)", 2.0, 4.5, 3.0)
petal_length = st.slider("Petal Length(cm)", 1.0, 7.0, 4.0)
petal_width = st.slider("Petal Width(cm)", 0.1, 2.5, 1.0)

predict = st.button("Predict Species")

if predict:
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    pred = model.predict(input_data)[0]
    st.success(f"Predicted Species: **{pred}**")
