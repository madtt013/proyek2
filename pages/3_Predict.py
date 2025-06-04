import streamlit as st
import numpy as np
import joblib

st.title("🧾 Prediksi Spesies Penguin")

# Load model
model = joblib.load("model/penguin_model.pkl")

st.write("Masukkan fitur-fitur berikut untuk memprediksi spesies penguin:")

culmen_length = st.number_input("Panjang Culmen (mm)", min_value=30.0, max_value=60.0, value=45.0)
culmen_depth = st.number_input("Kedalaman Culmen (mm)", min_value=13.0, max_value=21.0, value=17.0)
flipper_length = st.number_input("Panjang Sirip (mm)", min_value=170.0, max_value=230.0, value=200.0)
body_mass = st.number_input("Massa Tubuh (g)", min_value=2700.0, max_value=6300.0, value=4000.0)

if st.button("Prediksi"):
    features = np.array([[culmen_length, culmen_depth, flipper_length, body_mass]])
    prediction = model.predict(features)
    st.success(f"Spesies yang diprediksi: {prediction[0]}")
