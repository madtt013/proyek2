import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib

st.title("🤖 Pelatihan Model")

@st.cache_data
def load_data():
    return pd.read_csv("data/penguins_size.csv")

df = load_data()
df = df.dropna()

X = df[["culmen_length_mm", "culmen_depth_mm", "flipper_length_mm", "body_mass_g"]]
y = df["species"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

st.write("### Evaluasi Model")
st.text(classification_report(y_test, y_pred))

# Simpan model
joblib.dump(model, "model/penguin_model.pkl")
