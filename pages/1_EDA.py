import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

st.title("📊 Exploratory Data Analysis (EDA)")

@st.cache_data
def load_data():
    return pd.read_csv("data/penguins_size.csv")

df = load_data()
st.write("### Dataset", df)

st.write("### Informasi Dataset")
st.write(df.describe())

st.write("### Visualisasi Distribusi Spesies")
fig, ax = plt.subplots()
sns.countplot(data=df, x="species", ax=ax)
st.pyplot(fig)

st.write("### Korelasi Fitur Numerik")
fig, ax = plt.subplots()
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm", ax=ax)
st.pyplot(fig)
