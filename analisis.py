import streamlit as st
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load the saved model
with open("best_model.pkl", "rb") as f:
    loaded_model = pickle.load(f)

# Initialize LabelEncoder and StandardScaler
le = LabelEncoder()
scaler = StandardScaler()

# Pre-trained label encoder for 'Sex'
le.classes_ = np.array(['female', 'male'])

# Title of the app
st.title("Prediksi Kelangsungan Hidup Penumpang Titanic")

# Input fields for user
st.header("Masukkan Data Penumpang")

pclass = st.number_input("Kelas Tiket (Pclass) [1, 2, 3]:", min_value=1, max_value=3, step=1)
sex = st.selectbox("Jenis Kelamin (Sex):", ['male', 'female'])
age = st.number_input("Umur Penumpang (Age):", min_value=0.0, step=0.1)
sibsp = st.number_input("Jumlah Saudara/Pasangan (SibSp):", min_value=0, step=1)
parch = st.number_input("Jumlah Orang Tua/Anak (Parch):", min_value=0, step=1)
fare = st.number_input("Harga Tiket (Fare):", min_value=0.0, step=0.1)
embarked_q = st.selectbox("Embarked dari Queenstown (0 = Tidak, 1 = Ya):", [0, 1])
embarked_s = st.selectbox("Embarked dari Southampton (0 = Tidak, 1 = Ya):", [0, 1])

# Predict button
if st.button("Prediksi"):
    try:
        # Prepare user input
        user_input = {
            'Pclass': pclass,
            'Sex': sex,
            'Age': age,
            'SibSp': sibsp,
            'Parch': parch,
            'Fare': fare,
            'Embarked_Q': embarked_q,
            'Embarked_S': embarked_s
        }

        # Convert user input to DataFrame
        user_input_df = pd.DataFrame([user_input])

        # Encode and scale features
        user_input_df['Sex'] = le.transform(user_input_df['Sex'])
        user_input_df[['Age', 'Fare']] = scaler.fit_transform(user_input_df[['Age', 'Fare']])

        # Make prediction
        prediction = loaded_model.predict(user_input_df)

        # Display prediction result
        if prediction[0] == 1:
            st.success("Penumpang kemungkinan **Selamat**.")
        else:
            st.error("Penumpang kemungkinan **Tidak Selamat**.")
    except Exception as e:
        st.error(f"Terjadi kesalahan: {e}")
