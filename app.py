# app.py
import streamlit as st
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Load model
with open('best_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Load encoder dan scaler
le = LabelEncoder()
le.classes_ = ['Merchant_Type_A', 'Merchant_Type_B', 'Merchant_Type_C']  # Contoh classes
scaler = StandardScaler()

# Aplikasi Streamlit
st.title("Prediksi Transaksi UPI Payment")
st.write("Masukkan data transaksi untuk prediksi:")

# Input pengguna
transaction_amount = st.number_input("Jumlah Transaksi")
merchant_type = st.selectbox("Tipe Merchant", options=le.classes_)

# Preprocessing input
merchant_type_encoded = le.transform([merchant_type])
transaction_amount_scaled = scaler.fit_transform([[transaction_amount]])

# Prediksi
if st.button("Prediksi"):
    prediction = model.predict([[transaction_amount_scaled[0][0], merchant_type_encoded[0]]])
    st.write(f"Prediksi: {'Fraud' if prediction == 1 else 'Non-Fraud'}")
