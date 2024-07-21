import streamlit as st
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from flask import Flask, request, jsonify
import joblib


# Fungsi untuk memuat dan mempersiapkan model
def load_model():
    df = pd.read_csv('transactions.csv')
    model = joblib.load('best_model.pkl')
    df = df.dropna()
    label_encoder = LabelEncoder()
    df['Sender Name'] = label_encoder.fit_transform(df['Sender Name'])
    df['Receiver Name'] = label_encoder.fit_transform(df['Receiver Name'])
    df['Sender UPI ID'] = label_encoder.fit_transform(df['Sender UPI ID'])
    df['Receiver UPI ID'] = label_encoder.fit_transform(df['Receiver UPI ID'])
    df['status'] = label_encoder.fit_transform(df['status'])
    X = df.drop('status', axis=1)
    y = df['status']
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    model = LogisticRegression()
    model.fit(X, y)
    return model, scaler, label_encoder

model, scaler, label_encoder = load_model()

st.title('Prediksi Status Transaksi')

# Input user
sender_name = st.text_input('Nama Pengirim')
sender_upi = st.text_input('UPI ID Pengirim')
receiver_name = st.text_input('Nama Penerima')
receiver_upi = st.text_input('UPI ID Penerima')
amount = st.number_input('Jumlah (INR)', min_value=0)

if st.button('Prediksi'):
    data = np.array([[
        sender_name,
        sender_upi,
        receiver_name,
        receiver_upi,
        amount
    ]])
    data[:, 0] = label_encoder.transform(data[:, 0])
    data[:, 2] = label_encoder.transform(data[:, 2])
    data[:, 1] = label_encoder.transform(data[:, 1])
    data[:, 3] = label_encoder.transform(data[:, 3])
    data = scaler.transform(data)
    prediction = model.predict(data)
    st.write(f'Status Transaksi: {"Berhasil" if prediction[0] == 1 else "Gagal"}')
