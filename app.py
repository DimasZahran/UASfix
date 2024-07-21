import streamlit as st
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# Memuat model, scaler, dan encoder
model = joblib.load('best_model.pkl')
scaler = joblib.load('scaler.pkl')
label_encoder = joblib.load('label_encoder.pkl')

# Judul aplikasi
st.title('Prediksi Status Transaksi')

# Input dari pengguna
sender_name = st.text_input('Nama Pengirim')
sender_upi = st.text_input('UPI ID Pengirim')
receiver_name = st.text_input('Nama Penerima')
receiver_upi = st.text_input('UPI ID Penerima')
amount = st.number_input('Jumlah (INR)', min_value=0)

# Fungsi untuk prediksi
def predict_status(sender_name, sender_upi, receiver_name, receiver_upi, amount):
    data = np.array([[sender_name, sender_upi, receiver_name, receiver_upi, amount]])
    
    # Encode input data
    data[:, 0] = label_encoder.transform([sender_name])[0]
    data[:, 1] = label_encoder.transform([sender_upi])[0]
    data[:, 2] = label_encoder.transform([receiver_name])[0]
    data[:, 3] = label_encoder.transform([receiver_upi])[0]
    
    # Scale input data
    data = scaler.transform(data)
    
    # Predict
    prediction = model.predict(data)
    return "Berhasil" if prediction[0] == 1 else "Gagal"

# Tampilkan hasil prediksi jika tombol ditekan
if st.button('Prediksi'):
    status = predict_status(sender_name, sender_upi, receiver_name, receiver_upi, amount)
    st.write(f'Status Transaksi: {status}')
