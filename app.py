import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Fungsi untuk memuat model dan scaler
def load_model_and_scaler():
    # Memuat model terbaik dari file
    model = joblib.load('best_model.pkl')
    
    # Memuat scaler dan label encoder
    scaler = joblib.load('scaler.pkl')
    label_encoder = joblib.load('label_encoder.pkl')
    
    return model, scaler, label_encoder

# Fungsi untuk memuat data transaksi
@st.cache
def load_data():
    return pd.read_csv('transactions.csv')

def main():
    st.title('Aplikasi Prediksi Status Transaksi')

    # Menampilkan data dari transactions.csv
    st.subheader('Data Transaksi')
    data = load_data()
    st.dataframe(data)

    # Memuat model, scaler, dan encoder
    model, scaler, label_encoder = load_model_and_scaler()

    st.subheader('Prediksi Status Transaksi')

    # Input pengguna
    sender_name = st.text_input('Nama Pengirim')
    sender_upi = st.text_input('UPI ID Pengirim')
    receiver_name = st.text_input('Nama Penerima')
    receiver_upi = st.text_input('UPI ID Penerima')
    amount = st.number_input('Jumlah (INR)', min_value=0)

    if st.button('Prediksi'):
        # Mengubah data input ke format yang diperlukan
        data = np.array([[
            sender_name,
            sender_upi,
            receiver_name,
            receiver_upi,
            amount
        ]])

        # Mengkodekan data input
        data[:, 0] = label_encoder.transform([data[:, 0][0]])
        data[:, 2] = label_encoder.transform([data[:, 2][0]])
        data[:, 1] = label_encoder.transform([data[:, 1][0]])
        data[:, 3] = label_encoder.transform([data[:, 3][0]])

        # Menskalakan data input
        data = scaler.transform(data)

        # Memprediksi menggunakan model
        prediction = model.predict(data)
        
        # Menampilkan hasil prediksi
        st.write(f'Status Transaksi: {"Berhasil" if prediction[0] == 1 else "Gagal"}')

if __name__ == '__main__':
    main()
