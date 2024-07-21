import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Fungsi untuk memuat model dan scaler
def load_model_and_scaler():
    try:
        # Memuat model terbaik dari file
        model = joblib.load('best_model.pkl')
        
        # Memuat scaler dan label encoder
        scaler = joblib.load('scaler.pkl')
        label_encoder = joblib.load('label_encoder.pkl')
        
        return model, scaler, label_encoder
    except Exception as e:
        st.error(f"Error loading model or scaler: {e}")
        return None, None, None

# Fungsi untuk memuat data transaksi
@st.cache
def load_data():
    try:
        return pd.read_csv('transactions.csv')
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

def main():
    st.title('Aplikasi Prediksi Status Transaksi')

    # Menampilkan data dari transactions.csv
    st.subheader('Data Transaksi')
    data = load_data()
    if not data.empty:
        st.dataframe(data)

    # Memuat model, scaler, dan encoder
    model, scaler, label_encoder = load_model_and_scaler()

    if model is not None and scaler is not None and label_encoder is not None:
        st.subheader('Prediksi Status Transaksi')

        # Input pengguna
        sender_name = st.text_input('Nama Pengirim')
        sender_upi = st.text_input('UPI ID Pengirim')
        receiver_name = st.text_input('Nama Penerima')
        receiver_upi = st.text_input('UPI ID Penerima')
        amount = st.number_input('Jumlah (INR)', min_value=0)

        if st.button('Prediksi'):
            try:
                # Mengubah data input ke format yang diperlukan
                data_input = np.array([[
                    sender_name,
                    sender_upi,
                    receiver_name,
                    receiver_upi,
                    amount
                ]])

                # Mengkodekan data input
                data_input[:, 0] = label_encoder.transform([data_input[:, 0][0]])
                data_input[:, 2] = label_encoder.transform([data_input[:, 2][0]])
                data_input[:, 1] = label_encoder.transform([data_input[:, 1][0]])
                data_input[:, 3] = label_encoder.transform([data_input[:, 3][0]])

                # Menskalakan data input
                data_input = scaler.transform(data_input)

                # Memprediksi menggunakan model
                prediction = model.predict(data_input)
                
                # Menampilkan hasil prediksi
                st.write(f'Status Transaksi: {"Berhasil" if prediction[0] == 1 else "Gagal"}')
            except Exception as e:
                st.error(f"Error in prediction: {e}")
    else:
        st.error("Model atau scaler tidak berhasil dimuat.")

if __name__ == '__main__':
    main()
