import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Fungsi untuk memuat model
def load_model():
    try:
        model = joblib.load('best_model.pkl')
        return model
    except Exception as e:
        st.error(f"Terjadi kesalahan saat memuat model: {e}")
        return None

# Load model
model = load_model()

# Jika model berhasil dimuat, tampilkan antarmuka aplikasi
if model:
    st.title('Prediksi Status Transaksi')

    # Input jumlah transaksi
    amount = st.number_input('Jumlah Transaksi (INR)', min_value=0.0, format='%f')

    if st.button('Prediksi'):
        try:
            # Lakukan prediksi
            prediction = model.predict(np.array([[amount]]))
            status = 'Berhasil' if prediction[0] == 1 else 'Gagal'
            st.write('Prediksi Status:', status)
        except Exception as e:
            st.error(f"Terjadi kesalahan saat melakukan prediksi: {e}")
