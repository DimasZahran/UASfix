import streamlit as st
import joblib
import pandas as pd

# Coba memuat model dan tangani kesalahan
try:
    model = joblib.load('best_model.pkl')
except Exception as e:
    st.error(f"Terjadi kesalahan saat memuat model: {e}")

# Jika model berhasil dimuat, lanjutkan dengan aplikasi
if 'model' in locals():
    st.title('Prediksi Status Transaksi')
    amount = st.number_input('Jumlah Transaksi (INR)', min_value=0)
    
    if st.button('Prediksi'):
        try:
            prediction = model.predict([[amount]])
            st.write('Prediksi Status:', 'Berhasil' if prediction[0] == 1 else 'Gagal')
        except Exception as e:
            st.error(f"Terjadi kesalahan saat melakukan prediksi: {e}")
