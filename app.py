import streamlit as st
import joblib
import pandas as pd

# Load model
model = joblib.load('best_model.pkl')

st.title('Prediksi Status Transaksi')
amount = st.number_input('Jumlah Transaksi (INR)', min_value=0)
prediction = model.predict([[amount]])

st.write('Prediksi Status:', 'Berhasil' if prediction[0] == 1 else 'Gagal')
