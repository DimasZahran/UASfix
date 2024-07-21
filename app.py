import streamlit as st
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler

# Fungsi untuk memuat model
def load_model():
    try:
        model = joblib.load('best_model.pkl')
        return model
    except Exception as e:
        st.error(f"Terjadi kesalahan saat memuat model: {e}")
        return None

# Muat model
model = load_model()
scaler = StandardScaler()  # Anda harus memastikan scaler yang sama digunakan saat pelatihan

if model:
    st.title('Prediksi Status Transaksi UPI')

    # Input jumlah transaksi
    amount = st.number_input('Jumlah Transaksi (INR)', min_value=0.0, format='%f')

    if st.button('Prediksi'):
        try:
            # Pra-pemrosesan input
            amount_scaled = scaler.transform(np.array([[amount]]))
            prediction = model.predict(amount_scaled)
            status = 'Berhasil' if prediction[0] == 1 else 'Gagal'
            st.write('Prediksi Status:', status)
        except Exception as e:
            st.error(f"Terjadi kesalahan saat melakukan prediksi: {e}")
else:
    st.error("Model gagal dimuat.")
