import streamlit as st
import joblib
import numpy as np
import os


model_path = os.path.join('/mount/src/uasfix/app.py', 'best_model.pkl')
print(f"Attempting to load model from: {model_path}")

if os.path.isfile(model_path):
    print(f"File {model_path} found.")
else:
    print(f"File {model_path} does not exist.")
    # Tentukan jalur file yang benar atau periksa lokasi file di server

def load_model():
    model = joblib.load('best_model.pkl')
    scaler = joblib.load('scaler.pkl')
    label_encoder = joblib.load('label_encoder.pkl')
    return model, scaler, label_encoder

model, scaler, label_encoder = load_model()

st.title('Prediksi Status Transaksi')

sender_name = st.text_input('Nama Pengirim')
sender_upi = st.text_input('UPI ID Pengirim')
receiver_name = st.text_input('Nama Penerima')
receiver_upi = st.text_input('UPI ID Penerima')
amount = st.number_input('Jumlah (INR)', min_value=0)

if st.button('Prediksi'):
    data = np.array([[sender_name, sender_upi, receiver_name, receiver_upi, amount]])
    data[:, 0] = label_encoder.transform([sender_name])[0]
    data[:, 1] = label_encoder.transform([sender_upi])[0]
    data[:, 2] = label_encoder.transform([receiver_name])[0]
    data[:, 3] = label_encoder.transform([receiver_upi])[0]
    data = scaler.transform(data)
    prediction = model.predict(data)
    st.write(f'Status Transaksi: {"Berhasil" if prediction[0] == 1 else "Gagal"}')

if __name__ == "__main__":
    main()
