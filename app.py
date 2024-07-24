import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Memuat model yang disimpan
try:
    best_model = joblib.load('best_model.pkl')
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Memuat dataset untuk mendapatkan informasi prapemrosesan
df = pd.read_csv('transactions.csv')

# Menyederhanakan fitur numerik
scaler = StandardScaler()
df[['Amount (INR)']] = scaler.fit_transform(df[['Amount (INR)']])

# Mengkodekan UPI ID (karena mereka bisa berupa string)
label_encoder_sender = LabelEncoder()
df['Sender UPI ID'] = label_encoder_sender.fit_transform(df['Sender UPI ID'])

label_encoder_receiver = LabelEncoder()
df['Receiver UPI ID'] = label_encoder_receiver.fit_transform(df['Receiver UPI ID'])

# Fungsi prediksi menggunakan model yang disimpan
def predict(sender_upi_id, receiver_upi_id, amount_inr):
    # Menskalakan input
    scaled_amount = scaler.transform([[amount_inr]])
    # Mengkodekan input
    encoded_sender = label_encoder_sender.transform([sender_upi_id])
    encoded_receiver = label_encoder_receiver.transform([receiver_upi_id])
    # Membuat prediksi
    features = np.array([[encoded_sender[0], encoded_receiver[0], scaled_amount[0][0]]])
    prediction = best_model.predict(features)
    return prediction[0]

# Membuat antarmuka Streamlit
st.title("Prediksi Jumlah Transaksi UPI")
sender_upi_id = st.selectbox("Sender UPI ID:", df['Sender UPI ID'].unique())
receiver_upi_id = st.selectbox("Receiver UPI ID:", df['Receiver UPI ID'].unique())
amount_inr = st.number_input("Amount (INR):", min_value=0.0, step=0.01)

if st.button("Prediksi"):
    result = predict(sender_upi_id, receiver_upi_id, amount_inr)
    st.write(f"Jumlah Transaksi yang Diprediksi: {result}")
