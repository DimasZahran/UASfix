import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Memuat model yang disimpan
best_model = joblib.load('best_model.pkl')

# Memuat dataset untuk mendapatkan informasi prapemrosesan
df = pd.read_csv('transactions.csv')

# Mengkodekan variabel kategorikal
label_encoder = LabelEncoder()
df['transaction_type'] = label_encoder.fit_transform(df['transaction_type'])

# Menskalakan fitur numerik
scaler = StandardScaler()
df[['amount']] = scaler.fit_transform(df[['amount']])

# Fungsi prediksi menggunakan model yang disimpan
def predict(transaction_type, amount):
    # Menskalakan input
    scaled_amount = scaler.transform([[amount]])
    # Mengkodekan input
    encoded_type = label_encoder.transform([transaction_type])
    # Membuat prediksi
    features = np.array([[encoded_type[0], scaled_amount[0][0]]])
    prediction = best_model.predict(features)
    return prediction[0]

# Membuat antarmuka Streamlit
st.title("Prediksi Jumlah Transaksi UPI")
transaction_type = st.selectbox("Jenis Transaksi:", df['transaction_type'].unique())
amount = st.number_input("Jumlah Transaksi:", min_value=0.0, step=0.01)

if st.button("Prediksi"):
    result = predict(transaction_type, amount)
    st.write(f"Jumlah Transaksi yang Diprediksi: {result}")
