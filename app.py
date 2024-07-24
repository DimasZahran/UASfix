import streamlit as st

# Membuat aplikasi Streamlit
def predict(transaction_type, amount):
    # Menskalakan input
    scaled_amount = scaler.transform([[amount]])
    # Mengkodekan input
    encoded_type = label_encoder.transform([transaction_type])
    # Membuat prediksi
    features = [[encoded_type[0], scaled_amount[0][0]]]
    prediction = forest_reg.predict(features)
    return prediction[0]

# Membuat antarmuka Streamlit
st.title("Prediksi Jumlah Transaksi UPI")
transaction_type = st.selectbox("Jenis Transaksi:", df['transaction_type'].unique())
amount = st.number_input("Jumlah Transaksi:", min_value=0.0, step=0.01)

if st.button("Prediksi"):
    result = predict(transaction_type, amount)
    st.write(f"Jumlah Transaksi yang Diprediksi: {result}")
