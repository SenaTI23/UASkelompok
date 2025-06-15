import streamlit as st
import joblib
import pandas as pd

# Load model dan encoder
model = joblib.load('model_motor.joblib')
le_merk = joblib.load('le_merk.joblib')
le_tipe = joblib.load('le_tipe.joblib')
le_kondisi = joblib.load('le_kondisi.joblib')
le_warna = joblib.load('le_warna.joblib')

st.title("Prediksi Harga Motor Bekas Indonesia")

# Input user
merk_list = list(le_merk.classes_)
merk = st.selectbox("Pilih Merk Motor", merk_list)

# Filter tipe sesuai merk
tipe_all = list(le_tipe.classes_)
tipe_filtered = [t for t in tipe_all if t.startswith(merk)]
tipe = st.selectbox("Pilih Tipe Motor", tipe_filtered)

tahun = st.slider("Tahun Motor", 2015, 2024, 2020)
kilometer = st.number_input("Kilometer (km)", min_value=0, max_value=200000, value=10000, step=1000)

kondisi_list = list(le_kondisi.classes_)
kondisi = st.selectbox("Kondisi Motor", kondisi_list)

warna_list = list(le_warna.classes_)
warna = st.selectbox("Warna Motor", warna_list)

# Prepare input data untuk prediksi
data_input = {
    'Merk_enc': le_merk.transform([merk])[0],
    'Tipe_enc': le_tipe.transform([tipe])[0],
    'Tahun': tahun,
    'Kilometer': kilometer,
    'Kondisi_enc': le_kondisi.transform([kondisi])[0],
    'Warna_enc': le_warna.transform([warna])[0]
}

input_df = pd.DataFrame([data_input])

if st.button("Prediksi Harga"):
    harga_pred = model.predict(input_df)[0]
    st.success(f"Perkiraan Harga Motor Bekas: Rp {int(harga_pred):,}")
