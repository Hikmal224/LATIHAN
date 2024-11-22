import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Judul aplikasi
st.title('Prediksi Status Gizi Balita dengan KNN')

# Menampilkan penjelasan aplikasi
st.write("""
Aplikasi ini menggunakan model KNN untuk memprediksi status gizi balita
berdasarkan usia, jenis kelamin, dan tinggi badan.
""")

# Upload file CSV
uploaded_file = st.file_uploader("Unggah data CSV", type="csv")
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Data yang Diupload:", df.head())

    # Kolom dataset
    df.columns = ['Age', 'Gender', 'Height', 'Status']
    
    # Cleaning Data
    df = df.drop_duplicates()
    df['Gender'] = LabelEncoder().fit_transform(df['Gender'])
    df['Status'] = LabelEncoder().fit_transform(df['Status'])
    
    # Data Preparation
    x = df.drop(columns=['Status'])
    y = df['Status']
    
    # Split data train-test
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    
    # Modeling: KNN
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(x_train, y_train)
    
    # Prediksi dan Akurasi
    y_pred = knn.predict(x_test)
    acc = accuracy_score(y_pred, y_test)
    st.write(f"Akurasi Model: {acc*100:.2f}%")
    
    # Menampilkan classification report
    st.write("Laporan Klasifikasi:")
    st.text(classification_report(y_pred, y_test))
    
    # Menampilkan grafik distribusi umur
    st.write("Distribusi Umur")
    sns.histplot(df['Age'], kde=True)
    plt.xlabel('Age')
    plt.ylabel('Frekuensi')
    st.pyplot()

    # Input untuk prediksi
    st.write("Masukkan data untuk prediksi")
    
    age = st.number_input('Usia (Age)', min_value=0)
    gender = st.selectbox('Jenis Kelamin (Gender)', options=['Laki-Laki', 'Perempuan'])
    height = st.number_input('Tinggi Badan (Height)', min_value=0.0)

    # Encode Gender
    gender_encoded = 0 if gender == 'Laki-Laki' else 1

    if st.button('Prediksi'):
        # Prediksi Status Gizi berdasarkan input pengguna
        input_data = pd.DataFrame({
            'Age': [age],
            'Gender': [gender_encoded],
            'Height': [height]
        })
        
        pred = knn.predict(input_data)
        status = "Baik" if pred == 1 else "Buruk"  # Status gizi berdasarkan LabelEncoder (1 = Baik, 0 = Buruk)
        st.write(f"Hasil Prediksi Status Gizi: {status}")