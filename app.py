import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import time
from sklearn.impute import SimpleImputer

# Sidebar Menu
sidebar_option = st.sidebar.radio("Pilih Menu", ("Penjelasan", "Clustering K-Means"))

# Penjelasan Menu
if sidebar_option == "Penjelasan":
    st.sidebar.title("Penjelasan Aplikasi")
    st.write("""
    ### Preprocessing Data

    Data kartu kredit mentah tidak dapat langsung digunakan untuk pemodelan. Preprocessing melibatkan:
    - Menghapus fitur yang tidak relevan
    - Membersihkan nilai yang hilang
    - Melakukan normalisasi data

    #### Seleksi Fitur
    Semua fitur numerik secara otomatis digunakan untuk pemodelan.

    #### Penanganan Missing Value
    Nilai yang hilang diimputasi menggunakan rata-rata. 

    #### Normalisasi Data
    Data dinormalisasi menggunakan Min-Max Scaler untuk memastikan semua fitur memiliki rentang nilai yang sama.

    ### Data Mining

    Data mining adalah proses menemukan pengetahuan tersembunyi dalam basis data.

    #### K-Means Clustering
    K-Means adalah metode clustering populer yang membagi data ke dalam beberapa cluster berdasarkan jarak Euclidean.
    """)

# Clustering K-Means Menu
elif sidebar_option == "Clustering K-Means":
    # Judul Aplikasi
    st.title("Aplikasi Clustering K-Means")

    # Unggah file CSV
    uploaded_file = st.file_uploader("Unggah file CSV", type=["csv"])

    if uploaded_file:
        # Baca data
        data = pd.read_csv(uploaded_file)
        st.subheader("Data Pratinjau")
        st.dataframe(data.head())

        # Pilih semua kolom numerik secara otomatis
        numeric_columns = data.select_dtypes(include=np.number).columns.tolist()

        if numeric_columns:
            X = data[numeric_columns]

            # Tangani NaN (mengisi NaN dengan nilai rata-rata)
            imputer = SimpleImputer(strategy='mean')
            X_imputed = imputer.fit_transform(X)

            # Normalisasi Data menggunakan Min-Max Scaler
            scaler = MinMaxScaler()
            X_scaled = scaler.fit_transform(X_imputed)

            # Tentukan jumlah klaster
            n_clusters = st.slider("Jumlah Klaster (K)", min_value=2, max_value=10, value=3)

            # Catat waktu mulai
            start_time = time.time()

            # Buat model KMeans
            kmeans = KMeans(n_clusters=n_clusters, init='k-means++', random_state=42)
            labels = kmeans.fit_predict(X_scaled)

            # Catat waktu selesai
            end_time = time.time()
            execution_time = end_time - start_time

            # Tambahkan label ke data
            data['Cluster'] = labels
            st.subheader("Hasil Clustering")
            st.dataframe(data.head())

            # Evaluasi performa
            silhouette_avg = silhouette_score(X_scaled, labels)
            dbi_score = davies_bouldin_score(X_scaled, labels)

            st.subheader("Metrik Evaluasi")
            st.write(f"Silhouette Score: {silhouette_avg:.2f}")
            st.write(f"Davies-Bouldin Index: {dbi_score:.2f}")
            st.write(f"Execution Time: {execution_time:.2f} detik")

            # Visualisasi Hasil
            st.subheader("Visualisasi Hasil Clustering")
            pca = PCA(n_components=2)
            reduced_data = pca.fit_transform(X_scaled)

            plt.figure(figsize=(8, 5))
            sns.scatterplot(x=reduced_data[:, 0], y=reduced_data[:, 1], hue=data['Cluster'], palette='Set2', s=50)
            plt.title('Visualisasi Clustering')
            plt.xlabel('Komponen PCA 1')
            plt.ylabel('Komponen PCA 2')
            plt.legend(title='Cluster')
            st.pyplot(plt)
        else:
            st.warning("Dataset tidak memiliki kolom numerik untuk digunakan.")
    else:
        st.info("Silakan unggah file CSV untuk memulai.")
