# 📊 K-Means Customer Segmentation Dashboard

[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io)
[![Live Demo](https://img.shields.io/badge/Live-Demo-brightgreen.svg)](https://customer-behavior-analytics-hyfhi.streamlit.app)
[![Python](https://img.shields.io/badge/Python-3.12+-blue.svg?logo=python&logoColor=white)](https://python.org)
[![Machine Learning](https://img.shields.io/badge/Machine_Learning-K_Means-FF6F00?logo=scikitlearn&logoColor=white)](https://scikit-learn.org)

Aplikasi Web *Enterprise-grade* yang menyulap pemrosesan data Machine Learning kaku menjadi sebuah **Dashboard Interaktif**. Sistem ini menggunakan algoritma **K-Means Clustering** untuk memecah metrik perilaku konsumen (*Recency, Frequency, Monetary*) ke dalam tiga segmen bisnis utama.

---

> 🌐 **LIVE DEMO AVAILABLE**
>
> 🚀 **[KLIK DI SINI UNTUK MENCOBA DASHBOARD (STREAMLIT CLOUD)](https://customer-behavior-analytics-hyfhi.streamlit.app)** 🚀
> 
> *(Silakan unggah dataset ritel Anda dan cobalah berinteraksi dengan visualisasi 3D clustering secara real-time!)*

---

## ✨ Pembaruan Sistem (Enterprise Upgrade)

- **Anti-Hardcode File Upload**: Menggantikan *absolute path* yang rawan *error* dengan fitur *File Uploader* Streamlit yang dinamis dan aman.
- **Visualisasi Interaktif 3D**: Menggantikan output `.png` statis dengan Plotly 3D *Scatter Chart*. Pengguna dapat memutar, melakukan *zoom*, dan menyeleksi *cluster* secara *real-time* di *browser*.
- **Desain UI Premium**: Navigasi bersih dengan tema *Clean & White*, *Custom CSS*, efek *hover* pada kartu metrik, dan tipografi bergradasi elegan.
- **Kinerja Cepat**: Dioptimalkan dengan modul *caching* (`@st.cache_data` & `@st.cache_resource`) sehingga file berukuran puluhan megabyte dapat diproses tanpa memuat ulang model dari awal.

## 🛠️ Arsitektur Teknologi

- **Frontend**: Streamlit, Custom HTML/CSS
- **Visualisasi**: Plotly Express & Graph Objects
- **Machine Learning**: Scikit-Learn (K-Means, MinMaxScaler)
- **Pemrosesan Data**: Pandas, Numpy, Datetime

## 🚀 Cara Eksekusi Lokal

1. **Clone Repositori**
   ```bash
   git clone https://github.com/HyLuthfi/customer-behavior-analytics.git
   cd customer-behavior-analytics
   ```

2. **Instalasi Dependensi**
   Pastikan Anda menginstal modul yang tepat tanpa spesifikasi versi agar terhindar dari *conflict deployment*.
   ```bash
   pip install -r requirements.txt
   ```

3. **Menjalankan Dashboard**
   ```bash
   streamlit run main.py
   ```
   Setelah server menyala, unggah dataset `Online Retail.xlsx` Anda ke dalam tombol *upload* yang tersedia di layar.

## 📈 Metodologi

- **Preprocessing**: Menghilangkan `CustomerID` kosong, duplikasi data, dan anomali nilai negatif pada `Quantity`/`UnitPrice`.
- **RFM Extraction**: Menghitung seberapa baru (*Recency*), seberapa sering (*Frequency*), dan seberapa besar (*Monetary*) nilai transaksional setiap konsumen.
- **Clustering**: Membagi hasil normalisasi metrik ke dalam **3 Cluster Optimal**, yang secara logis dipetakan ke dalam Segmen *High-Value*, *Medium-Value*, dan *Low-Value*.

---
*Proyek ini telah direkonstruksi dari bentuk prosedural standar menuju arsitektur web modern agar memenuhi standar portofolio industri.*
