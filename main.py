import os
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import streamlit as st

st.set_page_config(page_title="Customer Segmentation Dashboard", layout="wide", initial_sidebar_state="collapsed")

css_kustom = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');
    
    html, body { 
        font-family: 'Plus Jakarta Sans', sans-serif; 
    }
    
    .stApp {
        background-color: #FAFAFA;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        display: flex;
        justify-content: center;
        gap: 30px;
        border-bottom: 1px solid #E5E7EB;
        padding-bottom: 10px;
        margin-bottom: 2rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: transparent;
        border-radius: 0px;
        padding: 10px 15px;
        color: #6B7280;
        font-weight: 600;
        font-size: 1.1rem;
        border: none !important;
        transition: all 0.3s;
    }
    
    .stTabs [aria-selected="true"] {
        color: #111827 !important;
        border-bottom: 3px solid #111827 !important;
        background-color: transparent !important;
    }

    .metric-card { 
        background-color: #FFFFFF; 
        border: 1px solid #E5E7EB; 
        border-radius: 12px; 
        padding: 32px 24px; 
        box-shadow: 0 1px 3px rgba(0,0,0,0.1); 
        text-align: center; 
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    .metric-card:hover { 
        transform: translateY(-5px); 
        box-shadow: 0 10px 25px -5px rgba(0,0,0,0.1); 
        border-color: #D1D5DB;
    }
    
    .gradient-text { 
        background: linear-gradient(135deg, #111827 0%, #4B5563 100%); 
        -webkit-background-clip: text; 
        -webkit-text-fill-color: transparent; 
        font-weight: 800 !important; 
    }
    
    h1, h2, h3 { color: #111827 !important; font-weight: 700 !important; }
</style>
"""
st.markdown(css_kustom, unsafe_allow_html=True)

@st.cache_data
def process_data(data):
    # Data Pre-processing
    data = data.dropna(subset=['CustomerID'])
    data = data[data['Quantity'] > 0]
    data = data[data['UnitPrice'] > 0]
    data = data.drop_duplicates().reset_index(drop=True)
    
    # Calculate RFM
    data['TotalAmount'] = data['Quantity'] * data['UnitPrice']
    data['InvoiceDate'] = pd.to_datetime(data['InvoiceDate'])
    tanggal_ref = data['InvoiceDate'].max() + pd.Timedelta(days=1)
    
    rfm = data.groupby('CustomerID').agg({
        'InvoiceDate': lambda x: (tanggal_ref - x.max()).days,
        'InvoiceNo': 'nunique',
        'TotalAmount': 'sum'
    })
    rfm.columns = ['Recency', 'Frequency', 'Monetary']
    rfm = rfm.reset_index()
    return data, rfm

@st.cache_resource
def run_kmeans(rfm, k=3):
    fitur = rfm[['Recency', 'Frequency', 'Monetary']].values
    scaler = MinMaxScaler()
    fitur_norm = scaler.fit_transform(fitur)
    
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    label = model.fit_predict(fitur_norm)
    rfm['Cluster'] = label
    
    cluster_monetary = rfm.groupby('Cluster')['Monetary'].mean().sort_values()
    mapping = {cluster_monetary.index[0]: 'Low-Value', cluster_monetary.index[1]: 'Medium-Value', cluster_monetary.index[2]: 'High-Value'}
    rfm['Segment'] = rfm['Cluster'].map(mapping)
    
    return rfm, fitur_norm, model

def render_panduan():
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("## Panduan Struktur Dataset")
    st.markdown("Agar model *Machine Learning* dapat mengekstraksi metrik **RFM (Recency, Frequency, Monetary)** dengan akurat, dataset yang diunggah harus memiliki format dan struktur kolom spesifik sebagai berikut:")
    
    panduan_data = {
        "Nama Kolom": ["InvoiceNo", "StockCode", "Description", "Quantity", "InvoiceDate", "UnitPrice", "CustomerID", "Country"],
        "Tipe Data": ["Kategorikal / String", "Kategorikal / String", "Teks", "Numerik (Integer)", "Tanggal (Datetime)", "Numerik (Float)", "Kategorikal / String", "Kategorikal / String"],
        "Deskripsi / Fungsi": [
            "Nomor unik tagihan. Digunakan untuk menghitung *Frequency* (seberapa sering pelanggan belanja).",
            "Kode unik barang jualan. (Tidak digunakan dalam perhitungan RFM).",
            "Nama barang/deskripsi. (Tidak digunakan dalam perhitungan RFM).",
            "Jumlah barang yang dibeli per transaksi. Digunakan untuk menghitung total pendapatan.",
            "Waktu terjadinya transaksi. Digunakan untuk menghitung *Recency* (hari sejak pembelian terakhir).",
            "Harga satuan barang. Digabungkan dengan Quantity untuk mendapatkan metrik *Monetary*.",
            "ID Unik Pelanggan. Komponen paling krusial untuk mengelompokkan riwayat belanja tiap orang.",
            "Negara asal pembeli. (Opsional untuk analisis geografis)."
        ]
    }
    st.dataframe(pd.DataFrame(panduan_data), use_container_width=True, hide_index=True)
    
    st.markdown("### Catatan Pre-processing Otomatis:")
    st.info("""
    Sistem Enterprise ini sudah dilengkapi algoritma pembersihan data (Data Cleaning) otomatis:
    1. Baris yang tidak memiliki **CustomerID** akan diabaikan (karena tidak bisa dilacak).
    2. Transaksi dengan **Quantity** bernilai negatif (pengembalian barang) atau nol akan dihapus untuk menjaga kemurnian analisis.
    3. Baris yang berduplikasi (*Duplicate records*) akan dihapus secara otomatis.
    """)

def main():
    st.markdown("<h1 class='gradient-text' style='text-align: center; margin-top: 1rem;'>Customer Segmentation Analytics</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; font-size: 1.1rem; color: #6B7280; margin-bottom: 2rem;'>K-Means Clustering on RFM (Recency, Frequency, Monetary) Variables</p>", unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Unggah Dataset (Excel / CSV)", type=['xlsx', 'csv'])
    
    tab_panduan, tab_analitik, tab_3d, tab_rfm = st.tabs(["Panduan Dataset", "Dashboard Bisnis", "3D Cluster Explorer", "Tabel RFM Mentah"])
    
    with tab_panduan:
        render_panduan()
        
    if uploaded_file:
        with st.spinner("Mengolah jutaan probabilitas algoritma K-Means..."):
            if uploaded_file.name.endswith('.csv'):
                data = pd.read_csv(uploaded_file, encoding='latin1')
            else:
                data = pd.read_excel(uploaded_file)
            
            clean_data, rfm = process_data(data)
            rfm_clustered, fitur_norm, model = run_kmeans(rfm, k=3)
        
        with tab_analitik:
            st.markdown("<br>", unsafe_allow_html=True)
            c1, c2, c3 = st.columns(3)
            with c1: 
                st.markdown(f"<div class='metric-card'><h3 style='color: #374151; font-size: 1.2rem;'>Total Pelanggan Aktif</h3><h2 style='margin: 0;'>{len(rfm_clustered):,}</h2></div>", unsafe_allow_html=True)
            with c2: 
                st.markdown(f"<div class='metric-card'><h3 style='color: #374151; font-size: 1.2rem;'>Total Pendapatan (Revenue)</h3><h2 style='margin: 0;'>£ {clean_data['TotalAmount'].sum():,.0f}</h2></div>", unsafe_allow_html=True)
            with c3: 
                st.markdown(f"<div class='metric-card'><h3 style='color: #374151; font-size: 1.2rem;'>Rata-rata Order (AOV)</h3><h2 style='margin: 0;'>£ {clean_data['TotalAmount'].mean():,.2f}</h2></div>", unsafe_allow_html=True)
            
            st.markdown("<hr>", unsafe_allow_html=True)
            
            col_chart1, col_chart2 = st.columns(2)
            with col_chart1:
                st.markdown("### Distribusi Segmen Pelanggan")
                fig = px.pie(rfm_clustered, names='Segment', hole=0.4, color_discrete_sequence=['#10B981', '#3B82F6', '#EF4444'])
                fig.update_layout(margin=dict(t=20, b=20, l=20, r=20))
                st.plotly_chart(fig, use_container_width=True)
                
            with col_chart2:
                st.markdown("### Karakteristik Rata-rata Segmen")
                karakteristik = rfm_clustered.groupby('Segment')[['Recency', 'Frequency', 'Monetary']].mean().reset_index()
                st.dataframe(karakteristik.style.format({
                    'Recency': '{:.1f} hari',
                    'Frequency': '{:.1f} kali',
                    'Monetary': '£ {:.2f}'
                }), use_container_width=True)
            
        with tab_3d:
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("### 3D Interactive Clustering Visualizer")
            st.markdown("Puntir, perbesar, dan geser grafik 3D di bawah ini untuk melihat pemisahan K-Means secara visual dan matematis.")
            
            fig3d = px.scatter_3d(
                rfm_clustered, x='Recency', y='Frequency', z='Monetary',
                color='Segment', size='Monetary', size_max=25,
                opacity=0.8, color_discrete_sequence=['#10B981', '#3B82F6', '#EF4444']
            )
            fig3d.update_layout(
                margin=dict(l=0, r=0, b=0, t=20),
                scene=dict(
                    xaxis_title='Recency (Days)',
                    yaxis_title='Frequency (Orders)',
                    zaxis_title='Monetary (£)'
                )
            )
            st.plotly_chart(fig3d, use_container_width=True)
            
        with tab_rfm:
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("### Tabel Data Hasil Analisis RFM")
            st.markdown("Berikut adalah tabel mentah untuk evaluasi individual pelanggan.")
            st.dataframe(rfm_clustered, use_container_width=True)
            
            csv_hasil = rfm_clustered.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Unduh Rekapitulasi CSV",
                data=csv_hasil,
                file_name='KMeans_Customer_Segments.csv',
                mime='text/csv',
            )
    else:
        with tab_analitik:
            st.info("Silakan unggah dataset Anda melalui kotak unggahan di atas untuk melihat analisis pada tab ini.")
        with tab_3d:
            st.info("Visualisasi 3D akan muncul setelah Anda mengunggah dataset melalui kotak unggahan di atas.")
        with tab_rfm:
            st.info("Tabel Data RFM akan diproses setelah Anda mengunggah dataset melalui kotak unggahan di atas.")

if __name__ == '__main__':
    main()
