# Customer Segmentation dengan K-Means Clustering

Analisis segmentasi pelanggan menggunakan metode RFM (Recency, Frequency, Monetary) dan algoritma K-Means Clustering pada dataset Online Retail.

## Dataset

Dataset: [Online Retail Dataset](https://archive.ics.uci.edu/ml/datasets/Online+Retail) dari UCI Machine Learning Repository

**Deskripsi:**
- Total records: 541,909 transaksi
- Periode: 01/12/2010 - 09/12/2011
- Fitur: InvoiceNo, StockCode, Description, Quantity, InvoiceDate, UnitPrice, CustomerID, Country

## Fitur Utama

- **Data Preprocessing**: Cleaning data (missing values, duplicates, negative values)
- **RFM Analysis**: Analisis Recency, Frequency, dan Monetary pelanggan
- **Elbow Method**: Menentukan jumlah cluster optimal
- **Silhouette Analysis**: Evaluasi kualitas clustering
- **K-Means Clustering**: Segmentasi pelanggan menjadi 3 cluster
- **Visualisasi**: 2D dan 3D clustering visualization

## Requirements

```
pandas
numpy
matplotlib
seaborn
scikit-learn
openpyxl
```

Install dependencies:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn openpyxl
```

## Output

**File Visualisasi:**
- `eda_analysis.png` - Exploratory Data Analysis
- `rfm_distributions.png` - Distribusi RFM
- `elbow_method.png` - Elbow Method untuk optimal K
- `silhouette_scores.png` - Silhouette Score analysis
- `kmeans_clustering.png` - 2D dan 3D clustering visualization
- `silhouette_analysis.png` - Detailed silhouette plot

**File Data:**
- `rfm_clusters.csv` - Data RFM dengan label cluster
- `clustering_summary.csv` - Summary evaluasi clustering

## Hasil Analisis

Optimal K = **3 cluster** berdasarkan:
- Elbow Method
- Silhouette Score Analysis

**Karakteristik Cluster:**
- Cluster 0: High-Value Active Customers
- Cluster 1: Low-Value Inactive Customers  
- Cluster 2: Medium-Value Customers

## Metrik Evaluasi

- **Silhouette Score**: Mengukur seberapa baik setiap data point cocok dengan clusternya
- **Davies-Bouldin Index**: Mengukur average similarity antara clusters
- **WCSS**: Within-Cluster Sum of Squares untuk Elbow Method

## Author

Luthfi Muthathohirin

## Referensi

- UCI Machine Learning Repository
- Scikit-learn Documentation
- RFM Analysis Methodology
