import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score, silhouette_samples
from mpl_toolkits.mplot3d import Axes3D

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("="*60)
print("LOADING DATASET")
print("="*60)

path = r"C:\Users\Lenovo\Downloads\online+retail\Online Retail.xlsx"
data = pd.read_excel(path, sheet_name='Online Retail')

print(f"\nDataset loaded successfully!")
print(f"Shape: {data.shape}")
print(f"Records: {data.shape[0]:,}")
print(f"Columns: {data.shape[1]}")

print("\n" + "="*60)
print("DATA UNDERSTANDING")
print("="*60)

print("\nInfo:")
print(data.info())

print("\nFirst 5 rows:")
print(data.head())

print("\nStatistics:")
print(data.describe())

print("\nMissing values:")
missing = data.isnull().sum()
pct_missing = (data.isnull().sum() / len(data)) * 100
tabel_missing = pd.DataFrame({
    'Column': missing.index,
    'Missing_Count': missing.values,
    'Missing_Percent': pct_missing.values
})
print(tabel_missing[tabel_missing['Missing_Count'] > 0])

print("\n" + "="*60)
print("DATA PRE-PROCESSING")
print("="*60)

total_awal = data.shape[0]

print("\nStep 1: Remove missing CustomerID")
data = data.dropna(subset=['CustomerID'])
print(f"Records after: {data.shape[0]:,}")

print("\nStep 2: Remove negative Quantity")
data = data[data['Quantity'] > 0]
print(f"Records after: {data.shape[0]:,}")

print("\nStep 3: Remove zero/negative UnitPrice")
data = data[data['UnitPrice'] > 0]
print(f"Records after: {data.shape[0]:,}")

print("\nStep 4: Remove duplicates")
data = data.drop_duplicates()
print(f"Records after: {data.shape[0]:,}")

data = data.reset_index(drop=True)
print(f"\nFinal shape: {data.shape}")

print("\n" + "="*60)
print("EXPLORATORY DATA ANALYSIS")
print("="*60)

print("\nTop 10 Countries:")
print(data['Country'].value_counts().head(10))

fig, ax = plt.subplots(2, 2, figsize=(15, 10))

ax[0, 0].hist(data['Quantity'], bins=50, edgecolor='black', alpha=0.7)
ax[0, 0].set_xlabel('Quantity')
ax[0, 0].set_ylabel('Frequency')
ax[0, 0].set_title('Distribution of Quantity')
ax[0, 0].set_xlim(0, data['Quantity'].quantile(0.95))

ax[0, 1].hist(data['UnitPrice'], bins=50, edgecolor='black', alpha=0.7, color='orange')
ax[0, 1].set_xlabel('Unit Price (GBP)')
ax[0, 1].set_ylabel('Frequency')
ax[0, 1].set_title('Distribution of Unit Price')
ax[0, 1].set_xlim(0, data['UnitPrice'].quantile(0.95))

data['Country'].value_counts().head(10).plot(kind='barh', ax=ax[1, 0], color='green', alpha=0.7)
ax[1, 0].set_xlabel('Number of Transactions')
ax[1, 0].set_title('Top 10 Countries')

data['InvoiceDate'] = pd.to_datetime(data['InvoiceDate'])
transaksi_bulanan = data.groupby(data['InvoiceDate'].dt.to_period('M')).size()
transaksi_bulanan.plot(ax=ax[1, 1], color='red', marker='o')
ax[1, 1].set_xlabel('Month')
ax[1, 1].set_ylabel('Number of Transactions')
ax[1, 1].set_title('Transactions Over Time')
ax[1, 1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('eda_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n" + "="*60)
print("RFM ANALYSIS")
print("="*60)

data['TotalAmount'] = data['Quantity'] * data['UnitPrice']
tanggal_ref = data['InvoiceDate'].max() + pd.Timedelta(days=1)

rfm = data.groupby('CustomerID').agg({
    'InvoiceDate': lambda x: (tanggal_ref - x.max()).days,
    'InvoiceNo': 'nunique',
    'TotalAmount': 'sum'
})

rfm.columns = ['Recency', 'Frequency', 'Monetary']
rfm = rfm.reset_index()

print(f"\nRFM calculated for {len(rfm):,} customers")
print("\nSample RFM:")
print(rfm.head(10))

print("\nRFM Statistics:")
print(rfm[['Recency', 'Frequency', 'Monetary']].describe())

rfm['R_Score'] = pd.qcut(rfm['Recency'], q=5, labels=[5, 4, 3, 2, 1], duplicates='drop')
rfm['F_Score'] = pd.qcut(rfm['Frequency'].rank(method='first'), q=5, labels=[1, 2, 3, 4, 5], duplicates='drop')
rfm['M_Score'] = pd.qcut(rfm['Monetary'].rank(method='first'), q=5, labels=[1, 2, 3, 4, 5], duplicates='drop')

rfm['R_Score'] = rfm['R_Score'].astype(int)
rfm['F_Score'] = rfm['F_Score'].astype(int)
rfm['M_Score'] = rfm['M_Score'].astype(int)

rfm['RFM_Score'] = (rfm['R_Score'] + rfm['F_Score'] + rfm['M_Score']) / 3

def segmentasi(score):
    if score > 4.5:
        return 'Top Customer'
    elif score > 4.0:
        return 'High-Value Customer'
    elif score > 3.0:
        return 'Medium-Value Customer'
    elif score > 1.6:
        return 'Low-Value Customer'
    else:
        return 'Lost Customer'

rfm['Customer_Segment'] = rfm['RFM_Score'].apply(segmentasi)

print("\nCustomer Segments:")
print(rfm['Customer_Segment'].value_counts())

fig, ax = plt.subplots(2, 2, figsize=(15, 10))

ax[0, 0].hist(rfm['Recency'], bins=50, edgecolor='black', alpha=0.7, color='skyblue')
ax[0, 0].set_xlabel('Recency (days)')
ax[0, 0].set_title('Distribution of Recency')

ax[0, 1].hist(rfm['Frequency'], bins=50, edgecolor='black', alpha=0.7, color='lightgreen')
ax[0, 1].set_xlabel('Frequency')
ax[0, 1].set_title('Distribution of Frequency')
ax[0, 1].set_xlim(0, rfm['Frequency'].quantile(0.95))

ax[1, 0].hist(rfm['Monetary'], bins=50, edgecolor='black', alpha=0.7, color='lightcoral')
ax[1, 0].set_xlabel('Monetary (GBP)')
ax[1, 0].set_title('Distribution of Monetary')
ax[1, 0].set_xlim(0, rfm['Monetary'].quantile(0.95))

ax[1, 1].hist(rfm['RFM_Score'], bins=30, edgecolor='black', alpha=0.7, color='gold')
ax[1, 1].set_xlabel('RFM Score')
ax[1, 1].set_title('Distribution of RFM Score')

plt.tight_layout()
plt.savefig('rfm_distributions.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n" + "="*60)
print("PREPARE DATA FOR CLUSTERING")
print("="*60)

fitur = rfm[['Recency', 'Frequency', 'Monetary']].values
print(f"Shape: {fitur.shape}")

scaler = MinMaxScaler()
fitur_norm = scaler.fit_transform(fitur)
print("Data normalized [0, 1]")

print("\n" + "="*60)
print("ELBOW METHOD")
print("="*60)

wcss = []
nilai_k = range(2, 11)

for k in nilai_k:
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    model.fit(fitur_norm)
    wcss.append(model.inertia_)
    print(f"K={k}: WCSS={model.inertia_:,.2f}")

plt.figure(figsize=(12, 7))
plt.plot(nilai_k, wcss, 'bo-', linewidth=2.5, markersize=10)

for k, val in zip(nilai_k, wcss):
    plt.annotate(f'{val:,.0f}', xy=(k, val), xytext=(0, 10), 
                textcoords='offset points', ha='center', fontsize=9)

plt.xlabel('Number of Clusters (K)', fontsize=13, fontweight='bold')
plt.ylabel('WCSS', fontsize=13, fontweight='bold')
plt.title('Elbow Method', fontsize=15, fontweight='bold')
plt.grid(True, alpha=0.4, linestyle='--')
plt.xticks(nilai_k)
plt.axvline(x=3, color='red', linestyle='--', linewidth=2, label='Optimal K=3')
plt.legend(fontsize=11)
plt.tight_layout()
plt.savefig('elbow_method.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n" + "="*60)
print("SILHOUETTE SCORE ANALYSIS")
print("="*60)

skor_silhouette = []

for k in nilai_k:
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    prediksi = model.fit_predict(fitur_norm)
    score = silhouette_score(fitur_norm, prediksi)
    skor_silhouette.append(score)
    print(f"K={k}: Silhouette={score:.4f}")

plt.figure(figsize=(12, 7))
plt.plot(nilai_k, skor_silhouette, 'go-', linewidth=2.5, markersize=10)

for k, score in zip(nilai_k, skor_silhouette):
    plt.annotate(f'{score:.4f}', xy=(k, score), xytext=(0, -15), 
                textcoords='offset points', ha='center', fontsize=9)

plt.xlabel('Number of Clusters (K)', fontsize=13, fontweight='bold')
plt.ylabel('Silhouette Score', fontsize=13, fontweight='bold')
plt.title('Silhouette Score Analysis', fontsize=15, fontweight='bold')
plt.grid(True, alpha=0.4, linestyle='--')
plt.xticks(nilai_k)

k_terbaik = nilai_k[skor_silhouette.index(max(skor_silhouette))]
plt.axvline(x=k_terbaik, color='red', linestyle='--', linewidth=2, 
           label=f'Best K={k_terbaik}')
plt.legend(fontsize=11)
plt.tight_layout()
plt.savefig('silhouette_scores.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n" + "="*60)
print("K-MEANS CLUSTERING (K=3)")
print("="*60)

jumlah_cluster = 3
model_akhir = KMeans(n_clusters=jumlah_cluster, random_state=42, n_init=10, max_iter=300)
label = model_akhir.fit_predict(fitur_norm)

rfm['Cluster'] = label

skor_sil = silhouette_score(fitur_norm, label)
skor_db = davies_bouldin_score(fitur_norm, label)

print(f"\nSilhouette Score: {skor_sil:.4f}")
print(f"Davies-Bouldin Index: {skor_db:.4f}")
print(f"Inertia: {model_akhir.inertia_:,.2f}")

print("\nCluster Distribution:")
print(rfm['Cluster'].value_counts().sort_index())

print("\nCluster Characteristics:")
ringkasan = rfm.groupby('Cluster').agg({
    'Recency': 'mean',
    'Frequency': 'mean',
    'Monetary': 'mean',
    'CustomerID': 'count'
}).round(2)
print(ringkasan)

pca = PCA(n_components=2, random_state=42)
fitur_2d = pca.fit_transform(fitur_norm)

fig, ax = plt.subplots(1, 2, figsize=(16, 6))

scatter1 = ax[0].scatter(fitur_2d[:, 0], fitur_2d[:, 1], c=label, 
                         cmap='viridis', alpha=0.6, edgecolors='black', linewidth=0.5, s=50)
centroid_2d = pca.transform(model_akhir.cluster_centers_)
ax[0].scatter(centroid_2d[:, 0], centroid_2d[:, 1], c='red', marker='X', 
               s=300, edgecolors='black', linewidth=2, label='Centroids')
ax[0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})', fontsize=11)
ax[0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})', fontsize=11)
ax[0].set_title(f'K-Means Clustering (K={jumlah_cluster})', fontsize=13, fontweight='bold')
ax[0].legend()
ax[0].grid(True, alpha=0.3)
plt.colorbar(scatter1, ax=ax[0], label='Cluster')

ax3d = fig.add_subplot(122, projection='3d')
scatter2 = ax3d.scatter(fitur_norm[:, 0], fitur_norm[:, 1], fitur_norm[:, 2],
                       c=label, cmap='viridis', alpha=0.6, 
                       edgecolors='black', linewidth=0.5, s=30)
ax3d.scatter(model_akhir.cluster_centers_[:, 0], model_akhir.cluster_centers_[:, 1], 
          model_akhir.cluster_centers_[:, 2], c='red', marker='X', 
          s=300, edgecolors='black', linewidth=2)
ax3d.set_xlabel('Recency', fontsize=10)
ax3d.set_ylabel('Frequency', fontsize=10)
ax3d.set_zlabel('Monetary', fontsize=10)
ax3d.set_title('K-Means (3D)', fontsize=13, fontweight='bold')
plt.colorbar(scatter2, ax=ax3d, label='Cluster', shrink=0.5)

plt.tight_layout()
plt.savefig('kmeans_clustering.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n" + "="*60)
print("SILHOUETTE ANALYSIS (K=3)")
print("="*60)

nilai_sil = silhouette_samples(fitur_norm, label)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

posisi_y = 10
warna = ['#FF6B6B', '#4ECDC4', '#45B7D1']

for i in range(jumlah_cluster):
    sil_cluster = nilai_sil[label == i]
    sil_cluster.sort()
    
    ukuran = sil_cluster.shape[0]
    akhir_y = posisi_y + ukuran
    
    ax1.fill_betweenx(np.arange(posisi_y, akhir_y), 0, sil_cluster,
                      facecolor=warna[i], edgecolor=warna[i], alpha=0.7)
    
    ax1.text(-0.05, posisi_y + 0.5 * ukuran, f'Cluster {i}', fontsize=11, fontweight='bold')
    posisi_y = akhir_y + 10

ax1.set_xlabel('Silhouette Coefficient', fontsize=12, fontweight='bold')
ax1.set_ylabel('Cluster Label', fontsize=12, fontweight='bold')
ax1.set_title(f'Silhouette Plot (K={jumlah_cluster})\nAvg: {skor_sil:.4f}', 
             fontsize=13, fontweight='bold')
ax1.axvline(x=skor_sil, color="red", linestyle="--", linewidth=2)
ax1.set_yticks([])
ax1.set_xlim([-0.1, 1])
ax1.grid(True, alpha=0.3, axis='x')

for i in range(jumlah_cluster):
    ax2.scatter(fitur_2d[label == i, 0], fitur_2d[label == i, 1],
               s=50, c=warna[i], label=f'Cluster {i}', alpha=0.6, 
               edgecolors='black', linewidth=0.5)

ax2.scatter(centroid_2d[:, 0], centroid_2d[:, 1], s=300, c='red', marker='X', 
           edgecolors='black', linewidth=2, label='Centroids', zorder=10)
ax2.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})', fontsize=12, fontweight='bold')
ax2.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})', fontsize=12, fontweight='bold')
ax2.set_title('Clustering Visualization', fontsize=13, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('silhouette_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n" + "="*60)
print("EXPORT RESULTS")
print("="*60)

kolom_export = ['CustomerID', 'Recency', 'Frequency', 'Monetary', 
               'R_Score', 'F_Score', 'M_Score', 'RFM_Score', 
               'Customer_Segment', 'Cluster']
rfm[kolom_export].to_csv('rfm_clusters.csv', index=False)

hasil = []
for k, wcss_val, sil in zip(nilai_k, wcss, skor_silhouette):
    hasil.append({
        'K': k,
        'WCSS': f'{wcss_val:,.2f}',
        'Silhouette': f'{sil:.4f}'
    })
pd.DataFrame(hasil).to_csv('clustering_summary.csv', index=False)

print("\nFiles saved:")
print("1. eda_analysis.png")
print("2. rfm_distributions.png")
print("3. elbow_method.png")
print("4. silhouette_scores.png")
print("5. kmeans_clustering.png")
print("6. silhouette_analysis.png")
print("7. rfm_clusters.csv")
print("8. clustering_summary.csv")
print("\nAnalysis completed!")
