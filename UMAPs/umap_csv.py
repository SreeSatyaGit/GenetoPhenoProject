import scanpy as sc
import numpy as np
import pandas as pd
import leidenalg
import umap
import umap.plot
import sklearn.datasets
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def umapCsv(csvData):
    mapper = umap.UMAP(metric='correlation', random_state=42, low_memory=True).fit_transform(csvData)
    y = ['CD+4 T','CD8+ T','B','MO','DC','NE','EO','BA']
    kmeans = KMeans(n_clusters=8)
    clusters = kmeans.fit_predict(mapper)
    clusters_no_zeros = clusters
    plt.scatter(mapper[:, 0], mapper[:, 1], c = clusters_no_zeros, cmap='viridis')
    cbar = plt.colorbar(boundaries=np.arange(9)-0.5)
    cbar.set_ticks(np.arange(8))
    cbar.set_ticklabels(y)
    plt.show()
    data_with_labels = np.concatenate((csvData, clusters_no_zeros[:, np.newaxis]),axis=1)
    df = pd.DataFrame(data=data_with_labels)
    column_names = ['CD3','CD4','CD8','CD2','CD45RA','CD57','CD16','CD14','CD11c','CD19','clusters']
    df.columns = column_names
    return df