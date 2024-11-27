import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster

def apply_clustering(data, target_column="Diabetes_012"):
    X = data.drop(target_column, axis=1)


    dbscan_scores = []

    # DBSCAN Clustering
    eps_range = np.linspace(0.5, 5, 25)
    dbscan_labels = None
    for eps in eps_range:
        dbscan = DBSCAN(eps=eps, min_samples=7)
        dbscan_labels = dbscan.fit_predict(X)
        if len(set(dbscan_labels)) > 1:
            score = silhouette_score(X, dbscan_labels)
            dbscan_scores.append(score)
        else:
            dbscan_scores.append(-1)
    optimal_eps = eps_range[np.argmax(dbscan_scores)]
    best_dbscan_score = np.max(dbscan_scores)
    print(f"Best Silhouette Score for DBSCAN: {best_dbscan_score:.4f}")


