import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import linkage, fcluster

"""
References for code used in this file:
https://www.w3schools.com/python/python_ml_k-means.asp
https://www.youtube.com/watch?v=pDaDbobxIkA&ab_channel=DataScienceTeacherBrandyn
https://www.kdnuggets.com/2022/08/implementing-dbscan-python.html
https://www.geeksforgeeks.org/dbscan-clustering-in-ml-density-based-clustering/
https://www.youtube.com/watch?v=-EFB6icKr7E&ab_channel=MazenAhmed
https://stackoverflow.com/questions/54705389/how-to-define-a-range-of-values-for-the-eps-parameter-of-sklearn-cluster-dbscan
https://www.w3schools.com/python/python_ml_hierarchial_clustering.asp
https://www.geeksforgeeks.org/scipy-cluster-hierarchy-dendrogram/
https://joernhees.de/blog/2015/08/26/scipy-hierarchical-clustering-and-dendrogram-tutorial/
https://plotly.com/python/pca-visualization/
"""

RANGE = range(2, 21)

def apply_clustering(data, target_column="Diabetes_012"):
    """
    Apply clustering algorithms (KMeans, DBSCAN, Hierarchical Clustering) to the data and evaluate their performance using silhouette scores. 
    @param data - the dataset
    @param target_column - the target column to drop from the data
    @return None
    """
    X = data.drop(target_column, axis=1) 

    kmeans_scores = []
    dbscan_scores = []
    hierarchical_scores = []
    
    # KMeans Clustering
    kmeans_labels = None
    for k in RANGE:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans_labels = kmeans.fit_predict(X)
        score = silhouette_score(X, kmeans_labels)
        kmeans_scores.append(score)
    best_kmeans_score = np.max(kmeans_scores)
    print(f"Best Silhouette Score for KMeans: {best_kmeans_score:.4f}")

    # DBSCAN Clustering
    dbscan_labels = None
    eps_range = np.linspace(0.5, 5, 25)
    
    for eps in eps_range:
        dbscan = DBSCAN(eps=eps, min_samples=9)
        dbscan_labels = dbscan.fit_predict(X)
        
        if len(set(dbscan_labels)) > 1:
            score = silhouette_score(X, dbscan_labels)
            dbscan_scores.append(score)
        else:
            dbscan_scores.append(-1)
    best_dbscan_score = np.max(dbscan_scores)
    print(f"Best Silhouette Score for DBSCAN: {best_dbscan_score:.4f}")
    
    # Hierarchical Clustering
    hierarchical_labels = None
    linkage_with_euclidean = linkage(X, method='ward', metric='euclidean')
    
    for i in RANGE:
        hierarchical_labels = fcluster(linkage_with_euclidean, i, criterion='maxclust')
        score = silhouette_score(X, hierarchical_labels)
        hierarchical_scores.append(score)
        
    top_score = np.max(hierarchical_scores)
    print(f"Best Silhouette Score for Hierarchical Clustering: {top_score:.4f}")

    plot_combined_silhouette_scores(kmeans_scores, dbscan_scores, hierarchical_scores, eps_range)
    plot_clustering_results(X, kmeans_labels, dbscan_labels, hierarchical_labels)


def plot_combined_silhouette_scores(kmeans_scores, dbscan_scores, hierarchical_scores, eps_range):
    """
    Plot the silhouette scores for KMeans, DBSCAN, and Hierarchical Clustering algorithms.
    @param kmeans_scores - Silhouette scores for KMeans algorithm
    @param dbscan_scores - Silhouette scores for DBSCAN algorithm
    @param hierarchical_scores - Silhouette scores for Hierarchical Clustering algorithm
    @param eps_range - Range of epsilon values for DBSCAN algorithm
    """
    _, silhouette_plot = plt.subplots(1, 3, figsize=(18, 6))

    # KMeans Silhouette Scores
    silhouette_plot[0].plot(RANGE, kmeans_scores, 'o-', label="Silhouette Score", color='orange')
    silhouette_plot[0].set_title("KMeans Silhouette Scores")
    silhouette_plot[0].set_ylabel("Silhouette Score")
    silhouette_plot[0].set_xlabel("Number of Clusters")
    silhouette_plot[0].grid(True)

    # DBSCAN Silhouette Scores
    silhouette_plot[1].plot(eps_range, dbscan_scores, 'o-', label="Silhouette Score", color='green')
    silhouette_plot[1].set_title("DBSCAN Silhouette Scores")
    silhouette_plot[1].set_ylabel("Silhouette Score")
    silhouette_plot[1].set_xlabel("Epsilon (eps)")
    silhouette_plot[1].grid(True)

    # Hierarchical Clustering Silhouette Scores
    silhouette_plot[2].plot(RANGE, hierarchical_scores, 'o-', label="Silhouette Score", color='purple')
    silhouette_plot[2].set_title("Hierarchical Clustering Silhouette Scores")
    silhouette_plot[2].set_ylabel("Silhouette Score")
    silhouette_plot[2].set_xlabel("Number of Clusters")
    silhouette_plot[2].grid(True)

    plt.tight_layout()
    plt.savefig("Silhouette_Scores.png")


def plot_clustering_results(X, kmeans_labels, dbscan_labels, hierarchical_labels):
    """
    Plot the clustering results using PCA for visualization.
    @param X - The data points
    @param kmeans_labels - Labels assigned by KMeans clustering
    @param dbscan_labels - Labels assigned by DBSCAN clustering
    @param hierarchical_labels - Labels assigned by Hierarchical clustering
    """
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    
    # Create a plot with 3 subplots for each clustering algo 
    _, cluster_plot = plt.subplots(1, 3, figsize=(18, 6))

    # KMeans clustering visualization
    cluster_plot[0].scatter(X_pca[:, 0], X_pca[:, 1], c=kmeans_labels, cmap='viridis', marker='o')
    cluster_plot[0].set_title("KMeans Clustering (PCA)")
    cluster_plot[0].set_ylabel("PCA Component 2")
    cluster_plot[0].set_xlabel("PCA Component 1")

    # DBSCAN clustering visualization
    cluster_plot[1].scatter(X_pca[:, 0], X_pca[:, 1], c=dbscan_labels, cmap='viridis', marker='o')
    cluster_plot[1].set_title("DBSCAN Clustering (PCA)")
    cluster_plot[1].set_ylabel("PCA Component 2")
    cluster_plot[1].set_xlabel("PCA Component 1")

    # Hierarchical clustering visualization
    cluster_plot[2].scatter(X_pca[:, 0], X_pca[:, 1], c=hierarchical_labels, cmap='viridis', marker='o')
    cluster_plot[2].set_title("Hierarchical Clustering (PCA)")
    cluster_plot[2].set_ylabel("PCA Component 2")
    cluster_plot[2].set_xlabel("PCA Component 1")
    
    plt.tight_layout()
    plt.savefig("Clustering_Results_PCA.png")