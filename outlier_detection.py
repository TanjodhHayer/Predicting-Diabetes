import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import EllipticEnvelope

"""
References for code used in this file:
https://scikit-learn.org/dev/modules/generated/sklearn.neighbors.LocalOutlierFactor.html
https://scikit-learn.org/stable/auto_examples/ensemble/plot_isolation_forest.html#sphx-glr-auto-examples-ensemble-plot-isolation-forest-py
https://github.com/spribylova/Python-Elliptic-Data/blob/main/elliptic_envelope.ipynb?source=post_page-----673a39e3b315--------------------------------
"""

def detect_outliers_lof(data, numerical_columns, n_neighbors=20, contamination=0.02):
    """
    Detect outliers in the given data using the Local Outlier Factor (LOF) algorithm.
    @param data - The dataset containing the numerical columns.
    @param numerical_columns - The numerical columns in the dataset.
    @param n_neighbors - The number of neighbors to consider (default is 20).
    @param contamination - The proportion of outliers in the dataset (default is 0.02).
    @return A boolean array indicating whether each data point is an outlier or not.
    """
    lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination)
    lof_flags = lof.fit_predict(data[numerical_columns])
    return lof_flags == -1  

def detect_outliers_isoforest(data, numerical_columns, contamination=0.02, random_state=42):
    """
    Detect outliers in the given data using Isolation Forest algorithm.
    @param data - The dataset containing the numerical columns.
    @param numerical_columns - The numerical columns in the dataset.
    @param contamination - The proportion of outliers in the data (default is 0.02).
    @param random_state - The random seed for reproducibility (default is 42).
    @return A boolean array indicating whether each data point is an outlier.
    """
    iso_forest = IsolationForest(contamination=contamination, random_state=random_state)
    iso_forest_flags = iso_forest.fit_predict(data[numerical_columns])
    return iso_forest_flags == -1

def detect_outliers_elliptic(data, numerical_columns, contamination=0.02, random_state=42):
    """
    Detect outliers in the data using the Elliptic Envelope method.
    @param data - The dataset containing the numerical columns.
    @param numerical_columns - The numerical columns in the dataset.
    @param contamination - The proportion of outliers to expect in the data.
    @param random_state - The random seed for reproducibility.
    @return A boolean array indicating whether each data point is an outlier.
    """
    elliptic = EllipticEnvelope(contamination=contamination, random_state=random_state)
    elliptic_flags = elliptic.fit_predict(data[numerical_columns])
    return elliptic_flags == -1

def plot_outliers(data, outliers, numerical_columns, method_name="Outlier Detection"):
    """
    Plot outliers in the data for each numerical column.
    @param data - The dataset containing the numerical columns
    @param outliers - A boolean array indicating whether each data point is an outlier
    @param numerical_columns - List of numerical columns to plot
    @param method_name - Name of the outlier detection method used (default is "Outlier Detection")
    """
    num_features = len(numerical_columns)
    rows = int(np.ceil(num_features / 2))
    cols = 2

    plt.figure(figsize=(15, 8))
    
    for i, feature in enumerate(numerical_columns):
        plt.subplot(rows, cols, i + 1)
        plt.scatter(
            data[feature],
            np.random.rand(len(data[feature])) * 0.1,
            c=np.where(outliers, 'red', 'blue'),
            alpha=0.6, 
            edgecolors='k', 
            label='Outliers/Inliers'
        )
        
        plt.title(f"{feature} - {method_name}")
        plt.xlabel(f"{feature} Values")
        plt.yticks([])
        plt.ylabel("")
    
    plt.tight_layout()
    plt.suptitle(f"{method_name}", fontsize=16)
    plt.subplots_adjust(top=0.85)
    plt.savefig(f"{method_name.replace(' ', '_')}_Plot.png")

def run_outlier_detection(data, numerical_columns):
    """
    Run outlier detection on the given data using Local Outlier Factor (LOF), Isolation Forest (IF), and Elliptic Envelope (EE) algorithms.
    @param data - The dataset to detect outliers in
    @param numerical_columns - The numerical columns in the dataset
    @return A tuple containing the outliers detected by LOF, IF, and EE algorithms
    """
    # Local Outlier Factor
    lof_outliers = detect_outliers_lof(data, numerical_columns)
    plot_outliers(data, lof_outliers, numerical_columns, "Local Outlier Factor (LOF)")
    
    # Isolation Forest
    iso_outliers = detect_outliers_isoforest(data, numerical_columns)
    plot_outliers(data, iso_outliers, numerical_columns, "Isolation Forest (IF)")
    
    # Elliptic Envelope
    ee_outliers = detect_outliers_elliptic(data, numerical_columns)
    plot_outliers(data, ee_outliers, numerical_columns, "Elliptic Envelope (EE)")

    return lof_outliers, iso_outliers, ee_outliers