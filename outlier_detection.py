import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import EllipticEnvelope

def detect_outliers_lof(data, numerical_columns, n_neighbors=20, contamination=0.01):
    lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination)
    lof_flags = lof.fit_predict(data[numerical_columns])
    return lof_flags == -1  

def detect_outliers_isoforest(data, numerical_columns, contamination=0.01, random_state=42):
    iso_forest = IsolationForest(contamination=contamination, random_state=random_state)
    iso_forest_flags = iso_forest.fit_predict(data[numerical_columns])
    return iso_forest_flags == -1

def plot_outliers(data, outliers, numerical_columns, method_name="Outlier Detection"):
    """
    Generalized plot function for visualizing outliers detected by different methods.
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
            alpha=0.6, edgecolors='k', label='Outliers/Inliers'
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