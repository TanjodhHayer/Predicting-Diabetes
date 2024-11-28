import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

"""
References for code used in this file:
https://stackoverflow.com/questions/25896964/centered-text-in-matplotlib-tables
https://www.datacamp.com/tutorial/how-to-make-a-seaborn-histogram
https://www.kaggle.com/code/saduman/eda-and-data-visualization-with-seaborn
https://stackoverflow.com/questions/43363389/share-axis-and-remove-unused-in-matplotlib-subplots
"""

def perform_eda(data):
    """
    Perform Exploratory Data Analysis (EDA) on the given dataset.
    @param data - The dataset to perform EDA on
    @return None
    """
    # We are going to make a directory within this repo to store all the plots that will be made
    complete_data = data
    output_dir = "eda_plots"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Plotting the summary statistics 
    summary_stats = data.describe().transpose()
    _, plot = plt.subplots(figsize=(12, 6))
    plot.axis('tight')
    plot.axis('off')
    
    # Centering the data 
    table = plot.table(
        cellText=summary_stats.round(2).values,
        colLabels=summary_stats.columns,
        rowLabels=summary_stats.index,
        loc='center',
        cellLoc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.auto_set_column_width(col=list(range(len(summary_stats.columns))))
    
    plt.title("Summary Statistics", fontsize=16)
    plt.savefig(os.path.join(output_dir, "eda_summary_statistics.png"), format="png", bbox_inches='tight', dpi=300)
    plt.close()
    
    # This is the save to the new directory containing the plots
    summary_stats.to_csv(os.path.join(output_dir, "eda_summary_statistics.csv"))

    # Select the key features that we use in our data
    key_features = ['GenHlth', 'MentHlth', 'Income', 'PhysHlth', 'Education', 'BMI', 'Age', 'HighBP']
    
    # Add target variable for contextual analysis
    data = data[key_features + ['Diabetes_012']]  

    # Histograms for our key features
    n_features = len(key_features)
    rows = (n_features + 2) // 3  
    _, axes = plt.subplots(rows, 3, figsize=(18, 5 * rows)) 
    axes = axes.flatten()  

    for i, col in enumerate(key_features):
        # Add histogram to the subplot
        sns.histplot(data[col], kde=True, ax=axes[i])  
        axes[i].set_title(f"Distribution of {col}")
        axes[i].set_xlabel(col)
        axes[i].set_ylabel("Frequency")

    # Dont include any subplots that were not used 
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')  

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "Key_Features_Histograms.png"))
    plt.close()

    # Creating the heatmap 
    correlation_matrix = complete_data.corr()
    plt.figure(figsize=(12, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "Correlation_Heatmap.png"))
    plt.clf()
