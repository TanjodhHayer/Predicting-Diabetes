import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def perform_eda(data):
    # Create a folder for saving plots if it doesn't exist
    complete_data = data
    output_dir = "eda_plots"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 1. Summary Statistics: Save as a Table Plot
    summary_stats = data.describe().transpose()
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis('tight')
    ax.axis('off')
    table = ax.table(
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
    summary_stats.to_csv(os.path.join(output_dir, "eda_summary_statistics.csv"))

    # Select key features
    key_features = ['GenHlth', 'MentHlth', 'Income', 'PhysHlth', 'Education', 'BMI', 'Age', 'HighBP']
    data = data[key_features + ['Diabetes_012']]  # Add target variable for contextual analysis

    # 2. Histograms for Key Features
    # Combined Histograms for Key Features
    n_features = len(key_features)
    rows = (n_features + 2) // 3  # Adjust number of rows for 3 columns per row
    fig, axes = plt.subplots(rows, 3, figsize=(18, 5 * rows))  # Adjust figure size as needed
    axes = axes.flatten()  # Flatten the axes array for easy iteration

    for i, col in enumerate(key_features):
        sns.histplot(data[col], kde=True, ax=axes[i])  # Add histogram to the subplot
        axes[i].set_title(f"Distribution of {col}")
        axes[i].set_xlabel(col)
        axes[i].set_ylabel("Frequency")

    # Hide any unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')  # Turn off empty axes

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "Key_Features_Histograms.png"))
    plt.close()

    # 3. Correlation Analysis: Heatmap
    correlation_matrix = complete_data.corr()
    plt.figure(figsize=(12, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "Correlation_Heatmap.png"))
    plt.clf()
