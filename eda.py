import os
import matplotlib.pyplot as plt
import seaborn as sns

def perform_eda(data):

    # Create a folder for saving plots if it doesn't exist
    output_dir = "eda_plots"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 1. Basic Overview of the Data
    _, ax = plt.subplots(figsize=(12, 6))  # Adjust the figure size if necessary
    ax.axis('off')  # Hide axes
    plt.title("Summary Statistics", fontsize=16)
    plt.savefig(os.path.join(output_dir, "eda_summary_statistics.png"), format="png", bbox_inches='tight', dpi=300)
    
    # 2. Feature Distribution: Histograms in Subplots
    numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns
    column_num = len(numeric_columns)
    sub_plt = 4  # Number of subplots per row
    rows = -(-column_num // sub_plt)  # Ceiling division for rows

    _, hist = plt.subplots(rows, sub_plt, figsize=(16, 4 * rows))
    hist = hist.flatten()  # Flatten to easily iterate
    for i, col in enumerate(numeric_columns):
        sns.histplot(data[col], kde=True, ax=hist[i])
        hist[i].set_title(f"Histogram of {col}")
    # Hide empty subplots if any
    for j in range(i + 1, len(hist)):
        hist[j].axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "Numeric_Features_Histogram.png"))
    plt.clf()

    # 3. Correlation Analysis: Heatmap
    correlation_matrix = data.corr()
    plt.figure(figsize=(12, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "Correlation_Heatmap.png"))
    plt.clf()

