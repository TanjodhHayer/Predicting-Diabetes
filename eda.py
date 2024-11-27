import os
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import table
import seaborn as sns
import sys

def perform_eda(data):

    # Create a folder for saving plots if it doesn't exist
    output_dir = "eda_plots"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 1. Basic Overview of the Data
    # Suppress info output and save describe() as a table
    describe = data.describe()
    fig, ax = plt.subplots(figsize=(12, 6))  # Adjust the figure size if necessary
    ax.axis('off')  # Hide axes
    summary_table = table(ax, describe, loc='center', colWidths=[0.2] * len(data.columns))
    plt.title("Summary Statistics", fontsize=16)
    plt.savefig(os.path.join(output_dir, "eda_summary_statistics.png"), format="png", bbox_inches='tight', dpi=300)

