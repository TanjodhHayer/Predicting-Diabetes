import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_classif
from sklearn.ensemble import RandomForestClassifier

"""
References for code used in this file:
https://www.datacamp.com/tutorial/random-forests-classifier-python
https://sklearner.com/scikit-learn-mutual_info_classif/
https://medium.com/@hertan06/which-features-to-use-in-your-model-350630a1e31c
"""

def select_features(data, final_features_count=6):
    """
    Select the top features from the given dataset based on Mutual Information and Random Forest feature importances.
    @param data - The dataset containing features and target variable.
    @param final_features_count - The number of final features to select (default is 6).
    @return A new dataset with the selected features and the target variable.
    """
    X = data.drop("Diabetes_012", axis=1)
    y = data["Diabetes_012"]

    # Performing Mutual Information
    mi = mutual_info_classif(X, y, random_state=42)
    mi_scores = pd.Series(mi, index=X.columns).sort_values(ascending=False)
    print(f"Top features by Mutual Information:")
    print(mi_scores)
    
    # Perofmring Feature Importance from Random Forest
    rf = RandomForestClassifier(random_state=42)
    rf.fit(X, y)
    rf_scores = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
    print(f"Feature Importances by Random Forest:")
    print(rf_scores)
    
    # Combine results by selecting top features from both methods
    # For simplicity, take top features from both methods and ensure we have the desired number of final features.
    mi_top_features = mi_scores.head(final_features_count).index.tolist()
    rf_top_features = rf_scores.head(final_features_count).index.tolist()
    
    # Combine the two lists and remove duplicates
    selected_features = list(set(mi_top_features + rf_top_features))[:final_features_count]
    
    print(f"Final Selected Features (Top {final_features_count}):")
    print(selected_features)
    
    # Analyze correlations for the selected features
    X_selected = X[selected_features]
    correlation_matrix = X_selected.corr()

    # Now we plot and save the correlations by plotting it
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', cbar=True, square=True)
    plt.title('Correlation Matrix for Selected Features')
    plt.savefig("Correlation_Matrix_Selected_Features.png")
    pd.DataFrame(selected_features, columns=["Selected Features"]).to_csv("Selected_Features.csv", index=False)

    return data[selected_features + ["Diabetes_012"]]
