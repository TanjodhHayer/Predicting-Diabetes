import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_classif, RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

def select_features(data, threshold=0.7, top_percentage=0.3):
    X = data.drop("Diabetes_012", axis=1)
    y = data["Diabetes_012"]

    # 1. Filter Method: Mutual Information
    mi = mutual_info_classif(X, y, random_state=42)
    mi_scores = pd.Series(mi, index=X.columns).sort_values(ascending=False)
    
    # Dynamically choose top features
    top_n_features = max(4, int(len(X.columns) * top_percentage))
    print(f"Top {top_n_features} features by Mutual Information:")
    print(mi_scores.head(top_n_features))
    
    # 2. Wrapper Method: Recursive Feature Elimination
    model = LogisticRegression(max_iter=2000, random_state=42)
    rfe = RFE(model, n_features_to_select=top_n_features)
    rfe.fit(X, y)
    rfe_features = X.columns[rfe.support_]
    print("Selected Features by RFE:")
    print(rfe_features)
    
    # 3. Embedded Method: Feature Importance from Random Forest
    rf = RandomForestClassifier(random_state=42)
    rf.fit(X, y)
    feature_importance = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
    print(f"Top {top_n_features} features by Random Forest:")
    print(feature_importance.head(top_n_features))
    
    # Combine results
    selected_features = list(set(mi_scores.head(top_n_features).index) | set(rfe_features) | set(feature_importance.head(top_n_features).index))
    
    print("Analyzing correlations...")
    X_selected = X[selected_features]
    correlation_matrix = X_selected.corr()

    # Visualize correlations
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', cbar=True, square=True)
    plt.title('Correlation Matrix for Selected Features')
    plt.savefig("Correlation_Matrix_Selected_Features.png")

    # Identify highly correlated pairs
    corr_pairs = correlation_matrix.unstack()
    high_corr = corr_pairs[(abs(corr_pairs) > threshold) & (corr_pairs < 1)]
    print("Highly Correlated Feature Pairs:")
    print(high_corr)

    # Remove less important features from highly correlated pairs
    features_to_drop = set()
    for feature_1, feature_2 in high_corr.index:
        if feature_1 not in features_to_drop and feature_2 not in features_to_drop:
            if feature_importance[feature_1] < feature_importance[feature_2]:
                features_to_drop.add(feature_1)
            else:
                features_to_drop.add(feature_2)

    new_selected_features = []
    for feature in selected_features:
        if feature not in features_to_drop:
            new_selected_features.append(feature)

    selected_features = new_selected_features

    print("Final Selected Features after Correlation Check:")
    print(selected_features)

    pd.DataFrame(selected_features, columns=["Selected Features"]).to_csv("Selected_Features.csv", index=False)

    return data[selected_features + ["Diabetes_012"]]
    