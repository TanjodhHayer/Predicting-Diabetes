import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest, mutual_info_classif, RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

def select_features(data, threshold=0.7, top_percentage=0.15):

    X = data.drop("Diabetes_012", axis=1)
    y = data["Diabetes_012"]


    # 1. Filter Method: Mutual Information
    mi = mutual_info_classif(X, y, random_state=42)
    mi_scores = pd.Series(mi, index=X.columns).sort_values(ascending=False)
    
    # Dynamically choose top features
    top_n_features = max(2, int(len(X.columns) * top_percentage))
    print(f"Top {top_n_features} features by Mutual Information:")
    print(mi_scores.head(top_n_features))
    
    # 2. Wrapper Method: Recursive Feature Elimination
    model = LogisticRegression(max_iter=2000, random_state=42)
    rfe = RFE(model, n_features_to_select=top_n_features)
    rfe.fit(X, y)
    rfe_features = X.columns[rfe.support_]
    print("Selected Features by RFE:")
    print(rfe_features)
    