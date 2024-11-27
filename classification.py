import numpy as np
from sklearn.model_selection import RandomizedSearchCV, train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, confusion_matrix, roc_curve)
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import randint

def run_classifiers(data, target_column="Diabetes_012"):
    # Split the dataset into features (X) and target (y)
    X = data.drop(columns=[target_column])
    y = data[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    # Random Forest Hyperparameter Tuning
    rf = RandomForestClassifier(random_state=42)
    param_distributions = {
        'n_estimators': randint(50, 200),
        'max_depth': [None, 10, 20, 30, 40],
        'min_samples_split': randint(2, 10),
        'min_samples_leaf': randint(1, 5),
        'max_features': ['sqrt', 'log2', None]
    }
    print("Performing Randomized Search for Hyperparameter Tuning on Random Forest...")
    randomized_search = RandomizedSearchCV(
        estimator=rf,
        param_distributions=param_distributions,
        n_iter=50,  # Number of combinations to try
        cv=5,  # 5-fold cross-validation
        scoring='f1_macro',  # Optimize for F1-score (macro-average)
        verbose=0,
        random_state=42,
        n_jobs=-1
    )
    randomized_search.fit(X_train, y_train)
    
    # Get the best Random Forest model
    best_rf = randomized_search.best_estimator_
    print("Best Hyperparameters for Random Forest:", randomized_search.best_params_)
    
    models = {
        "Random Forest (Tuned)": best_rf,
        "SVM": SVC(kernel='rbf', probability=True, random_state=42)
    }
    

def plot_combined_roc_curves(y_true_list, y_proba_list, model_names):
    n_models = len(model_names)  # Number of models
    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 6))  # 1 row, n_models columns
    
    for i, (y_true, y_proba, model_name) in enumerate(zip(y_true_list, y_proba_list, model_names)):
        if y_proba is None:
            print(f"ROC Curve not available for {model_name} (no probabilities predicted).")
            continue
        
        ax = axes[i]  # Select the current subplot axis
        
        if y_proba.ndim == 1:  # Binary classification
            fpr, tpr, _ = roc_curve(y_true, y_proba)
            auc = roc_auc_score(y_true, y_proba)
            ax.plot(fpr, tpr, label=f"AUC = {auc:.4f}")
        else:  # Multi-class classification
            for i_class in range(y_proba.shape[1]):
                fpr, tpr, _ = roc_curve(y_true == i_class, y_proba[:, i_class])
                auc = roc_auc_score(y_true == i_class, y_proba[:, i_class])
                ax.plot(fpr, tpr, label=f"Class {i_class} AUC = {auc:.4f}")
        
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random Guess")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(f"{model_name} ROC Curve")
        ax.legend(loc="lower right")
    
    plt.tight_layout()
    plt.savefig("combined_roc_curves.png")


def run_rf_with_cv(data, target_column="Diabetes_012"):
    # Split the dataset into features (X) and target (y)
    X = data.drop(columns=[target_column])
    y = data[target_column]
    
    # Initialize the Random Forest Classifier with default hyperparameters
    rf = RandomForestClassifier(random_state=42)
    
    # Perform 5-fold cross-validation
    print("Performing 5-fold cross-validation for Random Forest...")
    cv_scores = cross_val_score(rf, X, y, cv=5, scoring='f1_macro')  # You can change scoring as needed
    
    print(f"Cross-validation scores: {cv_scores}")
    print(f"Mean accuracy from 5-fold CV: {np.mean(cv_scores):.4f}")
    
    # Train the model on the full dataset
    rf.fit(X, y)
    
    # Make predictions on the same data (for demonstration, though this is typically not done in real evaluation)
    y_pred = rf.predict(X)
    y_proba = rf.predict_proba(X) if hasattr(rf, "predict_proba") else None
    
    # Evaluate model performance on the entire dataset
    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred, average='macro')
    recall = recall_score(y, y_pred, average='macro')
    f1 = f1_score(y, y_pred, average='macro')
    auc = roc_auc_score(y, y_proba, multi_class='ovr') if y_proba is not None else None
    
    print(f"Random Forest Accuracy: {accuracy:.4f}")
    print(f"Random Forest Precision: {precision:.4f}")
    print(f"Random Forest Recall: {recall:.4f}")
    print(f"Random Forest F1-score: {f1:.4f}")
    if auc is not None:
        print(f"Random Forest AUC-ROC: {auc:.4f}")
    
    # Plot confusion matrix and ROC curve
    plot_confusion_matrices([y], [y_pred], ["Random Forest"])
    plot_combined_roc_curves([y], [y_proba], ["Random Forest"])

    