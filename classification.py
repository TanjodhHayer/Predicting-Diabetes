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
        "KNN": KNeighborsClassifier(n_neighbors=1),
        "SVM": SVC(kernel='rbf', probability=True, random_state=42)
    }
    
    # Cross-validation setup (5-fold)
    cross_validation = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Storage for results
    predictions = []
    probabilities = []
    model_names = []
    
    for model_name, model in models.items():
        print(f"\nTraining and Evaluating {model_name}...")
        
        # Perform cross-validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=cross_validation, scoring='f1_macro')
        print(f"{model_name} Cross-Validation F1 Score (5-fold): Mean = {cv_scores.mean():.4f}, Std = {cv_scores.std():.4f}")
        
        # Fit the model on the entire training set
        model.fit(X_train, y_train)
        
        prediction = model.predict(X_test)
        probability = None
        if hasattr(model, "predict_proba"):
            probability = model.predict_proba(X_test)
       
        # Collect predictions and probabilities
        predictions.append(prediction)
        probabilities.append(probability)
        model_names.append(model_name)
        
        # Evaluate model
        accuracy = accuracy_score(y_test, prediction)
        precision = precision_score(y_test, prediction, average='macro')
        recall = recall_score(y_test, prediction, average='macro')
        f1 = f1_score(y_test, prediction, average='macro')
        auc = None
        
        if probability:
            auc = roc_auc_score(y_test, probability, multi_class='ovr')
        
        print(f"{model_name} Accuracy: {accuracy:.4f}")
        print(f"{model_name} Precision: {precision:.4f}")
        print(f"{model_name} Recall: {recall:.4f}")
        print(f"{model_name} F1-score: {f1:.4f}")
        if auc:
            print(f"{model_name} AUC-ROC: {auc:.4f}")
            
        y_true_list = [y_test] * len(models)
        plot_combined_roc_curves(y_true_list, probabilities, model_names)
    

def plot_combined_roc_curves(y_true_list, probabilities, model_names):
    n_models = len(model_names)  # Number of models
    _, roc_plot = plt.subplots(1, n_models, figsize=(6 * n_models, 6))  # 1 row, n_models columns
    
    for i, (y_true, prob, model_name) in enumerate(zip(y_true_list, probabilities, model_names)):
        if prob is None:
            print(f"ROC Curve not available for {model_name} (no probabilities predicted).")
            continue
        
        current = roc_plot[i]  # Select the current subplot axis
        
        if prob.ndim == 1:  # Binary classification
            fpr, tpr, _ = roc_curve(y_true, prob)
            auc = roc_auc_score(y_true, prob)
            current.plot(fpr, tpr, label=f"AUC = {auc:.4f}")
        else:  # Multi-class classification
            for i_class in range(prob.shape[1]):
                fpr, tpr, _ = roc_curve(y_true == i_class, prob[:, i_class])
                auc = roc_auc_score(y_true == i_class, prob[:, i_class])
                current.plot(fpr, tpr, label=f"Class {i_class} AUC = {auc:.4f}")
        
        current.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random Guess")
        current.set_xlabel("False Positive Rate")
        current.set_ylabel("True Positive Rate")
        current.set_title(f"{model_name} ROC Curve")
        current.legend(loc="lower right")
    
    plt.tight_layout()
    plt.savefig("ROC_Curves.png")
    
    
    


    