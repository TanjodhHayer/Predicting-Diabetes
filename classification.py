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

"""
References for code used in this file:
https://towardsdatascience.com/hyperparameter-tuning-the-random-forest-in-python-using-scikit-learn-28d2aa77dd74
https://www.geeksforgeeks.org/random-forest-hyperparameter-tuning-in-python/
https://www.datacamp.com/tutorial/k-nearest-neighbor-classification-scikit-learn
https://www.datacamp.com/tutorial/svm-classification-scikit-learn-python
https://www.geeksforgeeks.org/how-to-plot-roc-curve-in-python/
https://www.w3schools.com/python/python_ml_confusion_matrix.asp
https://stackoverflow.com/questions/12514890/python-numpy-test-for-ndarray-using-ndim
https://stackoverflow.com/questions/53782169/random-forest-tuning-with-randomizedsearchcv
https://rasbt.github.io/mlxtend/user_guide/plotting/plot_confusion_matrix/
https://stackoverflow.com/questions/45139163/roc-auc-score-only-one-class-present-in-y-true
"""

def run_classifiers(data, target_column="Diabetes_012"):
    """
    Run multiple classifiers on the given data and evaluate their performance.
    @param data - The dataset containing features and target column.
    @param target_column - The name of the target column in the dataset. Default is "Diabetes_012".
    @return None
    """
    # Split the dataset into features X and target y
    X = data.drop(columns=[target_column])
    y = data[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    # Random Forest 
    rf = RandomForestClassifier(random_state=42)
    param_distributions = {
        'n_estimators': randint(50, 200),
        'max_depth': [None, 10, 20, 30, 40],
        'min_samples_split': randint(2, 10),
        'min_samples_leaf': randint(1, 5),
        'max_features': ['sqrt', 'log2', None]
    }
    
    # Perofmring the Hyperparameter tuning
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

    # Store all the models in a dictionary so that we can iterate over it
    models = {
        "Random Forest (Tuned)": best_rf,  # Tuned RF
        "KNN": KNeighborsClassifier(),     
        "SVM": SVC(kernel='rbf', probability=True, random_state=42)       
    }

    # Cross-validation setup, we are using 5 folds
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    predictions = []
    probabilities = []
    model_names = []
    
    for model_name, model in models.items():
        print(f"\nTraining and Evaluating {model_name}...")
        
        # Perform cross-validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='f1_macro')
        print(f"{model_name} Cross-Validation F1 Score (5-fold): Mean = {cv_scores.mean():.4f}, Std = {cv_scores.std():.4f}")
        
        # Fit the models on the entire training set
        model.fit(X_train, y_train)
        prediction = model.predict(X_test)
        probability = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None
        
        # Store the predictions and probabilities
        predictions.append(prediction)
        probabilities.append(probability)
        model_names.append(model_name)
        
        # Now determine all the scores needed from the pdf requirements
        accuracy = accuracy_score(y_test, prediction)
        precision = precision_score(y_test, prediction, average='macro')
        recall = recall_score(y_test, prediction, average='macro')
        f1 = f1_score(y_test, prediction, average='macro')
        auc = roc_auc_score(y_test, probability, multi_class='ovr') if probability is not None else None
        
        print(f"{model_name} Accuracy: {accuracy:.4f}")
        print(f"{model_name} Precision: {precision:.4f}")
        print(f"{model_name} Recall: {recall:.4f}")
        print(f"{model_name} F1-score: {f1:.4f}")
        if auc is not None:
            print(f"{model_name} AUC-ROC: {auc:.4f}")
    
    # Plot combined confusion matrices and ROC curves
    plot_confusion_matrices([y_test] * len(models), predictions, model_names)
    plot_combined_roc_curves([y_test] * len(models), probabilities, model_names)

def plot_confusion_matrices(y_true_list, y_pred_list, model_names):
    """
    Plot confusion matrices for multiple models based on their true and predicted labels.
    @param y_true_list - List of true labels for each model
    @param y_pred_list - List of predicted labels for each model
    @param model_names - List of names for each model
    @return None
    """
    # Create a figure for all confusion matrices
    n_models = len(model_names)
    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 6))  # 1 row, n_models columns
    for i, (y_true, y_pred, model_name) in enumerate(zip(y_true_list, y_pred_list, model_names)):
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y_true), yticklabels=np.unique(y_true), ax=axes[i])
        axes[i].set_title(f"{model_name} Confusion Matrix")
        axes[i].set_xlabel("Predicted Labels")
        axes[i].set_ylabel("True Labels")
    plt.tight_layout()
    plt.savefig("combined_confusion_matrices.png")

def plot_combined_roc_curves(y_true_list, probabilities, model_names):
    """
    Plot combined ROC curves for multiple models.
    @param y_true_list - List of true labels for each model.
    @param probabilities - List of predicted probabilities for each model.
    @param model_names - List of names for each model.
    @return None. The ROC curves are plotted and saved as "combined_roc_curves.png".
    """
    n_models = len(model_names)  # Number of models
    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 6))  # 1 row, n_models columns
    
    for i, (y_true, probability, model_name) in enumerate(zip(y_true_list, probabilities, model_names)):
        if probability is None:
            print(f"ROC Curve not available for {model_name} (no probabilities predicted).")
            continue
        
        ax = axes[i]  # Select the current subplot axis
        
        if probability.ndim == 1:  # Binary classification
            fpr, tpr, _ = roc_curve(y_true, probability)
            auc = roc_auc_score(y_true, probability)
            ax.plot(fpr, tpr, label=f"AUC = {auc:.4f}")
        else:  # Multi-class classification
            for i_class in range(probability.shape[1]):
                fpr, tpr, _ = roc_curve(y_true == i_class, probability[:, i_class])
                auc = roc_auc_score(y_true == i_class, probability[:, i_class])
                ax.plot(fpr, tpr, label=f"Class {i_class} AUC = {auc:.4f}")
        
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random Guess")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(f"{model_name} ROC Curve")
        ax.legend(loc="lower right")
    
    plt.tight_layout()
    plt.savefig("combined_roc_curves.png")

def run_rf_with_cv(data, target_column="Diabetes_012"):
    """
    Run a Random Forest classifier with cross-validation and evaluate its performance metrics.
    @param data - The dataset containing features and target column.
    @param target_column - The name of the target column in the dataset. Default is "Diabetes_012".
    @return None
    """
    # Split the dataset into features (X) and target (y)
    X = data.drop(columns=[target_column])
    y = data[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    # Initialize the Random Forest Classifier with default hyperparameters
    rf = RandomForestClassifier(random_state=42)
    
    # Perform 5-fold cross-validation on the training set
    print("Performing 5-fold cross-validation for Random Forest Default")
    cv_scores = cross_val_score(rf, X_train, y_train, cv=5, scoring='f1_macro')  # Use training set only
    print(f"Mean F1-score from 5-fold CV: {np.mean(cv_scores):.4f}")
    
    # Train on the full training set and evaluate on the test set
    rf.fit(X_train, y_train)
    prediction = rf.predict(X_test)
    probability = rf.predict_proba(X_test)
    
    # Evaluate model performance on the test set
    accuracy = accuracy_score(y_test, prediction)
    precision = precision_score(y_test, prediction, average='macro')
    recall = recall_score(y_test, prediction, average='macro')
    f1 = f1_score(y_test, prediction, average='macro')
    auc = roc_auc_score(y_test, probability, multi_class='ovr') if probability is not None else None
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")
    if auc is not None:
        print(f"AUC-ROC: {auc:.4f}")
    
    # Plot confusion matrix and ROC curve (on the test set)
    plot_confusion_matrix_single(y_test, prediction, "Random Forest")
    plot_roc_curve_single(y_test, probability, "Random Forest")

def plot_confusion_matrix_single(y_true, y_pred, model_name):
    """
    Plot a single confusion matrix for a given model's predictions.
    @param y_true - True labels
    @param y_pred - Predicted labels
    @param model_name - Name of the model
    @return None
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y_true), yticklabels=np.unique(y_true))
    plt.title(f"{model_name} Confusion Matrix")
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.tight_layout()
    plt.savefig(f"{model_name.replace(' ', '_')}_Confusion_Matrix.png")
    plt.close()

def plot_roc_curve_single(y_true, probability, model_name):
    """
    Plot the ROC curve for a single model.
    @param y_true - True labels
    @param probability - Predicted probabilities
    @param model_name - Name of the model
    @return None
    """
    # This is for Binary classification
    if probability.ndim == 1:  
        fpr, tpr, _ = roc_curve(y_true, probability)
        auc = roc_auc_score(y_true, probability)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f"AUC = {auc:.4f}")
    else:  # This is for Multi-class classification
        plt.figure(figsize=(8, 6))
        for i in range(probability.shape[1]):
            fpr, tpr, _ = roc_curve(y_true == i, probability[:, i])
            auc = roc_auc_score(y_true == i, probability[:, i])
            plt.plot(fpr, tpr, label=f"Class {i} AUC = {auc:.4f}")
    
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random Guess")
    plt.title(f"{model_name} ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(f"{model_name.replace(' ', '_')}_ROC_Curve.png")
    plt.close()
