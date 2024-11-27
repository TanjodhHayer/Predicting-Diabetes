import os
import pandas as pd
import argparse
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from eda import perform_eda
from outlier_detection import run_outlier_detection
from clustering import apply_clustering
from feature_selection import select_features
from classification import run_classifiers
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE

def load_data(filepath):
    processed_file_path = filepath.replace('.csv', '_processed.csv')
    if os.path.exists(processed_file_path):
        print("Loading preprocessed data...")
        return pd.read_csv(processed_file_path)
    else:
        print("Preprocessing data...")
        data = pd.read_csv(filepath)
        return preprocess_data(data, filepath)

def preprocess_data(data, filepath):
    label_encoder = LabelEncoder()
    categorical_columns = ['Diabetes_012', 'HighBP', 'HighChol', 'CholCheck', 'Smoker', 'Stroke', 
                           'HeartDiseaseorAttack', 'PhysActivity', 'Fruits', 'Veggies', 
                           'HvyAlcoholConsump', 'AnyHealthcare', 'NoDocbcCost', 'DiffWalk', 'Sex']
    
    for col in categorical_columns:
        if col in data.columns:
            data[col] = label_encoder.fit_transform(data[col])
    
    data['Diabetes_012'] = data['Diabetes_012'].round().astype(int)
    X = data.drop('Diabetes_012', axis=1)
    y = data['Diabetes_012']
    
    # Impute missing values for features (X)
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X)

    # Standardize the features after imputation
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)

    # Convert back to DataFrame for easy handling
    data_processed = pd.DataFrame(X_scaled, columns=X.columns)

    # EDA: Class distribution before SMOTE
    class_distribution = y.value_counts()
    class_distribution.plot(kind='bar')
    plt.title("Class Distribution Before SMOTE")
    plt.xlabel("Diabetes Status (0: No Diabetes, 1: Diagnosed, 2: Prediabetic)")
    plt.ylabel("Number of Samples")
    plt.xticks(ticks=[0, 1, 2], labels=["No Diabetes", "Diagnosed", "Prediabetic"], rotation=0)
    plt.tight_layout()
    plt.savefig("Diabetes_Class_Distribution_Before_SMOTE.png")
    plt.close()

    # Apply SMOTE for balancing the dataset
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

    # Combine the resampled data back
    data_resampled = pd.DataFrame(X_resampled, columns=X.columns)
    data_resampled['Diabetes_012'] = y_resampled

    # Stratified Subsampling after SMOTE
    sample_size = 20000  # Define the desired sample size
    data_resampled, _ = train_test_split(
        data_resampled,
        test_size=(1 - sample_size / len(data_resampled)),
        stratify=data_resampled['Diabetes_012'],
        random_state=42
    )


    class_distribution_resampled = data_resampled['Diabetes_012'].value_counts()
    class_distribution_resampled.plot(kind='bar')
    plt.title("Class Distribution After SMOTE & Stratified Subsampling")
    plt.xlabel("Diabetes Status (0: No Diabetes, 1: Diagnosed, 2: Prediabetic)")
    plt.ylabel("Number of Samples")
    plt.xticks(ticks=[0, 1, 2], labels=["No Diabetes", "Diagnosed", "Prediabetic"], rotation=0)
    plt.tight_layout()
    plt.savefig("Diabetes_Class_Distribution_After_SMOTE_&_Stratifed_Sampling.png")

    processed_file_path = filepath.replace('.csv', '_processed.csv')
    data_resampled.to_csv(processed_file_path, index=False)
    
    return data_resampled

def clean_data_with_outliers(data, lof_outliers, iso_outliers, ee_outliers):
    data["Outlier_Flag"] = lof_outliers | iso_outliers | ee_outliers
    print(f"Number of rows before cleaning: {data.shape[0]}")
    cleaned_data = data[~data["Outlier_Flag"]]
    print(f"Number of rows after cleaning: {cleaned_data.shape[0]}")
    
    return cleaned_data

def main(args):
    data_resampled = load_data(args.data)
    print("EDA...")
    perform_eda(data_resampled)

    print("Performing Outlier Detection with LOF and Isolation Forest")
    numerical_columns = ['BMI', 'GenHlth', 'MentHlth', 'PhysHlth', 'Age', 'Education', 'Income']
    lof_outliers, iso_outliers, ee_outliers = run_outlier_detection(data_resampled, numerical_columns)

    print("Cleaning dataset by removing outliers detected by both methods...")
    data_resampled = clean_data_with_outliers(data_resampled, lof_outliers, iso_outliers, ee_outliers)

    print("Performing feature selection on cleaned dataset...")
    selected_data = select_features(data_resampled)

    print("Applying clustering...")
    apply_clustering(selected_data, target_column="Diabetes_012")

    run_classifiers(selected_data, target_column="Diabetes_012")
    
    # Call this function to run a default Random Forest
    #run_rf_default(selected_data, target_column="Diabetes_012")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Data Mining Project")
    parser.add_argument("--data", type=str, required=True, help="Path to the dataset file.")
    
    args = parser.parse_args()
    main(args)
