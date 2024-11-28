# Understanding What Factors Lead To Diabetes Using Machine Learning

## Tanjodh Hayer (301432974), Mohammad Haris Ahmad (301427462), Inderpreet Rangi (301433641)
## Simon Fraser University, CMPT 459 - Fall 2024

### Problem Statement
Diabetes is a common health issue that many people face. We wanted to figure out what were some of the leading factors that cause it such that at-risk individuals can be identified early, potentially reducing the prevalence or severity of diabetes. 

### Selected Dataset: 
Diabetes Health Indicators Dataset: [https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset]
This dataset came from Kaggle and it represents the connections between the human lifestyle and diabetes within the United States of America. Each row within this dataset is a US citizen who had participated in this survey conducted by the CDC. This dataset has 253681 entries and consist of the following features
Numerical:
- BMI (Body Mass Index)
- GenHlth (General health status, rated on a scale)
- MentHlth (Mental health status, rated on a scale)
- PhysHlth (Physical health status, rated on a scale)
- Age (Age category)
- Education (Education level)
- Income (Income level)
Categorical:
- Diabetes_012 (Diabetes classification: No diabetes,Prediabetes, Diabetes)
- HighBP (High blood pressure status: Yes or No)
- HighChol (High cholesterol status: Yes or No)
- CholCheck (Cholesterol check status: Yes or No)
- Smoker (Smoking status: Yes or No) 
- Stroke (Stroke status: Yes or No)
- HeartDiseaseorAttack (Heart disease or attack status: Yes or No)
- PhysActivity (Physical activity status: Yes or No)
- Fruits (Fruit consumption: Yes or No)
- Veggies (Vegetable consumption: Yes or No)
- HvyAlcoholConsump (Heavy alcohol consumption status: Yes or No)
- AnyHealthcare (Access to healthcare: Yes or No)
- NoDocbcCost (Cost barriers to healthcare: Yes or No)
- DiffWalk (Difficulty walking status: Yes or No)
- Sex (Gender: Female or Male)

 
To run this program enter the following command: 
python main.py --data "dataset.csv"

### Data Table (5 Rows)

| Diabetes_012 | HighBP | HighChol | CholCheck | BMI | Smoker | Stroke | HeartDiseaseorAttack | PhysActivity | Fruits | Veggies | HvyAlcoholConsump | AnyHealthcare | NoDocbcCost | GenHlth | PhysHlth | DiffWalk | Sex | Age | Education | Income |
|--------------|--------|----------|-----------|-----|--------|--------|----------------------|--------------|--------|---------|-------------------|---------------|-------------|---------|----------|----------|-----|-----|-----------|--------|
| 0            | 1      | 1        | 1         | 40  | 1      | 0      | 0                    | 0            | 0      | 1       | 0                 | 1             | 0           | 5       | 18       | 1       | 1   | 7   | 6         | 4      |
| 0            | 0      | 0        | 0         | 25  | 1      | 0      | 0                    | 0            | 1      | 0       | 0                 | 0             | 1           | 0       | 0        | 0        | 0   | 4   | 5         | 6      |
| 2            | 1      | 1        | 1         | 30  | 1      | 0      | 1                    | 1            | 0      | 0       | 1                 | 1            | 1           | 0       | 0        | 1        | 0   | 6   | 6         | 5      |
| 1            | 1      | 1        | 1         | 32  | 1      | 1      | 0                    | 1            | 0      | 0       | 1                 | 0             | 1           | 0       | 30       | 1        | 0   | 12   | 4         | 5      |
| 0            | 1      | 0        | 1         | 30  | 0      | 0      | 0                    | 1            | 1      | 1       | 0                 | 1             | 0           | 0       | 0        | 1        | 0   | 9   | 3         | 4      |



## Methodology
### Data Preprocessing
Data preprocessing is a crucial step in preparing raw data for the machine learning tasks that follow. Here's a detailed overview of the steps we implemented:

We began by converting all categorical columns into numeric values using label encoding. This transformation ensures compatibility with machine learning models, which generally require numerical inputs. Encoding these features enabled their effective use during the training of the three chosen models.

The target variable, Diabetes_012, was rounded and converted to an integer type, with the classes represented as follows:

    0: No Diabetes
    1: Diagnosed Diabetes
    2: Prediabetic.

This conversion was performed to maintain consistency and prevent errors during model training and classification.

To handle missing data, we imputed missing values in the features using the mean of each column. This approach preserves dataset completeness while providing reasonable estimates for missing entries.

Feature standardization was also applied, ensuring all features had a mean of 0 and a standard deviation of 1. This step is particularly important for algorithms like K-Means and Support Vector Machines (SVM), which are sensitive to feature scaling. Standardization improved model convergence and overall performance.

Our initial dataset contained over 200,000 samples, but there was a significant class imbalance. To address this, we applied SMOTE (Synthetic Minority Oversampling Technique) to balance the class distribution by generating synthetic samples for the minority classes. Without balancing, the models could become biased toward the majority class. However, applying SMOTE increased the dataset size to over 600,000 samples, which was computationally prohibitive. To resolve this, we reduced the dataset to 20,000 samples using stratified subsampling. This method preserved the class proportions, ensuring an equal class distribution while making the dataset manageable for machine learning tasks. By subsampling, we retained the dataset's integrity while enabling efficient model training, classification, and clustering.

To verify the effectiveness of our preprocessing, we visualized the class distributions before and after SMOTE and subsampling. These plots provide clear evidence of the process and outcomes.

Finally, to save time during subsequent runs, the processed dataset was saved as dataset_processed.csv. A utility function checks for the existence of this file:

    If the file exists, the code skips preprocessing and directly loads the cleaned, balanced, and scaled dataset.
    If the file does not exist, the preprocessing pipeline runs on the original dataset to regenerate and save the processed version.

## Class Distribution Before SMOTE
![Class Distribution Before SMOTE](/Diabetes_Class_Distribution_Before_SMOTE.png "Class Distribution Before SMOTE")

## Class Distribution After SMOTE and Stratified Sampling
![Class Distribution After SMOTE and Stratified Sampling](/Diabetes_Class_Distribution_After_SMOTE_&_Stratifed_Sampling.png "Class Distribution After SMOTE & Subsampling")

### Exploratory Data Analysis (EDA)

### Clustering

### Outlier Detection

### Feature Selection

### Classification

### Hyperparemeter Tuning

### Conclusion