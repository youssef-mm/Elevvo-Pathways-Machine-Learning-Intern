Predicting Loan Sanction 

This project aims to create machine learning models to predict loan approval by using the demographic and financial data of an applicant. 

Steps include: 

- Filling missing data
- Encoding categorical variables
- Feature scaling 
- Using SMOTE to fix class imbalance
- Training and testing of Logistic Regression and Decision Tree Classifier models

📂 Dataset

The dataset in use is loan_approval_dataset.csv, which contains both numerical and categorical variables. Important columns include:

- education
- self_employed
- loan_status (target variable)
- loan_id (removed from features)

⚙️ Project Workflow 

Data Preprocessing

- Imputed missing values of numerical features with their median.
- Imputed missing values of categorical features with the mode.
- Encoded categorical features with LabelEncoder.
- Standardized numerical features with StandardScaler.

Train-Test Split

- 80% of data was used for training and the rest 20% for testing.
- Used stratified sampling to maintain the proportion of classes. 

Imbalance Handling

- The dataset was balanced using SMOTE (Synthetic Minority Oversampling Technique).

Model Training 

- Logistic Regression with increased max_iter=1333.
- Decision Tree Classifier w
