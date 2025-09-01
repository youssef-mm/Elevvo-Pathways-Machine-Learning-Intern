# 📊 Loan Approval Prediction  

This project applies **Machine Learning** techniques to predict whether a loan application will be **approved or rejected** based on applicant information.  
It includes full preprocessing, class balancing, model training, and evaluation steps.  

---

## 📂 Project Structure  


---

## ⚙️ Workflow  

1. **Data Preprocessing**  
   - Fill missing values (median for numerical, mode for categorical).  
   - Encode categorical features using `LabelEncoder`.  
   - Standardize features with `StandardScaler`.  

2. **Train/Test Split**  
   - 80% training, 20% testing.  
   - Stratified sampling to preserve target distribution.  

3. **Imbalance Handling**  
   - Applied **SMOTE** to oversample minority class.  

4. **Models Trained**  
   - **Logistic Regression**  
   - **Decision Tree Classifier**  

5. **Evaluation**  
   - `classification_report` (Precision, Recall, F1-score).  
   - Confusion Matrix visualization.  
   - Metrics comparison in bar charts.  

---

## 📊 Results  

- Both models achieved solid results, with strengths in different metrics.  
- Logistic Regression is simpler and interpretable.  
- Decision Tree provides flexibility and handles imbalanced data better with depth control.  

📌 Metrics compared: **Precision, Recall, F1-score**.  

---

## 🛠️ Tools & Libraries  

- **pandas, numpy** → Data handling and analysis  
- **matplotlib, seaborn** → Visualization  
- **scikit-learn** → Preprocessing, models, and evaluation metrics  
- **imbalanced-learn (SMOTE)** → Handling class imbalance  

---

## ▶️ How to Run  

1. Clone the repository:  
   ```bash
   git clone https://github.com/youssef-mm/Elevvo-Pathways-Machine-learning/loan-approval-project.git
   cd loan-approval-project

pip install -r requirements.txt

python src/loan_approval.py

🚀 Future Work

Add more advanced models (Random Forest, XGBoost).

Hyperparameter tuning with GridSearchCV/RandomizedSearchCV.

Deploy model as an API (Flask/FastAPI).

Build interactive dashboard (Streamlit / Power BI)
