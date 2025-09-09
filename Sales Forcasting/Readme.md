# Walmart Sales Forecasting

This project focuses on **forecasting weekly sales** for Walmart stores using different Machine Learning techniques.  
It includes **data preprocessing, feature engineering, clustering, and time series analysis** to identify trends, seasonality, and improve predictive performance.

---

## 📂 Project Overview
- **Exploratory Data Analysis (EDA)**: Understanding correlations, missing values, and sales distribution.  
- **Feature Engineering**: Lag features, scaling, and handling categorical data.  
- **Modeling**:  
  - Linear Regression  
  - Decision Tree  
  - Random Forest  
  - K-Nearest Neighbors (KNN)  
  - XGBoost (with GridSearchCV for hyperparameter tuning)  
- **Time Series Decomposition**: Identifying **trend, seasonality, and residuals**.  
- **Clustering**: Using KMeans to segment sales behavior.  

---

## 🛠️ Technologies Used
- Python (Pandas, NumPy, Matplotlib, Seaborn)  
- Scikit-learn  
- XGBoost  
- Statsmodels  
- Streamlit (for interactive dashboard)  

---

## 📊 Key Visualizations
- Correlation Heatmap  
- Seasonal Decomposition (Trend, Seasonality, Residuals)  
- Clustering plots  
- Model evaluation metrics (RMSE, R²)  

---
## 🚀 How to Run  
1. Clone the repository:  
   ```bash
   git clone https://github.com/youssef-mm/Elevvo-Pathways-Machine-Learning-Intern.git

2. Navigate to the project folder:
   ```bash
   cd Sales Forcasting

3. Run the Jupyter Notebook:
   ```bash
   streamlit run app.py


## 📌 Results

Models compared using RMSE & R².

XGBoost with hyperparameter tuning achieved the best performance.


Seasonal decomposition highlighted strong yearly seasonality in sales.
