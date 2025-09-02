# 📚 Student Score Prediction App

This is a Streamlit web app that predicts **student exam scores** using different regression models.  
The app allows you to experiment with **Linear Regression**, **Polynomial Regression**, and **Multi-feature Linear Regression**, while also visualizing and comparing model performance.


## 🚀 Features
- Load and clean student performance dataset (`StudentPerformanceFactors.csv`)
- Train multiple regression models:
  - **Linear Regression** (using `Hours_Studied` only)
  - **Polynomial Regression (degree=2)**
  - **Multi-feature Linear Regression** (`Hours_Studied`, `Sleep_Hours`, `Attendance`)
- Make predictions based on user input
- Visualize data:
  - Scatter plot: Study Hours vs Exam Score
  - Bar charts: Compare **MSE** and **R²** across models
- Export visualizations as **PNG**

## 📦 Requirements
Install the dependencies before running the app:

```bash
pip install streamlit pandas scikit-learn matplotlib
▶️ How to Run

Clone this repository or download the project files.

Make sure the dataset StudentPerformanceFactors.csv is in the project folder.

Run the app using:

streamlit run app.py

Open the link in your browser (default: http://localhost:8501).

📊 Example Visualizations

Scatter plot: Study Hours vs Exam Score
Bar charts: Compare MSE and R² for each regression model

🗂 Project Structure
📁 StudentScoreApp
│── app.py                 # Main Streamlit app
│── StudentPerformanceFactors.csv   # Dataset
│── README.md              # Project documentation

🔮 Future Improvements

Add more machine learning models (Random Forest, Gradient Boosting, etc.)
Allow user to upload their own dataset
Add interactive feature importance analysis
