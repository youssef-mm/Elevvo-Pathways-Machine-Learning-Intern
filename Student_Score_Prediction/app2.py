import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, r2_score

# Load & clean data
@st.cache_data
def load_data():
    df = pd.read_csv("StudentPerformanceFactors.csv")
    df = df.dropna()
    return df

df = load_data()

st.title("📚 Student Score Prediction App")
st.write("Predict exam scores using different regression models.")

#Sidebar for model selection
model_choice = st.sidebar.selectbox(
    "Select Model",
    ["Linear Regression (StudyHours only)", "Polynomial Regression (degree=2)", "Linear Regression (Multi-features)"]
)

# Train Models
# Linear Regression (single feature)
X_single = df[["Hours_Studied"]]
y_single = df["Exam_Score"]
X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(X_single, y_single, test_size=0.2, random_state=42)

lin_model = LinearRegression()
lin_model.fit(X_train_s, y_train_s)
y_pred_lin = lin_model.predict(X_test_s)
mse_lin = mean_squared_error(y_test_s, y_pred_lin)
r2_lin = r2_score(y_test_s, y_pred_lin)

# Polynomial Regression
poly_model = make_pipeline(PolynomialFeatures(degree=2), LinearRegression())
poly_model.fit(X_train_s, y_train_s)
y_pred_poly = poly_model.predict(X_test_s)
mse_poly = mean_squared_error(y_test_s, y_pred_poly)
r2_poly = r2_score(y_test_s, y_pred_poly)

# Multi-feature Regression
features = ["Hours_Studied", "Sleep_Hours", "Attendance"]
X_multi = df[features]
y_multi = df["Exam_Score"]
X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(X_multi, y_multi, test_size=0.2, random_state=42)

multi_model = LinearRegression()
multi_model.fit(X_train_m, y_train_m)
y_pred_multi = multi_model.predict(X_test_m)
mse_multi = mean_squared_error(y_test_m, y_pred_multi)
r2_multi = r2_score(y_test_m, y_pred_multi)

# User Input
st.sidebar.header("Enter Student Data")
study_hours = st.sidebar.slider("Study Hours", 0.0, 12.0, 5.0, 0.5)

if model_choice == "Linear Regression (Multi-features)":
    sleep_hours = st.sidebar.slider("Sleep Hours", 0.0, 12.0, 7.0, 0.5)
    attendance = st.sidebar.slider("Attendance (%)", 0, 100, 80, 5)


# Prediction
if st.button("Predict Score"):
    if model_choice == "Linear Regression (StudyHours only)":
        input_data = pd.DataFrame([[study_hours]], columns=["Hours_Studied"])
        prediction = lin_model.predict(input_data)[0]
        st.success(f"Predicted Exam Score: {prediction:.2f}")
        st.write(f"Model Performance → MSE: {mse_lin:.2f}, R²: {r2_lin:.2f}")

    elif model_choice == "Polynomial Regression (degree=2)":
        input_data = pd.DataFrame([[study_hours]], columns=["Hours_Studied"])
        prediction = poly_model.predict(input_data)[0]
        st.success(f"Predicted Exam Score: {prediction:.2f}")
        st.write(f"Model Performance → MSE: {mse_poly:.2f}, R²: {r2_poly:.2f}")

    else:
        input_data = pd.DataFrame([[study_hours, sleep_hours, attendance]], columns=features)
        prediction = multi_model.predict(input_data)[0]
        st.success(f"Predicted Exam Score: {prediction:.2f}")
        st.write(f"Model Performance → MSE: {mse_multi:.2f}, R²: {r2_multi:.2f}")

# Visualization
st.subheader("📊 Visualization")

fig, ax = plt.subplots()

if model_choice == "Linear Regression (StudyHours only)":
    ax.scatter(df["Hours_Studied"], df["Exam_Score"], alpha=0.6, label="Actual Data")
    ax.plot(df["Hours_Studied"], lin_model.predict(df[["Hours_Studied"]]), color="red", label="Linear Fit")
    ax.set_title("Linear Regression (Study Hours)")
    ax.set_xlabel("Study Hours")
    ax.set_ylabel("Exam Score")

elif model_choice == "Polynomial Regression (degree=2)":
    ax.scatter(df["Hours_Studied"], df["Exam_Score"], alpha=0.6, label="Actual Data")
    X_sorted = np.sort(df[["Hours_Studied"]].values, axis=0)
    ax.plot(X_sorted, poly_model.predict(X_sorted), color="green", label="Polynomial Fit")
    ax.set_title("Polynomial Regression (Study Hours)")
    ax.set_xlabel("Study Hours")
    ax.set_ylabel("Exam Score")

else:  # Multi-feature
    ax.scatter(y_test_m, y_pred_multi, alpha=0.6, color="orange")
    ax.plot([y_test_m.min(), y_test_m.max()], [y_test_m.min(), y_test_m.max()], "r--")
    ax.set_title("Multi-feature Regression (Actual vs Predicted)")
    ax.set_xlabel("Actual Exam Score")
    ax.set_ylabel("Predicted Exam Score")

ax.legend()
st.pyplot(fig)

# Model Comparison (MSE & R² together)
st.subheader("📊 Model Comparison")

results = pd.DataFrame({
    "Model": [
        "Linear (Hours only)",
        "Polynomial (deg=2)",
        "Multi-feature"
    ],
    "MSE": [mse_lin, mse_poly, mse_multi],
    "R²": [r2_lin, r2_poly, r2_multi]
})

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Bar chart for MSE
axes[0].bar(results["Model"], results["MSE"], color="skyblue")
axes[0].set_title("Mean Squared Error (MSE)")
axes[0].set_ylabel("MSE")
axes[0].tick_params(axis='x', rotation=20)

# Bar chart for R²
axes[1].bar(results["Model"], results["R²"], color="lightgreen")
axes[1].set_title("R² Score")
axes[1].set_ylabel("R²")
axes[1].tick_params(axis='x', rotation=20)

st.pyplot(fig)
