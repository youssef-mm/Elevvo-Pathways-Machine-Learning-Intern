import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    ConfusionMatrixDisplay, 
    precision_score, 
    recall_score, 
    f1_score
)

st.title("📊 Loan Approval Prediction App")

# Upload Dataset
uploaded_file = st.file_uploader("Upload your CSV file", type="csv")
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("Dataset Preview")
    st.dataframe(df.head())
else:
    df = pd.read_csv("loan_approval_dataset.csv")

# Handle Missing Values
Cols_num = df.select_dtypes(include=['int64', 'float64']).columns
df[Cols_num] = df[Cols_num].fillna(df[Cols_num].median())

cols_categ = df.select_dtypes(include=['object']).columns
for col in cols_categ:
    df[col] = df[col].fillna(df[col].mode()[0])

# Encode categorical
encoder = LabelEncoder()
for col in [' education', ' self_employed', ' loan_status']:
    df[col] = encoder.fit_transform(df[col])

# Features & target
X = df.drop(columns=[' loan_status', 'loan_id'])
y = df[' loan_status']

# Save feature names
feature_names = X.columns

# Scale numeric
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=37, stratify=y
)

# SMOTE
smote = SMOTE(random_state=37)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)

# Sidebar: Choose Model
model_choice = st.sidebar.selectbox("Choose Model", ["Logistic Regression", "Decision Tree"])
if model_choice == "Logistic Regression":
    model = LogisticRegression(max_iter=1333, random_state=37)
else:
    model = DecisionTreeClassifier(max_depth=7, class_weight="balanced", random_state=37)

# Train
model.fit(X_train_sm, y_train_sm)
y_pred = model.predict(X_test)

# Evaluation
st.subheader(f"📌 {model_choice} - Classification Report")
report = classification_report(y_test, y_pred, output_dict=True)
st.dataframe(pd.DataFrame(report).transpose())

# Confusion Matrix
st.subheader("📌 Confusion Matrix")
fig, ax = plt.subplots()
ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax, cmap="Blues")
st.pyplot(fig)

# Metrics
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

st.subheader("📌 Metrics")
st.write(f"**Precision:** {precision:.2f}")
st.write(f"**Recall:** {recall:.2f}")
st.write(f"**F1-score:** {f1:.2f}")

# Save Confusion Matrix as PNG
fig.savefig("confusion_matrix.png", dpi=300, bbox_inches="tight")

# ------------------ Prediction Section ------------------
st.subheader("🔮 Make a Prediction")

# إدخال يدوي لكل feature
input_data = {}
for col in feature_names:
    val = st.number_input(f"Enter {col}", value=0.0)
    input_data[col] = val

if st.button("Predict"):
    new_df = pd.DataFrame([input_data])

    # Reorder columns to match training
    new_df = new_df[feature_names]

    # Scale with the same scaler
    new_scaled = scaler.transform(new_df)

    prediction = model.predict(new_scaled)[0]
    st.success(f"✅ Prediction: {'Approved' if prediction == 1 else 'Rejected'}")
