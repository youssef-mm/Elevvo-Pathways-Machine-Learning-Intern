import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN

# Load data
df = pd.read_csv("clustered_customers.csv")
cluster_summary = pd.read_csv("cluster_summary.csv")

# Helper function for prediction
def predict_customer_cluster(model, scaler, income, spending, algorithm="KMeans"):
    """
    Predicts the cluster for a new customer based on income and spending.
    Works for KMeans, returns None for DBSCAN (since no .predict method).
    """
    new_point = scaler.transform([[income, spending]])
    if algorithm == "KMeans":
        return model.predict(new_point)[0]
    else:
        return None

# Streamlit app
st.title("🛍️ Mall Customer Segmentation App")

st.markdown("""
This app segments customers into groups (clusters) based on their **Annual Income** and **Spending Score**  
using clustering algorithms like **KMeans**.
""")

# Show raw data
if st.checkbox("Show raw data"):
    st.dataframe(df.head(10))

# Cluster summary
st.subheader("📊 Cluster Summary (Averages)")
st.dataframe(cluster_summary)

# Plot clusters
st.subheader("🎨 Visualization of Clusters")
fig, ax = plt.subplots()
sns.scatterplot(
    x="Annual Income (k$)", y="Spending Score (1-100)",
    hue="KMeans_Cluster", palette="tab10", data=df, ax=ax
)
plt.title("Customer Segmentation by KMeans")

# Save plot as PNG
fig.savefig("clusters.png", dpi=300, bbox_inches="tight")

# Show in Streamlit
st.pyplot(fig)

# Predict new customer
st.subheader("🔮 Predict New Customer Cluster")
income = st.slider("Annual Income (k$)", 10, 150, 50)
spending = st.slider("Spending Score (1-100)", 1, 100, 50)

# Fit scaler and model on existing data
scaler = StandardScaler()
X = df[["Annual Income (k$)", "Spending Score (1-100)"]]
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=df["KMeans_Cluster"].nunique(), random_state=42)
kmeans.fit(X_scaled)

cluster = predict_customer_cluster(kmeans, scaler, income, spending, "KMeans")
st.success(f"This customer belongs to cluster: {cluster}")
