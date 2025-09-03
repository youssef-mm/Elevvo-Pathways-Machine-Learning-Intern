import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN

# Load clustered data
df = pd.read_csv("clustered_customers.csv")
kmeans_summary = pd.read_csv("kmeans_cluster_summary.csv")
dbscan_summary = pd.read_csv("dbscan_cluster_summary.csv")

# Helper function for prediction
def predict_customer_cluster(model, scaler, income, spending, algorithm="KMeans"):
    """
    Predicts the cluster for a new customer based on income and spending.
    Works for KMeans, returns None for DBSCAN (since DBSCAN has no .predict method).
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
using clustering algorithms like **KMeans** and **DBSCAN**.
""")

# Show raw data
if st.checkbox("📂 Show raw data"):
    st.dataframe(df.head(10))

# Cluster summaries
st.subheader("📊 Cluster Summaries")
st.write("**KMeans Clusters:**")
st.dataframe(kmeans_summary)

st.write("**DBSCAN Clusters:**")
st.dataframe(dbscan_summary)

# Visualize KMeans clusters
st.subheader("🎨 KMeans Clusters")
fig1, ax1 = plt.subplots()
sns.scatterplot(
    x="Annual Income (k$)", y="Spending Score (1-100)",
    hue="KMeans_Cluster", palette="tab10", data=df, ax=ax1
)
plt.title("Customer Segmentation by KMeans")
fig1.savefig("clusters_kmeans.png", dpi=300, bbox_inches="tight")
st.pyplot(fig1)

# Visualize DBSCAN clusters
st.subheader("🎨 DBSCAN Clusters")
fig2, ax2 = plt.subplots()
sns.scatterplot(
    x="Annual Income (k$)", y="Spending Score (1-100)",
    hue="DBSCAN_Cluster", palette="tab10", data=df, ax=ax2
)
plt.title("Customer Segmentation by DBSCAN")
fig2.savefig("clusters_dbscan.png", dpi=300, bbox_inches="tight")
st.pyplot(fig2)

# Bar Charts (Income & Spending per Cluster)
st.subheader("📊 Average Income & Spending per Cluster")

# KMeans
st.write("**KMeans Clusters:**")
fig3, ax3 = plt.subplots()
kmeans_summary.plot(kind="bar", ax=ax3, figsize=(8, 6))
plt.xlabel("KMeans Cluster")
plt.ylabel("Average Values")
plt.title("Average Income & Spending per KMeans Cluster")
plt.legend(["Average Income", "Average Spending"])
fig3.savefig("kmeans_avg_income_spending.png", dpi=300, bbox_inches="tight")
st.pyplot(fig3)

# DBSCAN
st.write("**DBSCAN Clusters:**")
fig4, ax4 = plt.subplots()
dbscan_summary.plot(kind="bar", ax=ax4, figsize=(8, 6))
plt.xlabel("DBSCAN Cluster")
plt.ylabel("Average Values")
plt.title("Average Income & Spending per DBSCAN Cluster")
plt.legend(["Average Income", "Average Spending"])
fig4.savefig("dbscan_avg_income_spending.png", dpi=300, bbox_inches="tight")
st.pyplot(fig4)

# Predict new customer (KMeans only)
st.subheader("🔮 Predict New Customer Cluster (KMeans)")
income = st.slider("Annual Income (k$)", 10, 150, 50)
spending = st.slider("Spending Score (1-100)", 1, 100, 50)

# Fit scaler and KMeans model on existing data
scaler = StandardScaler()
X = df[["Annual Income (k$)", "Spending Score (1-100)"]]
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=df["KMeans_Cluster"].nunique(), random_state=42)
kmeans.fit(X_scaled)

cluster = predict_customer_cluster(kmeans, scaler, income, spending, "KMeans")
st.success(f"This customer belongs to cluster: {cluster}")
