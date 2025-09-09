## Elevvo Pathways – Machine Learning Internship  

This repository documents my progress and solutions for the **Machine Learning Internship Pathways** program with **Elevvo**.  
Each task is focused on building practical skills in data science and machine learning using Python.  

## 📂 Repository Structure

Elevvo-Pathways-Machine-Learning-Intern/
│
├── Task-1-Customer-Segmentation/ # Customer segmentation using clustering
│ ├── Mall_Customer_Segmentation.ipynb # Jupyter notebook with analysis
│ ├── customer_segmentation.py # Python script version
│ ├── clustered_customers.csv # Output with customer cluster labels
│ ├── cluster_summary.csv # Summary averages of clusters
│ ├── kmeans_clusters.png # KMeans cluster visualization
│ ├── dbscan_clusters.png # DBSCAN cluster visualization
│ └── app.py # Streamlit app for interactive clustering
│
└── README.md

## 📊Task 2: Customer Segmentation  

-Objective
Cluster mall customers into meaningful segments based on their **Annual Income** and **Spending Score**.  

-Steps
1. **Data Exploration & Cleaning**  
   - Handle missing values (if any).  
   - Select relevant features: `Annual Income` and `Spending Score`.  

2. **Scaling & Preprocessing**  
   - Standardize data for clustering.  

3. **Clustering Algorithms**  
   - **K-Means**:  
     - Find optimal number of clusters using Elbow Method.  
     - Visualize customer groups in 2D plots.  
   - **DBSCAN (Bonus)**:  
     - Explore density-based clustering.  
     - Compare results with K-Means.  

4. **Analysis**  
   - Compute average income and spending per cluster.  
   - Save results in CSV files for further use.  

5. **Visualization**  
   - Save plots as `.png` for documentation.  

## 🚀 Streamlit App  

An interactive Streamlit app (`app.py`) is provided for experimenting with clustering.  
Run the app locally:  
```bash
streamlit run app.py

🛠️ Tech Stack
Python
Pandas
Matplotlib
Scikit-learn
Streamlit
