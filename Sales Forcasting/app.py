# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, TimeSeriesSplit, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error, roc_curve, auc
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from xgboost import XGBRegressor
from statsmodels.tsa.seasonal import seasonal_decompose

st.set_page_config(layout="wide")
st.title("Walmart Sales Forecast Dashboard")

# Sidebar
st.sidebar.header("User Inputs")
model_choice = st.sidebar.selectbox("Select Regression Model", ("KNN", "Linear Regression", "XGBoost"))

n_neighbors = st.sidebar.slider("KNN: Number of Neighbors", 1, 10, 5)
xgb_estimators = st.sidebar.slider("XGBoost: n_estimators", 50, 300, 100)
xgb_depth = st.sidebar.slider("XGBoost: max_depth", 3, 10, 5)
xgb_lr = st.sidebar.slider("XGBoost: learning_rate", 0.01, 0.3, 0.1)

# Load Dataset
df = pd.read_csv('Walmart.csv')
df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y')
df = df.sort_values('Date')

# Feature Engineering
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['WeekOfYear'] = df['Date'].dt.isocalendar().week
df['DayOfWeek'] = df['Date'].dt.dayofweek
df['Sales_Lag_1'] = df['Weekly_Sales'].shift(1)
df['Sales_Lag_2'] = df['Weekly_Sales'].shift(2)
df['Rolling_Mean_4'] = df['Weekly_Sales'].rolling(4).mean()
df.dropna(inplace=True)

features = ['Year','Month','WeekOfYear','DayOfWeek','Sales_Lag_1','Sales_Lag_2','Rolling_Mean_4']
X = df[features]
y = df['Weekly_Sales']

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model Selection
if model_choice == "KNN":
    model = KNeighborsRegressor(n_neighbors=n_neighbors)
elif model_choice == "Linear Regression":
    model = LinearRegression()
else:
    model = XGBRegressor(objective='reg:squarederror', n_estimators=xgb_estimators, max_depth=xgb_depth, learning_rate=xgb_lr, random_state=42)

model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
st.write(f"**{model_choice} Test RMSE:** {rmse:.2f}")

# Plot actual vs predicted
st.subheader("Actual vs Predicted Weekly Sales")
fig, ax = plt.subplots(figsize=(12,6))
ax.plot(df['Date'].iloc[-len(y_test):], y_test, label='Actual')
ax.plot(df['Date'].iloc[-len(y_test):], y_pred, label='Predicted')
ax.set_xlabel("Date")
ax.set_ylabel("Weekly Sales")
ax.legend()
st.pyplot(fig)

# Seasonal Decomposition Plot
st.subheader("Seasonal Decomposition")
result = seasonal_decompose(df['Weekly_Sales'], model='additive', period=52)
fig2, axs = plt.subplots(3,1, figsize=(15,8))
result.trend.plot(ax=axs[0], title='Trend', fontsize=12)
axs[0].title.set_position([.5, 1.05])  # تحريك العنوان أعلى
result.seasonal.plot(ax=axs[1], title='Seasonality', fontsize=12)
axs[1].title.set_position([.5, 1.05])
result.resid.plot(ax=axs[2], title='Residual', fontsize=12)
axs[2].title.set_position([.5, 1.05])

plt.tight_layout()
st.pyplot(fig2)

# Aggregate weekly total sales
weekly_total = df.groupby('Date', as_index=True)['Weekly_Sales'].sum().sort_index()

st.subheader("Weekly Total Sales Time Series")
st.write(weekly_total)  # دي هتعرض التاريخ وقيم المبيعات

# Seasonal decomposition only if enough points
if len(weekly_total) >= 104:
    st.write("Running seasonal decomposition (additive, yearly)")
    res = seasonal_decompose(weekly_total, model='additive', period=52)
    
    # Plot decomposition
    fig = res.plot()
    fig.set_size_inches(12,8)
    st.pyplot(fig)
else:
    st.write("Not enough points for seasonal decomposition")


# KMeans Clustering
st.subheader("Clustering (KMeans)")
X_cluster = df[features]
X_scaled_cluster = scaler.fit_transform(X_cluster)
kmeans = KMeans(n_clusters=3, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_scaled_cluster)

fig3, ax3 = plt.subplots(figsize=(8,6))
sns.scatterplot(x=df['Sales_Lag_1'], y=df['Weekly_Sales'], hue=df['Cluster'], palette='Set2', ax=ax3)
ax3.set_title("KMeans Clustering of Sales")
st.pyplot(fig3)
