import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.arima.model import ARIMA
from sklearn.cluster import KMeans

# Load dataset
df = pd.read_csv("Final crime against women.csv")

# Data preprocessing
df.columns = df.columns.str.strip()
df.rename(columns={"Total": "Total_Crime_Against_Women"}, inplace=True)

# Streamlit App Title
st.title("Crime Against Women Data Analysis")

# Crime Trend Analysis
st.header("Crime Trends (2001-2020)")
crime_trend = df.groupby("Year")["Total_Crime_Against_Women"].sum()
st.line_chart(crime_trend)

# Machine Learning: Crime Prediction
st.header("Crime Prediction for Next 5 Years")
model = ARIMA(crime_trend, order=(2,1,2))
model_fit = model.fit()
future_years = [2021, 2022, 2023, 2024, 2025]
forecast = model_fit.forecast(steps=5)
forecast_df = pd.DataFrame({"Year": future_years, "Predicted Crimes": forecast})
st.write(forecast_df)

# Clustering: Crime Hotspot Detection
st.header("Crime Hotspots (State-wise Analysis)")
state_crime = df.groupby("Area_Name")["Total_Crime_Against_Women"].sum().reset_index()
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
state_crime["Cluster"] = kmeans.fit_predict(state_crime[["Total_Crime_Against_Women"]])
st.bar_chart(state_crime.set_index("Area_Name")[["Total_Crime_Against_Women"]])

st.write("Clusters: 0 (Low), 1 (Medium), 2 (High)")
st.write(state_crime.sort_values("Cluster"))

# Run: streamlit run app.py
