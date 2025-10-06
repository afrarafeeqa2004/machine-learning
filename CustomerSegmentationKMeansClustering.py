#import packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from datetime import timedelta

#load dataset
df = pd.read_csv("/content/drive/MyDrive/Colab Notebooks/Online Retail.csv")

df.columns = df.columns.str.strip().str.replace("ï»¿", "")

df = df.dropna(subset=["CustomerID"])

df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], format="%d-%m-%Y %H:%M", errors="coerce")

df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]

reference_date = df["InvoiceDate"].max() + timedelta(days=1)

rfm = df.groupby("CustomerID").agg({
    "InvoiceDate": lambda x: (reference_date - x.max()).days,  
    "InvoiceNo": "nunique",                                    
    "TotalPrice": "sum"                                      
}).reset_index()

rfm.columns = ["CustomerID", "Recency", "Frequency", "Monetary"]

scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm[["Recency", "Frequency", "Monetary"]])

inertia = []
K = range(1, 11)
for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(rfm_scaled)
    inertia.append(kmeans.inertia_)

plt.figure(figsize=(6, 4))
plt.plot(K, inertia, "bx-")
plt.xlabel("Number of clusters (k)")
plt.ylabel("Inertia")
plt.title("Elbow Method For Optimal k")
plt.show()

kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)

#train the model
rfm["Cluster"] = kmeans.fit_predict(rfm_scaled)

cluster_summary = rfm.groupby("Cluster").agg({
    "Recency": "mean",
    "Frequency": "mean",
    "Monetary": "mean",
    "CustomerID": "count"
}).rename(columns={"CustomerID": "NumCustomers"})
print(cluster_summary)
