import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.cluster import KMeans

# Load data
df = pd.read_csv("Country-data.csv")

# Display title
st.markdown("<h1 style='color: blue;'>Unsupervised Learning on Country Data</h1>", unsafe_allow_html=True)

# Display dataset
st.header("Dataset")
st.write(df)

# Elbow plot
st.header("Elbow Curve for Optimal K")
k_values = list(range(1, 11))
inertia_values = []

for best_k in k_values:
    kmeans = KMeans(n_clusters=best_k, init="k-means++", n_init=10, max_iter=300, tol=0.0001, random_state=45)
    kmeans.fit(df.drop(["country"], axis=1))
    inertia_values.append(kmeans.inertia_)

# Display the Elbow Curve
fig, ax = plt.subplots()
ax.plot(k_values, inertia_values, marker="o")
ax.set(xlabel="Number of Clusters (k)", ylabel="Inertia", title="Elbow Curve")
st.pyplot(fig)

# Select number of clusters
st.header("Select Number of Clusters")
clust = st.slider("Choose the number of clusters:", 2, 10, 3, 1)

# K-means clustering and visualization
def perform_kmeans(best_k):
    kmeans = KMeans(n_clusters=best_k, init="k-means++", n_init=10, max_iter=300, tol=0.0001, random_state=45).fit(df.drop(["country"], axis=1))
    df["Cluster"] = kmeans.labels_

    fig = px.scatter_3d(x=df["child_mort"], y=df["health"], z=df["inflation"], color=df["Cluster"])

    fig.update_layout(
        title="K-means Clustering",
        scene=dict(
            xaxis_title="child_mort",
            yaxis_title="health",
            zaxis_title="inflation",
        ),
    )

    st.header("Cluster Plot")
    st.plotly_chart(fig)
    st.write(df)

# Perform K-means clustering
perform_kmeans(clust)
