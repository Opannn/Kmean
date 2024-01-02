import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.cluster import KMeans

df = pd.read_csv("Country-data.csv")

X = df.drop(["country"], axis=1)

st.header("isi dataset")
st.write(X)

k_values = list(range(1, 11))
inertia_values = []

for best_k in k_values:
    kmeans = KMeans(
        n_clusters=best_k,
        init="k-means++",
        n_init=10,
        max_iter=300,
        tol=0.0001,
        random_state=45,
    )
    kmeans.fit(X)
    inertia_values.append(kmeans.inertia_)

# Plot the elbow curve to find the optimal k value
plt.plot(k_values, inertia_values, marker="o")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Inertia")
plt.title("Elbow Curve")


st.set_option("deprecation.showPyplotGlobalUse", False)
elbo_plot = st.pyplot()

st.header("Nilai jumlah K")
clust = st.slider("Pilih jumlah cluster :", 2, 10, 3, 1)


def k_means(best_k):
    kmeans = KMeans(
        n_clusters=best_k,
        init="k-means++",
        n_init=10,
        max_iter=300,
        tol=0.0001,
        random_state=45,
    ).fit(X)
    X["Cluster"] = kmeans.labels_

    fig = px.scatter_3d(x=X["child_mort"], y=X["health"], z=X["inflation"], color=X["Cluster"])

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
    st.write(X)


k_means(clust)
