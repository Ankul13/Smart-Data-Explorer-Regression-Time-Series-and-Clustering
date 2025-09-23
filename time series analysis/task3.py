import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

st.set_page_config(page_title="🤖 Clustering Analysis", layout="wide")

st.title("🤖 K-Means Clustering")

# Upload File
uploaded_file = st.file_uploader("📂 Upload a CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file, low_memory=False)
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()

    if not numeric_cols:
        st.error("No numeric columns found!")
        st.stop()

    cluster_df = df[numeric_cols].dropna()
    scaler = StandardScaler()
    X = scaler.fit_transform(cluster_df)

    # Elbow method
    inertias = [KMeans(n_clusters=k, random_state=42, n_init=10).fit(X).inertia_ for k in range(1, 9)]
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.lineplot(x=range(1, 9), y=inertias, marker="o", ax=ax)
    ax.set_title("Elbow Method")
    st.pyplot(fig)

    # Choose number of clusters
    k = st.slider("Number of clusters", 2, 8, 3)
    labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(X)
    cluster_df["cluster"] = labels

    # PCA for visualization
    pca = PCA(n_components=2, random_state=42).fit_transform(X)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(x=pca[:, 0], y=pca[:, 1], hue=labels, palette="Set2", s=60, ax=ax)
    ax.set_title("Clusters (PCA Projection)")
    st.pyplot(fig)

    # Download clustered data
    st.download_button("📥 Download Clustered Data",
                       data=cluster_df.to_csv().encode("utf-8"),
                       file_name="clustered_results.csv",
                       mime="text/csv")
