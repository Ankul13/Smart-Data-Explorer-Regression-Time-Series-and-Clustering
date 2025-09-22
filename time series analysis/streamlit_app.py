import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from statsmodels.tsa.seasonal import STL, seasonal_decompose
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

st.set_page_config(page_title="Data Analysis  Dashboard", layout="wide")

# ------------------ File Upload ------------------
st.title("🚀  Data Analysis Dashboard")
uploaded_file = st.file_uploader("📂 Upload a CSV file", type=["csv"])

if not uploaded_file:
    st.info("👆 Upload a dataset to unlock analysis features")
    st.stop()

# ------------------ Load Data (cached) ------------------
@st.cache_data
def load_data(file):
    return pd.read_csv(file, low_memory=False)

df_raw = load_data(uploaded_file)
numeric_cols = df_raw.select_dtypes(include=[np.number]).columns.tolist()
if not numeric_cols:
    st.error("No numeric columns found in dataset!")
    st.stop()

# ------------------ Sidebar ------------------
st.sidebar.title("🔍 Navigation")
menu = st.sidebar.radio("Go to", ["🏠 Overview", "📈 Regression", "⏳ Time Series", "🤖 Clustering"])

# ------------------ Overview ------------------
if menu == "🏠 Overview":
    st.subheader("📘 Dataset Preview")
    st.dataframe(df_raw.head(10))
    st.write(f"Rows: {df_raw.shape[0]}, Columns: {df_raw.shape[1]}")

# ------------------ Regression ------------------
elif menu == "📈 Regression":
    st.subheader("📈 Simple Linear Regression")

    col1, col2 = st.columns(2)
    with col1:
        x_col = st.selectbox("Predictor (X)", numeric_cols)
    with col2:
        y_col = st.selectbox("Target (Y)", numeric_cols)

    if x_col != y_col:
        df = df_raw[[x_col, y_col]].dropna()

        # 🚀 Cache regression fit
        @st.cache_resource
        def fit_regression(X, y):
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            model = LinearRegression().fit(X_train, y_train)
            y_pred = model.predict(X_test)
            return model, X_test, y_test, y_pred

        model, X_test, y_test, y_pred = fit_regression(df[[x_col]], df[y_col])

        # Metrics
        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)

        st.metric("Coefficient", f"{model.coef_[0]:.4f}")
        st.metric("R² Score", f"{r2:.4f}")
        st.metric("MSE", f"{mse:.4f}")

        # 🚀 Limit scatterplot size for speed
        plot_df = pd.DataFrame({x_col: X_test[x_col], "Actual": y_test, "Predicted": y_pred})
        if len(plot_df) > 3000:
            plot_df = plot_df.sample(3000, random_state=42)

        # -------- Graph 1: Regression Fit --------
        fig1, ax1 = plt.subplots(figsize=(8, 5))
        sns.scatterplot(data=plot_df, x=x_col, y="Actual", alpha=0.6, ax=ax1, label="Actual")
        sns.lineplot(data=plot_df, x=x_col, y="Predicted", color="red", ax=ax1, label="Predicted")
        ax1.set_title("Regression Fit (X vs Y)")
        ax1.legend()
        st.pyplot(fig1)

        # -------- Graph 2: Actual vs Predicted --------
        fig2, ax2 = plt.subplots(figsize=(6, 6))
        sns.scatterplot(x=y_test, y=y_pred, alpha=0.6, ax=ax2)
        ax2.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--")  # 45-degree line
        ax2.set_xlabel("Actual Values")
        ax2.set_ylabel("Predicted Values")
        ax2.set_title("Actual vs Predicted")
        st.pyplot(fig2)

# ------------------ Time Series ------------------
elif menu == "⏳ Time Series":
    st.subheader("⏳ Time Series Analysis")

    # Detect date column
    date_col = next((c for c in df_raw.columns if "date" in c.lower() or c.lower() in ["timestamp", "time"]), None)
    df = df_raw.copy()
    if date_col:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.set_index(date_col).sort_index()
        df = df.groupby(df.index).mean(numeric_only=True)
        st.success(f"✅ Using **{date_col}** as time index")
    else:
        df.index = pd.RangeIndex(len(df))

    ts_col = st.selectbox("Choose Time Series Column", numeric_cols)
    ts = df[ts_col].dropna()

    # 🚀 Subsample for plotting
    if len(ts) > 5000:
        ts = ts.sample(5000, random_state=42)

    fig, ax = plt.subplots(figsize=(10, 4))
    sns.lineplot(x=ts.index, y=ts.values, ax=ax, color="darkblue")
    st.pyplot(fig)

    # STL Decomposition (cached)
    @st.cache_resource
    def decompose_series(series, period):
        try:
            return STL(series, period=period, robust=True).fit()
        except:
            return seasonal_decompose(series, model="additive", period=period)

    period = 252 if len(ts) >= 252 else (21 if len(ts) >= 60 else max(2, len(ts)//5))
    result = decompose_series(ts, period)
    fig = result.plot()
    fig.set_size_inches(10, 6)
    st.pyplot(fig)

# ------------------ Clustering ------------------
elif menu == "🤖 Clustering":
    st.subheader("🤖 K-Means Clustering")

    cluster_df = df_raw[numeric_cols].dropna()
    scaler = StandardScaler()
    X = scaler.fit_transform(cluster_df)

    # 🚀 Cache inertia computation
    @st.cache_resource
    def compute_inertia(X):
        return [KMeans(n_clusters=k, random_state=42, n_init=10).fit(X).inertia_ for k in range(1, 9)]

    inertias = compute_inertia(X)

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.lineplot(x=range(1, 9), y=inertias, marker="o", ax=ax)
    st.pyplot(fig)

    k = st.slider("Number of clusters", 2, 8, 3)
    labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(X)
    cluster_df["cluster"] = labels

    # 🚀 Cache PCA
    @st.cache_resource
    def fit_pca(X):
        return PCA(n_components=2, random_state=42).fit_transform(X)

    pca = fit_pca(X)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(x=pca[:, 0], y=pca[:, 1], hue=labels, palette="Set2", s=60, ax=ax)
    st.pyplot(fig)

    st.download_button("📥 Download Clustered Data",
                       data=cluster_df.to_csv().encode("utf-8"),
                       file_name="clustered_results.csv",
                       mime="text/csv")
