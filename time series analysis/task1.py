import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

st.set_page_config(page_title="📈 Regression Analysis", layout="wide")

st.title("📈 Simple Linear Regression")

# Upload File
uploaded_file = st.file_uploader("📂 Upload a CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file, low_memory=False)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if not numeric_cols:
        st.error("No numeric columns found!")
        st.stop()

    col1, col2 = st.columns(2)
    with col1:
        x_col = st.selectbox("Predictor (X)", numeric_cols)
    with col2:
        y_col = st.selectbox("Target (Y)", numeric_cols)

    if x_col != y_col:
        df = df[[x_col, y_col]].dropna()
        X_train, X_test, y_train, y_test = train_test_split(df[[x_col]], df[y_col], test_size=0.2, random_state=42)

        model = LinearRegression().fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Metrics
        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)

        st.metric("Coefficient", f"{model.coef_[0]:.4f}")
        st.metric("R² Score", f"{r2:.4f}")
        st.metric("MSE", f"{mse:.4f}")

        # Graph 1: Regression Fit
        fig1, ax1 = plt.subplots(figsize=(8, 5))
        sns.scatterplot(x=X_test[x_col], y=y_test, alpha=0.6, ax=ax1, label="Actual")
        sns.lineplot(x=X_test[x_col], y=y_pred, color="red", ax=ax1, label="Predicted")
        ax1.set_title("Regression Fit")
        st.pyplot(fig1)

        # Graph 2: Actual vs Predicted
        fig2, ax2 = plt.subplots(figsize=(6, 6))
        sns.scatterplot(x=y_test, y=y_pred, alpha=0.6, ax=ax2)
        ax2.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--")
        ax2.set_xlabel("Actual")
        ax2.set_ylabel("Predicted")
        ax2.set_title("Actual vs Predicted")
        st.pyplot(fig2)
