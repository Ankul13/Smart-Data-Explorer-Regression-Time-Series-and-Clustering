import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from statsmodels.tsa.seasonal import STL, seasonal_decompose

st.set_page_config(page_title="⏳ Time Series Analysis", layout="wide")

st.title("⏳ Time Series Analysis")

# Upload File
uploaded_file = st.file_uploader("📂 Upload a CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file, low_memory=False)

    # Detect date column
    date_col = next((c for c in df.columns if "date" in c.lower() or c.lower() in ["timestamp", "time"]), None)
    if date_col:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.set_index(date_col).sort_index()
        df = df.groupby(df.index).mean(numeric_only=True)
        st.success(f"✅ Using **{date_col}** as time index")
    else:
        df.index = pd.RangeIndex(len(df))

    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    ts_col = st.selectbox("Choose Time Series Column", numeric_cols)
    ts = df[ts_col].dropna()

    # Plot series
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.lineplot(x=ts.index, y=ts.values, ax=ax, color="darkblue")
    ax.set_title("Time Series Plot")
    st.pyplot(fig)

    # STL decomposition
    period = 252 if len(ts) >= 252 else (21 if len(ts) >= 60 else max(2, len(ts)//5))
    try:
        result = STL(ts, period=period, robust=True).fit()
    except:
        result = seasonal_decompose(ts, model="additive", period=period)

    fig = result.plot()
    fig.set_size_inches(10, 6)
    st.pyplot(fig)
