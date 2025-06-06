import streamlit as st
import pandas as pd
import altair as alt

st.set_page_config(page_title="E-Commerce Dashboard", layout="wide")

st.title("E-Commerce Sales Analysis Dashboard")
st.write("Upload a CSV file to explore your sales data with interactive visualizations.")

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("Data Preview")
    st.dataframe(df.head())

    # attempt to parse date columns
    for col in df.columns:
        if df[col].dtype == object:
            try:
                df[col] = pd.to_datetime(df[col])
            except Exception:
                pass

    date_cols = df.select_dtypes(include=["datetime64[ns]"]).columns.tolist()
    numeric_cols = df.select_dtypes(include="number").columns.tolist()

    if date_cols and numeric_cols:
        st.sidebar.header("Chart Options")
        date_col = st.sidebar.selectbox("Date column", date_cols)
        value_col = st.sidebar.selectbox("Numeric column", numeric_cols)
        freq = st.sidebar.selectbox(
            "Aggregation", ["D", "M", "Y"],
            format_func=lambda x: {"D": "Daily", "M": "Monthly", "Y": "Yearly"}[x]
        )
        df_group = df.set_index(date_col).resample(freq)[value_col].sum().reset_index()
        chart = alt.Chart(df_group).mark_line(point=True).encode(
            x=date_col,
            y=value_col,
            tooltip=[date_col, value_col]
        ).properties(width=800, height=400)
        st.altair_chart(chart, use_container_width=True)
    else:
        st.warning("Dataset must contain at least one date column and one numeric column")
else:
    st.info("Awaiting CSV file to be uploaded.")
