import streamlit as st
import pandas as pd
from data_funcs import load_and_validate, plot_monthly_sales, plot_top_products, plot_category_breakdown, plot_forecast, plot_anomalies

# Page config
st.set_page_config(
    page_title="E-Commerce Sales Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title
st.title("🛒 E-Commerce Sales Dashboard")
st.write("Upload your sales data to explore key metrics, trends, and anomalies.")

# 1. Upload data
uploaded = st.file_uploader(
    "Upload CSV or Excel",
    type=["csv", "xlsx"]
)
if not uploaded:
    st.stop()

# 2. Read raw DataFrame
try:
    if uploaded.name.lower().endswith('.csv'):
        df_raw = pd.read_csv(uploaded, encoding='latin1')
    else:
        df_raw = pd.read_excel(uploaded)
except Exception as e:
    st.error(f"Error reading file: {e}")
    st.stop()

st.subheader("Raw columns detected")
st.write(df_raw.columns.tolist())

# 3. Ingest and validate (auto-mapping)
manual_map = None
try:
    df = load_and_validate(df_raw)
    st.success("✅ Columns auto-detected and data cleaned.")
except Exception as e:
    st.warning(f"Auto-detection failed: {e}")
    st.sidebar.header("Manual Column Mapping")
    cols = df_raw.columns.tolist()
    date_col = st.sidebar.selectbox("Date column", cols)
    id_col = st.sidebar.selectbox("Order ID column", cols)
    prod_col = st.sidebar.selectbox("Product ID column", cols)
    price_col = st.sidebar.selectbox("Unit Price column", cols)
    qty_col = st.sidebar.selectbox("Quantity column", cols)
    # build manual mapping
    manual_map = {
        date_col: 'order_date',
        id_col: 'order_id',
        prod_col: 'product_id',
        price_col: 'unit_price',
        qty_col: 'quantity'
    }
    df = load_and_validate(df_raw, manual_map=manual_map)
    st.success("✅ Manual mapping applied and data cleaned.")

# Preview cleaned data
st.subheader("Cleaned Data Preview")
st.dataframe(df.head(5))

# 4. Compute summary tables
monthly_df = df.eda.compute_monthly_sales()
top10_df = df.eda.top_products(10)
cat_df = df.eda.sales_by_category()

# 5. Display key metrics
st.markdown("### Key Metrics")
col1, col2, col3 = st.columns(3)
col1.metric("Total Revenue", f"₹{df['revenue'].sum():,.2f}")
col2.metric("Total Orders", f"{df['order_id'].nunique():,}")
col3.metric("Average Order Value", f"₹{df['revenue'].sum()/df['order_id'].nunique():,.2f}")

# 6. Plots
st.markdown("---")
st.subheader("Monthly Revenue Trend")
fig1 = plot_monthly_sales(monthly_df)
st.plotly_chart(fig1, use_container_width=True)

st.subheader("Top 10 Products by Revenue")
fig2 = plot_top_products(top10_df)
st.plotly_chart(fig2, use_container_width=True)

if not cat_df.empty:
    st.subheader("Revenue by Category")
    fig3 = plot_category_breakdown(cat_df)
    st.plotly_chart(fig3, use_container_width=True)

# 7. Forecast & anomalies
with st.expander("Advanced Insights", expanded=False):
    if st.button("Run 3-Month Forecast"):
        st.info("Training forecast model…")
        fc = df.eda.forecast_sales(periods=3)
        fig_fc = plot_forecast(fc, monthly_df)
        st.plotly_chart(fig_fc, use_container_width=True)

    if st.button("Detect Anomalies (Daily)"):
        st.info("Detecting anomalies…")
        anom = df.eda.detect_anomalies(freq='D', window=7, threshold=2.0)
        fig_anom = plot_anomalies(anom)
        st.plotly_chart(fig_anom, use_container_width=True)

# 8. Downloads
st.markdown("---")
st.subheader("Download Data")
c1, c2 = st.columns(2)
with c1:
    st.download_button(
        "Download Cleaned Data",
        df.to_csv(index=False).encode('utf-8'),
        file_name='cleaned_data.csv',
        mime='text/csv'
    )
with c2:
    st.download_button(
        "Download Top 10 Products",
        top10_df.to_csv(index=False).encode('utf-8'),
        file_name='top_10_products.csv',
        mime='text/csv'
    )
