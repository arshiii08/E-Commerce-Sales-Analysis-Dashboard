import pandas as pd
import difflib
import numpy as np
from prophet import Prophet
from sklearn.ensemble import IsolationForest
import plotly.express as px
import plotly.graph_objects as go

# Column Mapping Utilities 
SYNS = {
    "order_date": ["InvoiceDate", "OrderDate", "Date", "Order_Timestamp", "Order_TS"],
    "order_id": ["InvoiceNo", "OrderID", "Order_No", "TransID", "TransactionID"],
    "product_id": ["StockCode", "ProductID", "SKU", "ItemCode"],
    "unit_price": ["UnitPrice", "Price", "Unit_Price", "SalePrice"],
    "quantity": ["Quantity", "Qty", "UnitsSold", "Quantity_Sold"]
}

def find_best_match(col_list, candidates, cutoff=0.6):
    best_col=None
    best_score=0.0

    for real_col in col_list:
        for cand in candidates:         #candidates: list of synonyms for logical field 
            score=difflib.SequenceMatcher(None, real_col.lower(), cand.lower().ratio())
            if score>best_score:
                best_score=score
                best_col=real_col
    return best_col if best_score>=cutoff else None     #cutoff: min similarity to accept a match

def auto_map_columns(df: pd.DataFrame)->pd.DataFrame:
    cols=df.columns.tolist()
    mapping={}
    for logical, synonyms in SYNS.items():
        match=find_best_match(cols, synonyms)
        if not match:
            raise ValueError(f"Could not auto-map `{logical}` (tried {synonyms})")
        mapping[match]=logical
    return df.rename(columns=mapping)

def load_and_validate(df_raw: pd.DataFrame, manual_map: dict=None)->pd.DataFrame:
    """
    df_raw: raw data frame read from user uploaded csv/xlsx
    manual map: dict mapping if user did manual mapping . if none, attempt auto-map
    """
    #1. Mapping columns
    if manual_map:
        df=df_raw.rename(columns=manual_map).copy()
    else:
        df=auto_map_columns(df_raw).copy()
    
    #2. Convert order date to datetime
    df['order_date']=pd.to_datetime(df['order_date'], errors='coerce')
    df=df.dropna(subset=['order_date'])

    #3. convert unit_price and quantity to numeric
    df['unit_price']=pd.to_numeric(df['unit_price'], errors='coerce').fillna(0.0)
    df['quantity']=pd.to_numeric(df['quantity'], errors='coerce').fillna(0).astype(int)

    #4. compute revenue
    df['revenue']=df['unit_price']*df['quantity']

    return df

# Aggregation / EDA Functions

@pd.api.extensions.register_dataframe_accessor("eda")
class EDAAccessor:
    def __init__(self, pandas_obj):
        self._df=pandas_obj

    def compute_monthly_sales(self)->pd.DataFrame:
        df=self._df.copy()
        df['year_month']=df['order_date'].dt.to_period('M').astype(str)
        monthly=(
            df.groupby('year_month')
                .agg(
                    total_revenue=('revenue', 'sum'),
                    order_count=('order_id', 'nunique')
                )
                .reset_index()
        )
        monthly['avg_order_value']=monthly['total_revenue']/monthly['order_count']
        return monthly
    
    def top_products(self, n: int=10)->pd.DataFrame:
        df=self._df.copy()
        agg=(
            df.groupby('product_id')
            .agg(
                total_revenue=('revenue', 'sum'),
                total_quantity=('quantity', 'sum')
            )
            .reset_index()
        )
        return agg.sort_values('total_revenue', ascending=False).head(n)
    
    def sales_by_category(self)->pd.DataFrame:
        df=self._df.copy()
        if 'category' not in df.columns:
            return pd.DataFrame(columns=['category', 'total_revenue'])
        cat=(
            df.groupby('category')
              .agg(total_revenue=('revenue', 'sum'))
              .reset_index()
              .sort_values('total_revenue', ascending=False)
        )
        return cat
    
    def forecast_sales(self, periods: int=3, order=(1,1,1), seasonal_order=(1,1,1,12))->pd.DataFrame:
        # 1. prepare time series
        monthly=self.compute_monthly_sales()
        ts=(
             monthly
                .assign(ds=pd.to_datetime(monthly['year_month']+'-01'))
                .set_index('ds')['total_revenue']
        )

        # 2. fit sarimax 
        from statsmodels.tsa.statespace.sarimax import SARIMAX
        model=SARIMAX(
             ts, 
             order=order,
             seasonal_order=seasonal_order,
             enforce_stationarity=False,
             enforce_invertibility=False
        )
        res=model.fit(disp=False)

        # 3. Forecast
        fc=res.get_forecast(steps=periods)
        fc_df=fc.summary_frame(alpha=0.05)

        # rename columns to match existing interface
        fc_df=(
             fc_df[['mean', 'mean_ci_lower', 'mean_ci_upper']]
                .rename(columns={
                     'mean': 'yhat',
                     'mean_ci_lower': 'yhat_lower',
                     'mean_ci_upper': 'yhat_upper'
                })
                .reset_index()
                .rename(columns={'index': 'ds'})
        )
        return fc_df
    
    def detect_anomalies(self, freq: str='D', window: int=7, threshold: float=2.0)->pd.DataFrame:       # 'D': group by day, window: rolling window size, threshold: no of standard devaitions from mean to flag anomaly 
        df=self._df.set_index('order_date')['revenue'].resample(freq).sum().reset_index()
        df.rename(columns={'order_date': 'date'}, inplace=True)
        df['rolling_mean']=df['revenue'].rolling(window=window).mean()
        df['rolling_std']=df['revenue'].rolling(window=window).std()
        df['is_anomaly']=(
             (df['rolling_std'].notna())&
             (
                (df['revenue']>df['rolling_mean']+threshold*df['rolling_std']) |
                (df['revenue']<df['rolling_mean']-threshold*df['rolling_std'])
            )
        )
        return df
    
# Plotly chart functions 
    
def plot_monthly_sales(monthly_df: pd.DataFrame):
        fig = px.line(
            monthly_df,
            x='year_month',
            y='total_revenue',
            title='Monthly Total Revenue',
            labels={'year_month': 'Month', 'total_revenue': 'Revenue'}
        ) 
        fig.update_layout(xaxis_tickangle=-45)  # tilt the x axis names to avoid cluttering 
        return fig 
    
def plot_top_products(top_df: pd.DataFrame):
        fig=px.bar(
            top_df, 
            x='product_id',
            y='total_revenue',
            title='Top Products by Revenue',
            labels={'product_id': 'Product ID', 'total_revenue': 'Revenue'}
        )
        return fig 
    
def plot_category_breakdown(cat_df: pd.DataFrame):
        fig=px.pie(
            cat_df,
            names='category',
            values='total_revenue',
            title='Revenue Distribution by Category'
        )
        return fig 
    
def plot_forecast(forecast_df: pd.DataFrame, monthly_df: pd.DataFrame):     # Overlays historical vs forecasted revenue
        hist=monthly_df.copy()
        hist['ds']=pd.to_datetime(hist['year_month']+'-01')
        hist_plot=hist[['ds', 'total_revenue']].rename(columns={'total_revenue': 'yhat'})

        combined=pd.concat([
            hist_plot.assign(type='Actual'),
            forecast_df[['ds', 'yhat']].assign(type='Forecast')
        ])
        fig=px.line(
            combined,
            x='ds',
            y='yhat',
            color='type',
            title='Actual vs Forecasted Revenue',
            labels={'ds': 'Date', 'yhat': 'Revenue'}
        )
        return fig 
    
def plot_anomalies(anomaly_df: pd.DataFrame):       # highlights anomalies on revenue time series 
        anomaly_df=anomaly_df.copy()
        anomaly_df['date']=pd.to_datetime(anomaly_df['date'])

        fig=go.Figure()
        # revenue line 
        fig.add_trace(go.Scatter(
            x=anomaly_df['date'],
            y=anomaly_df['revenue'],
            mode='lines',
            name='Revenue',
            line=dict(color='skyblue')
        ))
        # rolling mean dashed line 
        fig.add_trace(go.Scatter(
             x=anomaly_df['date'],
             y=anomaly_df['rolling_mean'],
             mode='lines',
             name='7-Day Rolling Mean',
             line=dict(dash='dash', color='orange')
        ))

        #Anomaly points 
        anomalies=anomaly_df[anomaly_df['is_anomaly']]
        if not anomalies.empty:
            fig.add_trace(go.Scatter(
                x=anomalies['date'],
                y=anomalies['revenue'],
                mode='markers',
                marker=dict(color='red', size=8),
                name='Anomaly'
            ))
        
        fig.update_layout(
             title="Revenue with Anomalies and Rolling Mean",
             xaxis_title="Date",
             yaxis_title="Revenue",
             template="plotly_dark",
             hovermode="x unified"
        )
        
        return fig 



        
