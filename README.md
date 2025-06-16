# E-Commerce Sales Analyzer

### Website Link: 
https://arshiii08-e-commerce-sales-analysis-dashboard-app-itzwbc.streamlit.app/

A Streamlit-powered dashboard for quick, interactive exploration of any e-commerce sales dataset. Upload a CSV/Excel file, map its columns, and get instant insights:

- **Key Metrics**: Total revenue, total orders, average order value  
- **Trends & Charts**:  
  - Monthly revenue time series  
  - Top N products by sales  
  - Category breakdown (if your data includes a `category` column)  
- **Advanced Analytics** (optional):  
  - 3-month sales forecast using SARIMA  
  - Daily anomaly detection with statistical rule-based method based on rolling mean and standard deviation
- **Exports**: Download cleaned data and summary tables as CSV  

---

## 🛠️ Tech Stack

- **Python 3.8+**  
- **Streamlit** for the web UI  
- **Pandas** for data manipulation  
- **Plotly** for interactive charts  
- **Statsmodels** (`SARIMAX`) for forecasting  
- **Rule-based Statistical (`rolling mean and standard deviation`)** for anomaly detection  

---

## 🚀 Quick Start

1. **Clone** the repo  
   ```bash
   git clone https://github.com/your-username/ecommerce-sales-analyzer.git
   cd ecommerce-sales-analyzer
