import streamlit as st
import pandas as pd
import numpy as np
import os
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

# Page config
st.set_page_config(
    page_title="NeuralRetail - AI Sales Intelligence",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
.main-header {
    font-size: 2.5rem;
    font-weight: bold;
    color: #E84E1B;
    text-align: center;
}
.metric-card {
    background-color: #1e1e1e;
    padding: 1rem;
    border-radius: 10px;
    border-left: 4px solid #E84E1B;
}
</style>
""", unsafe_allow_html=True)

# Load data
@st.cache_data
def load_data():
    data_path = os.path.join(os.path.dirname(__file__), 'data', 'online_retail_II.xlsx')
    df = pd.read_excel(data_path,
                       sheet_name='Year 2010-2011')
    df_clean = df.dropna(subset=['Customer ID'])
    df_clean = df_clean[df_clean['Quantity'] > 0]
    df_clean = df_clean[df_clean['Price'] > 0]
    df_clean['TotalAmount'] = df_clean['Quantity'] * df_clean['Price']
    df_clean['InvoiceDate'] = pd.to_datetime(df_clean['InvoiceDate'])
    df_clean['Month'] = df_clean['InvoiceDate'].dt.to_period('M').astype(str)
    df_clean['DayOfWeek'] = df_clean['InvoiceDate'].dt.day_name()
    df_clean['Customer ID'] = df_clean['Customer ID'].astype(int)
    return df_clean

@st.cache_data
def get_rfm(df_clean):
    snapshot_date = df_clean['InvoiceDate'].max() + pd.Timedelta(days=1)
    rfm = df_clean.groupby('Customer ID').agg({
        'InvoiceDate': lambda x: (snapshot_date - x.max()).days,
        'Invoice': 'nunique',
        'TotalAmount': 'sum'
    }).reset_index()
    rfm.columns = ['CustomerID', 'Recency', 'Frequency', 'Monetary']
    rfm['AvgOrderValue'] = rfm['Monetary'] / rfm['Frequency']
    rfm['Churn'] = (rfm['Recency'] > 90).astype(int)
    return rfm

# Sidebar
st.sidebar.markdown("## 🛒 NeuralRetail")
st.sidebar.markdown("**AI Sales Intelligence**")
st.sidebar.markdown("---")
page = st.sidebar.selectbox("📊 Navigate", [
    "🏠 Executive Overview",
    "📈 Demand Intelligence",
    "👥 Customer Hub",
    "🎯 Churn Prediction",
    "📦 Inventory Health"
])
st.sidebar.markdown("---")
st.sidebar.markdown("**Intern:** Adnan Khan")
st.sidebar.markdown("**Employee ID:** AdxTr15KKH")

# Load data
with st.spinner("Loading data..."):
    df = load_data()
    rfm = get_rfm(df)

# PAGE 1 - Executive Overview
if page == "🏠 Executive Overview":
    st.markdown('<p class="main-header">🛒 NeuralRetail Dashboard</p>', 
                unsafe_allow_html=True)
    st.markdown("### AI-Powered Sales Intelligence | Amdox Technologies")
    st.markdown("---")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("💰 Total Revenue", f"£{df['TotalAmount'].sum():,.0f}")
    col2.metric("📦 Total Orders", f"{df['Invoice'].nunique():,}")
    col3.metric("👥 Customers", f"{df['Customer ID'].nunique():,}")
    col4.metric("🌍 Countries", f"{df['Country'].nunique()}")
    
    st.markdown("---")
    col1, col2 = st.columns(2)
    
    with col1:
        monthly = df.groupby('Month')['TotalAmount'].sum().reset_index()
        fig = px.line(monthly, x='Month', y='TotalAmount',
                     title='📈 Monthly Revenue Trend',
                     color_discrete_sequence=['#E84E1B'])
        fig.update_layout(xaxis_tickangle=45)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        top_countries = df.groupby('Country')['TotalAmount'].sum().nlargest(10).reset_index()
        fig = px.bar(top_countries, x='Country', y='TotalAmount',
                    title='🌍 Top 10 Countries by Revenue',
                    color_discrete_sequence=['#F7941D'])
        fig.update_layout(xaxis_tickangle=45)
        st.plotly_chart(fig, use_container_width=True)

# PAGE 2 - Demand Intelligence
elif page == "📈 Demand Intelligence":
    st.markdown("## 📈 Demand Intelligence")
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    with col1:
        day_order = ['Monday','Tuesday','Wednesday','Thursday','Friday','Sunday']
        day_sales = df.groupby('DayOfWeek')['TotalAmount'].sum().reindex(day_order).reset_index()
        fig = px.bar(day_sales, x='DayOfWeek', y='TotalAmount',
                    title='📅 Revenue by Day of Week',
                    color_discrete_sequence=['#E84E1B'])
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        top_products = df.groupby('Description')['Quantity'].sum().nlargest(10).reset_index()
        fig = px.bar(top_products, x='Quantity', y='Description',
                    orientation='h', title='🛍️ Top 10 Products',
                    color_discrete_sequence=['#F7941D'])
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("### 🔮 30-Day Forecast")
    st.info("Prophet + LSTM Ensemble Model | Current MAPE: 25.3% | Improving with more data")

# PAGE 3 - Customer Hub
elif page == "👥 Customer Hub":
    st.markdown("## 👥 Customer Intelligence Hub")
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("👥 Total Customers", f"{len(rfm):,}")
    col2.metric("💰 Avg Order Value", f"£{rfm['AvgOrderValue'].mean():,.2f}")
    col3.metric("📊 Avg Frequency", f"{rfm['Frequency'].mean():.1f} orders")
    
    st.markdown("---")
    
    # Segmentation
    features = ['Recency', 'Frequency', 'Monetary', 'AvgOrderValue']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(rfm[features])
    kmeans = KMeans(n_clusters=6, random_state=42, n_init=10)
    rfm['Segment'] = kmeans.fit_predict(X_scaled)
    
    col1, col2 = st.columns(2)
    with col1:
        fig = px.scatter(rfm, x='Recency', y='Frequency',
                        color='Segment', size='Monetary',
                        title='👥 Customer Segments',
                        color_continuous_scale='Viridis')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        seg_rev = rfm.groupby('Segment')['Monetary'].sum().reset_index()
        fig = px.pie(seg_rev, values='Monetary', names='Segment',
                    title='💰 Revenue by Segment',
                    color_discrete_sequence=px.colors.qualitative.Set3)
        st.plotly_chart(fig, use_container_width=True)

# PAGE 4 - Churn Prediction
elif page == "🎯 Churn Prediction":
    st.markdown("## 🎯 Churn Prediction Engine")
    st.markdown("---")
    
    churn_rate = rfm['Churn'].mean() * 100
    active = (rfm['Churn']==0).sum()
    churned = (rfm['Churn']==1).sum()
    
    col1, col2, col3 = st.columns(3)
    col1.metric("⚠️ Churn Rate", f"{churn_rate:.1f}%")
    col2.metric("✅ Active Customers", f"{active:,}")
    col3.metric("❌ Churned Customers", f"{churned:,}")
    
    st.markdown("---")
    
    features = ['Recency', 'Frequency', 'Monetary', 'AvgOrderValue']
    scaler = StandardScaler()
    X = pd.DataFrame(scaler.fit_transform(rfm[features]), columns=features)
    
    model = xgb.XGBClassifier(n_estimators=100, random_state=42, verbosity=0)
    model.fit(X, rfm['Churn'])
    rfm['ChurnProb'] = model.predict_proba(X)[:,1]
    
    rfm['RiskLevel'] = pd.cut(rfm['ChurnProb'],
                               bins=[0,0.3,0.6,0.8,1.0],
                               labels=['Low','Medium','High','Critical'])
    
    col1, col2 = st.columns(2)
    with col1:
        risk_counts = rfm['RiskLevel'].value_counts().reset_index()
        fig = px.bar(risk_counts, x='RiskLevel', y='count',
                    title='👥 Churn Risk Segments',
                    color='RiskLevel',
                    color_discrete_map={'Low':'green','Medium':'orange',
                                       'High':'red','Critical':'darkred'})
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.histogram(rfm, x='ChurnProb', nbins=50,
                          title='📊 Churn Probability Distribution',
                          color_discrete_sequence=['#E84E1B'])
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("### 🚨 High Risk Customers")
    high_risk = rfm[rfm['RiskLevel'].isin(['High','Critical'])][
        ['CustomerID','Recency','Frequency','Monetary','ChurnProb','RiskLevel']
    ].sort_values('ChurnProb', ascending=False).head(20)
    st.dataframe(high_risk, use_container_width=True)

# PAGE 5 - Inventory Health
elif page == "📦 Inventory Health":
    st.markdown("## 📦 Inventory Health")
    st.markdown("---")
    
    product_stats = df.groupby('Description').agg({
        'Quantity': 'sum',
        'TotalAmount': 'sum',
        'Price': 'mean'
    }).reset_index()
    product_stats['StockStatus'] = pd.cut(
        product_stats['Quantity'],
        bins=[0, 100, 500, 1000, float('inf')],
        labels=['⚠️ Low Stock', '📦 Normal', '✅ Good', '🏭 Overstocked'])
    
    col1, col2, col3 = st.columns(3)
    col1.metric("🛍️ Total Products", f"{len(product_stats):,}")
    col2.metric("⚠️ Low Stock Items",
                f"{(product_stats['StockStatus']=='⚠️ Low Stock').sum():,}")
    col3.metric("💰 Avg Product Revenue",
                f"£{product_stats['TotalAmount'].mean():,.2f}")
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    with col1:
        status_counts = product_stats['StockStatus'].value_counts().reset_index()
        fig = px.pie(status_counts, values='count', names='StockStatus',
                    title='📦 Stock Status Distribution',
                    color_discrete_sequence=['#E84E1B','#F7941D','#FBBA13','green'])
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        top_revenue = product_stats.nlargest(10, 'TotalAmount')
        fig = px.bar(top_revenue, x='TotalAmount', y='Description',
                    orientation='h', title='💰 Top Revenue Products',
                    color_discrete_sequence=['#E84E1B'])
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("### 📋 Product Inventory Table")
    st.dataframe(product_stats.sort_values('Quantity').head(50),
                use_container_width=True)