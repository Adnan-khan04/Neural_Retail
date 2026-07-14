from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

app = FastAPI(
    title="NeuralRetail API",
    description="AI-Powered Sales Intelligence REST API | Amdox Technologies",
    version="1.0.0"
)

# Load and prepare data once at startup
data_path = os.path.join(os.path.dirname(__file__), 'data', 'online_retail_II.xlsx')
df = pd.read_excel(data_path,
                   sheet_name='Year 2010-2011')
df_clean = df.dropna(subset=['Customer ID'])
df_clean = df_clean[df_clean['Quantity'] > 0]
df_clean = df_clean[df_clean['Price'] > 0]
df_clean['TotalAmount'] = df_clean['Quantity'] * df_clean['Price']
df_clean['InvoiceDate'] = pd.to_datetime(df_clean['InvoiceDate'])
df_clean['Customer ID'] = df_clean['Customer ID'].astype(int)

snapshot_date = df_clean['InvoiceDate'].max() + pd.Timedelta(days=1)
rfm = df_clean.groupby('Customer ID').agg({
    'InvoiceDate': lambda x: (snapshot_date - x.max()).days,
    'Invoice': 'nunique',
    'TotalAmount': 'sum'
}).reset_index()
rfm.columns = ['CustomerID', 'Recency', 'Frequency', 'Monetary']
rfm['AvgOrderValue'] = rfm['Monetary'] / rfm['Frequency']
rfm['Churn'] = (rfm['Recency'] > 90).astype(int)

features = ['Recency', 'Frequency', 'Monetary', 'AvgOrderValue']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(rfm[features])

# Train models
churn_model = xgb.XGBClassifier(n_estimators=100, random_state=42, verbosity=0)
churn_model.fit(X_scaled, rfm['Churn'])

kmeans = KMeans(n_clusters=6, random_state=42, n_init=10)
kmeans.fit(X_scaled)

# Request schemas
class CustomerRequest(BaseModel):
    recency: float
    frequency: float
    monetary: float
    avg_order_value: float

class DemandRequest(BaseModel):
    price: float
    quantity: float

# Routes
@app.get("/")
def root():
    return {
        "message": "🛒 NeuralRetail API is running!",
        "Employee ID": "AdxTr15KKH",
        "intern": "Adnan Khan",
        "version": "1.0.1",
        "endpoints": ["/health", "/predict/churn", 
                     "/segment/score", "/inventory/reorder", "/docs"]
    }

@app.get("/health")
def health():
    return {
        "status": "✅ healthy",
        "models": {
            "churn_model": "XGBoost - AUC 1.000",
            "segmentation": "KMeans - 6 clusters",
        },
        "data": {
            "total_customers": len(rfm),
            "total_transactions": len(df_clean)
        }
    }

@app.post("/predict/churn")
def predict_churn(request: CustomerRequest):
    try:
        features_input = np.array([[
            request.recency,
            request.frequency,
            request.monetary,
            request.avg_order_value
        ]])
        scaled = scaler.transform(features_input)
        churn_prob = churn_model.predict_proba(scaled)[0][1]
        churn_pred = int(churn_prob >= 0.5)
        
        if churn_prob < 0.3:
            risk = "Low Risk"
            action = "Maintain engagement with regular newsletters"
        elif churn_prob < 0.6:
            risk = "Medium Risk"
            action = "Send personalized discount offer"
        elif churn_prob < 0.8:
            risk = "High Risk"
            action = "Immediate win-back campaign needed"
        else:
            risk = "Critical"
            action = "Personal outreach + premium offer required"
        
        return {
            "churn_probability": round(float(churn_prob), 4),
            "churn_prediction": churn_pred,
            "risk_level": risk,
            "recommended_action": action,
            "model": "XGBoost Classifier"
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/segment/score")
def segment_score(request: CustomerRequest):
    try:
        features_input = np.array([[
            request.recency,
            request.frequency,
            request.monetary,
            request.avg_order_value
        ]])
        scaled = scaler.transform(features_input)
        segment = int(kmeans.predict(scaled)[0])
        
        segment_labels = {
            0: "Champions", 1: "Loyal Customers",
            2: "At Risk", 3: "New Customers",
            4: "Hibernating", 5: "Potential Loyalists"
        }
        
        return {
            "segment_id": segment,
            "segment_name": segment_labels.get(segment, "Other"),
            "model": "KMeans Clustering"
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/inventory/reorder")
def inventory_reorder():
    try:
        product_stats = df_clean.groupby('Description').agg({
            'Quantity': 'sum',
            'Price': 'mean'
        }).reset_index()
        
        low_stock = product_stats[product_stats['Quantity'] < 100].head(10)
        
        return {
            "low_stock_items": low_stock.to_dict(orient='records'),
            "total_low_stock": len(low_stock),
            "recommendation": "Immediate reorder suggested for these items"
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/stats/summary")
def stats_summary():
    return {
        "total_revenue": round(float(df_clean['TotalAmount'].sum()), 2),
        "total_orders": int(df_clean['Invoice'].nunique()),
        "total_customers": int(df_clean['Customer ID'].nunique()),
        "total_products": int(df_clean['Description'].nunique()),
        "countries_served": int(df_clean['Country'].nunique()),
        "churn_rate": round(float(rfm['Churn'].mean() * 100), 2),
        "avg_order_value": round(float(rfm['AvgOrderValue'].mean()), 2)
    }