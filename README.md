# 🛒 NeuralRetail - AI Sales Intelligence Platform
### Amdox Technologies | AMX-DS-2026-04 | April 2026

![Python](https://img.shields.io/badge/Python-3.12-blue)
![MLflow](https://img.shields.io/badge/MLflow-3.13-orange)
![XGBoost](https://img.shields.io/badge/XGBoost-AUC_1.000-green)
![Prophet](https://img.shields.io/badge/Prophet-Forecasting-red)

## 🎯 Project Overview
NeuralRetail is an end-to-end AI-powered sales intelligence platform built for Amdox Technologies. It delivers demand forecasting, customer churn prediction, segmentation, and inventory optimization through an interactive dashboard and REST API.

## 🏆 Key Results
| Model | Metric | Score | Target |
|-------|--------|-------|--------|
| Churn Prediction (XGBoost) | AUC-ROC | 1.000 | ≥ 0.90 ✅ |
| Churn Prediction (LightGBM) | AUC-ROC | 1.000 | ≥ 0.90 ✅ |
| Demand Forecast (Prophet) | MAPE | 32.88% | ≤ 10% |
| Ensemble Forecast | MAPE | 25.30% | ≤ 10% |
| Customer Segmentation | Silhouette | Optimized | ≥ 0.55 |

## 🚀 Features
- **F01** Multi-source data ingestion & ETL
- **F02** Advanced customer segmentation (K-Means, DBSCAN, GMM)
- **F03** Demand forecasting (Prophet + LSTM ensemble)
- **F04** Churn prediction (XGBoost + LightGBM + SHAP)
- **F05** Revenue & price intelligence
- **F06** Inventory optimization
- **F07** Interactive 5-page Streamlit dashboard
- **F08** MLflow experiment tracking

## 🛠️ Tech Stack
- **Language:** Python 3.12
- **ML:** XGBoost, LightGBM, Prophet, PyTorch LSTM
- **Explainability:** SHAP
- **Dashboard:** Streamlit + Plotly
- **API:** FastAPI + Uvicorn
- **Tracking:** MLflow
- **Data:** Pandas, NumPy, Scikit-learn

## 📁 Project Structure