# 🚖 Uber Fare Prediction & Dynamic Pricing Analytics

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_svg?style=flat)](https://uberfare-regression-model-dktxuuf9grkdyyzbtphwar.streamlit.app/)
[![GitHub Repo](https://img.shields.io/badge/GitHub-Repository-blue?logo=github)](https://github.com/aadarshrdeshmukh/UberFare-Regression-Model)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A comprehensive data science and machine learning project that predicts Uber ride fares and analyzes dynamic pricing patterns in NYC. This project integrates three distinct machine learning models to provide real-time fare estimates, surge classification, and ride segmentation.

🔗 **Live Demo:** [Uber Analytics Dashboard](https://uberfare-regression-model-dktxuuf9grkdyyzbtphwar.streamlit.app/)

---

## � Business Problem Statement

In the hyper-competitive ride-hailing industry, **Pricing Optimization** is the primary lever for balancing supply and demand. Uber faces two critical challenges:

1.  **Revenue Loss from Static Pricing**: Fixed-rate pricing fails to account for traffic congestion, weather, and peak-hour demand, leading to missed revenue opportunities.
2.  **Supply-Demand Mismatch**: Without dynamic "surge" pricing, drivers have little incentive to operate in high-traffic or late-night zones, leading to long wait times and customer churn.

**Project Objective**: To build an intelligent pricing engine that accurately predicts fares, classifies high-demand (surge) scenarios, and segments consumers to optimize fleet allocation and maximize revenue per mile.

---

## 📈 Economic Concepts Applied

This project translates core economic theories into algorithmic logic:

*   **Dynamic Pricing (Surge)**: Implementing the concept of *Price Elasticity of Demand*. During peak hours, prices increase to satisfy the most urgent demand while simultaneously incentivizing the *Supply Side* (drivers) to enter the market.
*   **Price Discrimination (Segmentation)**: Using K-Means to identify distinct consumer clusters. By understanding which rides are "Premium" vs. "Budget," Uber can tailor marketing strategies and service levels.
*   **Temporal & Spatial Arbitrage**: Analyzing how value fluctuates based on *Time* (rush hour vs. midnight) and *Location* (Pickup Boroughs). The model captures the premium associated with high-value origins like Airports.
*   **Cost-Plus vs. Value-Based Pricing**: While the base fare follows a cost-plus model (distance + time), our XGBoost model incorporates value-based variables (peak status, location) to find the "Equilibrium Price."

---

## 🤖 AI & Machine Learning Techniques

The project utilizes a **Tri-Model Architecture** to provide holistic analytics:

### 1. Regression (XGBoost) - "The Price Predictor"
*   **Algorithm**: Extreme Gradient Boosting (XGBoost).
*   **Technique**: Optimized for minimizing Root Mean Squared Error (RMSE) on continuous fare data.
*   **Engine**: Handles non-linear relationships between distance, time, and fare better than standard Linear Regression.

### 2. Classification (Logistic Regression) - "The Surge Detector"
*   **Algorithm**: Logistic Regression with Standard Scaling.
*   **Technique**: Binary classification to predict if a ride falls into the top 25% of fare amounts (Surge/High-Fare).
*   **Output**: Provides a probability score (0-100%) for demand intensity.

### 3. Clustering (K-Means) - "The Market Segmenter"
*   **Algorithm**: K-Means Clustering (Unsupervised Learning).
*   **Technique**: Segregates rides into 4 tiers (Budget, Standard, Premium, Airport) based on fare-per-km, total fare, and trip distance.
*   **Business Use**: Helps in understanding the "Nature of the Trip" without manual labeling.

---

## �🌟 Key Features

The interactive dashboard (built with Streamlit) offers a 360-degree view of ride economics:

*   **💰 Multi-Model Fare Prediction**: Real-time estimates using a tuned **XGBoost** regressor, adjusted for distance, time, and location.
*   **⚡ Surge Classification**: A **Logistic Regression** model that predicts the probability of a ride being "High-Fare" based on demand patterns.
*   **🏷️ Ride Segmentation**: **K-Means Clustering** categorizes rides into four tiers.
*   **📊 Pricing Analytics**: Deep dive into NYC ride data, visualizing fare distributions and distance-price correlations.
*   **⏰ Demand Heatmaps**: Analysis of peak vs. off-peak hours and weekday vs. weekend patterns.

---

## 🧠 Model Performance Comparison

| Model | R² Score | RMSE | MAE |
| :--- | :--- | :--- | :--- |
| Linear Regression | ~0.62 | ~$6.80 | ~$4.50 |
| Random Forest | ~0.77 | ~$4.60 | ~$2.90 |
| **XGBoost (Tuned)** | **~0.84** | **~$3.90** | **~$2.40** |

---

## 📁 Project Structure

```text
.
├── app.py                  # Main Streamlit dashboard application
├── uber_fare_notebook.ipynb # Full research, EDA, and model training notebook
├── requirements.txt        # Project dependencies
├── uber_fare_model.pkl     # Pre-trained XGBoost Regressor
├── uber_surge_model.pkl    # Pre-trained Logistic Regression Classifier
├── uber_cluster_model.pkl  # Pre-trained K-Means Clusterer
└── cleaned_sample.csv      # Processed dataset for dashboard analytics
```

---

## 🚀 Installation & Local Setup

### 1. Clone the repository
```bash
git clone https://github.com/aadarshrdeshmukh/UberFare-Regression-Model.git
cd UberFare-Regression-Model
```

### 2. Set up a Virtual Environment (Recommended)
```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Dashboard
```bash
streamlit run app.py
```

---

## 📌 Dataset Source
The model is trained on the [Uber Fares Dataset — Kaggle (yasserh)](https://www.kaggle.com/datasets/yasserh/uber-fares-dataset), containing NYC pickup/dropoff coordinates, passenger counts, and fare amounts.

---
**Developed by [Aadarsh Deshmukh](https://github.com/aadarshrdeshmukh)**
