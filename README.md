# 🛡️ GraphShield AI

> AI-Powered Fraud Detection using Machine Learning, Graph Neural Networks, FastAPI and Streamlit.

![Python](https://img.shields.io/badge/Python-3.13-blue)
![XGBoost](https://img.shields.io/badge/XGBoost-ML-success)
![PyTorch](https://img.shields.io/badge/PyTorch-GNN-red)
![FastAPI](https://img.shields.io/badge/FastAPI-API-green)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-orange)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

# 📌 Project Overview

GraphShield AI is an end-to-end fraud detection system that combines **Machine Learning** and **Graph Neural Networks (GNNs)** to detect suspicious financial transactions.

The project analyzes transaction data, constructs a customer transaction graph, identifies fraud rings, computes graph-based risk scores, trains multiple ML/GNN models, explains predictions using SHAP, and serves predictions through a FastAPI backend with an interactive Streamlit dashboard.

---

# 🚀 Features

- ✅ Fraud Detection using Random Forest
- ✅ Fraud Detection using XGBoost
- ✅ Graph Convolutional Network (GCN)
- ✅ GraphSAGE
- ✅ Customer Transaction Graph
- ✅ Fraud Ring Detection
- ✅ Graph Analytics
- ✅ Customer Risk Scoring
- ✅ SHAP Explainability
- ✅ FastAPI REST API
- ✅ Swagger Documentation
- ✅ Streamlit Dashboard
- ✅ Live Fraud Prediction

---

# 🏗 Project Architecture

```
                 PaySim Dataset
                        │
                        ▼
              Data Validation & EDA
                        │
                        ▼
          Customer Transaction Graph
                        │
      ┌─────────────────┴─────────────────┐
      ▼                                   ▼
Graph Features                  Fraud Ring Detection
      │                                   │
      └───────────────┬───────────────────┘
                      ▼
          Customer Feature Engineering
                      │
      ┌───────────────┴──────────────────┐
      ▼                                  ▼
Traditional ML                    Graph Neural Networks
(Random Forest)                     (GCN, GraphSAGE)
(XGBoost)
      │                                  │
      └───────────────┬──────────────────┘
                      ▼
             Model Comparison
                      ▼
          SHAP Explainability
                      ▼
           FastAPI Prediction API
                      ▼
           Streamlit Dashboard
```

---

# 🛠 Tech Stack

### Programming

- Python

### Machine Learning

- Scikit-Learn
- XGBoost

### Graph Machine Learning

- PyTorch Geometric
- NetworkX

### Explainability

- SHAP

### Backend

- FastAPI
- Uvicorn

### Dashboard

- Streamlit

### Data Processing

- Pandas
- NumPy

### Visualization

- Matplotlib

---

# 📂 Project Structure

```
GraphShieldAI/
│
├── api/
├── dashboard/
├── src/
│
├── outputs/
│   ├── graphs/
│   ├── reports/
│   ├── explainability/
│   └── models/
│
├── main.py
├── requirements.txt
└── README.md
```

---

# 📊 Models Used

| Model | Purpose |
|--------|---------|
| Random Forest | Baseline Fraud Detection |
| XGBoost | Gradient Boosting Fraud Detection |
| GCN | Graph Neural Network |
| GraphSAGE | Inductive Graph Learning |

---

# 📈 Graph Analytics

The project computes:

- Degree
- In Degree
- Out Degree
- PageRank
- Betweenness Centrality
- Clustering Coefficient

These graph metrics are used to calculate customer risk scores.

---

# 🔍 Fraud Ring Detection

GraphShield AI identifies suspicious communities using graph connectivity.

Features:

- Weakly Connected Components
- Community Detection
- Fraud Ring Export
- Risk Ranking

---

# 🧠 Explainable AI

SHAP is used to explain predictions.

Generated outputs:

- SHAP Summary Plot
- Feature Importance Plot
- Feature Importance CSV

---

# 🌐 REST API

FastAPI provides REST endpoints.

### Swagger UI

```
http://127.0.0.1:8000/docs
```

### Prediction Endpoint

```
POST /predict
```

Example request

```json
{
    "type":"TRANSFER",
    "amount":50000,
    "oldbalanceOrg":100000,
    "newbalanceOrig":50000,
    "oldbalanceDest":0,
    "newbalanceDest":50000
}
```

Example response

```json
{
    "prediction":0,
    "fraud_probability":0.0003,
    "risk_level":"Low"
}
```

---

# 📺 Dashboard

The Streamlit dashboard includes:

- Dashboard Overview
- Model Comparison
- Graph Analytics
- Fraud Rings
- SHAP Explainability
- Live Prediction

Run

```bash
streamlit run dashboard/app.py
```

---

# ⚙️ Installation

Clone repository

```bash
git clone https://github.com/YOUR_USERNAME/GraphShieldAI.git
```

Go to project

```bash
cd GraphShieldAI
```

Install requirements

```bash
pip install -r requirements.txt
```

---

# ▶ Running the Project

### Train Models

```bash
python main.py
```

### Start FastAPI

```bash
uvicorn api.app:app --reload
```

### Start Streamlit

```bash
streamlit run dashboard/app.py
```

---

# 📁 Outputs

The project automatically generates:

- Graph Features
- Customer Risk Scores
- Fraud Rings Report
- Model Comparison Report
- SHAP Plots
- Network Graph
- Fraud Distribution Charts
- Trained Models

---

# 📚 Dataset

Dataset used:

**PaySim Mobile Money Transaction Dataset**

Source:

https://www.kaggle.com/datasets/ealaxi/paysim1

---

# 🚀 Future Improvements

- Graph Attention Networks (GAT)
- Temporal Graph Networks
- Neo4j Graph Database
- Docker Deployment
- Kubernetes Deployment
- AWS Cloud Deployment
- Real-Time Kafka Streaming
- Online Learning

---

# 👨‍💻 Author

**Akash Jadhav**

B.Sc. Computer Science

MIT World Peace University

Pune, India

---

# ⭐ If you like this project

Please consider giving this repository a ⭐ on GitHub.
