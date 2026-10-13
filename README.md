#  Customer Churn Prediction

> End-to-end machine learning project: predict telecom customer churn using XGBoost, SMOTE, and SHAP interpretability.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3+-orange?logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

##  Objective

Build a production-ready ML pipeline that identifies customers likely to churn, enabling proactive retention strategies.

##  Dataset

[Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) — IBM sample dataset (7 043 rows × 21 features).

##  Pipeline

| Step | Details |
|------|---------|
| **EDA** | Distribution analysis, correlation heatmap, churn rate by feature |
| **Feature Engineering** | Avg charges/month, tenure groups, service count |
| **Preprocessing** | One-hot encoding, StandardScaler, SMOTE oversampling |
| **Models** | Logistic Regression, Random Forest, Gradient Boosting, XGBoost |
| **Tuning** | GridSearchCV (5-fold stratified CV) |
| **Evaluation** | F1-score, ROC-AUC, confusion matrix, precision-recall curve |
| **Interpretability** | Feature importance + SHAP summary plot |

##  Quick Start

```bash
pip install numpy pandas matplotlib seaborn scikit-learn xgboost imbalanced-learn shap
jupyter notebook customer_churn_prediction.ipynb
```

##  Results

| Metric | Score |
|--------|-------|
| **F1-score** | ~0.62 |
| **ROC-AUC** | ~0.85 |
| **Top features** | Contract type, tenure, monthly charges |

##  Structure

```
├── customer_churn_prediction.ipynb   # Main notebook
├── README.md                         # This file
└── requirements.txt                  # Dependencies
```

##  License

MIT
