# Customer Churn Prediction

An end-to-end machine learning project to predict whether a telecom customer will churn (leave the service).

## Problem Statement
Customer churn is a critical business metric. This project builds a classification model to identify at-risk customers before they leave, enabling proactive retention strategies.

## Dataset
- 1,000 customer records with 10 features
- Features include: tenure, monthly charges, contract type, internet service, payment method
- Target: Churn (Yes/No) — 24.8% churn rate

## Project Structure
```
churn_prediction.py   ← Full pipeline (EDA + preprocessing + modeling + evaluation)
eda_plots.png         ← Exploratory Data Analysis visualizations
model_evaluation.png  ← Confusion matrix, ROC curve, feature importance
README.md             ← This file
```

## Approach
1. **EDA** — Analyzed churn patterns by contract type, tenure, monthly charges
2. **Preprocessing** — Label encoding for categoricals, StandardScaler for numeric features
3. **Modeling** — Compared Logistic Regression vs Random Forest
4. **Evaluation** — Accuracy, ROC-AUC, confusion matrix, cross-validation

## Key Findings
- Month-to-month contracts have 3x higher churn rate than two-year contracts
- Monthly charges and tenure are the top predictors of churn
- Fiber optic internet customers churn more than DSL customers

## Results
| Model | Accuracy | ROC-AUC | CV Score |
|---|---|---|---|
| Logistic Regression | 74.0% | 0.69 | 74.5% |
| Random Forest | 74.0% | 0.65 | 71.9% |

## Tech Stack
`Python` `Pandas` `NumPy` `Scikit-learn` `Matplotlib` `Seaborn`

## How to Run
```bash
pip install pandas numpy scikit-learn matplotlib seaborn
python churn_prediction.py
```

## Author
Alwin Anto — [LinkedIn](https://www.linkedin.com/in/alwin-anto-a333a9266/) | [GitHub](https://github.com/Alwyn10)
