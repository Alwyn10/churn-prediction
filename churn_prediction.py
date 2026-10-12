# ============================================================
# Customer Churn Prediction — End-to-End ML Project
# Author: Alwin Anto
# ============================================================

# ── 1. IMPORTS ──────────────────────────────────────────────
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, roc_auc_score, roc_curve)
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("  CUSTOMER CHURN PREDICTION — ML PROJECT")
print("=" * 60)

# ── 2. LOAD DATA ─────────────────────────────────────────────
# Using the famous Telco Customer Churn dataset (IBM)
# In real project: df = pd.read_csv('telco_churn.csv')
# We simulate it here with realistic data

np.random.seed(42)
n = 1000

df = pd.DataFrame({
    'gender':           np.random.choice(['Male', 'Female'], n),
    'SeniorCitizen':    np.random.choice([0, 1], n, p=[0.84, 0.16]),
    'tenure':           np.random.randint(1, 72, n),
    'MonthlyCharges':   np.round(np.random.uniform(18, 118, n), 2),
    'TotalCharges':     np.round(np.random.uniform(18, 8500, n), 2),
    'Contract':         np.random.choice(['Month-to-month', 'One year', 'Two year'], n, p=[0.55, 0.25, 0.20]),
    'InternetService':  np.random.choice(['DSL', 'Fiber optic', 'No'], n, p=[0.34, 0.44, 0.22]),
    'PaymentMethod':    np.random.choice(['Electronic check', 'Mailed check', 'Bank transfer', 'Credit card'], n),
    'TechSupport':      np.random.choice(['Yes', 'No', 'No internet service'], n, p=[0.29, 0.49, 0.22]),
    'OnlineSecurity':   np.random.choice(['Yes', 'No', 'No internet service'], n, p=[0.28, 0.50, 0.22]),
})
# Churn: more likely for month-to-month, high charges, low tenure
churn_prob = (
    0.05
    + 0.30 * (df['Contract'] == 'Month-to-month')
    + 0.10 * (df['MonthlyCharges'] > 70)
    - 0.15 * (df['tenure'] > 36)
    + 0.08 * (df['InternetService'] == 'Fiber optic')
)
churn_prob = churn_prob.clip(0.03, 0.85)
df['Churn'] = (np.random.rand(n) < churn_prob).astype(int)

print(f"\n📊 Dataset loaded: {df.shape[0]} rows × {df.shape[1]} columns")
print(f"   Churn rate: {df['Churn'].mean():.1%}")

# ── 3. EDA ────────────────────────────────────────────────────
print("\n── EDA ────────────────────────────────────────────────")
print(df[['tenure', 'MonthlyCharges', 'TotalCharges']].describe().round(2))
print(f"\nMissing values:\n{df.isnull().sum()}")

fig, axes = plt.subplots(2, 3, figsize=(15, 9))
fig.suptitle('Customer Churn — Exploratory Data Analysis', fontsize=14, fontweight='bold')

# Churn distribution
df['Churn'].value_counts().plot(kind='bar', ax=axes[0,0], color=['steelblue','tomato'])
axes[0,0].set_title('Churn Distribution')
axes[0,0].set_xticklabels(['No Churn', 'Churn'], rotation=0)
axes[0,0].set_ylabel('Count')

# Tenure vs Churn
df.groupby('Churn')['tenure'].plot(kind='hist', ax=axes[0,1], alpha=0.6, bins=20, legend=True)
axes[0,1].set_title('Tenure by Churn')
axes[0,1].set_xlabel('Tenure (months)')
axes[0,1].legend(['No Churn', 'Churn'])

# Monthly Charges vs Churn
sns.boxplot(x='Churn', y='MonthlyCharges', data=df, ax=axes[0,2], palette=['steelblue','tomato'])
axes[0,2].set_title('Monthly Charges by Churn')
axes[0,2].set_xticklabels(['No Churn', 'Churn'])

# Contract Type
contract_churn = df.groupby('Contract')['Churn'].mean().sort_values(ascending=False)
contract_churn.plot(kind='bar', ax=axes[1,0], color='coral')
axes[1,0].set_title('Churn Rate by Contract Type')
axes[1,0].set_ylabel('Churn Rate')
axes[1,0].tick_params(axis='x', rotation=15)

# Internet Service
internet_churn = df.groupby('InternetService')['Churn'].mean()
internet_churn.plot(kind='bar', ax=axes[1,1], color='steelblue')
axes[1,1].set_title('Churn Rate by Internet Service')
axes[1,1].tick_params(axis='x', rotation=15)

# Correlation heatmap (numeric only)
num_df = df[['tenure', 'MonthlyCharges', 'TotalCharges', 'SeniorCitizen', 'Churn']]
sns.heatmap(num_df.corr(), annot=True, fmt='.2f', cmap='coolwarm', ax=axes[1,2])
axes[1,2].set_title('Correlation Heatmap')

plt.tight_layout()
plt.savefig('/home/claude/churn_project/eda_plots.png', dpi=120, bbox_inches='tight')
plt.close()
print("   ✓ EDA plots saved")

# ── 4. PREPROCESSING ─────────────────────────────────────────
print("\n── Preprocessing ──────────────────────────────────────")
df_model = df.copy()

# Encode categorical columns
cat_cols = ['gender', 'Contract', 'InternetService', 'PaymentMethod', 'TechSupport', 'OnlineSecurity']
le = LabelEncoder()
for col in cat_cols:
    df_model[col] = le.fit_transform(df_model[col])

X = df_model.drop('Churn', axis=1)
y = df_model['Churn']

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train / test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y)

print(f"   Train size: {X_train.shape[0]} | Test size: {X_test.shape[0]}")

# ── 5. MODEL TRAINING ─────────────────────────────────────────
print("\n── Model Training ─────────────────────────────────────")

models = {
    'Logistic Regression': LogisticRegression(random_state=42),
    'Random Forest':       RandomForestClassifier(n_estimators=100, random_state=42)
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    preds  = model.predict(X_test)
    proba  = model.predict_proba(X_test)[:, 1]
    acc    = accuracy_score(y_test, preds)
    auc    = roc_auc_score(y_test, proba)
    cv     = cross_val_score(model, X_scaled, y, cv=5, scoring='accuracy').mean()
    results[name] = {'model': model, 'preds': preds, 'proba': proba,
                     'accuracy': acc, 'auc': auc, 'cv': cv}
    print(f"\n   {name}")
    print(f"     Accuracy : {acc:.3f}")
    print(f"     ROC-AUC  : {auc:.3f}")
    print(f"     CV Score : {cv:.3f}")

# ── 6. EVALUATION PLOTS ───────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle('Model Evaluation', fontsize=14, fontweight='bold')

# Confusion matrix — best model (Random Forest)
rf = results['Random Forest']
cm = confusion_matrix(y_test, rf['preds'])
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
            xticklabels=['No Churn','Churn'], yticklabels=['No Churn','Churn'])
axes[0].set_title('Confusion Matrix — Random Forest')
axes[0].set_ylabel('Actual')
axes[0].set_xlabel('Predicted')

# ROC Curves
for name, res in results.items():
    fpr, tpr, _ = roc_curve(y_test, res['proba'])
    axes[1].plot(fpr, tpr, label=f"{name} (AUC={res['auc']:.2f})")
axes[1].plot([0,1],[0,1],'k--')
axes[1].set_title('ROC Curve')
axes[1].set_xlabel('False Positive Rate')
axes[1].set_ylabel('True Positive Rate')
axes[1].legend()

# Feature Importance
rf_model = results['Random Forest']['model']
feat_imp = pd.Series(rf_model.feature_importances_, index=X.columns).sort_values(ascending=True)
feat_imp.tail(8).plot(kind='barh', ax=axes[2], color='steelblue')
axes[2].set_title('Top Feature Importances')

plt.tight_layout()
plt.savefig('/home/claude/churn_project/model_evaluation.png', dpi=120, bbox_inches='tight')
plt.close()
print("\n   ✓ Evaluation plots saved")

# ── 7. CLASSIFICATION REPORT ──────────────────────────────────
print("\n── Classification Report (Random Forest) ──────────────")
print(classification_report(y_test, rf['preds'], target_names=['No Churn', 'Churn']))

print("\n✅ Project complete! Files saved:")
print("   churn_prediction.py  — full source code")
print("   eda_plots.png        — EDA visualizations")
print("   model_evaluation.png — model performance plots")
