import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (roc_auc_score, classification_report,
                             confusion_matrix, roc_curve)
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ---- Load Data (FIXED PATH) ----
df = pd.read_csv('../data/msme_loans_clean.csv')

# ---- Feature Selection ----
feature_cols = [
    'loan_amount', 'interest_rate', 'tenure_months',
    'emi_to_income_ratio', 'credit_score',
    'bounce_count_6m', 'dpd_current', 'prior_defaults',
    'loan_age_months', 'emi_stress_flag', 'high_bounce_flag',
    'sector', 'state', 'gst_filing', 'collateral_type'
]

X = df[feature_cols].copy()
y = df['default']

# ---- Encode categorical columns ----
cat_cols = ['sector', 'state', 'gst_filing', 'collateral_type']
le = LabelEncoder()
for col in cat_cols:
    X[col] = le.fit_transform(X[col].astype(str))

# ---- Train/Test Split ----
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ---- Train XGBoost Model ----
model = XGBClassifier(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=3,
    eval_metric='auc',
    random_state=42,
    verbosity=0
)

model.fit(X_train, y_train,
          eval_set=[(X_test, y_test)],
          verbose=False)

# ---- Evaluate ----
y_pred_proba = model.predict_proba(X_test)[:, 1]
y_pred = (y_pred_proba > 0.35).astype(int)

print("=== MODEL PERFORMANCE ===")
print(f"AUC-ROC: {roc_auc_score(y_test, y_pred_proba):.4f}")
print(classification_report(y_test, y_pred))

# ---- Feature Importance (REPLACED SHAP) ----
feat_imp = pd.Series(model.feature_importances_, index=X.columns)
feat_imp = feat_imp.sort_values(ascending=False)

plt.figure(figsize=(10, 6))
feat_imp.head(10).plot(kind='bar')
plt.title('Top 10 Feature Importance')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('../outputs/feature_importance.png', dpi=120)
print("Feature importance chart saved.")

# ---- ROC Curve ----
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
auc = roc_auc_score(y_test, y_pred_proba)

plt.figure(figsize=(7, 5))
plt.plot(fpr, tpr, lw=2, label=f'XGBoost AUC = {auc:.3f}')
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve — MSME Default Prediction')
plt.legend()
plt.tight_layout()
plt.savefig('../outputs/roc_curve.png', dpi=120)
print("ROC curve saved.")

# ---- Generate PD Scores ----
df['pd_score'] = model.predict_proba(X)[:, 1].round(4)

df['predicted_risk'] = pd.cut(
    df['pd_score'],
    bins=[0, 0.15, 0.35, 0.60, 1.0],
    labels=['Green', 'Amber', 'Red', 'Critical']
)

# ---- Save scored dataset ----
df.to_csv('../data/msme_loans_scored.csv', index=False)

print("\nScored dataset saved: msme_loans_scored.csv")
print(df[['borrower_id', 'pd_score', 'predicted_risk']].head(10))