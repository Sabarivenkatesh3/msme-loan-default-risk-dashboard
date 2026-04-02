import pandas as pd
import numpy as np

# Load data
df = pd.read_csv('data/msme_loans.csv')

print("=== INITIAL INSPECTION ===")
print(df.shape)          # rows, columns
print(df.dtypes)         # data types
print(df.isnull().sum()) # missing values
print(df.describe())     # basic stats

# ---- 1. Fix data types ----
df['disbursal_date'] = pd.to_datetime(df['disbursal_date'])

# ---- 2. Handle missing values (if any) ----
# Numeric columns: fill with median
numeric_cols = ['credit_score', 'bounce_count_6m', 'dpd_current']
for col in numeric_cols:
    df[col] = df[col].fillna(df[col].median())

# Categorical: fill with mode
cat_cols = ['gst_filing', 'collateral_type', 'sector']
for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

# ---- 3. Remove duplicates ----
print(f"Duplicates: {df.duplicated('borrower_id').sum()}")
df = df.drop_duplicates(subset='borrower_id')

# ---- 4. Outlier treatment (cap at 99th percentile) ----
for col in ['loan_amount', 'monthly_income', 'emi_amount']:
    cap = df[col].quantile(0.99)
    df[col] = df[col].clip(upper=cap)

# ---- 5. Add derived/engineered features ----

# Loan age in months
df['loan_age_months'] = (
    (pd.Timestamp('2025-01-01') - df['disbursal_date']).dt.days / 30
).astype(int)

# Credit score bucket
df['credit_bucket'] = pd.cut(
    df['credit_score'],
    bins=[0, 620, 680, 740, 850],
    labels=['Poor', 'Fair', 'Good', 'Excellent']
)

# EMI stress flag
df['emi_stress_flag'] = (df['emi_to_income_ratio'] > 0.50).astype(int)

# High bounce flag
df['high_bounce_flag'] = (df['bounce_count_6m'] >= 3).astype(int)

# DPD bucket
df['dpd_bucket'] = pd.cut(
    df['dpd_current'],
    bins=[-1, 0, 30, 60, 90, 999],
    labels=['Current', 'DPD 1-30', 'DPD 31-60', 'DPD 61-90', 'DPD 90+']
)

# Loan size category
df['loan_size_cat'] = pd.cut(
    df['loan_amount'],
    bins=[0, 1000000, 3000000, 7000000, float('inf')],
    labels=['Small', 'Medium', 'Large', 'Enterprise']
)

# ---- 6. Validation checks ----
print("\n=== VALIDATION ===")
print(f"Rows: {len(df)}")
print(f"Default rate: {df['default'].mean():.1%}")
print(f"EMI stress: {df['emi_stress_flag'].mean():.1%}")
print(f"High bounce: {df['high_bounce_flag'].mean():.1%}")
print(df['loan_status'].value_counts())

# Save cleaned data
df.to_csv('data/msme_loans_clean.csv', index=False)
print("\nCleaned data saved!")