import pandas as pd
import numpy as np
from faker import Faker
import random
from datetime import datetime, timedelta

fake = Faker('en_IN')
np.random.seed(42)
random.seed(42)

n = 5000

# --- Reference Lists ---
states = ['Maharashtra', 'Gujarat', 'Tamil Nadu', 'Rajasthan',
          'Uttar Pradesh', 'Karnataka', 'Punjab', 'Telangana',
          'West Bengal', 'Madhya Pradesh', 'Delhi', 'Haryana']

sectors = ['Textile', 'Auto Parts', 'Agri Processing', 'Food & Beverage',
           'Construction', 'Retail Trade', 'Chemicals', 'Electronics']

# Sectors with historically higher default rates
sector_risk = {
    'Textile': 0.22, 'Auto Parts': 0.14, 'Agri Processing': 0.18,
    'Food & Beverage': 0.10, 'Construction': 0.25,
    'Retail Trade': 0.12, 'Chemicals': 0.16, 'Electronics': 0.08
}

loan_officers = [f'LO_{i:03d}' for i in range(1, 41)]
branches      = [f'BR_{i:02d}' for i in range(1, 21)]

# --- Disbursement Dates (last 3 years) ---
base_date = datetime(2022, 1, 1)
disbursal_dates = [base_date + timedelta(days=random.randint(0, 900))
                   for _ in range(n)]

# --- Core Features ---
sector_list   = np.random.choice(sectors, n)
loan_amount   = np.random.choice(
    [500000, 750000, 1000000, 1500000, 2000000,
     3000000, 5000000, 7500000, 10000000], n)

interest_rate = np.random.uniform(11.5, 18.5, n).round(2)
tenure_months = np.random.choice([12, 18, 24, 36, 48, 60], n)
monthly_income = (loan_amount * np.random.uniform(0.08, 0.35, n)).astype(int)
emi_amount    = ((loan_amount * (interest_rate/1200)) /
                  (1 - (1 + interest_rate/1200)**(-tenure_months))).astype(int)

emi_to_income = (emi_amount / monthly_income).round(3)
credit_score  = np.random.randint(580, 820, n)
bounce_count  = np.random.choice([0,0,0,1,1,2,3,4,5,6], n)
dpd_current   = np.random.choice(
    [0,0,0,0,0,15,30,45,60,75,90,120], n)
gst_filing_reg = np.random.choice(['Regular','Irregular','Non-Filer'], n,
                                    p=[0.60,0.28,0.12])
collateral_type = np.random.choice(
    ['Property','Machinery','Inventory','Gold','Unsecured'], n,
    p=[0.35,0.25,0.15,0.12,0.13])
prior_defaults = np.random.choice([0,0,0,1,2], n)

# --- Target Variable: Default (1 = defaulted) ---
# Probability influenced by multiple risk factors
base_prob = np.array([sector_risk[s] for s in sector_list])
base_prob += (bounce_count * 0.05)
base_prob += (dpd_current > 30).astype(float) * 0.25
base_prob += (emi_to_income > 0.50).astype(float) * 0.12
base_prob += (credit_score < 650).astype(float) * 0.10
base_prob += (gst_filing_reg == 'Non-Filer').astype(float) * 0.08
base_prob += (prior_defaults > 0).astype(float) * 0.15
base_prob  = np.clip(base_prob, 0, 0.95)
default    = (np.random.random(n) < base_prob).astype(int)

# --- Risk Category ---
def risk_cat(prob):
    if prob < 0.15:   return 'Green'
    elif prob < 0.30: return 'Amber'
    elif prob < 0.50: return 'Red'
    else:             return 'Critical'

risk_category = [risk_cat(p) for p in base_prob]

# --- Loan Status ---
status_map = lambda d, dpd: (
    'NPA'      if dpd >= 90  else
    'Stressed' if dpd >= 30  else
    'Defaulted'if d  == 1   else
    'Active'
)
loan_status = [status_map(d, dpd)
               for d, dpd in zip(default, dpd_current)]

# --- Build DataFrame ---
df = pd.DataFrame({
    'borrower_id'     : [f'BRW_{i:05d}' for i in range(1, n+1)],
    'borrower_name'   : [fake.name() for _ in range(n)],
    'state'           : np.random.choice(states, n),
    'sector'          : sector_list,
    'branch_id'       : np.random.choice(branches, n),
    'loan_officer_id' : np.random.choice(loan_officers, n),
    'disbursal_date'  : [d.strftime('%Y-%m-%d') for d in disbursal_dates],
    'loan_amount'     : loan_amount,
    'interest_rate'   : interest_rate,
    'tenure_months'   : tenure_months,
    'monthly_income'  : monthly_income,
    'emi_amount'      : emi_amount,
    'emi_to_income_ratio' : emi_to_income,
    'credit_score'    : credit_score,
    'bounce_count_6m' : bounce_count,
    'dpd_current'     : dpd_current,
    'gst_filing'      : gst_filing_reg,
    'collateral_type' : collateral_type,
    'prior_defaults'  : prior_defaults,
    'risk_category'   : risk_category,
    'loan_status'     : loan_status,
    'default'         : default
})

# Save files
df.to_csv('msme_loans.csv', index=False)
print(f"Dataset created: {len(df)} rows, {df.columns.tolist()}")
print(f"Default rate: {df['default'].mean():.1%}")
print(f"NPA count: {(df['loan_status']=='NPA').sum()}")