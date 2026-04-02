import pandas as pd

df = pd.read_csv('../data/msme_loans_scored.csv')

# Get Critical borrowers only
high_risk = df[df['predicted_risk'] == 'Critical'].copy()
high_risk = high_risk.sort_values('pd_score', ascending=False).head(30)

def rule_based_narrative(row):
    # Identify risk drivers
    drivers = []
    if row['bounce_count_6m'] >= 3:
        drivers.append(f"{int(row['bounce_count_6m'])} EMI bounces in last 6 months")
    if row['credit_score'] < 650:
        drivers.append(f"low credit score of {int(row['credit_score'])}")
    if row['gst_filing'] == 'Non-Filer':
        drivers.append("GST non-compliance")
    if row['dpd_current'] > 30:
        drivers.append(f"{int(row['dpd_current'])} days past due")
    if row['prior_defaults'] > 0:
        drivers.append(f"{int(row['prior_defaults'])} prior default(s) on record")

    driver_str = (", ".join(drivers)
                  if drivers else "elevated sector and portfolio risk")

    # Sentence 1 — Risk summary
    s1 = (f"Borrower {row['borrower_id']} operating in the "
          f"{row['sector']} sector ({row['state']}) shows critical "
          f"credit stress driven by {driver_str}.")

    # Sentence 2 — PD score context
    s2 = (f"The ML-based Probability of Default score is "
          f"{row['pd_score']*100:.1f}%, placing this account in the "
          f"Critical risk tier with ₹{row['loan_amount']/100000:.1f}L "
          f"exposure backed by {row['collateral_type']} collateral.")

    # Sentence 3 — Recommended action
    if row['pd_score'] >= 0.85:
        action = ("Immediate escalation required — assign senior "
                  "relationship manager, initiate restructuring discussion, "
                  "and verify collateral valuation within 7 days.")
    elif row['pd_score'] >= 0.75:
        action = ("Proactive contact within 3 days recommended — offer "
                  "EMI deferment or restructuring to prevent NPA formation.")
    else:
        action = ("Place on monthly watchlist — monitor next 2 EMI "
                  "payments closely and review GST filing status.")

    return f"{s1} {s2} Recommended Action: {action}"

# Generate narratives
print(f"Generating narratives for {len(high_risk)} critical borrowers...")
high_risk['risk_narrative'] = high_risk.apply(rule_based_narrative, axis=1)

# Preview 3 samples
for _, row in high_risk.head(3).iterrows():
    print(f"\n{'='*60}")
    print(f"Borrower : {row['borrower_id']}")
    print(f"Sector   : {row['sector']}")
    print(f"PD Score : {row['pd_score']*100:.1f}%")
    print(f"Narrative: {row['risk_narrative']}")

# Save to outputs folder
high_risk.to_csv('../outputs/critical_borrowers_with_narratives.csv',
                 index=False)
print(f"\nFile saved to outputs/critical_borrowers_with_narratives.csv")
print(f"Total records: {len(high_risk)}")
