import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as mtick
import warnings
warnings.filterwarnings('ignore')

# Style
plt.rcParams.update({
    'figure.facecolor': '#0d1117',
    'axes.facecolor'  : '#161b22',
    'text.color'      : '#e6edf3',
    'axes.labelcolor' : '#8b949e',
    'xtick.color'     : '#8b949e',
    'ytick.color'     : '#8b949e',
    'axes.edgecolor'  : '#30363d',
    'grid.color'      : '#21262d',
    'font.family'     : 'DejaVu Sans'
})
ORANGE = '#f0883e'
BLUE   = '#58a6ff'
GREEN  = '#3fb950'
RED    = '#ff7b72'
PURPLE = '#bc8cff'

df = pd.read_csv('../data/msme_loans_clean.csv')

fig, axes = plt.subplots(2, 4, figsize=(20, 10))
plt.suptitle('MSME Loan Portfolio — Risk Analytics EDA',
             fontsize=16, fontweight='bold', color='#e6edf3', y=1.01)

# --- Chart 1: Default Rate by Sector ---
ax1 = axes[0,0]
sec_dr = df.groupby('sector')['default'].mean().sort_values(ascending=True)*100
colors = [RED if v > 20 else ORANGE if v > 15 else GREEN
          for v in sec_dr]
ax1.barh(sec_dr.index, sec_dr.values, color=colors)
ax1.set_title('Default Rate by Sector', color='#e6edf3', fontweight='bold')
ax1.set_xlabel('Default Rate %')
ax1.axvline(15, color=ORANGE, linestyle='--', alpha=0.6, label='Threshold 15%')
ax1.legend(fontsize=8)

# --- Chart 2: Credit Score Distribution by Default ---
ax2 = axes[0,1]
df[df['default']==0]['credit_score'].hist(
    ax=ax2, bins=30, color=GREEN, alpha=0.6, label='No Default')
df[df['default']==1]['credit_score'].hist(
    ax=ax2, bins=30, color=RED, alpha=0.6, label='Default')
ax2.set_title('Credit Score: Default vs Non-Default', color='#e6edf3', fontweight='bold')
ax2.set_xlabel('Credit Score')
ax2.legend()

# --- Chart 3: EMI-to-Income Ratio vs Default Rate ---
ax3 = axes[0,2]
emi_bins = pd.cut(df['emi_to_income_ratio'],
                   bins=[0,0.2,0.3,0.4,0.5,0.6,0.7,1.0],
                   labels=['0-20%','20-30%','30-40%','40-50%','50-60%','60-70%','70%+'])
emi_dr = df.groupby(emi_bins)['default'].mean() * 100
ax3.bar(range(len(emi_dr)), emi_dr.values,
        color=[RED if v>20 else ORANGE for v in emi_dr])
ax3.set_xticks(range(len(emi_dr)))
ax3.set_xticklabels(emi_dr.index, rotation=35, fontsize=8)
ax3.set_title('Default Rate by EMI/Income Ratio', color='#e6edf3', fontweight='bold')
ax3.set_ylabel('Default Rate %')

# --- Chart 4: Bounce Count vs Default Rate ---
ax4 = axes[0,3]
bounce_dr = df.groupby('bounce_count_6m')['default'].mean() * 100
ax4.plot(bounce_dr.index, bounce_dr.values,
         marker='o', color=ORANGE, linewidth=2, markersize=7)
ax4.fill_between(bounce_dr.index, bounce_dr.values, alpha=0.15, color=ORANGE)
ax4.set_title('Default Rate vs EMI Bounce Count', color='#e6edf3', fontweight='bold')
ax4.set_xlabel('# EMI Bounces (last 6 months)')
ax4.set_ylabel('Default Rate %')

# --- Chart 5: Risk Category Donut ---
ax5 = axes[1,0]
risk_counts = df['risk_category'].value_counts()
rc_colors   = {'Green':GREEN, 'Amber':ORANGE, 'Red':RED, 'Critical':'#ff4444'}
pie_colors  = [rc_colors.get(k, BLUE) for k in risk_counts.index]
wedges, texts, autotexts = ax5.pie(
    risk_counts, labels=risk_counts.index, colors=pie_colors,
    autopct='%1.1f%%', startangle=90,
    wedgeprops=dict(width=0.5), textprops={'color':'#e6edf3', 'fontsize':9})
ax5.set_title('Portfolio by Risk Category', color='#e6edf3', fontweight='bold')

# --- Chart 6: State-wise Default Rate ---
ax6 = axes[1,1]
state_dr = (df.groupby('state')['default'].mean() * 100).sort_values(ascending=False)
ax6.bar(range(len(state_dr)), state_dr.values,
        color=[RED if v>20 else ORANGE if v>15 else GREEN
               for v in state_dr])
ax6.set_xticks(range(len(state_dr)))
ax6.set_xticklabels(state_dr.index, rotation=45, ha='right', fontsize=8)
ax6.set_title('Default Rate by State', color='#e6edf3', fontweight='bold')

# --- Chart 7: Correlation Heatmap ---
ax7 = axes[1,2]
corr_cols = ['credit_score','bounce_count_6m','dpd_current',
             'emi_to_income_ratio','prior_defaults','default']
corr = df[corr_cols].corr()
sns.heatmap(corr, ax=ax7, cmap='RdYlGn', center=0,
            annot=True, fmt='.2f', annot_kws={'size':8},
            xticklabels=['CrScore','Bounce','DPD','EMI/Inc','PriorDef','Default'],
            yticklabels=['CrScore','Bounce','DPD','EMI/Inc','PriorDef','Default'])
ax7.set_title('Feature Correlation Matrix', color='#e6edf3', fontweight='bold')

# --- Chart 8: Collateral Type Default Rate ---
ax8 = axes[1,3]
coll_dr = (df.groupby('collateral_type')['default'].mean() * 100).sort_values(ascending=False)
ax8.bar(coll_dr.index, coll_dr.values,
        color=[RED if v>20 else ORANGE if v>15 else GREEN
               for v in coll_dr])
ax8.set_title('Default Rate by Collateral Type', color='#e6edf3', fontweight='bold')
ax8.set_ylabel('Default Rate %')
ax8.tick_params(axis='x', rotation=25)

plt.tight_layout()
plt.savefig('../outputs/eda_charts.png', dpi=150, bbox_inches='tight',
            facecolor='#0d1117')
plt.show()
print("EDA charts saved to outputs/eda_charts.png")